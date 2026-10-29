"""Generative chemistry: the de-novo design rung (our Chemistry42 analog).

A graph genetic algorithm over RDKit molecules (GB-GA style, Jensen 2019) —
no trained generative network, no GPU, no paid API. It mutates and recombines
real molecules and keeps whatever a multi-objective reward favours, so it runs
anywhere RDKit does, including Apple Silicon.

Pipeline, matched to the rest of ddd:
- **Seeds come from the target.** Given a gene symbol we pull its ChEMBL
  actives as the starting population, so generated molecules explore the
  neighbourhood of things already known to bind. Seeds can also be supplied
  directly, or we fall back to a small drug-like starter set.
- **Reward reuses what we already score with.** QED (drug-likeness), a
  synthetic-accessibility estimate, Lipinski behaviour and Tanimoto similarity
  to the seed actives — the same chemistry layer the repurposing playbooks use.
  Docking (Vina) is optional and only run on the final elite, since it is slow.
- **Honest caveats.** This is reward-guided search over public chemistry, not a
  validated design engine. Novelty is measured as distance from the seeds;
  "drug-like" is QED/Ro5, not a synthesizability guarantee.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def embed3d(smiles: str, seed: int = 1, optimize: bool = True) -> dict:
    """Turn a flat SMILES into a real 3D conformer: ETKDGv3 distance-geometry
    embedding + MMFF94 minimisation (UFF fallback). Returns an MDL mol block
    (with explicit H) ready for a 3D viewer, plus the force-field energy.

    This is the geometry rung of generative design — the molecule's actual
    shape, not a 2D depiction. Conformer energy is a relative stability signal
    within one molecule, not a cross-molecule binding score.
    """
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return {"ok": False, "error": "invalid SMILES", "smiles": smiles}
    mh = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    cid = AllChem.EmbedMolecule(mh, params)
    if cid < 0:  # retry with random coords for strained systems
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mh, params)
    if cid < 0:
        return {"ok": False, "error": "could not embed a 3D conformer", "smiles": smiles}
    energy, ff_used = None, None
    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mh):
                AllChem.MMFFOptimizeMolecule(mh)
                props = AllChem.MMFFGetMoleculeProperties(mh)
                ff = AllChem.MMFFGetMoleculeForceField(mh, props)
                energy, ff_used = (round(ff.CalcEnergy(), 2) if ff else None), "MMFF94"
            else:
                AllChem.UFFOptimizeMolecule(mh)
                ff = AllChem.UFFGetMoleculeForceField(mh)
                energy, ff_used = (round(ff.CalcEnergy(), 2) if ff else None), "UFF"
        except Exception:
            ff_used = None
    return {
        "ok": True,
        "smiles": Chem.MolToSmiles(m),
        "molblock": Chem.MolToMolBlock(mh),
        "energy": energy,
        "forcefield": ff_used,
        "atoms": mh.GetNumAtoms(),
        "heavy_atoms": m.GetNumHeavyAtoms(),
    }

RDLogger.DisableLog("rdApp.*")

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Elements the GA is allowed to introduce / keep (drug-like organic space).
_ORGANIC = {1, 6, 7, 8, 9, 15, 16, 17, 35}  # H C N O F P S Cl Br
_ADD_ATOMS = [6, 6, 6, 7, 8, 9, 16, 17]       # bias toward carbon
_BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

# A tiny drug-like starter set, used only when no target/seeds are given.
_STARTERS = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCNCC1", "c1cc[nH]c1",
    "O=C(N)c1ccccc1", "CC(=O)Nc1ccccc1", "c1ccc(cc1)S(=O)(=O)N",
    "c1ccc2[nH]ccc2c1", "O=C1CCCCC1", "c1ccc(nc1)N", "C1COCCN1",
]

# --- synthetic accessibility -------------------------------------------------
# Prefer RDKit Contrib sascorer if present; otherwise a cheap size/complexity
# heuristic normalised to 0..1 (1 = easy to make).
_sascorer = None
try:  # pragma: no cover - availability depends on install
    from rdkit.Chem import RDConfig  # noqa
    import os, sys
    _sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if _sa_path not in sys.path:
        sys.path.append(_sa_path)
    import sascorer as _sascorer  # type: ignore
except Exception:
    _sascorer = None


def _sa_norm(m) -> float:
    """0..1, higher = more synthesizable."""
    if _sascorer is not None:
        try:
            raw = _sascorer.calculateScore(m)  # 1 (easy) .. 10 (hard)
            return max(0.0, min(1.0, (10.0 - raw) / 9.0))
        except Exception:
            pass
    # Heuristic fallback: penalise size, spiro/bridgehead, many rings, stereo.
    try:
        n = m.GetNumHeavyAtoms()
        rings = rdMolDescriptors.CalcNumRings(m)
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(m)
        bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
        stereo = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
        pen = 0.015 * max(0, n - 20) + 0.10 * max(0, rings - 3) + 0.15 * spiro + 0.15 * bridge + 0.05 * stereo
        return max(0.0, min(1.0, 1.0 - pen))
    except Exception:
        return 0.5


# --- helpers -----------------------------------------------------------------
def _canon(m) -> Optional[str]:
    try:
        s = Chem.MolToSmiles(m)
        return s if s else None
    except Exception:
        return None


def _valid(m, mw_cap: float) -> bool:
    if m is None:
        return False
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False
    for a in m.GetAtoms():
        if a.GetAtomicNum() not in _ORGANIC:
            return False
    if m.GetNumHeavyAtoms() < 5 or m.GetNumHeavyAtoms() > 60:
        return False
    if Descriptors.MolWt(m) > mw_cap:
        return False
    # reject disconnected fragments
    if len(Chem.GetMolFrags(m)) != 1:
        return False
    return True


def _fp(m):
    return _GEN.GetFingerprint(m)


def _nearest_sim(m, seed_fps) -> float:
    if not seed_fps:
        return 0.0
    fp = _fp(m)
    return max(DataStructs.TanimotoSimilarity(fp, s) for s in seed_fps)


# --- GA operators (GB-GA style) ----------------------------------------------
def _mutate(m, mw_cap):
    rw = Chem.RWMol(m)
    atoms = rw.GetNumAtoms()
    if atoms == 0:
        return None
    op = random.random()
    try:
        if op < 0.30:  # append an atom to a random atom with spare valence
            idx = random.randrange(atoms)
            a = rw.GetAtomWithIdx(idx)
            if a.GetImplicitValence() + a.GetNumExplicitHs() > 0:
                new = rw.AddAtom(Chem.Atom(random.choice(_ADD_ATOMS)))
                rw.AddBond(idx, new, Chem.BondType.SINGLE)
        elif op < 0.50:  # change an atom's element
            idx = random.randrange(atoms)
            a = rw.GetAtomWithIdx(idx)
            if not a.GetIsAromatic():
                a.SetAtomicNum(random.choice(_ADD_ATOMS))
        elif op < 0.68:  # change a bond order
            if rw.GetNumBonds():
                b = rw.GetBondWithIdx(random.randrange(rw.GetNumBonds()))
                if not b.GetIsAromatic():
                    b.SetBondType(random.choice(_BONDS))
        elif op < 0.84:  # delete a terminal atom
            terminals = [a.GetIdx() for a in rw.GetAtoms() if a.GetDegree() == 1]
            if terminals:
                rw.RemoveAtom(random.choice(terminals))
        else:  # add a ring-closing bond between two near atoms
            if atoms >= 4:
                i, j = random.sample(range(atoms), 2)
                if not rw.GetBondBetweenAtoms(i, j):
                    ai, aj = rw.GetAtomWithIdx(i), rw.GetAtomWithIdx(j)
                    if (ai.GetImplicitValence() > 0 and aj.GetImplicitValence() > 0):
                        rw.AddBond(i, j, Chem.BondType.SINGLE)
        cand = rw.GetMol()
        if _valid(cand, mw_cap):
            return cand
    except Exception:
        return None
    return None


def _crossover(m1, m2, mw_cap):
    """Cut one non-ring single bond in each parent, swap and reconnect."""
    try:
        frags = []
        for m in (m1, m2):
            bonds = [b.GetIdx() for b in m.GetBonds()
                     if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
            if not bonds:
                return None
            bidx = random.choice(bonds)
            frag = Chem.FragmentOnBonds(m, [bidx], addDummies=True)
            parts = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
            if len(parts) < 2:
                return None
            frags.append(random.choice(parts))
        # join the two fragments at their dummy attachment points
        combo = Chem.CombineMols(frags[0], frags[1])
        rw = Chem.RWMol(combo)
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) < 2:
            return None
        d1, d2 = dummies[0], dummies[1]
        n1 = [nb.GetIdx() for nb in rw.GetAtomWithIdx(d1).GetNeighbors()]
        n2 = [nb.GetIdx() for nb in rw.GetAtomWithIdx(d2).GetNeighbors()]
        if not n1 or not n2:
            return None
        rw.AddBond(n1[0], n2[0], Chem.BondType.SINGLE)
        for d in sorted([d1, d2], reverse=True):
            rw.RemoveAtom(d)
        cand = rw.GetMol()
        # strip any remaining dummies
        cand = Chem.DeleteSubstructs(cand, Chem.MolFromSmarts("[#0]"))
        if _valid(cand, mw_cap):
            return cand
    except Exception:
        return None
    return None


# --- reward ------------------------------------------------------------------
def _reward(m, seed_fps, weights, mw_cap):
    qed = QED.qed(m)
    sa = _sa_norm(m)
    sim = _nearest_sim(m, seed_fps)
    # Ro5 behaviour
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    hbd = rdMolDescriptors.CalcNumHBD(m)
    hba = rdMolDescriptors.CalcNumHBA(m)
    viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    ro5 = 1.0 - 0.25 * viol
    # novelty: reward being *near but not identical* to a known active
    nov = 1.0 if sim < 0.99 else 0.0
    score = (weights["qed"] * qed + weights["sa"] * sa +
             weights["sim"] * sim + weights["ro5"] * ro5 + weights["nov"] * nov)
    return score, {"qed": round(qed, 3), "sa": round(sa, 3), "sim_to_seed": round(sim, 3),
                   "ro5_violations": viol, "mol_weight": round(mw, 1), "logp": round(logp, 2)}


_DEFAULT_WEIGHTS = {"qed": 1.0, "sa": 0.6, "sim": 0.7, "ro5": 0.4, "nov": 0.3}


# --- seeds -------------------------------------------------------------------
def _seeds_for_target(gene, organism=None, limit=40):
    """ChEMBL actives of a target as starting molecules. Returns (smiles, meta)."""
    try:
        from .sources import chembl
    except Exception:
        return [], "chembl module unavailable"
    try:
        res = chembl.target_actives(gene, limit=limit, organism=organism)
        acts = res.get("actives", []) or []
        smis = [a["smiles"] for a in acts if a.get("smiles")]
        return smis, f"{len(smis)} ChEMBL actives of {gene}"
    except Exception as e:
        return [], f"ChEMBL unavailable: {str(e)[:80]}"


# --- public API --------------------------------------------------------------
def generate(target: str = "", seeds: Optional[list] = None, organism: Optional[str] = None,
             n: int = 12, generations: int = 20, pop: int = 60,
             mw_cap: float = 600.0, weights: Optional[dict] = None,
             seed_rng: int = 0, dock: bool = False, progress=None) -> dict:
    """Design novel drug-like molecules in the neighbourhood of a target's actives.

    target     : gene symbol whose ChEMBL actives seed the search (optional).
    seeds      : explicit seed SMILES (overrides/augments target actives).
    n          : how many final candidates to return.
    generations: GA iterations; pop: population size.
    dock       : if True and DDD_DOCK_VINA is set, dock the final elite vs target.
    Returns a dict with the ranked candidates and run metadata.
    """
    random.seed(seed_rng)
    np.random.seed(seed_rng)
    weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
    log = (lambda s: progress(s)) if progress else (lambda s: None)

    # 1) assemble seed population
    seed_smiles, meta = [], []
    if seeds:
        seed_smiles += [s for s in seeds if s]
        meta.append(f"{len([s for s in seeds if s])} supplied seeds")
    if target:
        tsm, tmeta = _seeds_for_target(target, organism=organism)
        seed_smiles += tsm
        meta.append(tmeta)
    if not seed_smiles:
        seed_smiles = list(_STARTERS)
        meta.append("no target/seeds — using drug-like starter set")

    seed_mols, seed_fps, seen = [], [], set()
    for s in seed_smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None or not _valid(m, mw_cap):
            continue
        c = _canon(m)
        if c in seen:
            continue
        seen.add(c)
        seed_mols.append(m)
        seed_fps.append(_fp(m))
    if not seed_mols:
        return {"ok": False, "error": "no valid seed molecules", "meta": meta}

    seed_set = set(_canon(m) for m in seed_mols)
    log(f"seeds: {len(seed_mols)} molecules")

    # population starts as the seeds (padded by mutation if small)
    population = list(seed_mols)
    while len(population) < pop:
        parent = random.choice(seed_mols)
        child = _mutate(parent, mw_cap)
        population.append(child if child else parent)

    # 2) evolve
    best = {}  # canonical smiles -> (score, detail, mol)
    for g in range(generations):
        scored = []
        for m in population:
            c = _canon(m)
            if c is None:
                continue
            if c in best:
                sc, det = best[c][0], best[c][1]
            else:
                sc, det = _reward(m, seed_fps, weights, mw_cap)
                best[c] = (sc, det, m)
            scored.append((sc, c, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        elite = [m for _, _, m in scored[: max(4, pop // 4)]]
        # next generation: elitism + crossover + mutation
        nxt = list(elite)
        tries = 0
        while len(nxt) < pop and tries < pop * 6:
            tries += 1
            if random.random() < 0.4 and len(elite) >= 2:
                a, b = random.sample(elite, 2)
                child = _crossover(a, b, mw_cap)
            else:
                child = _mutate(random.choice(elite), mw_cap)
            if child is not None:
                nxt.append(child)
        population = nxt if nxt else elite
        if progress and (g % 5 == 0 or g == generations - 1):
            log(f"gen {g+1}/{generations}: pool {len(best)} unique, best {scored[0][0]:.3f}")

    # 3) rank novel candidates (exclude exact seeds)
    cand = [(sc, c, det, m) for c, (sc, det, m) in best.items() if c not in seed_set]
    cand.sort(key=lambda x: x[0], reverse=True)
    top = cand[:n]

    # 4) optional docking of the elite vs the target
    docked = {}
    if dock and target and top:
        try:
            from .armory import dock as dockmod
            import os
            if os.environ.get("DDD_DOCK_VINA", "").strip():
                log(f"docking {len(top)} candidates vs {target}")
                res = dockmod.dock(target, [c for _, c, _, _ in top])
                for row in (res.get("results") or []):
                    if row.get("smiles") and row.get("affinity") is not None:
                        docked[Chem.CanonSmiles(row["smiles"])] = row["affinity"]
        except Exception as e:
            log(f"docking skipped: {str(e)[:80]}")

    out = []
    for rank, (sc, c, det, m) in enumerate(top, 1):
        row = {"rank": rank, "smiles": c, "score": round(sc, 3),
               "formula": rdMolDescriptors.CalcMolFormula(m), **det}
        if c in docked:
            row["dock_kcal_mol"] = docked[c]
        out.append(row)

    return {
        "ok": True,
        "target": target or None,
        "seeds": len(seed_mols),
        "seed_meta": meta,
        "generations": generations,
        "population": pop,
        "unique_explored": len(best),
        "weights": weights,
        "docked": bool(docked),
        "candidates": out,
        "note": "Reward-guided graph GA over public chemistry. Novelty = distance from seeds; "
                "QED/Ro5 are drug-likeness proxies, not synthesizability or activity guarantees.",
    }
