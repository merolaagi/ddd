"""Docking: the first emulator rung.

Given a target (gene symbol, UniProt accession or a structure file/URL) and
one or more ligand SMILES, fetch the protein structure on demand, prepare
receptor and ligands, and run AutoDock Vina / smina to estimate binding
affinity (kcal/mol; more negative = tighter).

Design, matched to the rest of ddd:
- **On demand, disk-safe.** Structures come from AlphaFold (one small PDB per
  target, cached under data/armory/structures); no bulk downloads.
- **Optional heavy deps, graceful.** None of vina / meeko / openbabel are in
  requirements.txt. When a piece is missing the tool returns a clear message
  with the exact install one-liner instead of crashing -- same as the
  Open Targets / ChEMBL fallback.
- **Honest caveats.** AlphaFold models are apo (no ligand-shaped pocket) and
  the default box is a blind box over the whole protein, so scores are a
  *relative triage signal*, not validated affinities. Supply a pocket centre
  to dock into a known site.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from .paths import DIRS, stamp_name

# Let the host point at a docking engine / Open Babel installed anywhere, e.g. a
# conda-forge env on Apple Silicon:  DDD_DOCK_VINA="micromamba run -n dock vina"
VINA_CMD = os.environ.get("DDD_DOCK_VINA", "").strip()
OBABEL_CMD = os.environ.get("DDD_DOCK_OBABEL", "").strip()

STRUCT = DIRS["har"].parent / "structures"
STRUCT.mkdir(parents=True, exist_ok=True)
DOCK = DIRS["har"].parent / "dock"
DOCK.mkdir(parents=True, exist_ok=True)

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
AF_FILE = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"


class ToolMissing(Exception):
    """A required external tool/library is not installed; carries the fix."""


def _which(name: str) -> str | None:
    return shutil.which(name)


# ------------------------------------------------------------------ structure

_UNIPROT_FIELDS = "accession,gene_primary,protein_name,organism_name,organism_id,reviewed"


def _parse_uniprot(e: dict) -> dict:
    acc = e.get("primaryAccession")
    gene = ""
    try:
        gene = e["genes"][0]["geneName"]["value"]
    except Exception:
        pass
    name = ""
    try:
        name = e["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except Exception:
        try:
            name = e["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
        except Exception:
            pass
    org, taxid = "", None
    try:
        org = e["organism"]["scientificName"]
        taxid = e["organism"]["taxonId"]
    except Exception:
        pass
    et = (e.get("entryType") or "").lower()
    reviewed = ("reviewed" in et) and ("unreviewed" not in et)
    return {"accession": acc, "gene": gene, "symbol": gene or None, "protein": name,
            "organism": org, "taxid": taxid, "reviewed": reviewed}


def _uniprot_search(query: str, size: int = 5) -> list[dict]:
    r = httpx.get(UNIPROT, timeout=30, params={
        "query": query, "fields": _UNIPROT_FIELDS, "format": "json", "size": size})
    r.raise_for_status()
    return [_parse_uniprot(e) for e in r.json().get("results", [])]


def _looks_accession(s: str) -> bool:
    import re
    return bool(re.match(r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$", s.upper()))


def resolve_uniprot(symbol: str, organism: str | None = None) -> dict:
    """Resolve a target to a UniProt accession — any organism (human, viral,
    bacterial), not just human. Accepts a gene symbol, a protein name, or an
    accession. For a bare gene symbol with no organism given, a human hit is
    preferred (backward compatible); otherwise the best match wins. Pass
    `organism` (name or NCBI taxid) to pin it, e.g. 'SARS-CoV-2' or 2697049."""
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("no target given")
    if _looks_accession(symbol):
        return {"accession": symbol.upper(), "symbol": symbol.upper(), "gene": None,
                "protein": None, "organism": None, "note": "used as accession"}
    org_clause = ""
    if organism:
        org_clause = (f" AND organism_id:{organism}" if str(organism).isdigit()
                      else f' AND organism_name:"{organism}"')
    phrase = " " in symbol
    base = f'protein_name:"{symbol}"' if phrase else f'(gene:{symbol} OR protein_name:{symbol})'
    attempts = [f"{base}{org_clause} AND reviewed:true",
                f"{base}{org_clause}",
                f'"{symbol}"{org_clause} AND reviewed:true',
                f'"{symbol}"{org_clause}']
    hits = []
    for q in attempts:
        try:
            hits = _uniprot_search(q)
        except Exception:
            hits = []
        if hits:
            break
    if not hits:
        raise ValueError(f"no UniProt entry for '{symbol}'"
                         + (f" in organism '{organism}'" if organism else ""))
    # choose: with no organism hint, prefer a human reviewed hit for a bare gene
    chosen = hits[0]
    if not organism:
        human = [h for h in hits if h.get("taxid") == 9606]
        if human:
            # prefer human only when the gene actually matches (keeps EGFR human,
            # but lets "spike glycoprotein" resolve to the virus)
            gm = [h for h in human if (h.get("gene") or "").upper() == symbol.upper()]
            chosen = gm[0] if gm else (human[0] if not phrase else hits[0])
    out = dict(chosen)
    out["symbol"] = chosen.get("gene") or symbol
    return out


def fetch_alphafold(acc: str) -> Path:
    """Cache the AlphaFold predicted structure for a UniProt accession. Resolves
    the current file URL via the AlphaFold API (version-proof), with versioned
    filename fallbacks."""
    dest = STRUCT / f"AF-{acc}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urls = []
    try:
        r = httpx.get(AF_API.format(acc=acc), timeout=30, follow_redirects=True)
        if r.status_code == 200 and r.json():
            u = r.json()[0].get("pdbUrl")
            if u:
                urls.append(u)
    except Exception:
        pass
    urls += [AF_FILE.format(acc=acc, v=v) for v in (6, 5, 4)]
    last = None
    for u in urls:
        try:
            rr = httpx.get(u, timeout=60, follow_redirects=True)
            if rr.status_code == 200 and rr.content:
                dest.write_bytes(rr.content)
                return dest
            last = f"HTTP {rr.status_code}"
        except Exception as e:
            last = str(e)[:120]
    raise ValueError(f"no AlphaFold model fetched for {acc} (last: {last})")


# ------------------------------------------------------------------ box

def box_from_pdb(pdb: Path, pad: float = 5.0, cap: float = 30.0) -> dict:
    """Blind-docking box: bounding box of CA atoms, padded, each side capped."""
    xs, ys, zs = [], [], []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            try:
                xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
            except ValueError:
                continue
    if not xs:
        raise ValueError("no CA atoms found in structure")
    def ctr_sz(v):
        c = (min(v) + max(v)) / 2.0
        s = min((max(v) - min(v)) + 2 * pad, cap)
        return round(c, 3), round(s, 3)
    cx, sx = ctr_sz(xs); cy, sy = ctr_sz(ys); cz, sz = ctr_sz(zs)
    return {"center": [cx, cy, cz], "size": [sx, sy, sz], "blind": True, "n_ca": len(xs)}


# ------------------------------------------------------------------ prep

def _prep_ligand_meeko(smiles: str, out: Path) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        raise ToolMissing("RDKit is required. Install: .venv/bin/pip install rdkit")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) != 0:
        raise ValueError(f"3D embedding failed for: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise ValueError(f"meeko ligand write failed: {err}")
        out.write_text(pdbqt)
        return
    except ImportError:
        pass
    # fall back to obabel CLI
    ob = _obabel()
    if ob:
        with tempfile.NamedTemporaryFile("w", suffix=".mol", delete=False) as f:
            f.write(Chem.MolToMolBlock(mol)); molf = f.name
        subprocess.run(ob + [molf, "-O", str(out), "-xh"], check=True,
                       capture_output=True, text=True)
        os.unlink(molf)
        return
    raise ToolMissing(
        "No ligand-prep tool. Install meeko (.venv/bin/pip install meeko) or "
        "Open Babel (conda-forge openbabel / brew install open-babel).")


def _prep_receptor(pdb: Path, out: Path) -> None:
    # meeko receptor prep if present, else obabel
    if _which("mk_prepare_receptor.py") or _which("mk_prepare_receptor"):
        exe = _which("mk_prepare_receptor.py") or _which("mk_prepare_receptor")
        subprocess.run([exe, "--read_pdb", str(pdb), "-o", str(out.with_suffix("")),
                        "-p"], check=True, capture_output=True, text=True)
        if out.exists():
            return
    ob = _obabel()
    if ob:
        subprocess.run(ob + [str(pdb), "-O", str(out), "-xr"], check=True,
                       capture_output=True, text=True)
        return
    raise ToolMissing(
        "No receptor-prep tool. Install Open Babel (conda-forge openbabel, or "
        "brew install open-babel) and set DDD_DOCK_OBABEL, or pip install meeko.")


# ------------------------------------------------------------------ engine

def _run_vina_py(rec: Path, lig: Path, box: dict, out: Path, exhaustiveness: int, n_poses: int):
    from vina import Vina
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec))
    v.set_ligand_from_file(str(lig))
    v.compute_vina_maps(center=box["center"], box_size=box["size"])
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    v.write_poses(str(out), n_poses=n_poses, overwrite=True)
    energies = v.energies(n_poses=n_poses)
    return [round(float(e[0]), 3) for e in energies]


def _run_cli(engine: str, rec: Path, lig: Path, box: dict, out: Path, exhaustiveness: int, n_poses: int):
    c, s = box["center"], box["size"]
    cmd = shlex.split(engine) + ["--receptor", str(rec), "--ligand", str(lig),
           "--center_x", str(c[0]), "--center_y", str(c[1]), "--center_z", str(c[2]),
           "--size_x", str(s[0]), "--size_y", str(s[1]), "--size_z", str(s[2]),
           "--exhaustiveness", str(exhaustiveness), "--num_modes", str(n_poses),
           "--out", str(out)]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    scores = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                scores.append(round(float(parts[1]), 3))
            except ValueError:
                pass
    return scores


def _obabel():
    """Open Babel invocation as a token list, or None. Honors DDD_DOCK_OBABEL."""
    if OBABEL_CMD:
        return shlex.split(OBABEL_CMD)
    w = _which("obabel")
    return [w] if w else None


def _engine():
    try:
        import vina  # noqa: F401
        return ("py", None)
    except Exception:
        pass
    if VINA_CMD:
        return ("cli", VINA_CMD)
    for cli in ("smina", "vina"):
        if _which(cli):
            return ("cli", cli)
    return (None, None)


# ------------------------------------------------------------------ orchestrator

def dock(target: str, smiles: str, exhaustiveness: int = 8, n_poses: int = 9,
         center: str = "", size: float = 0.0):
    """Dock one or more ligands (newline/comma-separated SMILES) into a target.

    center: optional "x,y,z" pocket centre; blank = blind box over whole protein.
    size:   optional cubic box side (A) used with an explicit centre.
    """
    kind, engine = _engine()
    if kind is None:
        return {"ok": False, "need": "docking engine",
                "install": "Apple Silicon (no pip wheel for vina): use conda-forge --\n"
                           "  brew install micromamba\n"
                           "  micromamba create -y -n dock -c conda-forge vina openbabel\n"
                           "  .venv/bin/pip install meeko\n"
                           "then in ~/Sites/ddd/.env set:\n"
                           "  DDD_DOCK_VINA=micromamba run -n dock vina\n"
                           "  DDD_DOCK_OBABEL=micromamba run -n dock obabel",
                "note": "No vina python module, no DDD_DOCK_VINA, no smina/vina on PATH."}

    import traceback
    try:
        info = resolve_uniprot(target)
        pdb = fetch_alphafold(info["accession"])
        if center.strip():
            cx, cy, cz = [float(x) for x in center.replace(" ", "").split(",")]
            side = size or 22.0
            box = {"center": [cx, cy, cz], "size": [side, side, side], "blind": False}
        else:
            box = box_from_pdb(pdb)
        ligs = [s.strip() for s in smiles.replace(",", "\n").splitlines() if s.strip()]
        if not ligs:
            return {"ok": False, "error": "no SMILES given"}
        rec_pdbqt = STRUCT / f"AF-{info['accession']}.pdbqt"
        if not rec_pdbqt.exists():
            _prep_receptor(pdb, rec_pdbqt)
    except ToolMissing as e:
        return {"ok": False, "need": "prep tool", "install": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"dock setup failed: {str(e)[:300]}",
                "stage": "resolve/fetch/receptor-prep", "engine": engine,
                "trace": traceback.format_exc()[-600:]}

    results = []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(ligs[:25]):
            lig_pdbqt = Path(td) / f"lig{i}.pdbqt"
            try:
                _prep_ligand_meeko(smi, lig_pdbqt)
            except ToolMissing as e:
                return {"ok": False, "need": "ligand prep", "install": str(e)}
            except Exception as e:
                results.append({"smiles": smi, "error": f"ligand prep: {str(e)[:250]}"})
                continue
            out = DOCK / stamp_name(f"{info['accession']}-lig{i}", "pdbqt")
            try:
                if kind == "py":
                    scores = _run_vina_py(rec_pdbqt, lig_pdbqt, box, out, exhaustiveness, n_poses)
                else:
                    scores = _run_cli(engine, rec_pdbqt, lig_pdbqt, box, out, exhaustiveness, n_poses)
            except subprocess.CalledProcessError as e:
                results.append({"smiles": smi, "error": "vina/engine: " + (e.stderr or str(e))[-250:]})
                continue
            except Exception as e:
                results.append({"smiles": smi, "error": f"docking: {str(e)[:250]}"})
                continue
            results.append({
                "smiles": smi,
                "best_affinity": min(scores) if scores else None,
                "poses": scores,
                "pose_file": out.name,
            })

    ranked = sorted([r for r in results if r.get("best_affinity") is not None],
                    key=lambda r: r["best_affinity"])
    return {
        "ok": True,
        "target": info,
        "structure": pdb.name,
        "engine": engine or "vina(python)",
        "box": box,
        "exhaustiveness": exhaustiveness,
        "results": results,
        "ranking": [{"smiles": r["smiles"], "best_affinity": r["best_affinity"]} for r in ranked],
        "caveat": "AlphaFold apo model; " + ("blind box over whole protein" if box.get("blind")
                  else "docked into supplied pocket") + ". Relative triage, not a validated affinity.",
    }
