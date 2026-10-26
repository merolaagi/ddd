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
ALPHAFOLD = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"


class ToolMissing(Exception):
    """A required external tool/library is not installed; carries the fix."""


def _which(name: str) -> str | None:
    return shutil.which(name)


# ------------------------------------------------------------------ structure

def resolve_uniprot(symbol: str) -> dict:
    """Gene symbol (human) -> reviewed UniProt accession."""
    if symbol and len(symbol) == 6 and symbol[0] in "OPQ" and symbol[1].isdigit():
        return {"accession": symbol, "symbol": symbol, "note": "used as accession"}
    if symbol and symbol.isalnum() and symbol[0].isalpha() and any(c.isdigit() for c in symbol) and len(symbol) >= 6:
        # looks like an accession (e.g. P16234) already
        if symbol[0] in "OPQ" or (symbol[0] in "A" and symbol[1].isdigit()):
            return {"accession": symbol, "symbol": symbol, "note": "used as accession"}
    q = f"gene:{symbol} AND organism_id:9606 AND reviewed:true"
    r = httpx.get(UNIPROT, timeout=30, params={
        "query": q, "fields": "accession,gene_primary,protein_name", "format": "json", "size": 1})
    r.raise_for_status()
    res = r.json().get("results", [])
    if not res:
        raise ValueError(f"no reviewed human UniProt entry for gene '{symbol}'")
    acc = res[0]["primaryAccession"]
    name = ""
    try:
        name = res[0]["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except Exception:
        pass
    return {"accession": acc, "symbol": symbol, "protein": name}


def fetch_alphafold(acc: str) -> Path:
    """Cache the AlphaFold predicted structure for a UniProt accession."""
    dest = STRUCT / f"AF-{acc}-F1-v4.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = httpx.get(ALPHAFOLD.format(acc=acc), timeout=60, follow_redirects=True)
    if r.status_code == 404:
        raise ValueError(f"no AlphaFold model for {acc} (not in the predicted set)")
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


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

    # resolve + fetch structure
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
        raise ValueError("no SMILES given")

    rec_pdbqt = STRUCT / f"AF-{info['accession']}-F1-v4.pdbqt"
    try:
        if not rec_pdbqt.exists():
            _prep_receptor(pdb, rec_pdbqt)
    except ToolMissing as e:
        return {"ok": False, "need": "receptor prep", "install": str(e)}

    results = []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(ligs[:25]):
            lig_pdbqt = Path(td) / f"lig{i}.pdbqt"
            try:
                _prep_ligand_meeko(smi, lig_pdbqt)
            except ToolMissing as e:
                return {"ok": False, "need": "ligand prep", "install": str(e)}
            except ValueError as e:
                results.append({"smiles": smi, "error": str(e)})
                continue
            out = DOCK / stamp_name(f"{info['accession']}-lig{i}", "pdbqt")
            try:
                if kind == "py":
                    scores = _run_vina_py(rec_pdbqt, lig_pdbqt, box, out, exhaustiveness, n_poses)
                else:
                    scores = _run_cli(engine, rec_pdbqt, lig_pdbqt, box, out, exhaustiveness, n_poses)
            except subprocess.CalledProcessError as e:
                results.append({"smiles": smi, "error": (e.stderr or str(e))[-300:]})
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
