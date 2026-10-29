"""Protein structure on demand — the AlphaFold view.

Given a target (gene symbol or UniProt accession) this fetches the *precomputed*
AlphaFold DB model and its PAE (predicted aligned error) matrix, so the Lab can
render the pLDDT-coloured ribbon and the PAE heatmap exactly like the AlphaFold
viewer. Disk-safe: the PDB is cached (one small file per target, reused by the
docking engine); the PAE is downsampled for transport and not written to disk.

Honest scope: this shows AlphaFold's *existing* prediction for a known protein
(200M+ structures in the public AlphaFold DB, CC-BY 4.0). It is **not** us running
AlphaFold to fold a brand-new sequence — de-novo folding of a novel sequence needs
the AlphaFold/ESMFold model itself. Use it for targets, not for arbitrary designed
sequences.
"""
from __future__ import annotations

import re

import httpx

from ..armory.dock import AF_API, resolve_uniprot, fetch_alphafold

_PAE_GRID_CAP = 200  # downsample PAE to at most this many rows/cols for transport
# UniProt accession pattern (so "EGFR" resolves by gene, but "P00533" is used directly)
_ACC = re.compile(r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")


def _mean_plddt(pdb_text: str):
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    if not vals:
        return None, 0
    return round(sum(vals) / len(vals), 1), len(vals)


def _plddt_bands(pdb_text: str):
    """Fraction of residues in each AlphaFold confidence band."""
    very, conf, low, vlow = 0, 0, 0, 0
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                b = float(line[60:66])
            except ValueError:
                continue
            if b >= 90:
                very += 1
            elif b >= 70:
                conf += 1
            elif b >= 50:
                low += 1
            else:
                vlow += 1
    tot = very + conf + low + vlow or 1
    return {"very_high": round(100 * very / tot), "confident": round(100 * conf / tot),
            "low": round(100 * low / tot), "very_low": round(100 * vlow / tot)}


def _downsample(mat, cap=_PAE_GRID_CAP):
    n = len(mat)
    if n <= cap:
        return mat, 1
    step = (n + cap - 1) // cap  # stride
    small = [[mat[i][j] for j in range(0, n, step)] for i in range(0, n, step)]
    return small, step


def _fetch_pae(pae_url: str):
    """AFDB PAE JSON -> NxN matrix + max. Handles both current and legacy formats."""
    try:
        r = httpx.get(pae_url, timeout=60, follow_redirects=True)
        if r.status_code != 200:
            return None, None
        data = r.json()
    except Exception:
        return None, None
    obj = data[0] if isinstance(data, list) and data else data
    if not isinstance(obj, dict):
        return None, None
    mat = obj.get("predicted_aligned_error")
    mx = obj.get("max_predicted_aligned_error")
    if mat is None and "distance" in obj:  # legacy flat format
        r1, r2, dist = obj.get("residue1"), obj.get("residue2"), obj.get("distance")
        if r1 and r2 and dist:
            n = max(r1)
            mat = [[0.0] * n for _ in range(n)]
            for a, b, d in zip(r1, r2, dist):
                mat[a - 1][b - 1] = d
    if mat is None:
        return None, None
    return mat, mx


def fetch(target: str, organism: str | None = None) -> dict:
    """Resolve a target to its AlphaFold model: pLDDT-coloured PDB + PAE matrix.
    `organism` (name or NCBI taxid) pins viral/bacterial targets, e.g. 'SARS-CoV-2'."""
    target = (target or "").strip()
    if not target:
        return {"ok": False, "error": "give a gene symbol or UniProt accession"}
    # resolve to a UniProt accession (accepts an accession directly too)
    acc, gene, orgname = target, None, None
    if not _ACC.match(target.upper()):
        try:
            info = resolve_uniprot(target, organism)
            acc, gene, orgname = info["accession"], info.get("symbol") or target, info.get("organism")
        except Exception as e:
            return {"ok": False, "error": f"couldn't resolve '{target}' to a protein: {str(e)[:120]}"}
    # structure (cached) + metadata from the AF API
    pdb_text, pae_url, af_url = None, None, None
    try:
        pdb_text = fetch_alphafold(acc).read_text()
    except Exception as e:
        return {"ok": False, "error": f"no AlphaFold model for {acc}: {str(e)[:120]}", "uniprot": acc, "gene": gene}
    try:
        r = httpx.get(AF_API.format(acc=acc), timeout=30, follow_redirects=True)
        if r.status_code == 200 and r.json():
            j = r.json()[0]
            pae_url = j.get("paeDocUrl") or j.get("paeUrl")
            af_url = j.get("pdbUrl")
            gene = gene or j.get("gene")
    except Exception:
        pass
    mean_plddt, length = _mean_plddt(pdb_text)
    pae, pae_max = (None, None)
    step = 1
    if pae_url:
        pae, pae_max = _fetch_pae(pae_url)
        if pae:
            pae, step = _downsample(pae)
    return {
        "ok": True,
        "target": target,
        "gene": gene,
        "organism": orgname,
        "uniprot": acc,
        "length": length,
        "mean_plddt": mean_plddt,
        "plddt_bands": _plddt_bands(pdb_text),
        "pdb": pdb_text,
        "pae": pae,
        "pae_max": pae_max,
        "pae_stride": step,
        "source": f"https://alphafold.ebi.ac.uk/entry/{acc}",
        "note": "Precomputed AlphaFold DB model (CC-BY 4.0). pLDDT is per-residue confidence "
                "(B-factor column); PAE is predicted aligned error in Å. Not a de-novo fold of a new sequence.",
    }
