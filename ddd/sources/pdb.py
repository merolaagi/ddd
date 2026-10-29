"""Experimental structures from the RCSB PDB — "has someone solved this for real?".

Given a UniProt accession, find deposited experimental structures (X-ray, cryo-EM,
NMR), with resolution, method, release date and whether any are ligand-bound
(a ligand-bound structure means a druggable pocket is already characterised).
Public RCSB Search + Data APIs, no key. On demand, nothing written to disk.
"""
from __future__ import annotations

import httpx

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdbid}"
_TIMEOUT = 30


def _search_ids(acc: str, rows: int = 25) -> list[str]:
    q = {
        "query": {
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match", "value": acc,
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows},
                            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]},
    }
    r = httpx.post(SEARCH, json=q, timeout=_TIMEOUT)
    if r.status_code == 204:        # no hits
        return []
    r.raise_for_status()
    return [row["identifier"] for row in r.json().get("result_set", [])]


def _entry(pdbid: str) -> dict | None:
    try:
        r = httpx.get(ENTRY.format(pdbid=pdbid), timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:
        return None
    info = j.get("rcsb_entry_info", {}) or {}
    res = info.get("resolution_combined") or []
    methods = j.get("exptl") or []
    rel = (j.get("rcsb_accession_info", {}) or {}).get("initial_release_date", "")
    nonpoly = info.get("nonpolymer_entity_count", 0) or 0
    return {
        "id": pdbid,
        "method": (methods[0].get("method") if methods else None),
        "resolution": (round(res[0], 2) if res else None),
        "released": rel[:10] if rel else None,
        "ligands": nonpoly,          # count of non-polymer entities (ligands/cofactors)
        "title": j.get("struct", {}).get("title", "")[:120],
    }


def structures_for_uniprot(acc: str, detail: int = 8) -> dict:
    """All experimental PDB entries mapped to a UniProt accession, newest first.
    Fetches per-entry detail for the first `detail` entries (resolution, method,
    ligands); the rest are counted only."""
    try:
        ids = _search_ids(acc)
    except Exception as e:
        return {"ok": False, "error": f"RCSB search failed: {str(e)[:120]}", "count": 0, "entries": []}
    entries = []
    best_res, ligand_bound = None, False
    for pid in ids[:detail]:
        e = _entry(pid)
        if not e:
            continue
        entries.append(e)
        if e["resolution"] is not None:
            best_res = e["resolution"] if best_res is None else min(best_res, e["resolution"])
        if e["ligands"]:
            ligand_bound = True
    return {
        "ok": True,
        "uniprot": acc,
        "count": len(ids),
        "shown": len(entries),
        "best_resolution": best_res,
        "ligand_bound": ligand_bound,
        "newest": entries[0]["released"] if entries else None,
        "entries": entries,
        "ids": ids,
    }
