"""ChEMBL enrichment via its public REST API.

ChEMBL holds millions of measured bioactivities — the real version of
Chemistry42's activity data. We query actives for a target (or a molecule's
activities) on demand and can cache them into Neo4j as ChEMBL-sourced edges.
Public API, no key, no bulk dump.
"""
from __future__ import annotations

import httpx

from .. import kg

BASE = "https://www.ebi.ac.uk/chembl/api/data"
_TIMEOUT = 40


def _get(path: str, params: dict) -> dict:
    r = httpx.get(f"{BASE}/{path}", params={**params, "format": "json"}, timeout=_TIMEOUT,
                  follow_redirects=True)
    r.raise_for_status()
    return r.json()


def resolve_target(symbol: str) -> dict | None:
    """Gene symbol -> ChEMBL single-protein human target."""
    d = _get("target/search", {"q": symbol, "limit": 10})
    human = [t for t in d.get("targets", [])
             if t.get("organism") == "Homo sapiens" and t.get("target_type") == "SINGLE PROTEIN"]
    pick = (human or d.get("targets", []) or [None])[0]
    if not pick:
        return None
    return {"target_chembl_id": pick["target_chembl_id"], "pref_name": pick.get("pref_name"),
            "organism": pick.get("organism")}


def target_actives(symbol: str, standard_type: str = "IC50", limit: int = 50,
                   min_pchembl: float = 6.0) -> dict:
    """Potent measured compounds for a target (pChEMBL >= min_pchembl ~ <=1 uM)."""
    tgt = resolve_target(symbol)
    if not tgt:
        return {"found": False, "query": symbol}
    d = _get("activity", {"target_chembl_id": tgt["target_chembl_id"],
                          "standard_type": standard_type, "pchembl_value__gte": min_pchembl,
                          "limit": min(limit, 200), "offset": 0})
    rows = []
    for a in d.get("activities", []):
        rows.append({"molecule_chembl_id": a.get("molecule_chembl_id"),
                     "pchembl": a.get("pchembl_value"),
                     "standard_type": a.get("standard_type"),
                     "value": a.get("standard_value"), "units": a.get("standard_units"),
                     "smiles": a.get("canonical_smiles")})
    return {"found": True, "symbol": symbol, "target_chembl_id": tgt["target_chembl_id"],
            "pref_name": tgt["pref_name"], "standard_type": standard_type,
            "count": len(rows), "actives": rows}


def enrich_target(symbol: str, standard_type: str = "IC50", limit: int = 50,
                  write: bool = True) -> dict:
    """Fetch a target's potent actives and (default) write Gene-active->Compound
    edges into Neo4j, tagged ChEMBL, with pChEMBL on the edge."""
    res = target_actives(symbol, standard_type, limit)
    if not res.get("found"):
        return {"written": 0, **res}
    written = 0
    if write and kg.status().get("connected"):
        g_idx = kg.ensure_node(symbol, "gene/protein", "ChEMBL")
        for a in res["actives"]:
            cid = a.get("molecule_chembl_id")
            if not cid:
                continue
            props = {"chembl_id": cid}
            if a.get("smiles"):
                props["smiles"] = a["smiles"]
            c_idx = kg.ensure_node(cid, "drug", "ChEMBL", props)
            kg.merge_edge(c_idx, g_idx, "chembl_active_on", "ChEMBL active",
                          "ChEMBL", {"pchembl": a.get("pchembl"), "standard_type": a.get("standard_type")})
            written += 1
    return {"written": written, "symbol": symbol, "target_chembl_id": res["target_chembl_id"],
            "count": res["count"], "actives": res["actives"][:limit]}


def molecule(name: str) -> dict:
    d = _get("molecule/search", {"q": name, "limit": 5})
    out = []
    for m in d.get("molecules", []):
        st = (m.get("molecule_structures") or {})
        out.append({"chembl_id": m.get("molecule_chembl_id"), "name": m.get("pref_name"),
                    "smiles": st.get("canonical_smiles"),
                    "max_phase": m.get("max_phase")})
    return {"query": name, "molecules": out}
