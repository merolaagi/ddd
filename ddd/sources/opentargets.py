"""Open Targets Platform enrichment via its public GraphQL API.

Open Targets gives evidence-weighted target-disease association scores (0-1),
which is a far better target-ranking signal than PrimeKG's drug-count proxy.
We query on demand and write the associations into Neo4j so the knowledgebase
grows around the diseases you actually study. Public API, no key, no bulk dump.
"""
from __future__ import annotations

import httpx

from .. import kg

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
_TIMEOUT = 40


def _gql(query: str, variables: dict) -> dict:
    r = httpx.post(ENDPOINT, json={"query": query, "variables": variables}, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError("; ".join(e.get("message", "?") for e in data["errors"])[:300])
    return data["data"]


def resolve_disease(name: str) -> dict | None:
    """Disease name -> {efo_id, name} via Open Targets search."""
    q = """query($q:String!){search(queryString:$q,entityNames:["disease"],page:{index:0,size:1}){
             hits{id name entity}}}"""
    hits = _gql(q, {"q": name}).get("search", {}).get("hits", [])
    if not hits:
        return None
    return {"efo_id": hits[0]["id"], "name": hits[0]["name"]}


def disease_targets(name: str, size: int = 25) -> dict:
    """Top associated targets for a disease, with Open Targets scores."""
    d = resolve_disease(name)
    if not d:
        return {"found": False, "query": name}
    q = """query($efo:String!,$size:Int!){
             disease(efoId:$efo){ id name
               associatedTargets(page:{index:0,size:$size}){
                 count rows{ score target{ id approvedSymbol } } } } }"""
    dd = _gql(q, {"efo": d["efo_id"], "size": size}).get("disease") or {}
    at = dd.get("associatedTargets", {})
    rows = [{"symbol": r["target"]["approvedSymbol"], "ensembl": r["target"]["id"],
             "score": round(r["score"], 4)} for r in at.get("rows", []) if r.get("target")]
    return {"found": True, "efo_id": d["efo_id"], "disease": dd.get("name", d["name"]),
            "total": at.get("count"), "rows": rows}


def target_known_drugs(symbol: str, size: int = 25) -> dict:
    q = """query($q:String!){search(queryString:$q,entityNames:["target"],page:{index:0,size:1}){hits{id name}}}"""
    hits = _gql(q, {"q": symbol}).get("search", {}).get("hits", [])
    if not hits:
        return {"found": False, "query": symbol}
    ens = hits[0]["id"]
    q2 = """query($ens:String!,$size:Int!){ target(ensemblId:$ens){ id approvedSymbol
              knownDrugs(size:$size){ count rows{ phase drug{ id name } mechanismOfAction } } } }"""
    t = _gql(q2, {"ens": ens, "size": size}).get("target") or {}
    kd = (t.get("knownDrugs") or {})
    rows = [{"drug": r["drug"]["name"], "phase": r.get("phase"),
             "moa": r.get("mechanismOfAction")} for r in kd.get("rows", []) if r.get("drug")]
    return {"found": True, "symbol": t.get("approvedSymbol", symbol), "ensembl": ens,
            "total": kd.get("count"), "rows": rows}


def enrich_disease(name: str, size: int = 25, write: bool = True) -> dict:
    """Fetch disease-target associations and (default) write them into Neo4j as
    OpenTargets-sourced edges with the association score on each edge."""
    res = disease_targets(name, size)
    if not res.get("found"):
        return {"written": 0, **res}
    written = 0
    if write and kg.status().get("connected"):
        d_idx = kg.ensure_node(res["disease"], "disease", "OpenTargets", {"efo_id": res["efo_id"]})
        for r in res["rows"]:
            g_idx = kg.ensure_node(r["symbol"], "gene/protein", "OpenTargets")
            kg.merge_edge(d_idx, g_idx, "ot_associated_with", "OT association",
                          "OpenTargets", {"score": r["score"]})
            written += 1
    return {"written": written, "disease": res["disease"], "efo_id": res["efo_id"],
            "total": res.get("total"), "targets": res["rows"]}
