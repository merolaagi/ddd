"""Neo4j knowledge-graph access (PrimeKG schema). Optional: everything else
works when Neo4j isn't running."""
from __future__ import annotations

from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

TYPE_LABEL = {
    "gene/protein": "Gene", "drug": "Drug", "disease": "Disease",
    "effect/phenotype": "Phenotype", "biological_process": "BioProcess",
    "molecular_function": "MolFunction", "cellular_component": "CellComponent",
    "pathway": "Pathway", "anatomy": "Anatomy", "exposure": "Exposure",
}

_driver = None


def driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def _q(cypher, **params):
    with driver().session() as s:
        return [r.data() for r in s.run(cypher, **params)]


def status():
    try:
        n = _q("match (n:Entity) return count(n) as nodes")[0]["nodes"]
        r = _q("match ()-[r:REL]->() return count(r) as rels")[0]["rels"]
        return {"connected": True, "uri": NEO4J_URI, "nodes": n, "relationships": r}
    except Exception as e:
        return {"connected": False, "uri": NEO4J_URI, "error": str(e)[:200]}


def search(name: str, limit: int = 10):
    return _q("""
        match (n:Entity) where toLower(n.name) contains toLower($name)
        return n.idx as idx, n.name as name, n.type as type, n.source as source
        order by size(n.name) limit $limit""", name=name, limit=limit)


def neighbors(name: str, relation: str | None = None, limit: int = 25):
    return _q("""
        match (n:Entity) where toLower(n.name) = toLower($name)
        with n limit 1
        match (n)-[r:REL]-(m:Entity)
        where $relation is null or r.relation = $relation or r.display = $relation
        return n.name as source, r.display as relation, m.name as target, m.type as target_type
        limit $limit""", name=name, relation=relation, limit=limit)


def relation_summary(name: str):
    return _q("""
        match (n:Entity) where toLower(n.name) = toLower($name)
        with n limit 1
        match (n)-[r:REL]-(m:Entity)
        return r.display as relation, m.type as target_type, count(*) as count
        order by count desc""", name=name)


def path(a: str, b: str, max_hops: int = 4):
    max_hops = max(1, min(int(max_hops), 5))
    rows = _q(f"""
        match (x:Entity) where toLower(x.name) = toLower($a) with x limit 1
        match (y:Entity) where toLower(y.name) = toLower($b) with x, y limit 1
        match p = shortestPath((x)-[:REL*..{max_hops}]-(y))
        return [n in nodes(p) | n.name + ' (' + n.type + ')'] as nodes,
               [r in relationships(p) | r.display] as relations""", a=a, b=b)
    return rows[0] if rows else {"nodes": [], "relations": []}


def drug_targets(drug: str, limit: int = 50):
    return _q("""
        match (d:Drug) where toLower(d.name) = toLower($drug) with d limit 1
        match (d)-[r:REL]-(g:Gene)
        return g.name as gene, r.display as relation limit $limit""", drug=drug, limit=limit)


def disease_candidate_targets(disease: str, limit: int = 25):
    """Naive target ranking: genes linked to the disease, scored by how many
    drugs already touch them (druggability proxy) and phenotype overlap."""
    return _q("""
        match (d:Disease) where toLower(d.name) = toLower($disease) with d limit 1
        match (d)-[:REL]-(g:Gene)
        optional match (g)-[:REL]-(drug:Drug)
        with g, count(distinct drug) as n_drugs
        optional match (g)-[:REL]-(p:Phenotype)
        return g.name as gene, n_drugs, count(distinct p) as n_phenotypes
        order by n_drugs desc, n_phenotypes desc limit $limit""", disease=disease, limit=limit)


def find_disease(name: str):
    """Best Disease node for a free-text name: exact match first, then shortest containing match."""
    rows = _q("""
        match (d:Disease) where toLower(d.name) = toLower($name) return d.name as name limit 1""", name=name)
    if rows:
        return rows[0]["name"]
    rows = _q("""
        match (d:Disease) where toLower(d.name) contains toLower($name)
        return d.name as name order by size(d.name) limit 1""", name=name)
    return rows[0]["name"] if rows else None


def gene_drugs(gene: str, limit: int = 8):
    return [r["drug"] for r in _q("""
        match (g:Gene) where g.name = $gene with g limit 1
        match (g)-[:REL]-(d:Drug) return distinct d.name as drug limit $limit""", gene=gene, limit=limit)]


def disease_drugs(disease: str, limit: int = 15):
    return _q("""
        match (d:Disease) where d.name = $disease with d limit 1
        match (d)-[r:REL]-(x:Drug)
        return x.name as drug, r.display as relation limit $limit""", disease=disease, limit=limit)


def subgraph(center: str, depth: int = 1, limit: int = 60, relation: str | None = None):
    """Nodes + edges around a center entity, shaped for graph rendering.
    depth 1 = direct neighbors; depth 2 also expands one hop from those."""
    depth = max(1, min(int(depth), 2))
    limit = max(5, min(int(limit), 250))
    rows = _q(f"""
        match (c:Entity) where toLower(c.name) = toLower($center) with c limit 1
        match p = (c)-[:REL*1..{depth}]-(m:Entity)
        with c, relationships(p) as rels, nodes(p) as ns
        unwind range(0, size(rels)-1) as i
        with ns[i] as a, ns[i+1] as b, rels[i] as r
        where $relation is null or r.display = $relation or r.relation = $relation
        return distinct
          a.idx as a_idx, a.name as a_name, a.type as a_type,
          b.idx as b_idx, b.name as b_name, b.type as b_type,
          r.display as rel
        limit $limit""", center=center, relation=relation, limit=limit)
    if not rows:
        return {"center": center, "found": False, "nodes": [], "edges": []}
    nodes, edges, seen = {}, [], set()
    center_l = center.lower()
    for row in rows:
        for pfx in ("a", "b"):
            idx = row[f"{pfx}_idx"]
            if idx not in nodes:
                nm = row[f"{pfx}_name"]
                nodes[idx] = {"id": idx, "name": nm, "type": row[f"{pfx}_type"],
                              "label": TYPE_LABEL.get(row[f"{pfx}_type"], "Other"),
                              "center": nm.lower() == center_l}
        key = (row["a_idx"], row["b_idx"], row["rel"])
        rkey = tuple(sorted((row["a_idx"], row["b_idx"]))) + (row["rel"],)
        if rkey not in seen:
            seen.add(rkey)
            edges.append({"source": row["a_idx"], "target": row["b_idx"], "rel": row["rel"]})
    return {"center": center, "found": True, "nodes": list(nodes.values()), "edges": edges,
            "counts": {"nodes": len(nodes), "edges": len(edges)}}
