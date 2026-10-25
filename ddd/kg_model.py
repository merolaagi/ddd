"""Models built FROM the knowledge graph — not fetched from anywhere.

Rung 3, tier 1: topological link prediction over PrimeKG (+ enrichment edges).
These are the standard, interpretable link-prediction scores (common neighbours,
Jaccard, Adamic-Adar, preferential attachment) computed directly on the graph.
They let you:

  * check a hypothesis  -> hypothesis(a, b): is this pair linked, by what path,
    and how plausible is a *direct* link given the shared neighbourhood?
  * generate hypotheses -> predict_links(entity, target): rank NOVEL nodes the
    graph implies should connect to `entity` but currently don't — i.e. new,
    testable predictions, each with the shared evidence that drives the score.

Everything runs on whatever is in Neo4j now, so the Open Targets / ChEMBL
enrichment edges feed straight into the predictions. No training, no deps.
Tier 2 (learned node embeddings) is train_embeddings() below, optional.
"""
from __future__ import annotations

import math

from . import kg

# node type -> graph label, whitelisted so `target` can't inject Cypher
LABELS = {
    "gene": "Gene", "gene/protein": "Gene", "protein": "Gene",
    "drug": "Drug", "disease": "Disease", "phenotype": "Phenotype",
    "pathway": "Pathway", "anatomy": "Anatomy", "exposure": "Exposure",
}


def _resolve(name: str):
    rows = kg._q("""
        match (n:Entity) where toLower(n.name) = toLower($name)
        return n.idx as idx, n.name as name, n.type as type,
               count{ (n)-[:REL]-() } as degree
        order by size(n.name) limit 1""", name=name)
    return rows[0] if rows else None


def _band(common, aa, path_len):
    """A qualitative read of topological plausibility. Honest, not causal."""
    if path_len == 1:
        return "known", "Already a direct edge in the graph."
    if common == 0:
        return "weak", "No shared neighbours; any link is distant."
    if aa >= 3.0 and common >= 5:
        return "strong", "Dense, specific shared neighbourhood."
    if aa >= 1.0 and common >= 2:
        return "moderate", "Several shared neighbours."
    return "tentative", "A few shared neighbours, mostly high-degree."


def hypothesis(a: str, b: str, max_hops: int = 4) -> dict:
    """Score the plausibility of a direct link between two named entities and
    show the evidence: shared neighbours, the shortest existing path, and the
    standard link-prediction indices."""
    na, nb = _resolve(a), _resolve(b)
    if not na or not nb:
        missing = a if not na else b
        return {"found": False, "error": f"'{missing}' is not in the graph. Try Search for the exact name."}

    direct = kg._q("""
        match (x:Entity {idx:$a})-[r:REL]-(y:Entity {idx:$b})
        return collect(distinct r.display) as rels, collect(distinct coalesce(r.source,'PrimeKG')) as sources
        """, a=na["idx"], b=nb["idx"])
    rels = direct[0]["rels"] if direct else []
    sources = direct[0]["sources"] if direct else []

    agg = kg._q("""
        match (x:Entity {idx:$a})-[:REL]-(m:Entity)-[:REL]-(y:Entity {idx:$b})
        where m.idx <> $a and m.idx <> $b
        with distinct m, count{ (m)-[:REL]-() } as md
        where md <= 5000
        return count(m) as common,
               sum(1.0/log(md + 2.0)) as adamic_adar,
               [x in collect({name:m.name, type:m.type, degree:md})[..15] | x] as shared
        """, a=na["idx"], b=nb["idx"])
    common = agg[0]["common"] if agg else 0
    aa = round(agg[0]["adamic_adar"], 3) if agg and agg[0]["adamic_adar"] else 0.0
    shared = agg[0]["shared"] if agg else []
    shared.sort(key=lambda s: s["degree"])  # most specific (lowest degree) first

    p = kg.path(na["name"], nb["name"], max_hops)
    path_len = max(0, len(p["nodes"]) - 1)
    union = na["degree"] + nb["degree"] - common
    jaccard = round(common / union, 4) if union > 0 else 0.0
    band, why = _band(common, aa, path_len)

    return {
        "found": True,
        "a": {"name": na["name"], "type": na["type"], "degree": na["degree"]},
        "b": {"name": nb["name"], "type": nb["type"], "degree": nb["degree"]},
        "direct_link": bool(rels),
        "direct_relations": rels,
        "direct_sources": sources,
        "shortest_path": p,
        "path_hops": path_len,
        "scores": {
            "common_neighbors": common,
            "adamic_adar": aa,
            "jaccard": jaccard,
            "preferential_attachment": na["degree"] * nb["degree"],
        },
        "shared_neighbors": shared,
        "plausibility": band,
        "reading": why,
    }


def predict_links(entity: str, target: str = "gene", k: int = 20,
                  hub_cap: int = 800, shared_cap: int = 400) -> dict:
    """Rank NOVEL target-type nodes that the graph implies should link to
    `entity` but currently don't — ordered by Adamic-Adar. New hypotheses.

    Bounded for high-degree entities: we take `entity`'s neighbours, drop hubs
    (degree > hub_cap, which are both uninformative and explosive), keep the
    `shared_cap` most specific, and only then expand to candidate nodes."""
    label = LABELS.get(target.lower())
    if not label:
        return {"error": f"unknown target type '{target}'. Use one of: {', '.join(sorted(set(LABELS)))}"}
    ne = _resolve(entity)
    if not ne:
        return {"error": f"'{entity}' is not in the graph. Try Search for the exact name."}
    k = max(1, min(int(k), 100))

    try:
        rows = kg._q(f"""
            match (d:Entity {{idx:$idx}})-[:REL]-(m:Entity)
            with d, m, count{{ (m)-[:REL]-() }} as md
            where md <= $hub_cap
            with d, m, md order by md asc limit $shared_cap
            match (m)-[:REL]-(c:{label})
            where c.idx <> $idx and NOT EXISTS {{ (d)-[:REL]-(c) }}
            with c, sum(1.0/log(md + 2.0)) as aa, count(distinct m) as common
            return c.name as name, c.type as type, common, round(aa, 3) as adamic_adar,
                   count{{ (c)-[:REL]-(:Drug) }} as known_drugs
            order by aa desc limit $k""",
            idx=ne["idx"], k=k, hub_cap=int(hub_cap), shared_cap=int(shared_cap))
    except Exception as e:
        return {"error": f"prediction query failed: {str(e)[:300]}",
                "entity": {"name": ne["name"], "type": ne["type"], "degree": ne["degree"]}}

    return {
        "entity": {"name": ne["name"], "type": ne["type"], "degree": ne["degree"]},
        "target_type": label,
        "note": "Novel (not already linked) nodes ranked by Adamic-Adar over the shared neighbourhood "
                f"(hubs over degree {hub_cap} dropped; top {shared_cap} most-specific shared neighbours). "
                "Topological signal, not validated causality.",
        "predictions": rows,
    }


# --------------------------------------------------------------- tier 2 (learned)

def _emb_path():
    from .config import DATA_DIR
    d = DATA_DIR / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / "node2vec.kv"


def train_embeddings(walks_per_node: int = 6, walk_len: int = 24, dims: int = 96,
                     max_nodes: int = 0, seed: int = 0):
    """Learn node2vec-style embeddings on the graph (in-memory random walks +
    skip-gram). Optional: needs gensim. Returns a summary. Heavier than tier 1,
    so start modestly; pass max_nodes to cap the graph for a quick first model."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        return {"ok": False, "need": "gensim",
                "install": ".venv/bin/pip install gensim",
                "note": "Tier-1 link prediction (hypothesis / predict_links) needs none of this."}
    import random
    rng = random.Random(seed)

    cap = "limit $cap" if max_nodes else ""
    edges = kg._q(f"""match (a:Entity)-[:REL]-(b:Entity)
                      return a.idx as a, b.idx as b {cap}""",
                  **({"cap": max_nodes * 40} if max_nodes else {}))
    adj: dict = {}
    for e in edges:
        adj.setdefault(str(e["a"]), []).append(str(e["b"]))
        adj.setdefault(str(e["b"]), []).append(str(e["a"]))
    nodes = list(adj)
    if max_nodes and len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        keep = set(nodes)
        adj = {n: [m for m in adj[n] if m in keep] for n in nodes}

    walks = []
    for n in nodes:
        for _ in range(walks_per_node):
            w, cur = [n], n
            for _ in range(walk_len - 1):
                nb = adj.get(cur)
                if not nb:
                    break
                cur = rng.choice(nb)
                w.append(cur)
            walks.append(w)

    model = Word2Vec(walks, vector_size=dims, window=5, min_count=1, sg=1,
                     workers=4, epochs=3, seed=seed)
    model.wv.save(str(_emb_path()))
    return {"ok": True, "nodes": len(nodes), "edges": len(edges), "walks": len(walks),
            "dims": dims, "saved": str(_emb_path())}


def _load_kv():
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        return None
    p = _emb_path()
    if not p.exists():
        return None
    return KeyedVectors.load(str(p))


def embed_score(a: str, b: str) -> dict:
    """Cosine similarity of two nodes in the learned embedding space."""
    kv = _load_kv()
    if kv is None:
        return {"ok": False, "note": "No embeddings yet. Run train_embeddings first (needs gensim)."}
    na, nb = _resolve(a), _resolve(b)
    if not na or not nb:
        return {"ok": False, "error": "entity not in graph"}
    ia, ib = str(na["idx"]), str(nb["idx"])
    if ia not in kv or ib not in kv:
        return {"ok": False, "error": "entity not covered by the trained embeddings"}
    return {"ok": True, "a": na["name"], "b": nb["name"],
            "cosine": round(float(kv.similarity(ia, ib)), 4)}
