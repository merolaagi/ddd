"""DEG / gene-set interpretation against the knowledge graph.

Take a differential-expression result (a list of gene symbols, or a pasted table
where the symbols are the first token per line) and interpret it in biological
context: which diseases and pathways the set is most connected to, and which of
the genes are already druggable. This is the "interpret your results" layer —
graph-based orientation, not a statistical enrichment test.
"""
from __future__ import annotations

import re

from .. import kg


def parse_genes(text):
    toks = text if isinstance(text, list) else re.split(r"[\s,;\t]+", str(text))
    out, seen = [], set()
    for t in toks:
        t = (t or "").strip().strip("\"'")
        if t and re.match(r"^[A-Za-z][A-Za-z0-9\-]{1,14}$", t) and not re.match(r"^(and|then|the|for|of|logfc|pvalue|gene|padj)$", t, re.I):
            u = t.upper()
            if u not in seen:
                seen.add(u); out.append(u)
    return out


def interpret(ctx, genes, disease: str = ""):
    syms = parse_genes(genes)
    if not syms:
        raise ValueError("Give a list of gene symbols (newline or comma separated).")
    syms = syms[:80]
    st = kg.status()
    if not st.get("connected"):
        raise RuntimeError("Knowledge graph is offline — start Neo4j and load PrimeKG, then rerun.")
    ctx.log(f"Interpreting {len(syms)} genes against the graph")

    rows = kg._q("""
        unwind $names as gn
        match (n:Gene) where toLower(n.name)=toLower(gn)
        return n.name as gene, count{ (n)-[:REL]-(:Disease) } as diseases,
               count{ (n)-[:REL]-(:Drug) } as drugs,
               count{ (n)-[:REL]-(:Phenotype) } as phenotypes""", names=syms)
    in_graph = {r["gene"].upper(): r for r in rows}
    dis = kg._q("""
        match (n:Gene)-[:REL]-(d:Disease) where n.name in $names
        return d.name as disease, count(distinct n) as genes order by genes desc limit 12""", names=syms)
    pw = kg._q("""
        match (n:Gene)-[:REL]-(p:Pathway) where n.name in $names
        return p.name as pathway, count(distinct n) as genes order by genes desc limit 12""", names=syms)

    covered = len(in_graph)
    druggable = sorted([r for r in rows if r["drugs"] > 0], key=lambda r: r["drugs"], reverse=True)

    lines = ["# DEG / gene-set interpretation", "",
             f"{len(syms)} genes submitted · {covered} found in PrimeKG"
             + (f" · context: {disease}" if disease else ""), "",
             "## What the set maps to", "",
             "Diseases most connected to your gene set — how many of your genes link to each.", "",
             "| Disease | Your genes linked |", "|---|---|"]
    lines += [f"| {d['disease']} | {d['genes']} |" for d in dis] or ["| — | — |"]
    lines += ["", "Pathways most connected:", "", "| Pathway | Your genes linked |", "|---|---|"]
    lines += [f"| {p['pathway']} | {p['genes']} |" for p in pw] or ["| — | — |"]
    lines += ["", "## Druggable genes in the set", "",
              "Genes in your set that already have drugs in the graph — the actionable ones.", "",
              "| Gene | Disease links | Drug links | Phenotype links |", "|---|---|---|---|"]
    for r in druggable[:20]:
        lines.append(f"| {r['gene']} | {r['diseases']} | {r['drugs']} | {r['phenotypes']} |")
    if not druggable:
        lines.append("| none found | | | |")
    missing = [g for g in syms if g not in in_graph]
    if missing:
        lines += ["", f"_{len(missing)} not in PrimeKG: {', '.join(missing[:20])}"
                  + ("…" if len(missing) > 20 else "") + "_"]
    lines += ["", "_Graph-based interpretation — which known biology your set touches, not a statistical "
              "enrichment test. Use it to orient, then confirm the leads._"]

    summary = {"genes": len(syms), "in_graph": covered, "druggable": len(druggable),
               "top_disease": dis[0]["disease"] if dis else None}
    return summary, lines
