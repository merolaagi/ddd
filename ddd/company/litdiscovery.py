"""Literature-driven discovery — agents that read research and return molecules.

The reading loop ddd runs for a condition:
  1. Pull the most-cited papers for the disease (Europe PMC).
  2. *Read* them with Europe PMC's text-mining annotations — the genes/proteins
     and chemicals each paper actually discusses (no paid LLM; EBI does the NLP).
  3. Rank targets by how many papers back them (evidence strength).
  4. Pull each target's drug-like chemical matter from ChEMBL — real compounds,
     many of them clinical, that are "as close to a drug" as public data offers.
  5. Hand the top target to generative design for novel analogs.

Output is a dossier: the targets the field converges on, the evidence-backed
candidate molecules, and the compounds the papers themselves name. It is a
prioritisation from public text + assays, not a validated hit list.
"""
from __future__ import annotations

# noise the text-miner tags that are not useful targets
_STOP = {"protein", "proteins", "gene", "genes", "receptor", "enzyme", "antibody",
         "antibodies", "peptide", "insulin", "albumin", "dna", "rna", "mrna",
         "cytokine", "cytokines", "hormone", "antigen", "ligand", "kinase"}


def discover(ctx, disease: str, max_papers: int = 20, max_targets: int = 6,
             max_compounds: int = 15, design: bool = False):
    from ..sources import literature
    from .playbooks import _compounds_for_targets, _compound_flags
    disease = (disease or "").strip()
    if not disease:
        raise ValueError("Give a disease or condition to read the literature on.")

    ctx.log(f"Searching Europe PMC for '{disease}' (most-cited)")
    s = literature.search(disease, limit=max_papers, sort="CITED desc")
    papers = s.get("papers") or []
    if not papers:
        raise ValueError(f"no papers found for '{disease}'")

    ctx.log(f"Reading {len(papers)} papers — text-mining genes/proteins & chemicals")
    ann = literature.annotate(papers)
    genes = [g for g in ann.get("gene_proteins", [])
             if g["name"].lower() not in _STOP and 2 <= len(g["name"]) <= 24]
    chems = [c for c in ann.get("chemicals", []) if c["name"].lower() not in _STOP]
    if not genes:
        raise ValueError("the papers yielded no usable protein/gene targets "
                         "(try a more specific disease term)")

    # keep targets backed by >1 paper when we have enough signal
    strong = [g for g in genes if g["papers"] >= 2] or genes
    top = strong[:max_targets]
    mx = max((g["papers"] for g in top), default=1)
    targets = [{"gene": g["name"], "weight": round(g["papers"] / mx, 3), "papers": g["papers"]}
               for g in top]
    ctx.log("Targets: " + ", ".join(f"{t['gene']}({t['papers']}p)" for t in targets))

    ranked, desc, tox, sol, agg = _compounds_for_targets(ctx, targets, max_compounds)

    # score each candidate for drug-likeness
    druglike = 0
    rows = []
    for c in ranked:
        d = desc.get(c["smiles"], {})
        flags = _compound_flags(d, tox.get(c["smiles"]), sol.get(c["smiles"]))
        clean = d.get("valid") and not flags and d.get("qed", 0) >= 0.4
        druglike += bool(clean)
        rows.append((c, d, flags, clean))
    # drug-like first, then by evidence/potency
    rows.sort(key=lambda r: (r[3], len(r[0]["targets_hit"]), r[0]["score"]), reverse=True)

    name = disease
    lines = [f"# Literature-driven discovery: {name}", "",
             f"Read the **{len(papers)}** most-cited papers for *{name}* via Europe PMC and text-mined "
             f"their targets and chemicals. Targets ranked by how many papers back them; candidate "
             f"molecules are those targets' drug-like ChEMBL actives (pChEMBL-weighted). "
             f"Evidence + public assays, not a validated hit list.", "",
             "## Targets the literature converges on", "",
             "| Target | Papers | ChEMBL actives |", "|---|---|---|"]
    for t in targets:
        lines.append(f"| {t['gene']} | {t['papers']} | {t.get('actives', 0)} |")

    lines += ["", f"## Candidate molecules — evidence-backed, drug-like ({druglike}/{len(rows)} clean)", "",
              "| # | SMILES | ChEMBL | pChEMBL | Targets hit | QED | MW | Flags |",
              "|---|---|---|---|---|---|---|---|"]
    for i, (c, d, flags, clean) in enumerate(rows, 1):
        if not d.get("valid"):
            lines.append(f"| {i} | `{c['smiles']}` | {c.get('chembl_id') or '—'} | {c['pchembl']} | "
                         f"{', '.join(sorted(c['targets_hit']))} | — | — | invalid |")
            continue
        lines.append(f"| {i} | `{d['smiles']}` | {c.get('chembl_id') or '—'} | {c['pchembl']} | "
                     f"{', '.join(sorted(c['targets_hit']))} | {d['qed']} | {d['mol_weight']} | "
                     f"{', '.join(flags) or '✓ clean'} |")

    if chems:
        lines += ["", "## Compounds the papers name", "",
                  "| Chemical | Papers |", "|---|---|"]
        for c in chems[:10]:
            lines.append(f"| {c['name']} | {c['papers']} |")

    lines += ["", "## Papers read (top-cited)", ""]
    for p in papers[:8]:
        cited = f" · cited {p['cited']}" if p.get("cited") is not None else ""
        link = f"[{p['title']}]({p['url']})" if p.get("url") else (p.get("title") or "untitled")
        lines.append(f"- {link} — {p.get('journal','')} {p.get('year','')}{cited}")

    lead = targets[0]["gene"] if targets else None
    lines += ["", "## Next step", "",
              f"- **Design novel analogs:** run generative design on **{lead}** "
              f"(it seeds from these actives and evolves new drug-like molecules).",
              f"- **Check crowding:** run target intelligence on **{lead}** to see if it's open space.",
              f"- **Go to protein:** fetch the AlphaFold structure of **{lead}** to set up docking."]

    summary = {"disease": name, "papers_read": len(papers), "targets": [t["gene"] for t in targets],
               "candidates": len(rows), "druglike": druglike, "lead_target": lead}

    # optional: chain straight into generative design on the lead target
    if design and lead:
        ctx.log(f"Designing novel analogs for lead target {lead}")
        try:
            from .. import generate as gen
            g = gen.generate(target=lead, n=8, generations=18, progress=ctx.log)
            if g.get("ok") and g.get("candidates"):
                lines += ["", "## Novel designed analogs (generative)", "",
                          "| # | SMILES | Score | QED | SA | Sim→seed |", "|---|---|---|---|---|---|"]
                for cc in g["candidates"]:
                    lines.append(f"| {cc['rank']} | `{cc['smiles']}` | {cc['score']} | {cc['qed']} | "
                                 f"{cc['sa']} | {cc['sim_to_seed']} |")
                summary["designed"] = len(g["candidates"])
        except Exception as e:
            ctx.log(f"generative step skipped: {str(e)[:80]}")

    return summary, lines
