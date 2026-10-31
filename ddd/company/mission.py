"""Discovery mission — the agents working in tandem until something useful.

One command runs the whole pipeline as a chain of agents and loops over targets
until a candidate clears the bar:

  Agent 1 · Reader        reads the literature -> ranked, validated targets
  Agent 2 · Scout         target intelligence  -> is it solved / crowded / open?
  Agent 3 · Designer      generative design    -> novel drug-like analogs
  (Agent 4 · Docking)     optional Vina pose    -> binding estimate

"Useful" = a designed molecule that is drug-like (QED >= 0.5, no Lipinski
violation, no tox flag) against a target with real chemical matter. The loop
tries targets in evidence order and stops at the first `want` useful hits (or
when it runs out of targets). Every stage logs a marked line (▶ start, ✓ hit,
– miss, ✗ error) so a live overlay can show the agents working in tandem.

Prioritisation from public text + assays + reward-guided design — not a
validated lead. The point is to surface the most promising place to look next.
"""
from __future__ import annotations

QED_BAR = 0.5


def run(ctx, disease: str, max_targets_try: int = 4, want: int = 1,
        design_n: int = 10, generations: int = 18, dock: bool = False):
    from . import litdiscovery as ld
    from . import target_intel as ti
    from .. import generate as gen

    disease = ld._clean_disease(disease)
    if not disease:
        raise ValueError("Give a disease or condition for the mission.")
    ctx.log(f"◆ Mission: find a drug-like candidate for {disease}")

    # --- Agent 1: Reader -------------------------------------------------
    ctx.log("▶ Agent 1 · Reader — reading the literature")
    lit_sum, _ = ld.discover(ctx, disease, design=False)
    targets = lit_sum.get("targets") or []
    if not targets:
        raise ValueError("Reader found no validated targets for this condition.")
    ctx.log(f"✓ Reader — {lit_sum.get('papers_read')} papers → targets: {', '.join(targets)}")

    found, trail = [], []
    for gene in targets[:max_targets_try]:
        verdict = None
        # --- Agent 2: Scout (target intelligence) ------------------------
        ctx.log(f"▶ Agent 2 · Scout — target intelligence on {gene}")
        try:
            isum, _ = ti.dossier(ctx, gene)
            verdict = isum.get("crowding")
            ctx.log(f"✓ Scout — {gene}: {verdict}"
                    + (f", {isum.get('pdb')} PDB, {isum.get('drugs')} known drug(s)" if isum else ""))
        except Exception as e:
            ctx.log(f"✗ Scout — {gene}: {str(e)[:70]}")

        # --- Agent 3: Designer (generative design) -----------------------
        ctx.log(f"▶ Agent 3 · Designer — evolving analogs for {gene}")
        clean = []
        try:
            g = gen.generate(target=gene, n=design_n, generations=generations,
                             dock=dock, progress=ctx.log)
            cands = g.get("candidates") or []
            clean = [c for c in cands
                     if c.get("qed", 0) >= QED_BAR and c.get("ro5_violations", 9) == 0]
            ctx.log(f"{'✓' if clean else '–'} Designer — {gene}: "
                    f"{len(clean)}/{len(cands)} clean drug-like candidate(s)")
        except Exception as e:
            ctx.log(f"✗ Designer — {gene}: {str(e)[:70]}")
            cands = []

        trail.append({"target": gene, "verdict": verdict, "clean": len(clean), "total": len(cands)})
        if clean:
            found.append({"target": gene, "verdict": verdict, "candidates": clean[:5]})
            ctx.log(f"★ Useful hit on {gene}: {len(clean)} candidate(s) clear the bar")
            if len(found) >= want:
                break
        else:
            ctx.log(f"– {gene}: nothing clean this pass — moving to the next target")

    # --- report ----------------------------------------------------------
    lines = [f"# Discovery mission: {disease}", ""]
    if found:
        lines.append(f"**Discovered {len(found)} promising lead(s).** Agents ran in tandem over "
                     f"{len(trail)} target(s); below are the ones that cleared the bar "
                     f"(drug-like: QED ≥ {QED_BAR}, no Lipinski violation).")
    else:
        lines.append(f"**No clean lead yet** across {len(trail)} target(s). "
                     "Either widen the search (more targets/generations) or the condition's top "
                     "targets are poorly druggable — see the trail below for where to push.")

    for hit in found:
        lines += ["", f"## Lead target: {hit['target']}",
                  f"- **Readiness:** {hit.get('verdict') or 'unknown'}", "",
                  "| # | SMILES | Score | QED | SA | Sim→seed |", "|---|---|---|---|---|---|"]
        for c in hit["candidates"]:
            lines.append(f"| {c['rank']} | `{c['smiles']}` | {c['score']} | {c['qed']} | "
                         f"{c['sa']} | {c['sim_to_seed']} |")

    lines += ["", "## Mission trail", "", "| Target | Readiness | Clean / designed |", "|---|---|---|"]
    for t in trail:
        lines.append(f"| {t['target']} | {t.get('verdict') or '—'} | {t['clean']} / {t['total']} |")

    if found:
        lead = found[0]
        lines += ["", "## Next step", "",
                  f"- **See it in 3D / dock it:** build the 3D conformer of the top candidate and dock "
                  f"it into **{lead['target']}** to get a binding estimate.",
                  f"- **Fold the target:** open the AlphaFold structure of **{lead['target']}** to set up the pocket."]
    lines += ["", "_Prioritisation from public literature, assays and reward-guided design — "
              "a place to look, not a validated lead._"]

    summary = {"disease": disease, "targets_tried": len(trail), "leads": len(found),
               "lead_target": (found[0]["target"] if found else None),
               "useful": bool(found)}
    return summary, lines
