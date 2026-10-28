"""Program Operations — the "Pipeline-in-a-Person" layer.

Turns an in-silico discovery result into a runnable drug program: staged
experiments from lead to validated candidate, a CRO/vendor task list, drafted
outreach, milestones with rough budget bands, and go/no-go gates. This is the
operations layer a solo founder needs to coordinate real wet-lab work — the
part that is orchestration, not computation.

Rules/template based: no model API. Budget bands are rough industry ballparks,
not quotes. Outreach drafts are generic RFQ templates with placeholders — they
name no real company and fabricate nothing.
"""
from __future__ import annotations


def _invivo_model(disease: str):
    d = (disease or "").lower()
    if "lama2" in d or ("muscular dystrophy" in d) or "dystroglycan" in d or "cmd" in d:
        return ("LAMA2-deficient mouse (e.g. dy^W/dy^W or dy/dy)",
                "grip strength, forelimb/triceps peak force, survival to weaning, and antibody/compound "
                "localization in muscle (target engagement)")
    if "fibros" in d or "ipf" in d:
        return ("bleomycin-induced lung fibrosis mouse",
                "hydroxyproline/collagen, Ashcroft fibrosis score, lung function")
    if "diabet" in d:
        return ("db/db or diet-induced obese mouse",
                "fasting glucose, glucose tolerance (OGTT), HbA1c, body weight")
    return ("a validated disease mouse model",
            "the disease's accepted functional endpoint plus target engagement in the relevant tissue")


def _stages(modality: str, disease: str):
    model, readout = _invivo_model(disease)
    if modality.startswith("antibod") or modality.startswith("biolog"):
        return [
            ("Target & epitope", "Pick the surface/matrix target and an accessible epitope",
             "ddd target ranking + literature; epitope accessibility", "ranked target + epitope rationale",
             "in-silico (ddd) + advisor", "2–3 wk", "Target is extracellular/accessible and genetically supported"),
            ("Antibody generation", "Raise a panel of binders",
             "immunization and/or phage/yeast display campaign", "panel of specific binders (dozens–hundreds)",
             "antibody-discovery CRO", "3–4 mo", "≥20 specific binders to the epitope"),
            ("Binding characterization", "Rank binders by kinetics & specificity",
             "SPR/BLI kinetics, epitope binning, cross-reactivity", "KD, kon/koff, specificity",
             "protein-science / biophysics CRO", "1–2 mo", "Sub-nM leads, clean specificity"),
            ("Functional in-vitro", "Show mechanism in a disease cell model",
             "disease-relevant cell assay (e.g. matrix-attachment / laminin bridging)", "dose-dependent functional restoration",
             "cell-biology CRO or academic lab", "2 mo", "Dose-dependent effect over isotype control"),
            ("In-vivo efficacy", f"Efficacy + target engagement in {model}",
             f"dose-ranging study in {model}", readout,
             "in-vivo pharmacology CRO", "2–3 mo", "Efficacy vs vehicle with tissue target engagement"),
            ("Developability & IND-enabling", "De-risk for the clinic",
             "immunogenicity, developability, PK, GLP tox, CMC", "developable, tolerated candidate",
             "DMPK/tox + CMC CROs", "9–12 mo", "Clean tox/PK; manufacturable"),
        ]
    # small molecule
    return [
        ("Target validation", "Confirm the target is worth pursuing",
         "ddd KG rationale + Open Targets + genetic evidence", "association score + mechanistic path",
         "in-silico (ddd) + advisor", "2–3 wk", "Strong, ideally genetic, disease link"),
        ("Hit identification", "Find starting chemical matter",
         "ChEMBL actives + docking/virtual screen (ddd), then confirm", "ranked, drug-like hits",
         "biochemical-assay CRO to confirm", "2–3 mo", "≥5 confirmed hits, pChEMBL≥6, drug-like"),
        ("Hit-to-lead", "Improve potency and selectivity",
         "SAR / med-chem cycles with assay", "lead series", "med-chem + assay CRO", "4–6 mo",
         "nM potency with selectivity window"),
        ("ADMET & safety", "De-risk DMPK early",
         "ddd ADMET baselines + in-vitro ADME panel (solubility, microsomal stability, hERG, CYP)",
         "DMPK profile", "in-vitro ADME CRO", "1–2 mo", "Acceptable DMPK, no hERG/CYP flags"),
        ("In-vivo efficacy", f"Prove efficacy in {model}",
         f"PK then dose-ranging efficacy in {model}", readout, "in-vivo pharmacology CRO", "2–3 mo",
         "Efficacy vs vehicle with exposure–response"),
        ("IND-enabling", "Prepare for the clinic", "GLP tox, PK, CMC", "IND package",
         "tox + CMC CROs", "9–12 mo", "Clean GLP tox; manufacturable"),
    ]


_CRO_TASKS = {
    "antibody": [
        ("Antibody-discovery CRO", "epitope/antigen, species, display vs immunization, deliverable panel size"),
        ("Protein-science / biophysics CRO", "SPR or BLI kinetics, epitope binning, specificity panel"),
        ("Cell-biology CRO / academic lab", "disease-relevant functional assay, readout, controls"),
        ("In-vivo pharmacology CRO", "mouse model, dosing, group sizes, randomization, blinded readouts"),
        ("DMPK / tox + CMC CROs", "developability, immunogenicity, PK, GLP tox, manufacturing"),
    ],
    "small molecule": [
        ("Biochemical/biophysical assay CRO", "target protein, assay format, hit-confirmation panel"),
        ("Med-chem CRO", "lead series, SAR cycles, synthesis & purification"),
        ("In-vitro ADME CRO", "solubility, microsomal stability, permeability, hERG, CYP panel"),
        ("In-vivo pharmacology CRO", "mouse model, PK then efficacy, group sizes, blinded readouts"),
        ("Tox + CMC CROs", "GLP tox, PK, manufacturing/scale-up"),
    ],
}


def _outreach(modality: str, disease: str, target: str):
    kind = "antibody" if (modality.startswith("antibod") or modality.startswith("biolog")) else "small molecule"
    drafts = []
    if kind == "antibody":
        drafts.append(("RFQ — antibody discovery", f"""Subject: RFQ — antibody discovery campaign ({target} / {disease})

Hello,

We are a small drug-discovery program seeking a quote for an antibody-discovery
campaign against [TARGET: {target}], an extracellular target implicated in {disease}.

Scope we'd like quoted:
- Platform: [phage/yeast display and/or immunization] — your recommendation welcome.
- Antigen: we can supply [antigen/epitope] or discuss antigen production.
- Deliverable: a characterized panel of [~N] specific binders with sequences.
- Timeline and staged pricing, with go/no-go after primary screening.

Could you share capabilities, a rough timeline, and an estimate? Happy to sign
an NDA first.

Thank you,
[NAME], [PROGRAM]"""))
        drafts.append(("RFQ — in-vivo efficacy", f"""Subject: RFQ — in-vivo efficacy study ({disease})

Hello,

We're seeking a quote for a dose-ranging efficacy study of a candidate antibody
in [{_invivo_model(disease)[0]}].

- Endpoints: {_invivo_model(disease)[1]}.
- Design: dose groups + vehicle, adequate group sizes, randomized and blinded readouts.
- Deliverables: raw data, statistics, and tissue target-engagement analysis.

Please share model availability, a suggested design, timeline and estimate. NDA fine.

Thank you,
[NAME], [PROGRAM]"""))
    else:
        drafts.append(("RFQ — in-vivo efficacy", f"""Subject: RFQ — PK + efficacy study ({disease})

Hello,

We're seeking a quote for PK followed by a dose-ranging efficacy study of a
small-molecule candidate in [{_invivo_model(disease)[0]}].

- Endpoints: {_invivo_model(disease)[1]}.
- Design: PK first, then dose groups + vehicle with blinded readouts and exposure–response.
- Deliverables: raw data, PK/PD analysis, statistics.

Please share model availability, a suggested design, timeline and estimate. NDA fine.

Thank you,
[NAME], [PROGRAM]"""))
    return drafts


_BUDGET = {
    "antibody": [
        ("Antibody discovery campaign", "$150k–$400k"),
        ("Biophysics / SPR panel", "$10k–$30k"),
        ("Functional cell assay", "$30k–$80k"),
        ("In-vivo efficacy study", "$80k–$200k"),
        ("Developability + IND-enabling (later)", "$1M+"),
    ],
    "small molecule": [
        ("Hit confirmation + assay", "$20k–$60k"),
        ("Med-chem hit-to-lead", "$150k–$500k"),
        ("In-vitro ADME panel", "$15k–$40k"),
        ("In-vivo efficacy study", "$60k–$180k"),
        ("IND-enabling tox + CMC (later)", "$1M+"),
    ],
}


def build_plan(disease: str, modality: str, targets=None, lead: str = "") -> tuple:
    kind = "antibody" if (modality.startswith("antibod") or modality.startswith("biolog")) else "small molecule"
    targets = targets or []
    top = targets[0] if targets else (lead or "[lead target]")
    stages = _stages(kind, disease)

    lines = [f"# Program plan: {disease} ({kind})", "",
             "Pipeline-in-a-Person operating plan — from an in-silico lead to a validated candidate. "
             "This is the orchestration layer: what to run, who runs it, in what order, and the go/no-go gates.", "",
             "## Program thesis", ""]
    if targets:
        lines.append(f"- Targets from ddd: {', '.join(targets[:8])}")
    lines += [f"- Lead focus: {top}",
              f"- Modality: {kind}",
              f"- In-vivo model: {_invivo_model(disease)[0]}",
              "- Humans choose targets, judge evidence, and own every go/no-go gate.", "",
              "## Stage plan", "",
              "| # | Stage | Goal | Key experiments | Readout | Who runs it | ~Time | Decision gate |",
              "|---|---|---|---|---|---|---|---|"]
    for i, (name, goal, exp, readout, who, time_, gate) in enumerate(stages, 1):
        lines.append(f"| {i} | {name} | {goal} | {exp} | {readout} | {who} | {time_} | {gate} |")

    lines += ["", "## CRO / vendor task list", "",
              "What to line up, and what each quote needs from you:", ""]
    for who, ask in _CRO_TASKS[kind]:
        lines.append(f"- **{who}** — provide: {ask}")

    lines += ["", "## Outreach drafts", "",
              "Generic RFQ templates — fill the [placeholders] and send after an NDA. They name no vendor.", ""]
    for title, body in _outreach(kind, disease, top):
        lines += [f"### {title}", "", "```", body, "```", ""]

    lines += ["## Milestones & rough budget bands", "",
              "Industry ballparks for a lean program — not quotes. Get real estimates from the RFQs above.", "",
              "| Milestone | Rough cost band |", "|---|---|"]
    for m, band in _BUDGET[kind]:
        lines.append(f"| {m} | {band} |")

    lines += ["", "## Honest limits", "",
              "- This is a plan, not a result. The real asset is capital + CROs + wet-lab data — the same "
              "roadblock in-silico can't cross alone.",
              "- Cost/timeline bands are rough; a real program is gated stage-by-stage on data.",
              "- A solo founder's job here is judgment and coordination; ddd drafts the plan and tracks it, "
              "you make the calls."]

    summary = {"disease": disease, "modality": kind, "stages": len(stages),
               "lead": top, "targets": targets[:8]}
    return summary, lines
