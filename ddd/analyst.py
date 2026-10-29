"""The Analyst — a plain-language front door to ddd.

Ask a question; the analyst works out what you mean, shows a short plan, asks a
clarifying question with options when it's ambiguous, then runs the right ddd
capability and hands back the result. Rules-based intent routing (no model API),
so it's instant and free; it orchestrates the same playbooks and graph tools the
rest of the company uses.
"""
from __future__ import annotations

import re

from . import kg, kg_model
from .company import engine
from .sources import literature

_VIRUS = re.compile(r"\b(viral|virus|influenza|flu|covid|sars|sars-cov-2|rsv|rhinovirus|parainfluenza|coronavirus)\b", re.I)
_BACT = re.compile(r"\b(bacteri\w*|tuberculosis|tb|mycobacter\w*|staph\w*|mrsa|aureus|e\.?\s?coli|escherichia|pseudomonas|aeruginosa)\b", re.I)
_ANTIBODY = re.compile(r"\b(antibod\w+|biologic\w*|mab|monoclonal)\b", re.I)
_SMALLMOL = re.compile(r"\b(small[- ]molecule|small molecules?)\b", re.I)


def _tail(q: str) -> str:
    """Best-effort entity: the text after a connector word, trimmed of clauses."""
    m = re.search(r"\b(?:for|of|against|about|on|to)\b\s+(.+)$", q, re.I)
    s = (m.group(1) if m else q).strip().strip("?.!,")
    s = re.split(r"\b(and then|then|using|with)\b|[,;?]", s, flags=re.I)[0].strip()
    # drop leading intent words that sometimes leak in
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.I).strip()
    return s


def _two(q: str):
    m = re.search(r"between\s+(.+?)\s+and\s+(.+)$", q, re.I) or \
        re.search(r"\bis\s+(.+?)\s+(?:linked|related|associated|connected)\s+to\s+(.+)$", q, re.I)
    if m:
        a = m.group(1).strip().strip("?.!,")
        b = re.split(r"[,;?]", m.group(2))[0].strip().strip("?.!,")
        return a, b
    return None, None


def _start(playbook: str, params: dict, plan: list, say: str) -> dict:
    try:
        rid = engine.start(playbook, params, trigger="analyst")
        return {"kind": "run", "say": say, "plan": plan, "playbook": playbook, "run_id": rid}
    except Exception as e:
        return {"kind": "answer", "say": f"Couldn't start that: {str(e)[:160]}"}


HELP = ("I can: rank **targets** for a disease, find **repurposing** candidates "
        "(human disease, virus, or bacterium), check a **hypothesis** between two things, "
        "**predict novel targets**, **design novel molecules** for a target, **build the cell**, "
        "**map the graph**, draft a **program plan**, "
        "or run a **benchmark**. Try: *\"repurposing candidates for rheumatoid arthritis\"* or "
        "*\"program plan for LAMA2 muscular dystrophy\"*.")


def handle(q: str, choice: str | None = None, pending: dict | None = None) -> dict:
    q = (q or "").strip()
    pending = pending or {}

    # ---- resolve a pending clarification ----
    if choice and pending:
        intent = pending.get("intent")
        ent = pending.get("entity", "")
        if intent == "repurpose":
            if choice == "virus":
                return _start("antiviral_repurposing", {"virus": ent},
                              ["Resolve virus → host factors", "Pull ChEMBL inhibitors", "Triage & rank", "Write dossier"],
                              f"Running antiviral repurposing for **{ent}**.")
            if choice == "bacterium":
                return _start("antibacterial_repurposing", {"bacterium": ent},
                              ["Resolve bacterial targets", "Pull ChEMBL inhibitors", "Triage & rank", "Write dossier"],
                              f"Running antibacterial repurposing for **{ent}**.")
            return _start("disease_repurposing", {"disease": ent},
                          ["Rank targets (graph + Open Targets)", "Pull ChEMBL modulators", "Triage & rank", "Write dossier"],
                          f"Running repurposing for **{ent}**.")
        if intent == "program_plan":
            return _start("program_plan", {"disease": ent, "modality": choice},
                          ["Pull target ranking", "Stage experiments", "List CROs & draft outreach", "Set milestones & gates"],
                          f"Drafting a **{choice}** program plan for **{ent}**.")
        return {"kind": "answer", "say": HELP}

    if not q:
        return {"kind": "answer", "say": HELP}
    ql = q.lower()

    # ---- program plan ----
    if re.search(r"\bprogram plan\b|\bplan a program\b|\bdevelop a (drug|therapy|program|treatment)\b|how (would|do) i develop", ql):
        ent = _tail(q)
        if _ANTIBODY.search(ql):
            mod = "antibody"
        elif _SMALLMOL.search(ql):
            mod = "small molecule"
        else:
            return {"kind": "clarify", "say": f"A program plan for **{ent}** — which modality?",
                    "options": [{"label": "Small molecule", "value": "small molecule"},
                                {"label": "Antibody / biologic", "value": "antibody"}],
                    "pending": {"intent": "program_plan", "entity": ent}}
        return _start("program_plan", {"disease": ent, "modality": mod},
                      ["Pull target ranking", "Stage experiments", "List CROs & draft outreach", "Set milestones & gates"],
                      f"Drafting a **{mod}** program plan for **{ent}**.")

    # ---- literature ----
    if re.search(r"literature|papers?|publications?|what'?s known about|pubmed|studies on|evidence (for|on)", ql):
        ent = _tail(q)
        try:
            r = literature.search(ent, 8)
            return {"kind": "data", "say": f"Key literature on **{ent}**:",
                    "plan": ["Search Europe PMC", "Rank by citations"],
                    "result": {"type": "literature", "data": r}}
        except Exception as e:
            return {"kind": "answer", "say": f"Literature search failed: {str(e)[:140]}"}

    # ---- DEG / gene-set interpretation ----
    if re.search(r"\bdeg\b|differential expression|gene set|interpret these genes|rna-?seq", ql):
        return {"kind": "answer", "say": "Paste your gene list into the **DEG / gene-set interpretation** panel below and run it — "
                "it maps the set onto diseases, pathways and druggable genes."}

    # ---- generative design / de-novo molecules (before repurposing, which also matches "compounds for") ----
    if re.search(r"\b(design|generate|create|invent|evolve|come up with|make me)\b.{0,30}\b(molecule|compound|ligand|drug|inhibitor|candidate|chemistry|chemical|analog\w*)s?\b", ql) \
            or re.search(r"\b(de[- ]?novo|generative)\b.{0,20}\b(design|chemistry|molecule)", ql) \
            or re.search(r"\bnew (molecule|compound|inhibitor|drug|ligand)s?\b", ql):
        ent = _tail(q)
        return _start("generative_design", {"target": ent},
                      ["Seed from target's ChEMBL actives", "Evolve via graph genetic algorithm",
                       "Score by QED + synthesizability + similarity", "Rank novel candidates"],
                      f"Designing novel molecules for **{ent}**.")

    # ---- repurposing / "drugs for" / "treatment for" ----
    if re.search(r"\brepurpos\w+|candidate drugs?|drugs? for|treat(ment)? for|therap(y|ies) for|compounds? for|cure for\b", ql):
        ent = _tail(q)
        if _VIRUS.search(ql):
            return _start("antiviral_repurposing", {"virus": ent},
                          ["Resolve virus → host factors", "Pull ChEMBL inhibitors", "Triage & rank", "Write dossier"],
                          f"Running antiviral repurposing for **{ent}**.")
        if _BACT.search(ql):
            return _start("antibacterial_repurposing", {"bacterium": ent},
                          ["Resolve bacterial targets", "Pull ChEMBL inhibitors", "Triage & rank", "Write dossier"],
                          f"Running antibacterial repurposing for **{ent}**.")
        return {"kind": "clarify", "say": f"Repurposing for **{ent}** — what kind?",
                "options": [{"label": "Human disease", "value": "disease"},
                            {"label": "Virus", "value": "virus"},
                            {"label": "Bacterium", "value": "bacterium"}],
                "pending": {"intent": "repurpose", "entity": ent}}

    # ---- target dossier ----
    if re.search(r"\btargets? for\b|target dossier|key genes? for|what (are the )?targets", ql) \
            and not re.search(r"novel|predict|new target", ql):
        ent = _tail(q)
        return _start("bio_target_dossier", {"disease": ent},
                      ["Find disease node", "Rank targets (Open Targets + PrimeKG)", "Pull ChEMBL actives", "Write dossier"],
                      f"Building a target dossier for **{ent}**.")

    # ---- novel target prediction ----
    if re.search(r"novel targets?|predict targets?|new targets?|what (does|might) the graph (imply|predict)", ql):
        ent = _tail(q)
        try:
            r = kg_model.predict_links(ent, "gene", 12)
            if r.get("error"):
                return {"kind": "answer", "say": r["error"]}
            return {"kind": "data", "say": f"Novel targets the graph predicts for **{ent}**:",
                    "plan": ["Resolve disease", "Score novel genes by Adamic-Adar"],
                    "result": {"type": "predict", "data": r}}
        except Exception as e:
            return {"kind": "answer", "say": f"Graph unavailable: {str(e)[:140]}"}

    # ---- hypothesis ----
    if re.search(r"hypothesis|connection between|link(ed)? between|is .+ (linked|related|associated|connected) to", ql):
        a, b = _two(q)
        if not a or not b:
            return {"kind": "answer", "say": "Tell me the two things, e.g. *\"is TERT linked to idiopathic pulmonary fibrosis\"*."}
        try:
            r = kg_model.hypothesis(a, b)
            if not r.get("found"):
                return {"kind": "answer", "say": r.get("error", "Couldn't resolve those in the graph.")}
            return {"kind": "data", "say": f"Hypothesis **{a} ↔ {b}**:",
                    "plan": ["Resolve both entities", "Shared neighbours + shortest path", "Link-prediction scores"],
                    "result": {"type": "hypothesis", "data": r}}
        except Exception as e:
            return {"kind": "answer", "say": f"Graph unavailable: {str(e)[:140]}"}

    # ---- protein structure (AlphaFold view) ----
    if re.search(r"\b(structure|alphafold|alpha ?fold|fold|3d structure|ribbon|cartoon|pae|crystal structure)\b", ql) \
            and re.search(r"\b(structure|alphafold|alpha ?fold|fold|protein|ribbon|cartoon|pae)\b", ql):
        ent = _tail(q)
        return {"kind": "ui", "say": f"Fetching the AlphaFold structure for **{ent}**.",
                "plan": ["Resolve target → UniProt", "Fetch AlphaFold model + PAE", "Render pLDDT ribbon + error map"],
                "ui": {"action": "structure", "entity": ent}}

    # ---- build the cell ----
    if re.search(r"build (a |the )?cell|show (the )?cell|virtual cell|cell (for|of)", ql):
        ent = _tail(q)
        return {"kind": "ui", "say": f"Building the cell for **{ent}**.",
                "plan": ["Pull targets + subcellular compartments", "Render the living cell"],
                "ui": {"action": "cell", "entity": ent}}

    # ---- map the graph ----
    if re.search(r"\bmap\b|graph of|neighbou?rs of|visuali[sz]e|show (the )?graph", ql):
        ent = _tail(q)
        return {"kind": "ui", "say": f"Mapping **{ent}** in the knowledge graph.",
                "plan": ["Fetch the subgraph", "Render it"],
                "ui": {"action": "graph", "entity": ent}}

    # ---- benchmark ----
    if re.search(r"benchmark|leaderboard|ddd score|evaluate the models|run.*eval", ql):
        return _start("bench_full_sweep", {"include_llm": False},
                      ["Score every baseline on every task", "Rank by ddd score", "Flag regressions"],
                      "Running the full benchmark sweep.")

    return {"kind": "answer", "say": "I'm not sure which analysis you mean. " + HELP}
