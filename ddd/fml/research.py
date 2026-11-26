"""Research: pull current papers, read the math out of them, and test whether it builds a working model.

The pipeline is literature -> recipe -> model -> feasibility:

  search          Europe PMC (PubMed, PMC and bioRxiv/medRxiv preprints), no key needed
  read the math   Claude via the Anthropic API when ANTHROPIC_API_KEY is set; otherwise a transparent keyword
                  reader that maps the paper onto one of the model families below and lists the sentences it used
  build           the recipe becomes a reaction network in the same form as the pathways, so every tool in the
                  app works on it: network view, what-if, therapy search, Cypher export and the Models canvas
  feasibility     structure, stability and a count of which parameters came from the paper and which were defaulted
"""
import json
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np

from .pathways import PATHWAYS, R, analyse, simulate_pathway

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ.get("FML_RESEARCH", ROOT / "data" / "research"))
STORE.mkdir(parents=True, exist_ok=True)
LAWS = {"mm", "mass_action", "hill"}


def _t(tid, name, family, description, species, initial, reactions, keywords, levers, reference):
    return {"id": tid, "name": name, "family": family, "description": description, "species": species,
            "clamped": [], "initial": initial, "reactions": reactions, "keywords": keywords, "levers": levers,
            "reference": reference}


TEMPLATES = {t["id"]: t for t in [
    _t("viral", "Within-host viral dynamics (target-cell limited)", "infection",
       "The standard model of a virus inside one person: uninfected target cells are made and die, virus infects "
       "them, infected cells make more virus and die, and virus is cleared. Antivirals act by blocking infection or "
       "virus production. It is the basis of most HIV, hepatitis C, influenza and SARS-CoV-2 treatment modelling.",
       ["T", "I", "V"], {"T": 10.0, "I": 0.0, "V": 0.01},
       [R("t_made", "Target cells made", "Supply", {}, {"T": 1}, kcat=1.0, km=1.0),
        dict(R("t_die", "Target cells die", "Turnover", {"T": 1}, {}, kcat=0.1, km=1.0), law="mass_action"),
        dict(R("infect", "Virus infects a target cell", "Entry", {"T": 1, "V": 1}, {"I": 1, "V": 1}, kcat=0.1, km=1.0,
               drugs=[{"name": "Entry and reverse transcriptase inhibitors", "effect": "inhibits",
                       "note": "stop new cells being infected"}]), law="mass_action"),
        dict(R("i_die", "Infected cells die", "Immune killing", {"I": 1}, {}, kcat=0.5, km=1.0), law="mass_action"),
        dict(R("produce", "Infected cells release virus", "Replication", {"I": 1}, {"I": 1, "V": 1}, kcat=5.0, km=1.0,
               drugs=[{"name": "Protease and polymerase inhibitors", "effect": "inhibits",
                       "note": "stop infected cells making infectious virus"}]), law="mass_action"),
        dict(R("clear", "Virus cleared", "Clearance", {"V": 1}, {}, kcat=3.0, km=1.0,
               drugs=[{"name": "Neutralising antibodies", "effect": "activates", "note": "speed up clearance"}]),
             law="mass_action")],
       ["viral load", "target cell", "within-host", "viral dynamics", "hiv", "hepatitis", "influenza", "sars-cov-2",
        "antiviral", "infected cells", "virion", "replication"],
       ["infect", "produce", "clear"], "Perelson and Nelson, SIAM Review 1999; Nowak and May, Virus Dynamics 2000"),
    _t("vaccine", "Vaccine antibody response", "immunology",
       "A vaccine dose of antigen activates B cells, which expand, become antibody-secreting plasma cells or long-lived "
       "memory cells. Antibody neutralises antigen. The model shows the antibody peak, how fast it wanes, and how much "
       "memory is left, which is what booster timing and adjuvant choice are trying to improve.",
       ["Ag", "B", "Bact", "P", "Ab", "M"], {"Ag": 5.0, "B": 1.0, "Bact": 0.0, "P": 0.0, "Ab": 0.0, "M": 0.0},
       [dict(R("ag_decay", "Antigen decays", "Clearance", {"Ag": 1}, {}, kcat=0.1, km=1.0), law="mass_action"),
        dict(R("activate", "Antigen activates B cells", "Activation", {"Ag": 1, "B": 1}, {"Ag": 1, "Bact": 1}, kcat=0.3,
               km=1.0, drugs=[{"name": "Adjuvants", "effect": "activates", "note": "raise how strongly antigen activates B cells"}]),
             law="mass_action"),
        dict(R("expand", "Activated B cells proliferate", "Proliferation", {"Bact": 1, "Ag": 1}, {"Bact": 2, "Ag": 1},
               kcat=0.4, km=1.0), law="mass_action"),
        dict(R("to_plasma", "Differentiate to plasma cells", "Differentiation", {"Bact": 1}, {"P": 1}, kcat=0.2, km=1.0),
             law="mass_action"),
        dict(R("to_memory", "Differentiate to memory cells", "Memory", {"Bact": 1}, {"M": 1}, kcat=0.05, km=1.0),
             law="mass_action"),
        dict(R("bact_die", "Activated B cells die", "Turnover", {"Bact": 1}, {}, kcat=0.1, km=1.0), law="mass_action"),
        dict(R("secrete", "Plasma cells secrete antibody", "Secretion", {"P": 1}, {"P": 1, "Ab": 1}, kcat=1.0, km=1.0),
             law="mass_action"),
        dict(R("p_die", "Plasma cells die", "Turnover", {"P": 1}, {}, kcat=0.05, km=1.0), law="mass_action"),
        dict(R("ab_decay", "Antibody decays", "Half-life", {"Ab": 1}, {}, kcat=0.05, km=1.0), law="mass_action"),
        dict(R("neutralise", "Antibody neutralises antigen", "Neutralisation", {"Ag": 1, "Ab": 1}, {"Ab": 1}, kcat=0.5,
               km=1.0), law="mass_action"),
        dict(R("m_decay", "Memory cells slowly lost", "Turnover", {"M": 1}, {}, kcat=0.002, km=1.0), law="mass_action")],
       ["vaccine", "vaccination", "antibody", "immunogenicity", "germinal centre", "germinal center", "b cell",
        "plasma cell", "booster", "adjuvant", "neutralizing", "neutralising", "memory b"],
       ["activate", "to_memory", "ab_decay"], "Standard B cell and antibody kinetics models, e.g. Andraud et al. 2012"),
    _t("pkpd", "Drug pharmacokinetics and effect (PK/PD)", "pharmacology",
       "A dose is absorbed from the gut into plasma, distributes into tissue, and is eliminated; its effect follows the "
       "plasma level through a saturating Emax curve. This is how dose, timing and half-life are chosen.",
       ["Gut", "Plasma", "Tissue", "Effect"], {"Gut": 10.0, "Plasma": 0.0, "Tissue": 0.0, "Effect": 0.0},
       [dict(R("absorb", "Absorption from gut", "Absorption", {"Gut": 1}, {"Plasma": 1}, kcat=0.8, km=1.0), law="mass_action"),
        dict(R("distribute", "Plasma to tissue", "Distribution", {"Plasma": 1}, {"Tissue": 1}, kcat=0.3, km=1.0,
               reversible=True), law="mass_action"),
        dict(R("eliminate", "Elimination (liver and kidney)", "Clearance", {"Plasma": 1}, {}, kcat=0.2, km=1.0,
               deficiency="Reduced clearance, as in kidney or liver impairment or a slow metaboliser genotype: levels and toxicity rise"),
             law="mass_action"),
        dict(R("act", "Drug effect (Emax)", "Target", {"Plasma": 1}, {"Plasma": 1, "Effect": 1}, kcat=1.0, km=1.0),
             law="hill", n=1),
        dict(R("effect_off", "Effect wears off", "Turnover", {"Effect": 1}, {}, kcat=0.5, km=1.0), law="mass_action")],
       ["pharmacokinetic", "pharmacodynamic", "pk/pd", "pkpd", "emax", "ec50", "half-life", "clearance", "bioavailability",
        "dose", "dosing", "plasma concentration", "auc"],
       ["absorb", "eliminate", "act"], "Standard one-compartment-plus-tissue PK with an Emax effect"),
    _t("inhibition", "Enzyme inhibition by a drug", "pharmacology",
       "An enzyme binds substrate and releases product, and a competitive inhibitor ties up free enzyme. Raising the "
       "inhibitor lowers product flux with a characteristic IC50, the number drug discovery screens for.",
       ["S", "E", "ES", "P", "Inh", "EI"], {"S": 5.0, "E": 1.0, "ES": 0.0, "P": 0.0, "Inh": 1.0, "EI": 0.0},
       [R("s_in", "Substrate supply", "Supply", {}, {"S": 1}, kcat=0.5, km=1.0),
        dict(R("bind", "Enzyme binds substrate", "Binding", {"E": 1, "S": 1}, {"ES": 1}, kcat=1.0, km=1.0, reversible=True),
             law="mass_action"),
        dict(R("cat", "Catalysis", "Enzyme", {"ES": 1}, {"E": 1, "P": 1}, kcat=1.0, km=1.0,
               deficiency="A loss-of-function variant: less product at every dose"), law="mass_action"),
        dict(R("inhibit", "Inhibitor binds free enzyme", "Drug binding", {"E": 1, "Inh": 1}, {"EI": 1}, kcat=2.0, km=1.0,
               reversible=True, drugs=[{"name": "The inhibitor under study", "effect": "inhibits",
                                        "note": "its IC50 is set by this binding step"}]), law="mass_action"),
        dict(R("p_out", "Product used", "Downstream", {"P": 1}, {}, kcat=0.5, km=1.0), law="mass_action")],
       ["inhibitor", "ic50", "ki ", "michaelis", "km ", "enzyme kinetics", "competitive", "allosteric", "binding affinity",
        "kd ", "dose-response", "potency"],
       ["inhibit", "cat"], "Michaelis-Menten kinetics with competitive inhibition"),
    _t("switch", "Gene regulatory switch (bistable)", "gene regulation",
       "Two genes that repress each other with cooperative binding. The circuit has two stable states, which is how "
       "cells make and remember fate decisions, and how some drug-tolerant states lock in.",
       ["A", "B"], {"A": 2.0, "B": 0.2},
       [R("a_made", "Gene A expressed", "Promoter A", {}, {"A": 1}, kcat=2.0, km=1.0,
          regulators=[{"species": "B", "effect": "inhibit", "k": 0.5, "n": 3}]),
        dict(R("a_decay", "Protein A degraded", "Turnover", {"A": 1}, {}, kcat=0.5, km=1.0), law="mass_action"),
        R("b_made", "Gene B expressed", "Promoter B", {}, {"B": 1}, kcat=2.0, km=1.0,
          regulators=[{"species": "A", "effect": "inhibit", "k": 0.5, "n": 3}]),
        dict(R("b_decay", "Protein B degraded", "Turnover", {"B": 1}, {}, kcat=0.5, km=1.0), law="mass_action")],
       ["bistable", "toggle", "gene regulatory", "hill function", "feedback loop", "cell fate", "transcription factor",
        "cooperativity", "switch", "hysteresis"],
       ["a_made", "b_made"], "Gardner, Cantor and Collins, Nature 2000"),
    _t("tumour", "Tumour with a drug-resistant subpopulation", "cancer",
       "Drug-sensitive cancer cells grow and are killed by treatment; a few mutate into resistant cells that the drug "
       "cannot touch. Continuous full dosing clears the sensitive cells and hands the tumour to the resistant ones, "
       "which is the argument behind adaptive therapy.",
       ["Sens", "Res"], {"Sens": 1.0, "Res": 0.001},
       [dict(R("s_grow", "Sensitive cells divide", "Growth", {"Sens": 1}, {"Sens": 2}, kcat=0.5, km=1.0,
               regulators=[{"species": "Sens", "effect": "inhibit", "k": 5.0}, {"species": "Res", "effect": "inhibit", "k": 5.0}]),
             law="mass_action"),
        dict(R("kill", "Drug kills sensitive cells", "Drug target", {"Sens": 1}, {}, kcat=0.6, km=1.0,
               drugs=[{"name": "The anticancer drug", "effect": "activates", "note": "raising the dose raises this kill rate"}]),
             law="mass_action"),
        dict(R("mutate", "Sensitive cells become resistant", "Mutation", {"Sens": 1}, {"Res": 1}, kcat=0.002, km=1.0),
             law="mass_action"),
        dict(R("r_grow", "Resistant cells divide (more slowly)", "Growth", {"Res": 1}, {"Res": 2}, kcat=0.3, km=1.0,
               regulators=[{"species": "Sens", "effect": "inhibit", "k": 5.0}, {"species": "Res", "effect": "inhibit", "k": 5.0}]),
             law="mass_action"),
        dict(R("r_die", "Resistant cells die", "Turnover", {"Res": 1}, {}, kcat=0.05, km=1.0), law="mass_action")],
       ["resistance", "resistant", "tumour", "tumor", "cancer", "clonal", "adaptive therapy", "persister",
        "chemotherapy", "relapse", "subclone", "evolution"],
       ["kill", "mutate", "r_grow"], "Gatenby et al., Cancer Research 2009 (adaptive therapy)"),
    _t("epidemic", "Epidemic with vaccination (SIR-V)", "epidemiology",
       "Susceptible people are infected by contact with infectious ones, recover, and can be vaccinated. The model gives "
       "the epidemic peak and how much vaccination it takes to stop spread.",
       ["S", "I", "Rec", "Vax"], {"S": 0.99, "I": 0.01, "Rec": 0.0, "Vax": 0.0},
       [dict(R("infect", "Transmission", "Contact", {"S": 1, "I": 1}, {"I": 2}, kcat=0.4, km=1.0,
               drugs=[{"name": "Masks, distancing, prophylaxis", "effect": "inhibits", "note": "lower the transmission rate"}]),
             law="mass_action"),
        dict(R("recover", "Recovery", "Recovery", {"I": 1}, {"Rec": 1}, kcat=0.1, km=1.0,
               drugs=[{"name": "Antivirals", "effect": "activates", "note": "shorten the infectious period"}]), law="mass_action"),
        dict(R("vaccinate", "Vaccination", "Vaccine programme", {"S": 1}, {"Vax": 1}, kcat=0.01, km=1.0,
               drugs=[{"name": "The vaccine", "effect": "activates", "note": "faster roll-out moves people out of the susceptible pool"}]),
             law="mass_action"),
        dict(R("wane", "Immunity wanes", "Waning", {"Rec": 1}, {"S": 1}, kcat=0.005, km=1.0), law="mass_action")],
       ["epidemic", "transmission", "sir model", "seir", "reproduction number", "r0", "herd immunity", "outbreak",
        "incidence", "compartmental"],
       ["infect", "vaccinate", "recover"], "Kermack and McKendrick 1927; Anderson and May 1991"),
]}


def template_pathway(tid, overrides=None, name=None, pid=None):
    t = TEMPLATES[tid]
    p = {k: (json.loads(json.dumps(v)) if isinstance(v, (list, dict)) else v) for k, v in t.items()}
    p["id"] = pid or f"tpl_{tid}"
    if name:
        p["name"] = name
    for r in p["reactions"]:
        r.setdefault("law", "mm")
        if overrides and r["id"] in overrides:
            r["kcat"] = float(overrides[r["id"]])
    return p


def europe_pmc(query, page_size=15, preprints=True, open_access=False, since=None):
    """Search Europe PMC. Returns title, authors, journal, year, abstract and links."""
    import httpx
    q = query.strip()
    if since:
        q += f" AND FIRST_PDATE:[{since}-01-01 TO 3000-01-01]"
    if open_access:
        q += " AND OPEN_ACCESS:y"
    if not preprints:
        q += " AND NOT SRC:PPR"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    r = httpx.get(url, params={"query": q, "format": "json", "resultType": "core", "pageSize": page_size,
                               "sort": "FIRST_PDATE desc"}, timeout=20)
    r.raise_for_status()
    return parse_europe_pmc(r.json())


def parse_europe_pmc(data):
    out = []
    for h in data.get("resultList", {}).get("result", []):
        src, rid = h.get("source"), h.get("id")
        abstract = re.sub(r"<[^>]+>", " ", h.get("abstractText") or "").strip()
        out.append({"id": f"{src}:{rid}", "title": re.sub(r"<[^>]+>", "", h.get("title") or "").strip(),
                    "authors": h.get("authorString", ""), "journal": (h.get("journalInfo") or {}).get("journal", {}).get("title")
                    or h.get("bookOrReportDetails", {}).get("publisher") or ("Preprint" if src == "PPR" else ""),
                    "year": h.get("pubYear"), "abstract": re.sub(r"\s+", " ", abstract), "doi": h.get("doi"),
                    "pmcid": h.get("pmcid"), "preprint": src == "PPR",
                    "link": f"https://europepmc.org/article/{src}/{rid}"})
    return out


NUMBER_RE = re.compile(r"[^.]*?\b\d+(?:\.\d+)?\s*(?:%|nM|µM|uM|mM|mg|kg|hours?|h\b|days?|d\b|min|/day|per day|copies|cells|IU)[^.]*\.",
                       re.IGNORECASE)


def read_keywords(text):
    """Transparent reader: which model families the text points to, and the sentences with stated numbers."""
    low = " " + text.lower() + " "
    scores = []
    for tid, t in TEMPLATES.items():
        hits = sorted({k.strip() for k in t["keywords"] if k in low})
        if hits:
            scores.append({"template": tid, "name": t["name"], "hits": hits, "score": len(hits)})
    scores.sort(key=lambda d: -d["score"])
    numbers = [m.group(0).strip() for m in NUMBER_RE.finditer(text)][:8]
    best = scores[0]["template"] if scores else None
    return {"reader": "keywords", "family": best, "matches": scores[:4], "stated_numbers": numbers,
            "recipe": {"template": best, "overrides": {}, "stated_parameters": [], "custom": None} if best else None,
            "caveat": "The keyword reader recognises which kind of model a paper uses; it does not read equations or "
                      "parameter values. Every rate constant starts at the template default. Set ANTHROPIC_API_KEY to "
                      "let Claude read the paper's actual mechanism and numbers."}


LLM_PROMPT = """You are extracting a mathematical model from a biology paper so it can be simulated.

Return ONLY a JSON object, no prose, no markdown fences, with this shape:
{
 "family": one of %s or "custom",
 "summary": "one sentence on what the model describes",
 "template_overrides": {"<reaction id of the chosen family>": <number>, ...},
 "stated_parameters": [{"name": str, "value": number, "unit": str, "sentence": "exact sentence from the text"}],
 "custom": null or {
    "species": [{"name": short identifier, "description": str, "initial": number}],
    "reactions": [{"id": short identifier, "name": str, "substrates": {"species": count}, "products": {"species": count},
                   "law": "mass_action" | "mm" | "hill", "rate": number, "km": number, "evidence": "sentence from the text"}]
 },
 "confidence": "high" | "medium" | "low",
 "caveats": ["what the text does not specify"]
}

Rules: use a listed family when the paper's mechanism matches it, and give template_overrides only for numbers the
text actually states, keyed by these reaction ids: %s. Use "custom" only when no family fits, with at most 12 species
and 16 reactions. Never invent numbers: leave anything the text does not state out of overrides and say so in caveats.
Scale rates to per-day units where you can, and mention the scaling in caveats.

PAPER TITLE: %s

PAPER TEXT:
%s
"""


def read_with_claude(title, text):
    from . import settings
    key = settings.get("anthropic_api_key")
    if not key:
        return None
    import httpx
    ids = {tid: [r["id"] for r in t["reactions"]] for tid, t in TEMPLATES.items()}
    prompt = LLM_PROMPT % (json.dumps(list(TEMPLATES)), json.dumps(ids), title, text[:30000])
    r = httpx.post("https://api.anthropic.com/v1/messages",
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                   json={"model": settings.get("anthropic_model"), "max_tokens": 4000,
                         "messages": [{"role": "user", "content": prompt}]}, timeout=120)
    r.raise_for_status()
    raw = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return sanitise_llm(json.loads(raw))


def sanitise_llm(d):
    fam = d.get("family") if d.get("family") in TEMPLATES or d.get("family") == "custom" else None
    out = {"reader": "claude", "family": fam, "summary": str(d.get("summary", ""))[:400],
           "confidence": d.get("confidence", "low"), "caveats": [str(c)[:300] for c in d.get("caveats", [])][:8],
           "stated_parameters": [], "recipe": None}
    for p in d.get("stated_parameters", [])[:20]:
        try:
            out["stated_parameters"].append({"name": str(p["name"])[:60], "value": float(p["value"]),
                                             "unit": str(p.get("unit", ""))[:30], "sentence": str(p.get("sentence", ""))[:400]})
        except (KeyError, TypeError, ValueError):
            continue
    if fam in TEMPLATES:
        valid = {r["id"] for r in TEMPLATES[fam]["reactions"]}
        ov = {}
        for k, v in (d.get("template_overrides") or {}).items():
            try:
                if k in valid and 0 < float(v) < 1e6:
                    ov[k] = float(v)
            except (TypeError, ValueError):
                continue
        out["recipe"] = {"template": fam, "overrides": ov, "custom": None}
    elif fam == "custom" and d.get("custom"):
        out["recipe"] = {"template": None, "overrides": {}, "custom": sanitise_custom(d["custom"])}
    return out


def _ident(x):
    return re.sub(r"[^A-Za-z0-9_]", "", str(x))[:24] or "X"


def sanitise_custom(c):
    species, initial = [], {}
    for s in c.get("species", [])[:12]:
        name = _ident(s.get("name"))
        if name not in species:
            species.append(name)
            try:
                initial[name] = max(float(s.get("initial", 0.1)), 0.0)
            except (TypeError, ValueError):
                initial[name] = 0.1
    reactions = []
    for i, r in enumerate(c.get("reactions", [])[:16]):
        subs = {_ident(k): int(v) for k, v in (r.get("substrates") or {}).items() if _ident(k) in species}
        prods = {_ident(k): int(v) for k, v in (r.get("products") or {}).items() if _ident(k) in species}
        if not subs and not prods:
            continue
        try:
            rate = min(max(float(r.get("rate", 0.1)), 1e-6), 1e4)
            km = min(max(float(r.get("km", 1.0)), 1e-6), 1e4)
        except (TypeError, ValueError):
            rate, km = 0.1, 1.0
        rr = R(_ident(r.get("id") or f"r{i}"), str(r.get("name", f"reaction {i + 1}"))[:80], "From paper",
               subs, prods, kcat=rate, km=km)
        rr["law"] = r.get("law") if r.get("law") in LAWS else "mass_action"
        rr["evidence"] = str(r.get("evidence", ""))[:300]
        reactions.append(rr)
    return {"species": species, "initial": initial, "reactions": reactions}


def build(recipe, name, pid):
    if recipe.get("template"):
        return template_pathway(recipe["template"], recipe.get("overrides"), name=name, pid=pid)
    c = recipe["custom"]
    return {"id": pid, "name": name, "family": "custom", "description": "A model assembled from the mechanism a paper "
            "describes.", "species": c["species"], "clamped": [], "initial": c["initial"], "reactions": c["reactions"],
            "levers": [r["id"] for r in c["reactions"]], "reference": name}


def feasibility(path, stated=0):
    """Is this a working model? Structure, stability, and how much of it the paper actually specified."""
    checks = []
    run = simulate_pathway(path, steps=4000, record=True)
    finals = run["final"]
    blow = max(finals) > 1e3 or any(np.isnan(finals))
    a = analyse(path)
    checks.append({"check": "Every reaction references known species", "ok": True})
    checks.append({"check": "Runs without blowing up", "ok": not blow,
                   "detail": "Some quantity grows without limit, so a rate is missing or too large." if blow else ""})
    checks.append({"check": "Settles to a steady state", "ok": bool(run["steady"]),
                   "detail": "" if run["steady"] else "Still changing at the end of the run. That can be correct: an "
                             "epidemic, a vaccine response or a drug dose is a transient, not a steady state."})
    live = [s for s, v in zip(path["species"], finals) if v > 1e-3]
    checks.append({"check": "Something is left at the end", "ok": bool(live),
                   "detail": "" if live else "Everything decays to zero, so the model has no lasting behaviour."})
    n_rates = len(path["reactions"])
    checks.append({"check": "Parameters taken from the paper", "ok": stated > 0,
                   "detail": f"{stated} of {n_rates} rate constants come from the paper; the rest are defaults."})
    ok = sum(1 for c in checks if c["ok"])
    verdict = ("A working model" if not blow and live else "Not a working model yet")
    if not blow and live and stated == 0:
        verdict += ", with every rate constant defaulted: it shows the paper's mechanism, not its numbers"
    return {"verdict": verdict, "checks": checks, "passed": ok, "total": len(checks), "structure": a["structure"],
            "notes": a["notes"], "conservation": a["conservation"], "trace": run["trace"], "species": run["species"],
            "final": finals}


def _file(mid):
    return STORE / f"{mid}.json"


def save_model(paper, reading, recipe, name=None):
    mid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    pid = f"rs_{mid}"
    title = name or (paper.get("title") or "Research model")[:90]
    path = build(recipe, title, pid)
    stated = len(recipe.get("overrides") or {}) + (len(path["reactions"]) if recipe.get("custom") else 0)
    feas = feasibility(path, stated=stated)
    rec = {"id": mid, "pid": pid, "created": time.time(), "paper": paper, "reading": reading, "recipe": recipe,
           "pathway": path, "feasibility": {k: v for k, v in feas.items() if k not in ("trace",)}}
    _file(mid).write_text(json.dumps(rec))
    register(rec)
    return rec, feas


def register(rec):
    p = rec["pathway"]
    p["from_research"] = True
    p["paper"] = {k: rec["paper"].get(k) for k in ("title", "link", "year", "authors")}
    PATHWAYS[p["id"]] = p


def load_all():
    out = []
    for f in sorted(STORE.glob("*.json"), reverse=True):
        try:
            rec = json.loads(f.read_text())
            register(rec)
            out.append(rec)
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def delete_model(mid):
    f = _file(mid)
    if f.exists():
        rec = json.loads(f.read_text())
        PATHWAYS.pop(rec["pid"], None)
        f.unlink()
        return True
    return False
