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

import re

# generic protein/gene words the text-miner tags that are not useful targets
_STOP = {"protein", "proteins", "gene", "genes", "receptor", "receptors", "enzyme",
         "enzymes", "antibody", "antibodies", "peptide", "peptides", "albumin",
         "dna", "rna", "mrna", "trna", "rrna", "cytokine", "cytokines", "hormone",
         "antigen", "antigens", "ligand", "ligands", "kinase", "kinases", "channel",
         "transporter", "factor", "complex", "subunit", "domain", "isoform"}

# common metabolites / elements / reagents the chemical miner tags — never drug leads
_CHEM_STOP = {"oxygen", "water", "iron", "glucose", "hydrogen", "lipid", "lipids",
              "zinc", "salt", "polyunsaturated fatty acid", "fatty acid", "amino acid",
              "amino acids", "calcium", "sodium", "potassium", "nitrogen", "carbon",
              "cholesterol", "atp", "adp", "amp", "nad", "nadh", "nadph", "gtp",
              "reactive oxygen species", "hydrogen peroxide", "nitric oxide",
              "superoxide", "peroxide", "copper", "magnesium", "manganese", "chloride",
              "phosphate", "sulfate", "sugar", "sucrose", "fructose", "lactate",
              "pyruvate", "ethanol", "methanol", "dmso", "glutamate", "glycine",
              "serine", "alanine", "glutamine", "cysteine", "glutathione", "urea",
              "ammonia", "bicarbonate", "carbon dioxide", "selenium", "iodine",
              "fluoride", "nitrate", "oxygen species", "metal", "ion", "ions",
              "electron", "proton", "acid", "base", "cholesterol", "creatinine"}

# systematic locus-tag / non-human-symbol patterns to drop (rice, yeast, E. coli, Ensembl…)
_LOCUS = re.compile(
    r"^(os\d|sp[abc][abc]\d|y[a-p][lr]\d|b\d{4}|jw\d{4}|at\d?g\d|loc\d|ensg\d|slr\d|sll\d|"
    r"pa\d{4}|rv\d{3,}|mb\d{3,}|cg\d{4,}|dmel|zmo|psed|cbu|n\d{3,})", re.I)

_ACTION_TAIL = re.compile(
    r"\b(and|to|,|then|that|which|&)?\s*"
    r"(find|finds?|discover|identif\w+|get|suggest\w*|design\w*|make|yield\w*|give|"
    r"come up with|look for|search for)\b.*$", re.I)
_TRAIL_NOUN = re.compile(
    r"\b(molecules?|compounds?|drugs?|candidates?|targets?|treatments?|therap\w+|cures?|"
    r"inhibitors?|leads?)\b[\s,]*$", re.I)


def _clean_disease(s: str) -> str:
    """Strip action phrasing that leaks in from chat, e.g. 'ALS and find molecules' -> 'ALS'."""
    s = (s or "").strip()
    s2 = _ACTION_TAIL.sub("", s).strip(" ,.-")
    s2 = _TRAIL_NOUN.sub("", s2).strip(" ,.-")
    return s2 or s


def _plausible_target(name: str) -> bool:
    n = name.strip()
    low = n.lower()
    if low in _STOP or len(n) < 2 or len(n) > 28:
        return False
    if _LOCUS.match(n):
        return False
    # protein-name phrases (with spaces) are allowed; single tokens must look like a gene symbol
    if " " not in n:
        if not re.match(r"^[A-Za-z][A-Za-z0-9]{1,13}$", n):
            return False
        if re.search(r"\d{3,}", n):          # long digit runs => systematic id
            return False
    return True


def discover(ctx, disease: str, max_papers: int = 20, max_targets: int = 6,
             max_compounds: int = 15, design: bool = False):
    from ..sources import literature
    from .playbooks import _compounds_for_targets, _compound_flags
    disease = _clean_disease(disease)
    if not disease:
        raise ValueError("Give a disease or condition to read the literature on.")

    # scope the search to the disease so text mining stays on-topic; relevance-sorted
    query = f'"{disease}" AND (treatment OR target OR therap* OR drug OR inhibitor)'
    ctx.log(f"Searching Europe PMC for '{disease}'")
    s = literature.search(query, limit=max_papers, sort="")
    papers = s.get("papers") or []
    if not papers:                       # fall back to a plain disease query
        s = literature.search(disease, limit=max_papers, sort="CITED desc")
        papers = s.get("papers") or []
    if not papers:
        raise ValueError(f"no papers found for '{disease}'")

    ctx.log(f"Reading {len(papers)} papers — text-mining genes/proteins & chemicals")
    ann = literature.annotate(papers)
    raw_genes = ann.get("gene_proteins", [])
    genes = [g for g in raw_genes if _plausible_target(g["name"])]
    dropped = len(raw_genes) - len(genes)
    chems = [c for c in ann.get("chemicals", [])
             if c["name"].lower() not in _CHEM_STOP and c["name"].lower() not in _STOP
             and not re.match(r"^[A-Z][a-z]?\d?$", c["name"])      # element symbols
             and len(c["name"]) > 2]
    if not genes:
        raise ValueError("the papers yielded no usable protein/gene targets "
                         "(try a more specific disease term)")

    # keep targets backed by >1 paper when we have enough signal
    strong = [g for g in genes if g["papers"] >= 2] or genes
    # consider a few extra; we'll keep the ones that actually have chemical matter
    cand = strong[:max_targets + 6]
    mx = max((g["papers"] for g in cand), default=1)
    probe = [{"gene": g["name"], "weight": round(g["papers"] / mx, 3), "papers": g["papers"]}
             for g in cand]
    ctx.log("Candidate targets: " + ", ".join(f"{t['gene']}({t['papers']}p)" for t in probe))

    ranked, desc, tox, sol, agg = _compounds_for_targets(ctx, probe, max_compounds)
    # a target is "real" for our purposes if it has druggable chemical matter in ChEMBL;
    # keep those, preserving evidence order, and fall back to top-evidence if none do
    with_matter = [t for t in probe if t.get("actives", 0) > 0]
    targets = (with_matter or probe)[:max_targets]
    ctx.log("Targets with chemical matter: "
            + (", ".join(f"{t['gene']}({t['actives']})" for t in with_matter[:max_targets]) or "none"))

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
    if dropped:
        lines.append("")
        lines.append(f"_Filtered out {dropped} text-mining noise tag(s) (cross-organism locus IDs, "
                     "generic terms) and kept targets with druggable chemical matter._")

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
