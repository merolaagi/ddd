"""Curated bacterial-target seed for antibacterial repurposing.

Antibiotic targets are BACTERIAL proteins (cell-wall synthesis, the ribosome,
DNA gyrase, folate pathway), so they are not in PrimeKG, which is human. This is
a small, validated set of druggable antibacterial targets per priority pathogen,
each with the name most likely to resolve in ChEMBL (`query`). Written into the
graph on demand (source="BacTarget"). A live resource (e.g. BV-BRC / PATRIC,
CARD) can extend it later.
"""
from __future__ import annotations

from .. import kg

BACTERIA = {
    "m. tuberculosis": {
        "display": "Mycobacterium tuberculosis",
        "aliases": ["tb", "mtb", "tuberculosis", "m tuberculosis", "m. tuberculosis", "mycobacterium tuberculosis"],
        "disease": "tuberculosis",
        "targets": [
            {"gene": "InhA", "role": "enoyl-ACP reductase (mycolic acid synthesis)", "query": "InhA"},
            {"gene": "DprE1", "role": "decaprenylphosphoryl-β-D-ribose oxidase", "query": "Decaprenylphosphoryl-beta-D-ribose oxidase"},
            {"gene": "RpoB", "role": "RNA polymerase β (rifampicin)", "query": "DNA-directed RNA polymerase subunit beta"},
            {"gene": "GyrB", "role": "DNA gyrase B (fluoroquinolone-class)", "query": "DNA gyrase subunit B"},
            {"gene": "KasA", "role": "β-ketoacyl-ACP synthase", "query": "3-oxoacyl-[acyl-carrier-protein] synthase 1"},
        ]},
    "s. aureus": {
        "display": "Staphylococcus aureus",
        "aliases": ["mrsa", "staph", "s aureus", "s. aureus", "staphylococcus aureus"],
        "disease": "staphylococcal infectious disease",
        "targets": [
            {"gene": "PBP2a (mecA)", "role": "penicillin-binding protein 2a (methicillin resistance)", "query": "Penicillin-binding protein 2a"},
            {"gene": "GyrB", "role": "DNA gyrase B", "query": "DNA gyrase subunit B"},
            {"gene": "DHFR", "role": "dihydrofolate reductase (trimethoprim)", "query": "Dihydrofolate reductase"},
            {"gene": "FabI", "role": "enoyl-ACP reductase", "query": "Enoyl-[acyl-carrier-protein] reductase [NADH] FabI"},
        ]},
    "p. aeruginosa": {
        "display": "Pseudomonas aeruginosa",
        "aliases": ["pseudomonas", "p aeruginosa", "p. aeruginosa"],
        "disease": "Pseudomonas aeruginosa infectious disease",
        "targets": [
            {"gene": "GyrB", "role": "DNA gyrase B", "query": "DNA gyrase subunit B"},
            {"gene": "LpxC", "role": "UDP-3-O-acyl-GlcNAc deacetylase (LPS/outer membrane)", "query": "UDP-3-O-acyl-N-acetylglucosamine deacetylase"},
            {"gene": "MurA", "role": "enolpyruvyl transferase (peptidoglycan)", "query": "UDP-N-acetylglucosamine 1-carboxyvinyltransferase"},
        ]},
    "e. coli": {
        "display": "Escherichia coli",
        "aliases": ["e coli", "e. coli", "escherichia coli"],
        "disease": "Escherichia coli infectious disease",
        "targets": [
            {"gene": "GyrB", "role": "DNA gyrase B", "query": "DNA gyrase subunit B"},
            {"gene": "DHFR (folA)", "role": "dihydrofolate reductase", "query": "Dihydrofolate reductase"},
            {"gene": "MurA", "role": "peptidoglycan biosynthesis", "query": "UDP-N-acetylglucosamine 1-carboxyvinyltransferase"},
            {"gene": "FabI", "role": "enoyl-ACP reductase", "query": "Enoyl-[acyl-carrier-protein] reductase [NADH]"},
        ]},
}


def resolve_bacterium(name: str):
    n = (name or "").strip().lower()
    if not n:
        return None, None
    for key, v in BACTERIA.items():
        if n == key or n in v["aliases"] or n == v["display"].lower():
            return key, v
    for key, v in BACTERIA.items():
        hay = v["aliases"] + [key, v["display"].lower()]
        if any(a in n or n in a for a in hay):
            return key, v
    return None, None


def list_bacteria():
    return [{"key": k, "display": v["display"], "disease": v["disease"], "targets": len(v["targets"])}
            for k, v in BACTERIA.items()]


def link_bacterium_targets(bacterium: str, write: bool = True) -> dict:
    key, v = resolve_bacterium(bacterium)
    if not v:
        return {"found": False, "bacterium": bacterium, "known": list(BACTERIA)}
    written = 0
    if write:
        try:
            bidx = kg.ensure_node(v["display"], "exposure", "BacTarget", {"kind": "bacterium"})
            for t in v["targets"]:
                tidx = kg.ensure_node(t["gene"], "gene/protein", "BacTarget", {"bacterial": True})
                kg.merge_edge(bidx, tidx, "drug_target", "antibacterial target", "BacTarget", {"role": t["role"]})
                written += 1
        except Exception:
            written = 0
    return {"found": True, "key": key, "bacterium": v["display"], "disease": v["disease"],
            "targets": v["targets"], "written": written}
