"""Curated virus -> host-factor seed for the antiviral repurposing pipeline.

This is NOT a comprehensive interactome. It is a small, literature-grounded set
of host dependency and entry factors for the major respiratory viruses — the
human proteins each virus needs, which are the host-directed antiviral targets
(host-directed hits resist viral mutation far better than direct-acting ones).

Written into the knowledge graph on demand (source="VirusHost"), so virus->host
edges show up in the graph and provenance. A live VirHostNet / IntAct connector
can extend or replace this later; the pipeline doesn't change.
"""
from __future__ import annotations

from .. import kg

VIRUSES = {
    "sars-cov-2": {
        "display": "SARS-CoV-2",
        "aliases": ["covid", "covid-19", "covid19", "sars-cov-2", "sarscov2", "sars cov 2", "coronavirus"],
        "disease": "COVID-19",
        "hosts": [
            {"gene": "ACE2", "role": "entry receptor"},
            {"gene": "TMPRSS2", "role": "entry protease"},
            {"gene": "CTSL", "role": "endosomal entry protease"},
            {"gene": "FURIN", "role": "spike priming protease"},
            {"gene": "NRP1", "role": "entry co-receptor"},
        ]},
    "influenza": {
        "display": "Influenza A",
        "aliases": ["influenza", "flu", "influenza a", "h1n1", "h3n2", "influenza virus"],
        "disease": "influenza",
        "hosts": [
            {"gene": "TMPRSS2", "role": "HA cleavage / entry protease"},
            {"gene": "ANP32A", "role": "polymerase cofactor"},
            {"gene": "ANP32B", "role": "polymerase cofactor"},
            {"gene": "ST6GAL1", "role": "sialic-acid receptor synthesis"},
            {"gene": "KPNA1", "role": "nuclear import of viral RNP"},
        ]},
    "rsv": {
        "display": "Respiratory syncytial virus",
        "aliases": ["rsv", "respiratory syncytial virus", "hrsv"],
        "disease": "respiratory syncytial virus infectious disease",
        "hosts": [
            {"gene": "NCL", "role": "entry receptor (nucleolin)"},
            {"gene": "IGF1R", "role": "entry co-receptor"},
            {"gene": "RHOA", "role": "fusion / syncytium formation"},
            {"gene": "ICAM1", "role": "cell adhesion / spread"},
        ]},
    "rhinovirus": {
        "display": "Rhinovirus",
        "aliases": ["rhinovirus", "common cold virus", "hrv"],
        "disease": "common cold",
        "hosts": [
            {"gene": "ICAM1", "role": "major-group receptor"},
            {"gene": "LDLR", "role": "minor-group receptor"},
            {"gene": "CDHR3", "role": "RV-C receptor"},
        ]},
    "parainfluenza": {
        "display": "Human parainfluenza virus",
        "aliases": ["parainfluenza", "hpiv", "human parainfluenza"],
        "disease": "parainfluenza",
        "hosts": [
            {"gene": "ST6GAL1", "role": "sialic-acid receptor"},
            {"gene": "TMPRSS2", "role": "F-protein cleavage"},
        ]},
}


def resolve_virus(name: str):
    n = (name or "").strip().lower()
    if not n:
        return None, None
    for key, v in VIRUSES.items():
        if n == key or n in v["aliases"] or n == v["display"].lower():
            return key, v
    for key, v in VIRUSES.items():
        hay = v["aliases"] + [key, v["display"].lower()]
        if any(a in n or n in a for a in hay):
            return key, v
    return None, None


def list_viruses():
    return [{"key": k, "display": v["display"], "disease": v["disease"], "hosts": len(v["hosts"])}
            for k, v in VIRUSES.items()]


def link_virus_hosts(virus: str, write: bool = True) -> dict:
    """Resolve a virus to its curated host factors, and (best-effort) write the
    virus -> host-factor edges into the graph tagged source=VirusHost."""
    key, v = resolve_virus(virus)
    if not v:
        return {"found": False, "virus": virus, "known": list(VIRUSES)}
    written = 0
    if write:
        try:
            vidx = kg.ensure_node(v["display"], "exposure", "VirusHost", {"kind": "virus"})
            for h in v["hosts"]:
                gidx = kg.ensure_node(h["gene"], "gene/protein", "VirusHost")
                kg.merge_edge(vidx, gidx, "host_factor", "host factor", "VirusHost", {"role": h["role"]})
                written += 1
        except Exception:
            written = 0  # graph offline is fine; the seed is still usable
    return {"found": True, "key": key, "virus": v["display"], "disease": v["disease"],
            "hosts": v["hosts"], "written": written}
