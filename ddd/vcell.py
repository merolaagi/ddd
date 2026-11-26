"""Virtual Cell: a Chorus-style multi-scale view. A schema + scaffold whose
intracellular scale is the pathway rule engine woven in from Fate Memory Lab
(druggable enzymes and the drugs that modulate them), shown inline. It does not
simulate the other scales yet; it defines the six scales and where each one's
model will plug in, so the UI and MCP contract are stable while the engines are
filled in.
"""
from __future__ import annotations

import copy

SCALES = [
    {"id": "molecular", "name": "Molecular", "zoom": 0,
     "models": "RDKit descriptors, baseline property models (this bench)",
     "status": "live"},
    {"id": "intracellular", "name": "Intracellular", "zoom": 1,
     "models": "Pathway rule engine — enzyme reactions and the drugs that modulate them",
     "status": "live", "view": "pathways"},
    {"id": "intercellular", "name": "Intercellular", "zoom": 2,
     "models": "cell-cell signalling (planned)", "status": "planned"},
    {"id": "tissue", "name": "Tissue", "zoom": 3,
     "models": "tissue-level aggregation (planned)", "status": "planned"},
    {"id": "organ", "name": "Organ", "zoom": 4,
     "models": "organ response (planned)", "status": "planned"},
    {"id": "organism", "name": "Organism / population", "zoom": 5,
     "models": "biological-age conditioning (planned)", "status": "planned"},
]


def _scales():
    """Scales with runtime detail for the UI: the live Molecular scale opens the
    real-time cell view; the live Intracellular scale shows the woven-in pathway
    rule engine inline (druggable enzymes and the drugs that modulate them)."""
    out = []
    for s in SCALES:
        s = copy.deepcopy(s)
        if s["status"] == "live" and s["id"] == "molecular":
            s["view"] = "cell"          # the UI opens the real-time Virtual Cell animation
        out.append(s)
    return out


def overview() -> dict:
    return {
        "name": "Virtual Cell",
        "note": ("A multi-scale scaffold. Biological age is the intended conditioning variable. "
                 "The intracellular scale is the woven-in pathway rule engine (druggable enzymes "
                 "and their modulators); other scales are stubs with a fixed contract so engines "
                 "can be added without breaking callers."),
        "scales": _scales(),
        "conditioning": ["biological_age", "perturbation (compound)", "genotype"],
        "live_scales": [s["id"] for s in SCALES if s["status"] == "live"],
    }


def scale(scale_id: str) -> dict:
    for s in SCALES:
        if s["id"] == scale_id:
            return s
    raise KeyError(f"unknown scale '{scale_id}'. Known: {', '.join(s['id'] for s in SCALES)}")
