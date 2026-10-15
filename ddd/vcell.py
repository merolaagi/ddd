"""Virtual Cell: a Chorus-style multi-scale view. v0.3 is a schema + static
scaffold that links out to Fate Memory Lab (the cell-biology rule engine) for
the intracellular scale. It does not simulate biology yet; it defines the six
scales and where each one's model will plug in, so the UI and MCP contract are
stable while the engines are filled in.
"""
from __future__ import annotations

SCALES = [
    {"id": "molecular", "name": "Molecular", "zoom": 0,
     "models": "RDKit descriptors, baseline property models (this bench)",
     "status": "live"},
    {"id": "intracellular", "name": "Intracellular", "zoom": 1,
     "models": "Fate Memory Lab rule engine (membrane intake, drug/chemical triggers)",
     "status": "linked", "links_to": "fate-memory-lab"},
    {"id": "intercellular", "name": "Intercellular", "zoom": 2,
     "models": "cell-cell signalling (planned)", "status": "planned"},
    {"id": "tissue", "name": "Tissue", "zoom": 3,
     "models": "tissue-level aggregation (planned)", "status": "planned"},
    {"id": "organ", "name": "Organ", "zoom": 4,
     "models": "organ response (planned)", "status": "planned"},
    {"id": "organism", "name": "Organism / population", "zoom": 5,
     "models": "biological-age conditioning (planned)", "status": "planned"},
]


def overview() -> dict:
    return {
        "name": "Virtual Cell",
        "note": ("A multi-scale scaffold. Biological age is the intended conditioning variable. "
                 "The intracellular scale links to your Fate Memory Lab; other scales are stubs "
                 "with a fixed contract so engines can be added without breaking callers."),
        "scales": SCALES,
        "conditioning": ["biological_age", "perturbation (compound)", "genotype"],
        "live_scales": [s["id"] for s in SCALES if s["status"] == "live"],
    }


def scale(scale_id: str) -> dict:
    for s in SCALES:
        if s["id"] == scale_id:
            return s
    raise KeyError(f"unknown scale '{scale_id}'. Known: {', '.join(s['id'] for s in SCALES)}")
