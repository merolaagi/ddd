"""The company charter: who ddd is, what it does, what it won't do.
Edits made from the HQ page are saved to data/company/charter.json."""
from __future__ import annotations

import copy
import json

from .paths import COMPANY_DIR

DEFAULT = {
    "name": "ddd",
    "tagline": "A headless drug-discovery intelligence company",
    "mission": ("Measure, map and explain AI for drug discovery in the open: rigorous benchmarks, "
                "a living biomedical knowledge graph and fast molecule triage, run by agents on our own hardware."),
    "model": ("Headless: no staff. Departments are agents that run playbooks on a schedule or on request. "
              "A human (the founder) sets the charter and reviews reports; Claude can operate the company "
              "through its MCP tools."),
    "scope": [
        "Competitive intelligence on public AI drug-discovery platforms (Insilico DDDBench, O3DC, Virtual Aging Cell)",
        "Open benchmark suite with scaffold splits and documented caveats",
        "PrimeKG knowledge graph for target reasoning",
        "Molecule triage: properties, ADMET-style predictions and red flags",
        "Publishing: an open benchmark index and leaderboard",
    ],
    "non_goals": [
        "No wet lab, clinical or medical advice; predictions are research signals, not assay results",
        "No copying of other companies' proprietary benchmark items, data or models",
        "No recon of logged-in, paywalled or non-public pages",
        "No paid model APIs; local models and the founder's own Claude subscription only",
    ],
    "principles": [
        "Every number ships with its split, sample size and known caveat",
        "Every run leaves a report",
        "Reproducible on one Mac",
    ],
    "watchlist": [
        {"label": "Insilico DDDBench", "url": "https://dddbench.insilico.com/"},
        {"label": "Insilico Virtual Aging Cell (Chorus v3)", "url": "https://virtualcell.insilico.com/chorus_v3.html"},
        {"label": "O3DC Benchmark Index", "url": "https://o3dc.org/"},
    ],
    "focus_diseases": ["idiopathic pulmonary fibrosis", "type 2 diabetes mellitus", "Alzheimer disease"],
}

_PATH = COMPANY_DIR / "charter.json"


def load() -> dict:
    c = copy.deepcopy(DEFAULT)
    if _PATH.exists():
        try:
            c.update(json.loads(_PATH.read_text()))
        except Exception:
            pass
    return c


def save(update: dict) -> dict:
    allowed = set(DEFAULT)
    c = load()
    for k, v in update.items():
        if k in allowed and isinstance(v, type(DEFAULT[k])):
            c[k] = v
    _PATH.write_text(json.dumps(c, indent=2))
    return c
