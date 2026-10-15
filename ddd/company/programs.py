"""The 'headless company': named agent roles that coordinate existing tools
(bench, PrimeKG, armory) into end-to-end discovery programs.

Everything here is deterministic orchestration over public data. A role may
optionally consult a local Ollama model for its written rationale; nothing
requires a paid API, and no step is a substitute for lab validation.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass

from .. import kg, store
from ..chem import describe
from ..runner import predict
from ..tasks import TASKS

ROLES = [
    {"id": "scout", "name": "Target Scout",
     "does": "Ranks candidate targets for a disease from the PrimeKG graph.",
     "tools": ["kg_candidate_targets", "kg_relation_summary"]},
    {"id": "chemist", "name": "Chemist",
     "does": "Describes and triages candidate molecules (properties, QED, Lipinski).",
     "tools": ["describe_molecule"]},
    {"id": "evaluator", "name": "Evaluator",
     "does": "Screens molecules through the ADMET/activity baselines.",
     "tools": ["predict_property"]},
    {"id": "curator", "name": "Curator",
     "does": "Writes the program report and records it to the run log.",
     "tools": ["benchmark_index"]},
]

# ADMET screen applied to every candidate molecule, with a plain-language read.
SCREEN = [
    ("bbbp", "CNS exposure (blood-brain barrier)"),
    ("clintox", "clinical-tox failure risk"),
    ("bace", "BACE-1 activity"),
    ("hiv", "HIV activity"),
]


@dataclass
class Program:
    id: str
    disease: str
    molecules: list[str]
    status: str = "queued"

    def dict(self):
        return self.__dict__


def _flag(task_id, p):
    """Turn a probability into a short verdict for the report."""
    if p is None:
        return "n/a"
    if task_id == "clintox":
        return "watch" if p >= 0.5 else "ok"
    return "high" if p >= 0.5 else "low"


def run_program(disease: str, molecules: list[str], top_targets: int = 15,
                on_step=None) -> dict:
    """Disease -> candidate targets (graph) -> molecule triage + ADMET screen -> report."""
    t0 = time.time()
    steps = []

    def step(role, msg):
        steps.append({"role": role, "msg": msg, "t": round(time.time() - t0, 1)})
        if on_step:
            on_step(steps[-1])

    # Scout
    targets, kg_note = [], None
    step("scout", f"Querying PrimeKG for targets associated with '{disease}'.")
    try:
        targets = kg.disease_candidate_targets(disease, top_targets)
        step("scout", f"Found {len(targets)} candidate target(s)." if targets
             else "No graph match; the knowledge graph may be empty or the disease name unmatched.")
    except Exception as e:
        kg_note = str(e)[:200]
        step("scout", f"Knowledge graph unavailable ({kg_note}). Continuing with molecule screen only.")

    # Chemist + Evaluator
    mol_rows = []
    for smi in molecules:
        d = describe(smi)
        if not d.get("valid"):
            mol_rows.append({"smiles": smi, "valid": False})
            continue
        step("chemist", f"{d['formula']} · MW {d['mol_weight']} · QED {d['qed']} · Lipinski {d['lipinski_violations']}")
        screen = {}
        for task_id, label in SCREEN:
            try:
                p = predict(task_id, [smi], "rf")[0].get("prediction")
            except Exception:
                p = None
            screen[task_id] = {"label": label, "p": p, "flag": _flag(task_id, p)}
        step("evaluator", f"Screened {d['formula']} through {len(SCREEN)} models.")
        mol_rows.append({"smiles": d["smiles"], "valid": True, "props": d, "screen": screen})

    # Curator
    step("curator", "Compiling program report.")
    report = _report(disease, targets, mol_rows, kg_note)
    rid = store.save(f"program:{disease}", "company", "program", "molecules",
                     float(len([m for m in mol_rows if m.get("valid")])),
                     len(targets), len(molecules),
                     {"disease": disease, "n_targets": len(targets), "seconds": round(time.time() - t0, 1)})
    return {"id": rid, "disease": disease, "targets": targets, "molecules": mol_rows,
            "steps": steps, "report": report, "kg_note": kg_note,
            "seconds": round(time.time() - t0, 1)}


def _report(disease, targets, mol_rows, kg_note) -> str:
    L = [f"# Discovery program · {disease}", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by the ddd headless pipeline. "
         "Computational triage on public data; not validated, not a clinical recommendation._", ""]
    L += ["## Candidate targets (PrimeKG)", ""]
    if targets:
        L += ["| Gene | drugs touching it | phenotype links |", "|---|---|---|"]
        L += [f"| {t['gene']} | {t.get('n_drugs', 0)} | {t.get('n_phenotypes', 0)} |" for t in targets]
        L += ["", "_Ranked by how many drugs already touch the gene (druggability proxy); a starting shortlist, not a verdict._"]
    else:
        L.append(f"No targets ({'graph offline: ' + kg_note if kg_note else 'no match in graph'}).")
    L += ["", "## Molecule screen", ""]
    valid = [m for m in mol_rows if m.get("valid")]
    if valid:
        heads = [s[1] for s in SCREEN]
        L += ["| Molecule | MW | QED | Lipinski | " + " | ".join(heads) + " |",
              "|---|---|---|---|" + "---|" * len(heads)]
        for m in valid:
            cells = []
            for task_id, _ in SCREEN:
                s = m["screen"][task_id]
                cells.append(f"{s['flag']} ({s['p']:.2f})" if s["p"] is not None else "n/a")
            p = m["props"]
            L.append(f"| `{m['smiles']}` | {p['mol_weight']} | {p['qed']} | {p['lipinski_violations']} | " + " | ".join(cells) + " |")
    invalid = [m for m in mol_rows if not m.get("valid")]
    if invalid:
        L += ["", "Invalid SMILES: " + ", ".join(f"`{m['smiles']}`" for m in invalid)]
    L += ["", "## How to read this", "",
          "- Flags are baseline model probabilities on scaffold-split public data, not assays.",
          "- `clintox` watch = elevated modelled clinical-tox risk; others high/low = modelled activity/exposure.",
          "- Next step is to pick targets, pull known actives, and retrain the relevant baseline before trusting any screen."]
    return "\n".join(L) + "\n"
