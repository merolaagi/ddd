"""The company engine: departments, playbook registry, run execution,
scheduler and the HQ overview."""
from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .. import kg, store
from ..armory import rea as rea_bridge
from ..armory.paths import list_artifacts
from ..config import OLLAMA_URL
from ..models import ollama_models
from ..tasks import TASKS
from . import charter as charter_mod
from . import db, playbooks as pb
from .paths import REPORTS, report_path

NARRATOR = os.environ.get("DDD_NARRATOR", "").strip()


@dataclass
class Department:
    id: str
    name: str
    agent: str
    mission: str
    kpis: list


DEPARTMENTS = [
    Department("intel", "Intelligence", "Scout",
               "Watch public AI drug-discovery platforms, capture what they expose and report what changed.",
               ["sites watched", "changes detected"]),
    Department("eval", "Evaluation", "Judge",
               "Run the benchmark suite on every model we can run, keep the leaderboard honest and catch regressions.",
               ["tasks covered", "models ranked"]),
    Department("bio", "Target Biology", "Cartographer",
               "Turn the knowledge graph into ranked, explained target hypotheses for the focus diseases.",
               ["dossiers written"]),
    Department("chem", "Chemistry", "Triage",
               "Profile candidate molecules fast: properties, ADMET-style predictions and red flags.",
               ["molecules triaged"]),
    Department("programs", "Discovery Programs", "Lead",
               "Run end-to-end programs: Scout ranks targets, Chemist and Evaluator screen molecules, Curator writes it up.",
               ["programs run"]),
    Department("pub", "Publishing", "Herald",
               "Publish the weekly digest, the open benchmark index and the public leaderboard page.",
               ["digests published"]),
]
DEPT = {d.id: d for d in DEPARTMENTS}


@dataclass
class Playbook:
    name: str
    title: str
    department: str
    description: str
    params: dict = field(default_factory=dict)
    fn: Callable = None

    def meta(self):
        return {k: v for k, v in self.__dict__.items() if k != "fn"}


PLAYBOOKS = {p.name: p for p in [
    Playbook("intel_watchlist_sweep", "Watchlist sweep", "intel",
             "Recon every watchlist page (HAR + analysis + REA) and diff against the previous capture.",
             {"run_rea": {"type": "bool", "default": True}}, pb.intel_watchlist_sweep),
    Playbook("intel_recon", "Recon one page", "intel",
             "Recon a single public page and compare it with its last capture.",
             {"url": {"type": "str"}, "run_rea": {"type": "bool", "default": True}}, pb.intel_recon),
    Playbook("bench_full_sweep", "Full benchmark sweep", "eval",
             "Every available task × every baseline (and optionally every Ollama model); flags regressions.",
             {"include_llm": {"type": "bool", "default": False}, "llm_sample": {"type": "int", "default": 50}},
             pb.bench_full_sweep),
    Playbook("bio_target_dossier", "Target dossier", "bio",
             "Ranked candidate targets, known drugs and graph context for a disease (needs Neo4j + PrimeKG).",
             {"disease": {"type": "str", "default": "idiopathic pulmonary fibrosis"}, "top": {"type": "int", "default": 12},
              "enrich": {"type": "bool", "default": True}},
             pb.bio_target_dossier),
    Playbook("chem_molecule_triage", "Molecule triage", "chem",
             "Properties, six baseline predictions and red flags for up to 50 SMILES.",
             {"smiles": {"type": "text", "default": "CC(=O)Oc1ccccc1C(=O)O\nCN1CCC[C@H]1c1cccnc1"},
              "label": {"type": "str", "default": ""}}, pb.chem_molecule_triage),
    Playbook("antiviral_repurposing", "Antiviral repurposing", "programs",
             "For a respiratory virus: host-directed targets (entry/replication factors) → their ChEMBL "
             "inhibitors as repurposing candidates, triaged and ranked. Host-directed, not validated.",
             {"virus": {"type": "str", "default": "sars-cov-2"},
              "top": {"type": "int", "default": 8},
              "max_compounds": {"type": "int", "default": 20}},
             pb.antiviral_repurposing),
    Playbook("antibacterial_repurposing", "Antibacterial repurposing", "programs",
             "For a priority bacterium: validated bacterial targets (cell wall, DNA gyrase, folate, FAS) → "
             "their ChEMBL inhibitors as repurposing candidates, triaged and ranked. Not validated.",
             {"bacterium": {"type": "str", "default": "m. tuberculosis"},
              "max_compounds": {"type": "int", "default": 20}},
             pb.antibacterial_repurposing),
    Playbook("disease_repurposing", "Disease repurposing", "programs",
             "Any human disease (autoimmune, cancer, metabolic, neuro…): targets from the graph + Open "
             "Targets (plus novel link-prediction targets) → ChEMBL modulators ranked as repurposing "
             "candidates. Needs Neo4j + PrimeKG. Not validated.",
             {"disease": {"type": "str", "default": "rheumatoid arthritis"},
              "top": {"type": "int", "default": 10},
              "max_compounds": {"type": "int", "default": 20},
              "novel": {"type": "bool", "default": True}},
             pb.disease_repurposing),
    Playbook("discovery_program", "Discovery program", "programs",
             "Disease → PrimeKG targets → molecule triage and ADMET/activity screen → program report. "
             "Runs the molecule screen even when the graph is offline.",
             {"disease": {"type": "str", "default": "idiopathic pulmonary fibrosis"},
              "molecules": {"type": "text", "default": "CC(=O)Oc1ccccc1C(=O)O\nCN1CCC[C@H]1c1cccnc1"},
              "top_targets": {"type": "int", "default": 15}}, pb.discovery_program),
    Playbook("pub_weekly_digest", "Weekly digest", "pub",
             "Summarise the week's runs, refresh repo activity for the open index and rebuild the public page.",
             {"days": {"type": "int", "default": 7}, "check_repos": {"type": "bool", "default": True}},
             pb.pub_weekly_digest),
]}

_slots = threading.Semaphore(2)


class Ctx:
    def __init__(self, run_id):
        self.run_id = run_id

    def log(self, line):
        db.run_log(self.run_id, line)


def _narrate(lines):
    """Optional executive summary from a local Ollama model (DDD_NARRATOR=<model>)."""
    if not NARRATOR:
        return None
    text = "\n".join(lines)[:12000]
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", timeout=240, json={
            "model": NARRATOR, "stream": False, "options": {"temperature": 0.2},
            "prompt": "Write a 3-5 sentence executive summary of this report for the founder. "
                      "Only state what the report says; no new facts.\n\n" + text})
        return r.json().get("response", "").strip() or None
    except Exception:
        return None


def _execute(run_id, book: Playbook, params):
    ctx = Ctx(run_id)
    with _slots:
        try:
            ctx.log(f"{DEPT[book.department].agent} starts {book.title}")
            summary, lines = book.fn(ctx, **params)
            exec_sum = _narrate(lines)
            if exec_sum:
                lines = lines[:1] + ["", f"**Executive summary ({NARRATOR}):** {exec_sum}", ""] + lines[1:]
            path = report_path(book.name, run_id)
            path.write_text("\n".join(lines) + f"\n\n---\nRun r{run_id} · {book.name} · {time.strftime('%Y-%m-%d %H:%M')}\n")
            db.run_finish(run_id, "done", summary, path.name)
            ctx.log("Done")
        except Exception as e:
            ctx.log(f"Error: {e}")
            ctx.log(traceback.format_exc()[-800:])
            db.run_finish(run_id, "error", {"error": str(e)[:300]})


def _clean(book: Playbook, params):
    params = dict(params or {})
    out = {}
    for k, spec in book.params.items():
        if k in params:
            out[k] = params[k]
        elif "default" in spec:
            out[k] = spec["default"]
        else:
            raise ValueError(f"{book.name} needs '{k}'")
    return out


def start(name: str, params: dict | None = None, trigger: str = "manual") -> int:
    if name not in PLAYBOOKS:
        raise KeyError(f"unknown playbook '{name}'. Known: {', '.join(PLAYBOOKS)}")
    book = PLAYBOOKS[name]
    clean = _clean(book, params)
    run_id = db.run_create(name, DEPT[book.department].name, clean, trigger)
    threading.Thread(target=_execute, args=(run_id, book, clean), daemon=True).start()
    return run_id


# ---------------------------------------------------------------- scheduler

_sched_started = False


def _scheduler_loop():
    while True:
        try:
            for s in db.due_schedules():
                db.schedule_advance(s["id"], s["every"])
                try:
                    start(s["playbook"], s["params"], trigger=f"schedule #{s['id']} ({s['every']})")
                except Exception as e:
                    rid = db.run_create(s["playbook"], "Operations", s["params"], f"schedule #{s['id']}")
                    db.run_finish(rid, "error", {"error": str(e)[:300]})
        except Exception:
            pass
        time.sleep(30)


def start_scheduler():
    global _sched_started
    if _sched_started:
        return
    _sched_started = True
    db.mark_orphans()
    db.seed_default_schedules()
    threading.Thread(target=_scheduler_loop, daemon=True).start()


# ---------------------------------------------------------------- overview

def overview() -> dict:
    c = charter_mod.load()
    runs = db.runs(300)
    board = store.leaderboard(TASKS)
    week = time.time() - 7 * 86400
    depts = []
    for d in DEPARTMENTS:
        mine = [r for r in runs if r["department"] == d.name]
        last = mine[0] if mine else None
        depts.append({**d.__dict__,
                      "playbooks": [p.meta() for p in PLAYBOOKS.values() if p.department == d.id],
                      "runs_total": len(mine),
                      "last_run": {k: last[k] for k in ("id", "playbook", "status", "started", "finished", "report", "summary")} if last else None})
    arts = list_artifacts()
    k = kg.status()
    scheds = db.schedules()
    nxt = sorted((s for s in scheds if s["enabled"]), key=lambda s: s["next_run"] or 1e18)
    return {
        "charter": c,
        "departments": depts,
        "kpis": {
            "tasks_benchmarked": len(board),
            "models_ranked": len({r["model"] for rows in board.values() for r in rows}),
            "runs_7d": sum((r["started"] or 0) >= week for r in runs),
            "failed_7d": sum((r["started"] or 0) >= week and r["status"] == "error" for r in runs),
            "captures": len(arts["har"]),
            "reports": len(list(REPORTS.glob("*.md"))),
            "kg_nodes": k.get("nodes") if k.get("connected") else None,
        },
        "systems": {
            "knowledge_graph": k,
            "ollama_models": ollama_models(),
            "narrator": NARRATOR or None,
            "rea": rea_bridge.available(),
        },
        "next_scheduled": [{"playbook": s["playbook"], "every": s["every"], "next_run": s["next_run"]} for s in nxt[:5]],
    }
