"""Web UI + JSON API for the bench."""
from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import __version__, kg, store
from .armory import PRESETS, TOOLS, run_tool
from .armory import rea as rea_bridge
from .armory.paths import list_artifacts, resolve
from .chem import describe
from .config import MCP_PORT
from .models import available_models
from .runner import predict, run
from .tasks import TASKS, available_tasks

app = FastAPI(title="ddd", version=__version__)
STATIC = Path(__file__).resolve().parent.parent / "static"
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


class RunReq(BaseModel):
    task: str
    model: str
    sample: int | None = None


class PredictReq(BaseModel):
    task: str
    smiles: list[str]
    model: str = "rf"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {"version": __version__, "mcp_port": MCP_PORT, "kg": kg.status(), "rea": rea_bridge.available()}


@app.get("/api/tasks")
def tasks():
    return available_tasks()


@app.get("/api/models")
def models():
    return available_models()


def _job(job_id, req: RunReq):
    def prog(done, total):
        JOBS[job_id]["progress"] = f"{done}/{total}"
    try:
        JOBS[job_id]["status"] = "running"
        res = run(req.task, req.model, req.sample, prog)
        JOBS[job_id].update(status="done", result=res)
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e), trace=traceback.format_exc()[-1500:])


@app.post("/api/run")
def start_run(req: RunReq):
    if req.task not in TASKS:
        raise HTTPException(404, f"unknown task {req.task}")
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        JOBS[job_id] = {"id": job_id, "status": "queued", "task": req.task, "model": req.model, "progress": ""}
    threading.Thread(target=_job, args=(job_id, req), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/jobs")
def jobs():
    return sorted(JOBS.values(), key=lambda j: j["id"])[-30:]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


@app.get("/api/leaderboard")
def leaderboard():
    return store.leaderboard(TASKS)


@app.get("/api/results")
def results(task: str | None = None):
    return store.all_results(task)


@app.get("/api/molecule")
def molecule(smiles: str):
    return describe(smiles)


@app.post("/api/predict")
def do_predict(req: PredictReq):
    try:
        return predict(req.task, req.smiles, req.model)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/kg/status")
def kg_status():
    return kg.status()


@app.get("/api/kg/search")
def kg_search(q: str):
    return _kg(kg.search, q)


@app.get("/api/kg/neighbors")
def kg_neighbors(name: str, relation: str | None = None, limit: int = 25):
    return _kg(kg.neighbors, name, relation, limit)


@app.get("/api/kg/summary")
def kg_summary(name: str):
    return _kg(kg.relation_summary, name)


@app.get("/api/kg/path")
def kg_path(a: str, b: str, hops: int = 4):
    return _kg(kg.path, a, b, hops)


@app.get("/api/kg/targets")
def kg_targets(disease: str, limit: int = 25):
    return _kg(kg.disease_candidate_targets, disease, limit)


def _kg(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        raise HTTPException(503, f"knowledge graph unavailable: {str(e)[:200]}")


# ---------- Tools Armory ----------

class ToolReq(BaseModel):
    args: dict = {}


@app.get("/api/armory")
def armory():
    return {"tools": [t.meta() for t in TOOLS.values()], "presets": PRESETS}


def _tool_job(job_id, name, args):
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id].update(status="done", result=run_tool(name, args))
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e)[:400], trace=traceback.format_exc()[-1500:])


@app.post("/api/armory/{name}")
def armory_run(name: str, req: ToolReq):
    if name not in TOOLS:
        raise HTTPException(404, f"unknown tool {name}")
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        JOBS[job_id] = {"id": job_id, "status": "queued", "task": "armory", "model": name, "progress": "", "tool": name}
    threading.Thread(target=_tool_job, args=(job_id, name, req.args), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/armory-files")
def armory_files():
    return list_artifacts()


@app.get("/api/armory-files/{kind}/{name}")
def armory_file(kind: str, name: str, download: bool = False):
    try:
        p = resolve(kind, name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    if kind == "reports" and not download:
        return PlainTextResponse(p.read_text())
    return FileResponse(p, filename=p.name)
