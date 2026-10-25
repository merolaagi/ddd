"""Web UI + JSON API for the bench."""
from __future__ import annotations

import base64
import hmac
import os
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, kg, store
from .armory import PRESETS, TOOLS, run_tool
from .armory import rea as rea_bridge
from . import index_catalog, vcell
from .company import programs as company_programs
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

_WEB_PASSWORD = os.environ.get("DDD_WEB_PASSWORD", "").strip()


@app.middleware("http")
async def _basic_auth(request, call_next):
    """When DDD_WEB_PASSWORD is set, require HTTP Basic auth on every request.
    Lets the whole web UI be exposed over the tunnel safely. Any username is
    accepted; the password must match. With no password set, the app stays
    open (localhost-only use) exactly as before."""
    if _WEB_PASSWORD and request.url.path != "/healthz":
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8", "replace")
                _, _, pw = decoded.partition(":")
                ok = hmac.compare_digest(pw, _WEB_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, content='{"error":"unauthorized"}',
                            media_type="application/json",
                            headers={"WWW-Authenticate": 'Basic realm="ddd"'})
    return await call_next(request)


class RunReq(BaseModel):
    task: str
    model: str
    sample: int | None = None


class PredictReq(BaseModel):
    task: str
    smiles: list[str]
    model: str = "rf"


@app.get("/")
def hq():
    return FileResponse(STATIC / "hq.html")


@app.get("/lab")
def lab():
    return FileResponse(STATIC / "lab.html")


@app.get("/healthz")
def healthz():
    """Unauthenticated readiness probe (serve.sh), so Basic auth on the UI
    doesn't make startup look like a failure."""
    return {"ok": True, "version": __version__}


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


@app.get("/api/scoreboard")
def scoreboard():
    return store.scoreboard(TASKS)


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


@app.get("/api/kg/subgraph")
def kg_subgraph(center: str, depth: int = 1, limit: int = 60, relation: str | None = None):
    return _kg(kg.subgraph, center, depth, limit, relation)


@app.get("/api/kg/provenance")
def kg_provenance():
    return _kg(kg.provenance)


@app.get("/api/kg/hypothesis")
def kg_hypothesis(a: str, b: str, max_hops: int = 4):
    from . import kg_model
    return _kg(kg_model.hypothesis, a, b, max_hops)


@app.get("/api/kg/predict")
def kg_predict(entity: str, target: str = "gene", k: int = 20):
    from . import kg_model
    return _kg(kg_model.predict_links, entity, target, k)


@app.get("/api/vcell/cell")
def vcell_cell(disease: str, limit: int = 22):
    return _kg(kg.cell_view, disease, limit)


@app.get("/api/sources/opentargets/disease")
def ot_disease(disease: str, size: int = 25, write: bool = True):
    from .sources import opentargets
    try:
        return opentargets.enrich_disease(disease, size, write)
    except Exception as e:
        raise HTTPException(502, f"Open Targets: {str(e)[:200]}")


@app.get("/api/sources/opentargets/target_drugs")
def ot_target_drugs(symbol: str, size: int = 25):
    from .sources import opentargets
    try:
        return opentargets.target_known_drugs(symbol, size)
    except Exception as e:
        raise HTTPException(502, f"Open Targets: {str(e)[:200]}")


@app.get("/api/sources/chembl/target")
def chembl_target(symbol: str, standard_type: str = "IC50", limit: int = 50, write: bool = True):
    from .sources import chembl
    try:
        return chembl.enrich_target(symbol, standard_type, limit, write)
    except Exception as e:
        raise HTTPException(502, f"ChEMBL: {str(e)[:200]}")


@app.get("/api/sources/chembl/molecule")
def chembl_molecule(name: str):
    from .sources import chembl
    try:
        return chembl.molecule(name)
    except Exception as e:
        raise HTTPException(502, f"ChEMBL: {str(e)[:200]}")


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


# ---------- Headless company ----------

class ProgramReq(BaseModel):
    disease: str
    molecules: list[str] = []
    top_targets: int = 15


class IndexEntry(BaseModel):
    name: str
    category: str
    maintainer: str = ""
    repo: str = ""
    url: str = ""
    measures: str
    caveat: str


@app.get("/api/company/roles")
def company_roles():
    return {"roles": company_programs.ROLES, "screen": [{"task": t, "label": l} for t, l in company_programs.SCREEN]}


def _program_job(job_id, req: ProgramReq):
    def step(s):
        JOBS[job_id].setdefault("steps", []).append(s)
        JOBS[job_id]["progress"] = f"{s['role']}: {s['msg'][:60]}"
    try:
        JOBS[job_id]["status"] = "running"
        res = company_programs.run_program(req.disease, req.molecules, req.top_targets, step)
        JOBS[job_id].update(status="done", result=res)
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e)[:400], trace=traceback.format_exc()[-1500:])


@app.post("/api/company/program")
def company_program(req: ProgramReq):
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        JOBS[job_id] = {"id": job_id, "status": "queued", "task": "program", "model": req.disease, "progress": "", "tool": "program"}
    threading.Thread(target=_program_job, args=(job_id, req), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/index")
def index_list():
    return {"entries": index_catalog.load_index(), "categories": index_catalog.categories()}


@app.post("/api/index")
def index_add(e: IndexEntry):
    try:
        return index_catalog.add_entry(e.dict())
    except ValueError as ex:
        raise HTTPException(400, str(ex))


@app.get("/api/vcell")
def vcell_overview():
    return vcell.overview()


# ---------- Company ----------

from .company import charter as charter_mod  # noqa: E402
from .company import db as cdb  # noqa: E402
from .company import engine  # noqa: E402
from .company.paths import PUBLIC, safe_report  # noqa: E402
from .company.playbooks import open_index  # noqa: E402

app.mount("/public", StaticFiles(directory=PUBLIC, html=True, check_dir=False), name="public")
app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")


@app.on_event("startup")
def _boot():
    engine.start_scheduler()


class PlaybookReq(BaseModel):
    params: dict = {}


class ScheduleReq(BaseModel):
    playbook: str
    every: str
    params: dict = {}


class ToggleReq(BaseModel):
    enabled: bool


@app.get("/api/company")
def company_overview():
    return engine.overview()


@app.get("/api/company/charter")
def get_charter():
    return charter_mod.load()


@app.put("/api/company/charter")
def put_charter(update: dict):
    return charter_mod.save(update)


@app.get("/api/company/playbooks")
def playbooks():
    return [p.meta() for p in engine.PLAYBOOKS.values()]


@app.post("/api/company/run/{name}")
def run_playbook(name: str, req: PlaybookReq):
    try:
        return {"run_id": engine.start(name, req.params, "manual")}
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/company/runs")
def company_runs(limit: int = 40):
    out = cdb.runs(limit)
    for r in out:
        r.pop("log", None)
    return out


@app.get("/api/company/runs/{run_id}")
def company_run(run_id: int):
    r = cdb.run_get(run_id)
    if not r:
        raise HTTPException(404)
    return r


@app.get("/api/company/reports/{name}")
def company_report(name: str):
    try:
        return PlainTextResponse(safe_report(name).read_text())
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


@app.get("/api/company/schedules")
def company_schedules():
    return cdb.schedules()


@app.post("/api/company/schedules")
def add_schedule(req: ScheduleReq):
    if req.playbook not in engine.PLAYBOOKS:
        raise HTTPException(400, f"unknown playbook {req.playbook}")
    try:
        return {"id": cdb.schedule_add(req.playbook, req.params, req.every)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/api/company/schedules/{sid}")
def toggle_schedule(sid: int, req: ToggleReq):
    cdb.schedule_set(sid, enabled=req.enabled)
    return {"ok": True}


@app.delete("/api/company/schedules/{sid}")
def delete_schedule(sid: int):
    cdb.schedule_set(sid, delete=True)
    return {"ok": True}


@app.get("/api/company/index")
def company_index():
    return open_index(False)
