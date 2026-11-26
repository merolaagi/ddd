import csv
import io
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from typing import Literal

from .engine import ARMS, DEFAULTS, MEMBRANES, TASKS, Cancelled, run_job
from .explain import LABEL, explain
from .pathways import (PATHWAYS, analyse, cypher, deeper, graph as pathway_graph, merged, simulate_pathway,
                        benchmark as run_benchmark, strategies, target_search, what_if)
from .workspace import (LAYER_TYPES, code_workspace, compile_workspace, default_workspace, from_model,
                        from_pathway, new_layer, run_workspace, waveform)
from . import research as RS
from . import settings as ST
from .model import THERAPY, code_numpy, code_torch, graph, probe, reachability, simulate, therapy

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FML_DATA", ROOT / "data" / "runs"))
DATA.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"
WORKERS = int(os.environ.get("FML_WORKERS", max(1, (os.cpu_count() or 2) - 2)))

_lock = threading.Lock()
_queue = []
_wake = threading.Event()


class RunConfig(BaseModel):
    name: str = ""
    tasks: list[str] = Field(default_factory=lambda: list(DEFAULTS["tasks"]))
    arms: list[str] = Field(default_factory=lambda: list(DEFAULTS["arms"]))
    hidden: int = Field(DEFAULTS["hidden"], ge=4, le=128)
    substeps: int = Field(DEFAULTS["substeps"], ge=1, le=10)
    iters: int = Field(DEFAULTS["iters"], ge=10, le=20000)
    lr: float = Field(DEFAULTS["lr"], gt=0, le=1)
    batch: int = Field(DEFAULTS["batch"], ge=1, le=512)
    train_len: int = Field(DEFAULTS["train_len"], ge=10, le=1000)
    test_len: int = Field(DEFAULTS["test_len"], ge=10, le=5000)
    pulse_prob: float = Field(DEFAULTS["pulse_prob"], gt=0, lt=1)
    self_excitation: float = Field(DEFAULTS["self_excitation"], ge=0, le=20)
    coupling: float = Field(DEFAULTS["coupling"], ge=0, le=5)
    drift: list[float] = Field(default_factory=lambda: list(DEFAULTS["drift"]))
    noise: list[float] = Field(default_factory=lambda: list(DEFAULTS["noise"]))
    division: list[float] = Field(default_factory=lambda: list(DEFAULTS["division"]))
    inhibitor: list[float] = Field(default_factory=lambda: list(DEFAULTS["inhibitor"]))
    membrane: Literal["direct", "transporter", "gated"] = DEFAULTS["membrane"]
    seed: int = DEFAULTS["seed"]
    feedback_edges: int = Field(DEFAULTS["feedback_edges"], ge=0, le=20)
    pool_size: int = Field(DEFAULTS["pool_size"], ge=2, le=16)
    repeats: int = Field(DEFAULTS["repeats"], ge=1, le=10)


def run_path(rid):
    return DATA / f"{rid}.json"


def progress_dir(rid):
    return DATA / f"{rid}.progress"


def load(rid):
    p = run_path(rid)
    if not p.exists() or "/" in rid or ".." in rid:
        raise HTTPException(404, "Run not found")
    return json.loads(p.read_text())


def save(run):
    tmp = run_path(run["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(run))
    tmp.replace(run_path(run["id"]))


def _job(task, arm, rep, cfg, pdir):
    pdir = Path(pdir)
    pfile = pdir / f"{task}__{arm}__{rep}"
    stop = pdir / "CANCEL"

    def progress(frac):
        if stop.exists():
            raise Cancelled()
        pfile.write_text(f"{frac:.4f}")

    return run_job(task, arm, cfg, progress, rep=rep)


def _execute(rid):
    run = load(rid)
    if run["status"] == "cancelled":
        return
    cfg = run["config"]
    pdir = progress_dir(rid)
    pdir.mkdir(exist_ok=True)
    run["status"] = "running"
    run["started"] = time.time()
    save(run)
    jobs = [(t, a, r) for t in cfg["tasks"] for a in cfg["arms"] for r in range(cfg["repeats"])]
    status, error = "done", None
    try:
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as ex:
            futs = [ex.submit(_job, t, a, r, cfg, str(pdir)) for t, a, r in jobs]
            for f in as_completed(futs):
                try:
                    res = f.result()
                except Cancelled:
                    status = "cancelled"
                    continue
                with _lock:
                    run = load(rid)
                    run["results"].append(res)
                    save(run)
    except Exception:
        status, error = "failed", traceback.format_exc(limit=3)
    if (pdir / "CANCEL").exists():
        status = "cancelled"
    run = load(rid)
    run["status"] = status
    if error:
        run["error"] = error
    run["finished"] = time.time()
    save(run)
    shutil.rmtree(pdir, ignore_errors=True)


def _worker():
    while True:
        _wake.wait()
        while True:
            with _lock:
                if not _queue:
                    _wake.clear()
                    break
                rid = _queue.pop(0)
            try:
                _execute(rid)
            except Exception:
                traceback.print_exc()


def _recover():
    for p in sorted(DATA.glob("*.json")):
        run = json.loads(p.read_text())
        if run.get("status") in ("queued", "running"):
            run["status"] = "queued"
            run["results"] = []
            run["config"].setdefault("repeats", 1)
            save(run)
            shutil.rmtree(progress_dir(run["id"]), ignore_errors=True)
            _queue.append(run["id"])
    if _queue:
        _wake.set()


_workers_started = False


def runs_summary(limit: int = 20):
    """Compact list of recent FML runs, for ddd's MCP/agents (the rule engine)."""
    out = []
    for p in sorted(DATA.glob("*.json"), reverse=True)[:max(1, limit)]:
        try:
            r = json.loads(p.read_text())
            cfg = r.get("config", {}) or {}
            out.append({"id": r.get("id"), "name": r.get("name"), "status": r.get("status"),
                        "created": r.get("created"), "tasks": cfg.get("tasks"), "arms": cfg.get("arms")})
        except Exception:
            pass
    return out


def ensure_workers():
    """Idempotent startup — called by ddd when FML is mounted (sub-app lifespans
    don't auto-run), and by FML's own lifespan when run standalone."""
    global _workers_started
    if _workers_started:
        return
    _workers_started = True
    try:
        _recover()
        RS.load_all()
    except Exception:
        pass
    threading.Thread(target=_worker, daemon=True).start()


@asynccontextmanager
async def lifespan(app):
    ensure_workers()
    yield


app = FastAPI(title="Fate Memory Lab", version=__version__, lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "queued": len(_queue)}


@app.get("/api/meta")
def meta():
    return {"version": __version__, "defaults": DEFAULTS, "arms": ARMS, "tasks": list(TASKS),
            "membranes": MEMBRANES, "workers": WORKERS}


@app.post("/api/runs")
def create_run(cfg: RunConfig):
    c = cfg.model_dump()
    c["tasks"] = [t for t in c["tasks"] if t in TASKS]
    c["arms"] = [a for a in c["arms"] if a in ARMS]
    if not c["tasks"] or not c["arms"]:
        raise HTTPException(400, "Pick at least one task and one arm")
    c["drift"] = [float(e) for e in c["drift"] if 0 <= float(e) <= 3][:6]
    c["noise"] = [float(e) for e in c["noise"] if 0 <= float(e) <= 1][:6]
    c["division"] = [float(e) for e in c["division"] if 0 <= float(e) <= 3][:6]
    c["inhibitor"] = [float(e) for e in c["inhibitor"] if 0 < float(e) <= 1][:6]
    rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    run = {"id": rid, "name": c.pop("name") or rid, "created": time.time(), "config": c,
           "status": "queued", "results": [], "version": __version__}
    save(run)
    with _lock:
        _queue.append(rid)
    _wake.set()
    return run


@app.get("/api/runs")
def list_runs():
    out = []
    for p in sorted(DATA.glob("*.json"), reverse=True):
        r = json.loads(p.read_text())
        c = r["config"]
        out.append({"id": r["id"], "name": r["name"], "status": r["status"], "created": r["created"],
                    "tasks": c["tasks"], "arms": c["arms"], "iters": c["iters"], "repeats": c.get("repeats", 1),
                    "membrane": c.get("membrane", "direct")})
    return out


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    run = load(rid)
    run["config"].setdefault("repeats", 1)
    prog = {}
    pdir = progress_dir(rid)
    if pdir.exists():
        for f in pdir.iterdir():
            if f.name == "CANCEL":
                continue
            try:
                prog[f.name] = float(f.read_text() or 0)
            except ValueError:
                pass
    for r in run["results"]:
        prog[f"{r['task']}__{r['arm']}__{r.get('rep', 0)}"] = 1.0
    run["progress"] = prog
    return run


@app.post("/api/runs/{rid}/cancel")
def cancel_run(rid: str):
    run = load(rid)
    with _lock:
        if rid in _queue:
            _queue.remove(rid)
            run["status"] = "cancelled"
            save(run)
            return run
    if run["status"] != "running":
        raise HTTPException(409, f"This run is {run['status']}, so there is nothing to stop.")
    pdir = progress_dir(rid)
    pdir.mkdir(exist_ok=True)
    (pdir / "CANCEL").write_text("1")
    return {"id": rid, "status": "stopping"}


@app.get("/api/runs/{rid}/export.json")
def export_json(rid: str):
    load(rid)
    return FileResponse(run_path(rid), media_type="application/json", filename=f"fate-memory-{rid}.json")


@app.get("/api/runs/{rid}/export.csv")
def export_csv(rid: str):
    run = load(rid)
    buf = io.StringIO()
    w = csv.writer(buf)
    c = run["config"]
    drift_eps, noise_eps, div_eps = c["drift"], c.get("noise", []), c.get("division", [])
    inh_eps = c.get("inhibitor", [])
    w.writerow(["task", "arm", "rep", "seed", "membrane", "params", "acc_train_len", "acc_test_len", "settled",
                "attractors", "settle_steps", "negative_edges", "seconds"]
               + [f"drift_{e}_acc" for e in drift_eps] + [f"drift_{e}_settled" for e in drift_eps]
               + [f"noise_{e}_acc" for e in noise_eps] + [f"division_{e}_acc" for e in div_eps]
               + [f"inhibitor_{e}_acc" for e in inh_eps])
    for r in sorted(run["results"], key=lambda r: (r["task"], r["arm"], r.get("rep", 0))):
        w.writerow([r["task"], r["arm"], r.get("rep", 0), r.get("seed", ""), r.get("membrane", "direct"), r["params"], r["acc_train_len"],
                    r["acc_test_len"], r["settled"], r["attractors"], r.get("settle_steps", ""),
                    r["negative_edges"], r["seconds"]]
                   + [d["acc"] for d in r["drift"]] + [d["settled"] for d in r["drift"]]
                   + [d["acc"] for d in r.get("noise", [])] + [d["acc"] for d in r.get("division", [])]
                   + [d["acc"] for d in r.get("inhibitor", [])])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="fate-memory-{rid}.csv"'})


@app.delete("/api/runs/{rid}")
def delete_run(rid: str):
    run = load(rid)
    if run["status"] == "running":
        raise HTTPException(409, "This run is still training. Stop it first, then delete it.")
    with _lock:
        if rid in _queue:
            _queue.remove(rid)
    run_path(rid).unlink(missing_ok=True)
    shutil.rmtree(progress_dir(rid), ignore_errors=True)
    return {"deleted": rid}


@app.get("/api/runs/{rid}/report")
def report(rid: str):
    return explain(load(rid))


def _pick(run, task, arm=None, rep=None):
    rs = [r for r in run["results"] if r["task"] == task]
    if not rs:
        raise HTTPException(404, f"No results for {task} in this run.")
    rs = [r for r in rs if "model" in r]
    if not rs:
        raise HTTPException(409, "This run was trained before version 0.5.0 and did not save its weights. Start a new run to build a model.")
    if arm is None:
        means = {}
        for r in rs:
            means.setdefault(r["arm"], []).append(r["acc_test_len"])
        arm = max(means, key=lambda a: sum(means[a]) / len(means[a]))
    pool = [r for r in rs if r["arm"] == arm]
    if not pool:
        raise HTTPException(404, f"No {arm} results for {task}.")
    if rep is None:
        return max(pool, key=lambda r: r["acc_test_len"]), rs
    for r in pool:
        if r.get("rep", 0) == rep:
            return r, rs
    raise HTTPException(404, "No such seed.")


def _title(run, r):
    return (f"Fate Memory Lab cell model: {r['task']} task, {LABEL[r['arm']]} arm, seed {r.get('rep', 0) + 1}, "
            f"{r.get('membrane', 'direct')} membrane, from run {run['id']} (v{run.get('version', '?')})")


@app.get("/api/runs/{rid}/model")
def model_info(rid: str, task: str, arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, rs = _pick(run, task, arm, rep)
    return {"task": task, "arm": r["arm"], "rep": r.get("rep", 0), "membrane": r.get("membrane", "direct"),
            "acc_test_len": r["acc_test_len"], "title": _title(run, r),
            "candidates": sorted([{"arm": x["arm"], "rep": x.get("rep", 0), "acc": x["acc_test_len"]} for x in rs],
                                 key=lambda x: (-x["acc"])),
            "graph": graph(r["model"], task, r), "has_probe": task in ("antagonist", "commit"),
            "has_therapy": task in THERAPY}


@app.get("/api/runs/{rid}/model/therapy")
def model_therapy(rid: str, task: str, arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    out = therapy(r["model"], task)
    if out is None:
        raise HTTPException(404, "No therapy search for this task.")
    return out


@app.get("/api/runs/{rid}/model/simulate")
def model_simulate(rid: str, task: str, arm: str | None = None, rep: int | None = None, seed: int = 0, length: int = 200):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    return simulate(r["model"], task, T=max(20, min(length, 1000)), seed=seed, p=run["config"]["pulse_prob"])


@app.get("/api/runs/{rid}/model/probe")
def model_probe(rid: str, task: str, arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    out = probe(r["model"], task)
    if out is None:
        raise HTTPException(404, "No probe for this task.")
    return out


@app.get("/api/runs/{rid}/model/export")
def model_export(rid: str, task: str, kind: Literal["numpy", "torch", "json"] = "numpy", arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    base = f"fate-cell-model-{task}-{r['arm']}-seed{r.get('rep', 0) + 1}-{rid}"
    title = _title(run, r)
    if kind == "json":
        body = json.dumps({"title": title, "task": task, "graph": graph(r["model"], task, r), "spec": r["model"]}, indent=1)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{base}.json"'})
    code = code_numpy(r["model"], task, title) if kind == "numpy" else code_torch(r["model"], task, title)
    suffix = "numpy" if kind == "numpy" else "torch"
    return Response(code, media_type="text/x-python",
                    headers={"Content-Disposition": f'attachment; filename="{base}-{suffix}.py"'})


@app.get("/api/runs/{rid}/model/reach")
def model_reach(rid: str, task: str, arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    return reachability(r["model"], task)


WS = Path(os.environ.get("FML_WORKSPACES", ROOT / "data" / "workspaces"))
WS.mkdir(parents=True, exist_ok=True)


class WorkspaceCreate(BaseModel):
    name: str = ""
    pathway: str | None = None
    run_id: str | None = None
    task: str | None = None
    arm: str | None = None
    rep: int | None = None


class Waveform(BaseModel):
    kind: str = "pulse"
    amp: float = 1.0
    start: int = 10
    width: int = 20
    period: int = 60


class SimRequest(BaseModel):
    length: int = Field(120, ge=10, le=2000)
    channels: list[Waveform] = Field(default_factory=list)


_MERGED = {}


def _path(pid):
    if pid == "all":
        if "m" not in _MERGED:
            _MERGED["m"] = merged()
        return _MERGED["m"]
    if pid not in PATHWAYS:
        raise HTTPException(404, "No such pathway")
    return PATHWAYS[pid]


@app.get("/api/pathways")
def list_pathways():
    out = [{"id": p["id"], "name": p["name"], "description": p["description"], "species": len(p["species"]),
            "reactions": len(p["reactions"]), "research": bool(p.get("from_research"))} for p in PATHWAYS.values()]
    m = _path("all")
    out.append({"id": "all", "name": m["name"], "description": m["description"], "species": len(m["species"]),
                "reactions": len(m["reactions"])})
    return out


@app.get("/api/pathways/{pid}")
def get_pathway(pid: str):
    return analyse(_path(pid))


@app.get("/api/pathways/{pid}/deeper")
def pathway_deeper(pid: str):
    return deeper(_path(pid))


@app.get("/api/pathways/{pid}/graph")
def pathway_graph_api(pid: str):
    return pathway_graph(_path(pid))


@app.get("/api/pathways/{pid}/cypher")
def pathway_cypher(pid: str, download: bool = False):
    text = cypher(_path(pid))
    if download:
        return Response(text, media_type="text/plain",
                        headers={"Content-Disposition": f'attachment; filename="fml-{pid}.cypher"'})
    return {"cypher": text, "statements": len(text.splitlines())}


@app.get("/api/pathways/{pid}/whatif")
def pathway_whatif(pid: str, node: str):
    out = what_if(_path(pid), node)
    if out is None:
        raise HTTPException(404, "No such node in this pathway")
    return out


def _neo4j_settings():
    return {"uri": ST.get("neo4j_uri"), "user": ST.get("neo4j_user"), "password": ST.get("neo4j_password"),
            "database": ST.get("neo4j_database")}


@app.get("/api/neo4j/status")
def neo4j_status():
    cfg = _neo4j_settings()
    try:
        import neo4j  # noqa: F401
        driver_installed = True
    except ImportError:
        driver_installed = False
    return {"driver_installed": driver_installed, "configured": bool(cfg["uri"] and cfg["password"]),
            "uri": cfg["uri"], "database": cfg["database"]}


@app.post("/api/neo4j/push")
def neo4j_push(body: dict):
    cfg = _neo4j_settings()
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise HTTPException(409, "The neo4j driver is not installed. Run: .venv/bin/pip install neo4j")
    if not cfg["uri"] or not cfg["password"]:
        raise HTTPException(409, "Add your Neo4j address and password in Settings before pushing.")
    pids = [body.get("pathway")] if body.get("pathway") in PATHWAYS else list(PATHWAYS)
    done = {}
    try:
        with GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])) as driver:
            with driver.session(database=cfg["database"]) as session:
                for pid in pids:
                    stmts = [x for x in cypher(PATHWAYS[pid]).split(";\n") if x.strip() and not x.strip().startswith("//")]
                    for st in stmts:
                        session.run(st.rstrip(";"))
                    done[pid] = len(stmts)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Neo4j rejected the load: {e}")
    return {"pushed": done, "uri": cfg["uri"], "database": cfg["database"]}


@app.get("/api/pathways/{pid}/strategies")
def pathway_strategies(pid: str, node: str):
    p = _path(pid)
    rid = node.split(":", 1)[1] if ":" in node else node
    out = strategies(p, rid)
    if out is None:
        raise HTTPException(404, "No such enzyme step in this pathway")
    return out


@app.get("/api/pathways/{pid}/simulate")
def sim_pathway(pid: str, steps: int = 2500, knockdown: str | None = None, factor: float = 0.2):
    p = _path(pid)
    scale = None
    if knockdown:
        scale = [factor if r["id"] == knockdown else 1.0 for r in p["reactions"]]
        if all(x == 1.0 for x in scale):
            raise HTTPException(404, "No such reaction")
    out = simulate_pathway(p, steps=max(50, min(steps, 4000)), enzyme_scale=scale)
    out["knockdown"] = knockdown
    return out


def ws_path(wid):
    return WS / f"{wid}.json"


def load_ws(wid):
    p = ws_path(wid)
    if not p.exists() or "/" in wid:
        raise HTTPException(404, "Workspace not found")
    return json.loads(p.read_text())


def save_ws(w):
    ws_path(w["id"]).write_text(json.dumps(w))


@app.get("/api/workspaces")
def list_workspaces():
    out = []
    for p in sorted(WS.glob("*.json"), reverse=True):
        w = json.loads(p.read_text())
        out.append({"id": w["id"], "name": w["name"], "created": w.get("created", 0),
                    "layers": [l["type"] for l in w["layers"]]})
    return out


@app.get("/api/workspaces/meta")
def workspace_meta():
    lt = json.loads(json.dumps(LAYER_TYPES))
    for f in lt["pathway"]["fields"]:
        if f[0] == "pathway":
            f[2] = list(PATHWAYS)
    return {"layer_types": lt, "pathways": [{"id": p["id"], "name": p["name"], "species": p["species"]}
                                            for p in PATHWAYS.values()]}


@app.post("/api/workspaces")
def create_workspace(req: WorkspaceCreate):
    if req.pathway:
        w = from_pathway(req.pathway, req.name or None)
    elif req.run_id and req.task:
        run = load(req.run_id)
        r, _ = _pick(run, req.task, req.arm, req.rep)
        w = from_model(r["model"], req.task, req.name or f"{req.task} model from run {run['id']}")
    else:
        w = default_workspace(req.name or "New model")
    w["id"] = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    w["created"] = time.time()
    w["compile"] = compile_workspace(w)
    save_ws(w)
    return w


@app.get("/api/workspaces/{wid}")
def get_workspace(wid: str):
    return load_ws(wid)


@app.put("/api/workspaces/{wid}")
def put_workspace(wid: str, body: dict):
    w = load_ws(wid)
    w["name"] = body.get("name", w["name"])
    if "layers" in body:
        w["layers"] = body["layers"]
    for key in ("input_names", "output_names"):
        if key in body:
            w[key] = body[key]
    w["compile"] = compile_workspace(w)
    save_ws(w)
    return w


@app.post("/api/workspaces/{wid}/layers")
def add_layer(wid: str, body: dict):
    w = load_ws(wid)
    kind = body.get("type")
    if kind not in LAYER_TYPES:
        raise HTTPException(400, "Unknown layer type")
    at = int(body.get("after", len(w["layers"]) - 1))
    w["layers"].insert(max(0, min(at + 1, len(w["layers"]))), new_layer(kind))
    w["compile"] = compile_workspace(w)
    save_ws(w)
    return w


@app.delete("/api/workspaces/{wid}/layers/{lid}")
def del_layer(wid: str, lid: str):
    w = load_ws(wid)
    w["layers"] = [l for l in w["layers"] if l["id"] != lid]
    w["compile"] = compile_workspace(w)
    save_ws(w)
    return w


@app.delete("/api/workspaces/{wid}")
def delete_workspace(wid: str):
    load_ws(wid)
    ws_path(wid).unlink(missing_ok=True)
    return {"deleted": wid}


@app.post("/api/workspaces/{wid}/simulate")
def simulate_workspace(wid: str, req: SimRequest):
    w = load_ws(wid)
    c = w["compile"] = compile_workspace(w)
    save_ws(w)
    if c["errors"]:
        raise HTTPException(409, "Fix the model first: " + " ".join(c["errors"]))
    nin = int(w["layers"][0]["params"].get("channels", 1))
    waves = (req.channels + [Waveform() for _ in range(nin)])[:nin]
    u = np.stack([waveform(v.kind, req.length, v.amp, v.start, v.width, v.period) for v in waves], axis=-1)[:, None, :]
    out, traces = run_workspace(w, u)
    pw = next((l for l in w["layers"] if l["type"] == "pathway"), None)
    species = PATHWAYS[pw["params"]["pathway"]]["species"] if pw else []
    return {"inputs": np.round(u[:, 0, :], 4).T.tolist(), "output": np.round(out[:, 0, :], 4).T.tolist(),
            "species": species, "trace": traces.get(pw["id"], []) if pw else [],
            "input_names": w.get("input_names", [f"channel {i + 1}" for i in range(nin)]),
            "output_names": w.get("output_names", [f"output {i + 1}" for i in range(out.shape[-1])])}


@app.get("/api/workspaces/{wid}/code")
def workspace_code(wid: str, download: bool = False):
    w = load_ws(wid)
    code = code_workspace(w)
    if download:
        return Response(code, media_type="text/x-python",
                        headers={"Content-Disposition": f'attachment; filename="fml-workspace-{wid}.py"'})
    return {"code": code}


@app.post("/api/neo4j/query")
def neo4j_query(body: dict):
    cfg = _neo4j_settings()
    q = (body.get("cypher") or "").strip()
    if not q:
        raise HTTPException(400, "Write a query first.")
    banned = ("create ", "merge ", "delete ", "detach", "set ", "remove ", "drop ", "load csv", "call db.", "apoc.")
    if any(b in q.lower() for b in banned):
        raise HTTPException(400, "This panel runs read-only queries. Use Push to Neo4j to write.")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise HTTPException(409, "The neo4j driver is not installed. Run: .venv/bin/pip install neo4j")
    if not cfg["uri"] or not cfg["password"]:
        raise HTTPException(409, "Add your Neo4j address and password in Settings before querying.")
    try:
        with GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])) as driver:
            with driver.session(database=cfg["database"]) as session:
                res = session.run(q)
                rows = [dict(r) for _, r in zip(range(200), res)]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Neo4j rejected the query: {e}")
    cols = sorted({k for r in rows for k in r})
    return {"columns": cols, "rows": [{k: str(r.get(k, "")) for k in cols} for r in rows], "count": len(rows)}


BENCH = Path(os.environ.get("FML_BENCH", ROOT / "data" / "benchmark.json"))
_bench_state = {"status": "idle", "progress": 0.0, "where": ""}


def _bench_worker():
    def progress(frac, where):
        _bench_state.update(progress=round(frac, 3), where=where)
    try:
        out = run_benchmark(progress=progress, steps=1500)
        out["finished"] = time.time()
        BENCH.write_text(json.dumps(out))
        _bench_state.update(status="done", progress=1.0, where="")
    except Exception as e:
        _bench_state.update(status="failed", where=str(e)[:200])


@app.post("/api/benchmark")
def start_benchmark():
    if _bench_state["status"] == "running":
        return _bench_state
    _bench_state.update(status="running", progress=0.0, where="starting")
    threading.Thread(target=_bench_worker, daemon=True).start()
    return _bench_state


@app.get("/api/benchmark")
def get_benchmark():
    out = dict(_bench_state)
    if BENCH.exists():
        out["result"] = json.loads(BENCH.read_text())
        if out["status"] == "idle":
            out["status"] = "done"
    return out


class Paper(BaseModel):
    id: str = ""
    title: str = ""
    abstract: str = ""
    authors: str = ""
    journal: str | None = ""
    year: str | None = ""
    link: str | None = ""
    doi: str | None = ""
    pmcid: str | None = ""
    preprint: bool = False


class ReadRequest(BaseModel):
    paper: Paper
    text: str = ""
    reader: Literal["auto", "keywords", "claude"] = "auto"


class BuildRequest(BaseModel):
    paper: Paper
    reading: dict = Field(default_factory=dict)
    recipe: dict
    name: str = ""


@app.get("/api/research/status")
def research_status():
    return {"claude": bool(ST.get("anthropic_api_key")), "model": ST.get("anthropic_model"),
            "templates": len(RS.TEMPLATES)}


@app.get("/api/research/templates")
def research_templates():
    return [{"id": t["id"], "name": t["name"], "family": t["family"], "description": t["description"],
             "species": t["species"], "reactions": [{"id": r["id"], "name": r["name"], "law": r.get("law", "mm"),
                                                    "rate": r["kcat"]} for r in t["reactions"]],
             "reference": t["reference"]} for t in RS.TEMPLATES.values()]


@app.get("/api/research/search")
def research_search(q: str, since: int | None = None, oa: bool = False, preprints: bool = True, n: int = 15):
    if not q.strip():
        raise HTTPException(400, "Type something to search for.")
    try:
        return RS.europe_pmc(q, page_size=max(1, min(n, 40)), preprints=preprints, open_access=oa, since=since)
    except Exception as e:
        raise HTTPException(502, f"Europe PMC could not be reached: {e}")


@app.post("/api/research/read")
def research_read(req: ReadRequest):
    text = (req.paper.title + ". " + req.paper.abstract + "\n\n" + req.text).strip()
    use_claude = req.reader == "claude" or (req.reader == "auto" and ST.get("anthropic_api_key"))
    if use_claude:
        try:
            out = RS.read_with_claude(req.paper.title, text)
            if out is not None:
                out["keywords"] = RS.read_keywords(text)["matches"]
                return out
            if req.reader == "claude":
                raise HTTPException(409, "Add your Anthropic API key in Settings to read papers with Claude.")
        except HTTPException:
            raise
        except Exception as e:
            fallback = RS.read_keywords(text)
            fallback["caveat"] = f"Claude could not read this paper ({e}); fell back to the keyword reader. " + fallback["caveat"]
            return fallback
    return RS.read_keywords(text)


def _model_view(rec, feas=None):
    feas = feas or RS.feasibility(rec["pathway"], stated=len(rec["recipe"].get("overrides") or {}))
    return {"id": rec["id"], "pid": rec["pid"], "name": rec["pathway"]["name"], "paper": rec["paper"],
            "reading": rec["reading"], "recipe": rec["recipe"], "family": rec["pathway"].get("family"),
            "reactions": [{"id": r["id"], "name": r["name"], "law": r.get("law", "mm"), "rate": r["kcat"],
                           "substrates": r["substrates"], "products": r["products"], "evidence": r.get("evidence", "")}
                          for r in rec["pathway"]["reactions"]],
            "feasibility": feas, "created": rec["created"]}


@app.post("/api/research/models")
def research_build(req: BuildRequest):
    rec_recipe = req.recipe
    if not rec_recipe.get("template") and not rec_recipe.get("custom"):
        raise HTTPException(400, "Nothing to build: the reading did not find a model.")
    if rec_recipe.get("template") and rec_recipe["template"] not in RS.TEMPLATES:
        raise HTTPException(400, "Unknown model family.")
    rec, feas = RS.save_model(req.paper.model_dump(), req.reading, rec_recipe, req.name or None)
    return _model_view(rec, feas)


@app.post("/api/research/models/template")
def research_from_template(body: dict):
    tid = body.get("template")
    if tid not in RS.TEMPLATES:
        raise HTTPException(400, "Unknown model family.")
    paper = {"title": RS.TEMPLATES[tid]["name"], "abstract": RS.TEMPLATES[tid]["description"],
             "authors": RS.TEMPLATES[tid]["reference"], "link": ""}
    rec, feas = RS.save_model(paper, {"reader": "library"}, {"template": tid, "overrides": body.get("overrides") or {}},
                              body.get("name") or None)
    return _model_view(rec, feas)


@app.get("/api/research/models")
def research_models():
    return [{"id": r["id"], "pid": r["pid"], "name": r["pathway"]["name"], "family": r["pathway"].get("family"),
             "verdict": r.get("feasibility", {}).get("verdict", ""), "created": r["created"]} for r in RS.load_all()]


@app.get("/api/research/models/{mid}")
def research_model(mid: str):
    f = RS.STORE / f"{mid}.json"
    if not f.exists() or "/" in mid:
        raise HTTPException(404, "No such research model")
    return _model_view(json.loads(f.read_text()))


@app.delete("/api/research/models/{mid}")
def research_delete(mid: str):
    if not RS.delete_model(mid):
        raise HTTPException(404, "No such research model")
    return {"deleted": mid}


@app.get("/api/pathways/{pid}/targets")
def pathway_targets(pid: str, species: str, direction: Literal["lower", "raise"] = "lower"):
    out = target_search(_path(pid), species, direction)
    if out is None:
        raise HTTPException(404, "No such species in this model")
    return out


def _may_change_settings(request: Request):
    """Settings hold secrets, so changes are refused from the open internet.

    Allowed from the Mac itself, or through Cloudflare Tunnel only when Cloudflare Access has signed the visitor in.
    Requests that came through the tunnel carry a Cf-Connecting-Ip header; Access adds the signed-in email.
    """
    through_tunnel = "cf-connecting-ip" in request.headers
    if not through_tunnel:
        host = request.client.host if request.client else ""
        return host in ("127.0.0.1", "::1", "localhost", "testclient")
    return bool(request.headers.get("cf-access-authenticated-user-email"))


@app.get("/api/settings")
def get_settings(request: Request):
    out = ST.view()
    out["can_edit"] = _may_change_settings(request)
    return out


@app.put("/api/settings")
def put_settings(body: dict, request: Request):
    if not _may_change_settings(request):
        raise HTTPException(403, "Settings can only be changed from the Mac itself (http://127.0.0.1:47431) or through "
                                 "Cloudflare Access. Protect fml.fueldeskpro.com with an Access policy to edit them remotely.")
    model = (body.get("values") or {}).get("anthropic_model")
    if model and model not in ST.MODELS:
        raise HTTPException(400, "Unknown model.")
    out = ST.update(body)
    out["can_edit"] = True
    return out


@app.post("/api/settings/test/anthropic")
def test_anthropic(request: Request):
    if not _may_change_settings(request):
        raise HTTPException(403, "Testing the key is only allowed where settings can be changed.")
    key = ST.get("anthropic_api_key")
    if not key:
        raise HTTPException(409, "No Anthropic API key is set.")
    import httpx
    try:
        r = httpx.post("https://api.anthropic.com/v1/messages",
                       headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                       json={"model": ST.get("anthropic_model"), "max_tokens": 8,
                             "messages": [{"role": "user", "content": "Reply with the word ok."}]}, timeout=30)
    except Exception as e:
        raise HTTPException(502, f"Could not reach Anthropic: {e}")
    if r.status_code == 401:
        raise HTTPException(400, "Anthropic rejected the key. Check it was copied in full.")
    if r.status_code == 404:
        raise HTTPException(400, f"The key works but the model {ST.get('anthropic_model')} is not available to it.")
    if r.status_code >= 400:
        raise HTTPException(400, f"Anthropic returned {r.status_code}: {r.text[:200]}")
    return {"ok": True, "model": ST.get("anthropic_model")}


@app.post("/api/settings/test/neo4j")
def test_neo4j(request: Request):
    if not _may_change_settings(request):
        raise HTTPException(403, "Testing the connection is only allowed where settings can be changed.")
    cfg = _neo4j_settings()
    if not cfg["uri"] or not cfg["password"]:
        raise HTTPException(409, "Add the Neo4j address and password first.")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise HTTPException(409, "The neo4j driver is not installed. Run: .venv/bin/pip install neo4j")
    try:
        with GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])) as driver:
            driver.verify_connectivity()
    except Exception as e:
        raise HTTPException(400, f"Neo4j refused the connection: {e}")
    return {"ok": True, "uri": cfg["uri"]}
