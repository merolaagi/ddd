"""MCP server: the agent layer. Any MCP client (Claude Desktop, claude.ai custom
connector via your Cloudflare Tunnel, local agents) can call these tools.

Run:  python -m ddd.mcp_server   (streamable HTTP on 127.0.0.1:47521/mcp)
"""
from __future__ import annotations

import asyncio
import hmac
import os

from mcp.server.mcpserver import MCPServer

from . import __version__, kg, store
from .armory import TOOLS, run_tool
from .company import charter as charter_mod
from .company import db as cdb
from .company import engine
from .company.paths import safe_report
from . import index_catalog, vcell
from .company import programs as company_programs
from .chem import describe
from .config import MCP_PORT
from .models import available_models
from .runner import predict, run
from .tasks import TASKS, available_tasks

mcp = MCPServer(
    name="ddd",
    version=__version__,
    instructions=(
        "ddd is a headless drug-discovery company. Start with company_overview to see the charter, "
        "departments, KPIs and schedules. company_run starts a department playbook and returns a run id; "
        "poll company_run_status, then read the report with company_report. "
        "Lower-level tools: describe_molecule for RDKit properties, "
        "predict_property for trained baseline predictions, run_benchmark to score a model "
        "on a task, leaderboard to compare, and the kg_* tools to reason over the PrimeKG "
        "knowledge graph (genes, drugs, diseases, phenotypes, pathways). armory_list and "
        "armory_run pull up on-demand tools such as site_recon (HAR capture + analysis + REA). "
        "Predictions are "
        "fingerprint baselines on public data, not validated assays."
    ),
)


@mcp.tool()
def list_tasks() -> list[dict]:
    """List benchmark tasks with domain, metric, split rule and known caveat."""
    return available_tasks()


@mcp.tool()
def list_models() -> list[str]:
    """List models that can be benchmarked (local baselines and Ollama models)."""
    return available_models()


@mcp.tool()
def describe_molecule(smiles: str) -> dict:
    """RDKit description of a molecule: weight, logP, TPSA, H-bond donors/acceptors,
    QED, Lipinski violations and Bemis-Murcko scaffold."""
    return describe(smiles)


@mcp.tool()
def predict_property(task: str, smiles: list[str]) -> list[dict]:
    """Predict a task's property (e.g. bbbp, esol, lipo, clintox, bace, hiv) for up to
    50 SMILES with the random-forest baseline. Trains it on first use."""
    return predict(task, smiles[:50], "rf")


@mcp.tool()
async def run_benchmark(task: str, model: str = "rf", sample: int | None = None) -> dict:
    """Train/evaluate a model on a task's held-out scaffold split and record the score.
    model: 'rf', 'logreg' or 'ollama:<name>'. sample limits LLM test molecules (default 100)."""
    return await asyncio.to_thread(run, task, model, sample)


@mcp.tool()
def leaderboard(task: str | None = None) -> dict:
    """Best recorded score per model, for one task or all tasks."""
    board = store.leaderboard(TASKS)
    if task:
        board = {task: board.get(task, [])}
    return {t: [{"model": r["model"], "metric": r["metric"], "score": r["score"],
                 "n_test": r["n_test"]} for r in rows] for t, rows in board.items()}


@mcp.tool()
def kg_search(name: str, limit: int = 10) -> list[dict]:
    """Find knowledge-graph entities whose name contains the text."""
    return kg.search(name, limit)


@mcp.tool()
def kg_neighbors(name: str, relation: str | None = None, limit: int = 25) -> list[dict]:
    """Neighbors of an entity (exact name), optionally filtered by relation
    (e.g. 'target', 'indication', 'contraindication', 'associated with')."""
    return kg.neighbors(name, relation, limit)


@mcp.tool()
def kg_relation_summary(name: str) -> list[dict]:
    """Counts of an entity's relations by type, to see what is known about it."""
    return kg.relation_summary(name)


@mcp.tool()
def kg_path(a: str, b: str, max_hops: int = 4) -> dict:
    """Shortest mechanistic path between two entities (e.g. a drug and a disease)."""
    return kg.path(a, b, max_hops)


@mcp.tool()
def kg_candidate_targets(disease: str, limit: int = 25) -> list[dict]:
    """Naive ranked target list for a disease: associated genes ordered by how many
    drugs already touch them and phenotype links. A starting point, not a verdict."""
    return kg.disease_candidate_targets(disease, limit)


@mcp.tool()
def armory_list() -> list[dict]:
    """List Tools Armory tools (recon, HAR capture/summary/payload, REA inspect, artifacts)
    with their parameters."""
    return [t.meta() for t in TOOLS.values()]


@mcp.tool()
async def armory_run(tool: str, args: dict | None = None) -> dict:
    """Run a Tools Armory tool by name, e.g. armory_run('site_recon', {'url': 'https://o3dc.org/'})
    or armory_run('har_payload', {'har': '<file>.har', 'match': 'data.json'})."""
    return await asyncio.to_thread(run_tool, tool, args or {})


@mcp.tool()
def company_roles() -> dict:
    """List the headless-company agent roles (scout, chemist, evaluator, curator)
    and the ADMET/activity models each discovery program applies."""
    return {"roles": company_programs.ROLES, "screen": [{"task": t, "label": l} for t, l in company_programs.SCREEN]}


@mcp.tool()
async def run_discovery_program(disease: str, molecules: list[str], top_targets: int = 15) -> dict:
    """Run a full program: disease -> candidate targets from PrimeKG -> molecule triage
    and ADMET/activity screen -> markdown report. Computational triage on public data,
    not validated and not a clinical recommendation."""
    return await asyncio.to_thread(company_programs.run_program, disease, molecules[:50], top_targets)


@mcp.tool()
def benchmark_index(category: str | None = None) -> dict:
    """The O3DC-style local benchmark index: open benchmarks, each with a known caveat.
    Optionally filter by category."""
    entries = index_catalog.load_index()
    if category:
        entries = [e for e in entries if e["category"].lower() == category.lower()]
    return {"entries": entries, "categories": index_catalog.categories()}


@mcp.tool()
def virtual_cell() -> dict:
    """The Virtual Cell multi-scale scaffold: six biological scales, which are live,
    and where each engine (e.g. Fate Memory Lab) plugs in."""
    return vcell.overview()


@mcp.tool()
def company_overview() -> dict:
    """The company at a glance: charter, departments with their playbooks and last runs,
    KPIs, system status and the next scheduled runs."""
    return engine.overview()


@mcp.tool()
def company_run(playbook: str, params: dict | None = None) -> dict:
    """Start a department playbook in the background and return its run id. Playbooks:
    intel_watchlist_sweep, intel_recon {url}, bench_full_sweep {include_llm, llm_sample},
    bio_target_dossier {disease, top}, chem_molecule_triage {smiles, label},
    discovery_program {disease, molecules, top_targets}, pub_weekly_digest {days}."""
    return {"run_id": engine.start(playbook, params or {}, trigger="mcp"),
            "next": "poll company_run_status(run_id); when done, company_report(report)"}


@mcp.tool()
def company_run_status(run_id: int) -> dict:
    """Status, summary, report file name and the agent log of one run."""
    r = cdb.run_get(run_id)
    if not r:
        return {"error": f"no run {run_id}"}
    r["log"] = (r.get("log") or "")[-3000:]
    return r


@mcp.tool()
def company_runs(limit: int = 20) -> list[dict]:
    """Recent runs across all departments (newest first)."""
    out = cdb.runs(limit)
    for r in out:
        r.pop("log", None)
    return out


@mcp.tool()
def company_report(name: str) -> str:
    """Read a run report (markdown) by file name."""
    return safe_report(name).read_text()


@mcp.tool()
def company_schedule(action: str, playbook: str | None = None, every: str | None = None,
                     params: dict | None = None, schedule_id: int | None = None) -> list[dict]:
    """Manage the company's schedules. action: 'list' | 'add' (playbook, every, params) |
    'pause' | 'resume' | 'delete' (schedule_id). every: 'hourly', 'daily HH:MM' or 'weekly mon HH:MM'
    in the Mac's local time. Returns the schedule list."""
    if action == "add":
        if playbook not in engine.PLAYBOOKS:
            raise ValueError(f"unknown playbook {playbook}")
        cdb.schedule_add(playbook, params or {}, every or "")
    elif action in ("pause", "resume"):
        cdb.schedule_set(schedule_id, enabled=(action == "resume"))
    elif action == "delete":
        cdb.schedule_set(schedule_id, delete=True)
    return cdb.schedules()


@mcp.tool()
def company_update_charter(update: dict) -> dict:
    """Change charter fields (mission, scope, non_goals, principles, watchlist, focus_diseases ...).
    Only the founder should ask for this; confirm with them before calling."""
    return charter_mod.save(update)


class _BearerGate:
    """ASGI wrapper: when DDD_MCP_TOKEN is set, every request must send
    'Authorization: Bearer <token>' (claude.ai custom connector -> Request headers)."""

    def __init__(self, app, token):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, b"Bearer " + self.token):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


def main():
    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    token = os.environ.get("DDD_MCP_TOKEN", "").strip()
    public_host = os.environ.get("DDD_MCP_PUBLIC_HOST", "").strip()
    security = None
    if public_host:
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1", f"127.0.0.1:{MCP_PORT}", "localhost", f"localhost:{MCP_PORT}", public_host],
            allowed_origins=[f"https://{public_host}", "https://claude.ai"])
        if not token:
            raise SystemExit("DDD_MCP_PUBLIC_HOST is set but DDD_MCP_TOKEN is empty: refusing to serve publicly without a token.")
    app = mcp.streamable_http_app(streamable_http_path="/mcp", transport_security=security)
    if token:
        app = _BearerGate(app, token)
    uvicorn.run(app, host="127.0.0.1", port=MCP_PORT, log_level="info")


if __name__ == "__main__":
    main()
