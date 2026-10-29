"""MCP server: the agent layer. Any MCP client (Claude Desktop, claude.ai custom
connector via your Cloudflare Tunnel, local agents) can call these tools.

Run:  python -m ddd.mcp_server   (streamable HTTP on 127.0.0.1:47521/mcp)
"""
from __future__ import annotations

import asyncio
import hmac
import os
import time

from mcp.server.mcpserver import MCPServer

from . import __version__, kg, kg_model, store
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
def generate_molecules(target: str = "", seeds: list[str] | None = None,
                       organism: str | None = None, n: int = 12,
                       generations: int = 20, dock: bool = False) -> dict:
    """De-novo molecule design (our Chemistry42 analog). A graph genetic algorithm
    evolves novel, drug-like molecules in the neighbourhood of a target's known
    ChEMBL actives, scored by QED + synthetic accessibility + Ro5 + similarity.
    target: gene symbol to seed from (e.g. 'EGFR'); or pass explicit `seeds` SMILES.
    dock=True also docks the final elite vs the target if a Vina engine is configured.
    Returns ranked candidates with per-molecule reward components. Reward-guided search
    over public chemistry, not validated activity."""
    from . import generate as _gen
    return _gen.generate(target=target, seeds=seeds, organism=organism,
                         n=min(n, 30), generations=min(generations, 40), dock=dock)


@mcp.tool()
def molecule_3d(smiles: str, seed: int = 1) -> dict:
    """Generate a real 3D conformer for a molecule: ETKDGv3 distance-geometry
    embedding + MMFF94 minimisation (UFF fallback). Returns an MDL mol block with
    explicit hydrogens and the force-field energy — the molecule's actual shape,
    ready for a 3D viewer or docking. Energy is relative within one molecule, not
    a binding score."""
    from . import generate as _gen
    return _gen.embed3d(smiles, seed=seed)


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
def scoreboard() -> dict:
    """DDDBench-style aggregate: each model's 0-100 ddd score, broken down by
    category, averaged over all tasks run. A relative ranking, not absolute quality."""
    return store.scoreboard(TASKS)


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
def kg_subgraph(center: str, depth: int = 1, limit: int = 60, relation: str | None = None) -> dict:
    """Nodes and edges around an entity, shaped for graph rendering (the Lab's graph view
    uses this). depth 1 = direct neighbors, 2 also expands one more hop."""
    return kg.subgraph(center, depth, limit, relation)


@mcp.tool()
def kg_provenance() -> list[dict]:
    """Edge counts by source (PrimeKG vs enrichment from Open Targets, ChEMBL)."""
    return kg.provenance()


@mcp.tool()
def kg_hypothesis(a: str, b: str, max_hops: int = 4) -> dict:
    """Check a hypothesis: how plausible is a DIRECT link between two entities?
    Returns whether they're already linked, the shortest existing path, the
    shared-neighbour evidence and the standard link-prediction scores
    (common neighbours, Jaccard, Adamic-Adar). A model computed from the graph."""
    return kg_model.hypothesis(a, b, max_hops)


@mcp.tool()
def kg_predict_links(entity: str, target: str = "gene", k: int = 20) -> dict:
    """Generate hypotheses: rank NOVEL target-type nodes (gene/drug/disease/...)
    that the graph implies should link to `entity` but currently don't, by
    Adamic-Adar over the shared neighbourhood. New, testable predictions."""
    return kg_model.predict_links(entity, target, k)


@mcp.tool()
def opentargets_disease(disease: str, size: int = 25, write: bool = True) -> dict:
    """Open Targets evidence-weighted target associations for a disease (score 0-1).
    By default writes them into the knowledge graph as OpenTargets-sourced edges.
    Needs internet on the host."""
    from .sources import opentargets
    return opentargets.enrich_disease(disease, size, write)


@mcp.tool()
def opentargets_target_drugs(symbol: str, size: int = 25) -> dict:
    """Known drugs for a target from Open Targets (name, phase, mechanism of action)."""
    from .sources import opentargets
    return opentargets.target_known_drugs(symbol, size)


@mcp.tool()
def chembl_target_actives(symbol: str, standard_type: str = "IC50", limit: int = 50, write: bool = True) -> dict:
    """Potent measured compounds (pChEMBL>=6) for a target from ChEMBL. By default
    writes Compound-active->Gene edges into the graph. Needs internet on the host."""
    from .sources import chembl
    return chembl.enrich_target(symbol, standard_type, limit, write)


@mcp.tool()
def chembl_molecule(name: str) -> dict:
    """Look up a molecule in ChEMBL by name (ChEMBL id, SMILES, max clinical phase)."""
    from .sources import chembl
    return chembl.molecule(name)


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


class _TokenGate:
    """ASGI wrapper used when DDD_MCP_TOKEN is set. A request is let in if either
    - its path is /mcp/<token> (the token is part of the connector URL), or
    - it sends 'Authorization: Bearer <token>'.
    OAuth discovery paths get 404 so clients don't assume a sign-in is needed.
    Every request is logged with the token redacted, plus why it was refused."""

    OAUTH_PATHS = ("/.well-known/oauth-protected-resource", "/.well-known/oauth-authorization-server",
                   "/.well-known/openid-configuration", "/register", "/authorize", "/token")

    def __init__(self, app, token):
        self.app, self.token = app, token
        self.secret_prefix = f"/mcp/{token}"

    @staticmethod
    async def _reply(send, status, body):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    def _log(self, scope, status, note=""):
        client = (scope.get("client") or ("?", 0))[0]
        path = scope.get("path", "").replace(self.token, "<token>")
        print(f"{time.strftime('%H:%M:%S')} {client} {scope.get('method')} {path} -> {status}{' (' + note + ')' if note else ''}",
              flush=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path.startswith(self.OAUTH_PATHS):
            self._log(scope, 404, "no OAuth on this server")
            return await self._reply(send, 404, b'{"error":"not found"}')

        ok, how = False, ""
        if path == self.secret_prefix or path.startswith(self.secret_prefix + "/"):
            ok, how = True, "token in URL"
            rest = path[len(self.secret_prefix):] or ""
            scope = dict(scope, path="/mcp" + rest, raw_path=("/mcp" + rest).encode())
        else:
            auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
            if hmac.compare_digest(auth.encode(), f"Bearer {self.token}".encode()):
                ok, how = True, "bearer header"
            elif not auth:
                how = "no Authorization header and no token in URL"
            elif not auth.startswith("Bearer "):
                how = "Authorization header present but doesn't start with 'Bearer '"
            elif path.startswith("/mcp/"):
                how = "token in URL doesn't match"
            else:
                how = f"bearer token doesn't match (got {len(auth) - 7} chars, expected {len(self.token)})"
        if not ok:
            self._log(scope, 401, how)
            return await self._reply(send, 401, b'{"error":"unauthorized"}')

        status = {}

        async def _send(msg):
            if msg["type"] == "http.response.start":
                status["code"] = msg["status"]
            await send(msg)
        await self.app(scope, receive, _send)
        self._log(scope, status.get("code", "?"), how)


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
        app = _TokenGate(app, token)
    uvicorn.run(app, host="127.0.0.1", port=MCP_PORT, log_level="info", access_log=not token)


if __name__ == "__main__":
    main()
