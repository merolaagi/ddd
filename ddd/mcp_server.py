"""MCP server: the agent layer. Any MCP client (Claude Desktop, claude.ai custom
connector via your Cloudflare Tunnel, local agents) can call these tools.

Run:  python -m ddd.mcp_server   (streamable HTTP on 127.0.0.1:47521/mcp)
"""
from __future__ import annotations

import asyncio

from mcp.server.mcpserver import MCPServer

from . import __version__, kg, store
from .armory import TOOLS, run_tool
from .chem import describe
from .config import MCP_PORT
from .models import available_models
from .runner import predict, run
from .tasks import TASKS, available_tasks

mcp = MCPServer(
    name="ddd",
    version=__version__,
    instructions=(
        "Headless drug-discovery bench. Use describe_molecule for RDKit properties, "
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


def main():
    mcp.run(transport="streamable-http", host="127.0.0.1", port=MCP_PORT)


if __name__ == "__main__":
    main()
