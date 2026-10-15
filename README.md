# ddd · v0.2.0

Headless drug discovery on one Mac: open benchmarks, a PrimeKG knowledge graph,
an MCP agent layer, and a **Tools Armory** of on-demand tools. No paid API.

| Part | Port |
|---|---|
| Web UI + JSON API (`ddd/app.py`) | 47520 |
| MCP server, streamable HTTP at `/mcp` (`ddd/mcp_server.py`) | 47521 |
| Neo4j with PrimeKG (optional) | 7687 |
| Ollama local LLMs (optional) | 11434 |

## Install, upgrade and run

Every release ships as `ddd-vX.Y.Z.tar.gz`. The same line installs the first
time and upgrades later; data, `.venv` and `.env` are kept.

```
cd ~/Downloads && tar -xzf ddd-v0.2.0.tar.gz && bash ddd-v0.2.0/install.sh && cd ~/Sites/ddd && ./serve.sh && git push -u origin main
```

`install.sh` creates the venv, installs packages and headless Chromium,
initialises git, creates `merolaagi/ddd` with `gh` if it is installed and
signed in (otherwise create the empty repo once on github.com), and commits the
release.

## Tools Armory

Tools ddd pulls up when it needs them, from the UI, the API
(`POST /api/armory/<tool>`) or MCP (`armory_list`, `armory_run`).

| Tool | What it does |
|---|---|
| `site_recon` | HAR capture → summary → REA → markdown report, in one call |
| `har_capture` | Load a public URL in headless Chromium, scroll, click tab-like controls (skips submit/join/sign-in style buttons), save HAR with cookies/auth redacted |
| `har_summary` | Hosts, first-party JSON endpoints with inferred schemas, JS bundles with embedded-data hints, third-party calls, failed requests |
| `har_payload` | Body of the first entry whose URL contains a substring |
| `rea_inspect` | `rea inspect-web-network-capture <har> har --json` via npx (Node 22+); HARs over 30 MB are slimmed first |
| `artifacts` | List captures, REA evidence and reports |

Captures live in `data/armory/` and are git-ignored, so third-party content
never lands in the public repo. Use recon on public pages only.

To add a tool: write a function in `ddd/armory/`, register it in `TOOLS` in
`ddd/armory/__init__.py`. UI, API and MCP pick it up automatically.

## Benchmarks

MoleculeNet sets (BBBP, ESOL, Lipophilicity, ClinTox, BACE, HIV) with a
deterministic Bemis-Murcko scaffold split; `ogbg-molhiv` with OGB's official
split after `pip install ogb`. Each task carries a known caveat. Scaffold
splits test generalisation but are not decontamination; time-split tasks are
planned.

Models: `rf`, `logreg` (fingerprint baselines) and `ollama:<name>` (zero-shot).

## Knowledge graph (optional)

```
brew install neo4j
neo4j-admin dbms set-initial-password YOUR_PASSWORD
brew services start neo4j
```

Set `NEO4J_PASSWORD` in `~/Sites/ddd/.env`, then
`.venv/bin/python scripts/load_primekg.py --download` and `./serve.sh`.
PrimeKG: Chandak, Huang & Zitnik, *Scientific Data* 2023, doi:10.7910/DVN/IXA7BM.

## Security

Both servers bind to 127.0.0.1 with no auth. Don't route them through a
public tunnel until a token check is added.
