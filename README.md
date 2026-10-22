# ddd · v0.6.1

**A headless drug-discovery intelligence company on one Mac.** No staff: six
departments run as agents executing playbooks on a schedule or on request. The
founder sets the charter and reads the reports. Claude can operate the company
through MCP. No paid model APIs.

| Part | Where |
|---|---|
| HQ: charter, KPIs, departments, activity, schedules, open index | http://127.0.0.1:47520/ |
| Lab: benchmarks, molecule tools, Tools Armory, KG explorer, programs, Virtual Cell | http://127.0.0.1:47520/lab |
| Public page: leaderboard + open benchmark index (rebuilt by Publishing) | http://127.0.0.1:47520/public/ |
| MCP server, streamable HTTP at `/mcp` | 127.0.0.1:47521 |
| Neo4j with PrimeKG (optional) | 7687 |
| Ollama local LLMs (optional) | 11434 |

## Install, upgrade and run

Every release ships as `ddd-vX.Y.Z.tar.gz`. The same line installs the first
time and upgrades later; data, `.venv` and `.env` are kept.

```
cd ~/Downloads && tar -xzf ddd-v0.6.0.tar.gz && bash ddd-v0.6.0/install.sh && cd ~/Sites/ddd && ./serve.sh && git push
```

## The company

**Charter** (editable on HQ, saved to `data/company/charter.json`): mission,
operating model, scope, non-goals, principles, the intelligence watchlist and
the focus diseases.

| Department | Agent | Playbooks |
|---|---|---|
| Intelligence | Scout | `intel_watchlist_sweep`: recon every watchlist page and diff against the last capture (new/removed endpoints, schema changes, redeploys) · `intel_recon {url}` |
| Evaluation | Judge | `bench_full_sweep`: every task × every baseline (optionally every Ollama model), flags regressions |
| Target Biology | Cartographer | `bio_target_dossier {disease}`: ranked targets, linked drugs and graph context from PrimeKG |
| Chemistry | Triage | `chem_molecule_triage {smiles}`: properties, six baseline predictions, red flags |
| Discovery Programs | Lead | `discovery_program {disease, molecules}`: Scout → Chemist → Evaluator → Curator, end to end |
| Publishing | Herald | `pub_weekly_digest`: the week's runs, repo activity for the index, rebuilds the public page |

Every run is recorded (status, params, trigger, agent log) and leaves a
markdown report in `data/company/reports/`.

**Standing schedule** (seeded on first start, editable on HQ, local time):
watchlist sweep Mon 06:00, benchmark sweep Sun 03:00, digest Mon 08:00.
Schedules fire while the server runs. Formats: `hourly`, `daily HH:MM`,
`weekly mon HH:MM`.

**Optional narrator:** set `DDD_NARRATOR=<ollama model>` in `.env` and every
report gets a short executive summary written by that local model.

## Operating it from Claude (no API spend)

The MCP server exposes `company_overview`, `company_run`,
`company_run_status`, `company_runs`, `company_report`, `company_schedule`,
`company_update_charter`, plus the lower-level bench, KG, armory, index,
program and Virtual Cell tools.

To reach it from claude.ai it must be public, so it needs a token:

1. In `~/Sites/ddd/.env` set `DDD_MCP_TOKEN=<long random string>` and
   `DDD_MCP_PUBLIC_HOST=<hostname>` (e.g. `ddd-mcp.fueldeskpro.com`), then `./serve.sh`.
2. Route that hostname to `http://localhost:47521` in your Cloudflare Tunnel.
3. In claude.ai: Customize → Connectors → Add custom connector, URL
   `https://<hostname>/mcp/<token>`, Authentication "No sign in". (A request
   header `Authorization: Bearer <token>` on `https://<hostname>/mcp` also works.)

OAuth discovery paths return 404, so clients don't assume a sign-in. Every
request is logged to `logs/mcp.log` with the token redacted and, when refused,
the reason (no header, wrong value, and so on).

The server refuses to start with a public host and no token. The web UI
(47520) has no auth **unless** you set `DDD_WEB_PASSWORD` in `.env`: with it
set, every web request needs HTTP Basic auth (any username, that password),
so the UI can be exposed over the tunnel safely. Left empty, the UI stays
open for localhost-only use.

## Tools Armory

On-demand tools behind the departments, from the Lab, the API
(`POST /api/armory/<tool>`) or MCP (`armory_list`, `armory_run`):
`site_recon`, `har_capture`, `har_summary`, `har_payload`, `rea_inspect`,
`artifacts`. Captures stay in `data/armory/` (git-ignored). Before REA runs, ddd
writes a temporary copy with server IP addresses normalised (REA's HAR schema
rejects bracketed IPv6) and, over 30 MB, media bodies dropped; REA failures are
reported with REA's own reason. Recon refuses
private, loopback and link-local addresses, so a remote caller can't use the
Mac to browse your home network.

## Benchmarks

21 tasks, DDDBench-style: each is tagged with a drug-R&D **phase** (target
discovery → hit ID → hit-to-lead → lead-opt ADME → lead-opt safety → clinical)
and a **category**, and carries a known caveat.

- **MoleculeNet** (BBBP, ESOL, Lipophilicity, ClinTox, BACE, HIV) and
  `ogbg-molhiv` (OGB official split, after `pip install ogb`).
- **TDC ADMET/safety** (14 tasks): Caco-2, AqSolDB solubility, BBB, oral
  bioavailability, P-gp, plasma protein binding, CYP3A4/2D6/2C9 inhibition,
  hERG, Ames, DILI, half-life, hepatocyte clearance. Downloaded from Harvard
  Dataverse on first use; metrics follow TDC (MAE, Spearman, AUROC).

All sets are re-split here with a deterministic Bemis-Murcko scaffold split, so
scores test generalisation but are **not** comparable to TDC's own 5-seed
leaderboard, and scaffold splitting is not decontamination. Models: `rf`,
`logreg` and `ollama:<name>` (zero-shot).

**ddd score** (`/api/scoreboard`, MCP `scoreboard`, and the public page): each
task's models are min-max normalised to 0-100 (direction-aware) and averaged
per model across categories — a DDDBench-style aggregate. It's a *relative*
ranking among the models run, not an absolute quality score; only tasks with
2+ scored models count.

## Open benchmark index

16 curated public benchmarks and services plus the bench's own tasks, each
with a known caveat. Add entries with `POST /api/index` (a real caveat is
required); they're stored in `data/benchmark_index.json`.

## Virtual Cell

A six-scale scaffold (molecular → organism) with biological age as the
conditioning variable. The molecular scale is live, the intracellular scale
links to Fate Memory Lab, the rest are stubs with a fixed contract.

## Knowledge graph viewer

The Lab's **Knowledge graph** panel renders PrimeKG subgraphs interactively,
in-app (no external Neo4j Browser): type an entity (disease, gene, drug) and
press **Visualize** for a force-directed graph coloured by node type, with the
centre node ringed and edges labelled by relation. Double-click any node to
expand its neighbours. Served by `/api/kg/subgraph` and the MCP `kg_subgraph`
tool; the vis-network library is vendored at `static/vendor/`, so it works
offline with no CDN.

## Knowledge graph (optional)

```
brew install neo4j
neo4j-admin dbms set-initial-password YOUR_PASSWORD
brew services start neo4j
```

Set `NEO4J_PASSWORD` in `~/Sites/ddd/.env`, then
`.venv/bin/python scripts/load_primekg.py --download` and `./serve.sh`.
PrimeKG: Chandak, Huang & Zitnik, *Scientific Data* 2023, doi:10.7910/DVN/IXA7BM.
