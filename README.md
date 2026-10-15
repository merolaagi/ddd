# ddd · v0.4.2

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
cd ~/Downloads && tar -xzf ddd-v0.4.2.tar.gz && bash ddd-v0.4.2/install.sh && cd ~/Sites/ddd && ./serve.sh && git push
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
(47520) has no auth: keep it local.

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

MoleculeNet sets (BBBP, ESOL, Lipophilicity, ClinTox, BACE, HIV) with a
deterministic Bemis-Murcko scaffold split; `ogbg-molhiv` with OGB's official
split after `pip install ogb`. Each task carries a known caveat. Scaffold
splits test generalisation but are not decontamination; time-split tasks are
planned. Models: `rf`, `logreg` and `ollama:<name>` (zero-shot).

## Open benchmark index

16 curated public benchmarks and services plus the bench's own tasks, each
with a known caveat. Add entries with `POST /api/index` (a real caveat is
required); they're stored in `data/benchmark_index.json`.

## Virtual Cell

A six-scale scaffold (molecular → organism) with biological age as the
conditioning variable. The molecular scale is live, the intracellular scale
links to Fate Memory Lab, the rest are stubs with a fixed contract.

## Knowledge graph (optional)

```
brew install neo4j
neo4j-admin dbms set-initial-password YOUR_PASSWORD
brew services start neo4j
```

Set `NEO4J_PASSWORD` in `~/Sites/ddd/.env`, then
`.venv/bin/python scripts/load_primekg.py --download` and `./serve.sh`.
PrimeKG: Chandak, Huang & Zitnik, *Scientific Data* 2023, doi:10.7910/DVN/IXA7BM.
