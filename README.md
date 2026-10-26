# ddd · v0.10.0

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

Delivered over git. The same line upgrades in place; data, `.venv` and `.env`
are kept.

```
cd ~/Sites/ddd && git pull origin main && .venv/bin/pip install -q -r requirements.txt && ./serve.sh
```

## The company

**Charter** (editable on HQ, saved to `data/company/charter.json`): mission,
operating model, scope, non-goals, principles, the intelligence watchlist and
the focus diseases.

| Department | Agent | Playbooks |
|---|---|---|
| Intelligence | Scout | `intel_watchlist_sweep`: recon every watchlist page and diff against the last capture (new/removed endpoints, schema changes, redeploys) · `intel_recon {url}` |
| Evaluation | Judge | `bench_full_sweep`: every task × every baseline (optionally every Ollama model), flags regressions |
| Target Biology | Cartographer | `bio_target_dossier {disease}`: targets ranked by Open Targets association score, known drugs with ChEMBL activity counts, graph context from PrimeKG |
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
`company_update_charter`, plus the lower-level bench, KG, knowledge-source
enrichment (`opentargets_*`, `chembl_*`, `kg_provenance`), armory, index,
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
`dock` (see Emulators), `artifacts`. Captures stay in `data/armory/` (git-ignored). Before REA runs, ddd
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

## Models from the graph

Rung 3: predictions computed **from the knowledge graph itself**, not fetched
from anywhere. Standard link-prediction over PrimeKG plus every enrichment edge.

- **Check a hypothesis** (`/api/kg/hypothesis`, MCP `kg_hypothesis`, Lab panel):
  how plausible is a *direct* link between two entities? Returns whether they're
  already linked and by what source, the shortest existing path, the shared
  neighbours, and the standard indices (common neighbours, Jaccard, Adamic-Adar,
  preferential attachment) with a plain-language reading.
- **Generate hypotheses** (`/api/kg/predict`, MCP `kg_predict_links`): rank
  NOVEL nodes of a type (gene/drug/disease/phenotype) that the graph implies
  should link to an entity but currently don't, by Adamic-Adar over the shared
  neighbourhood — new, testable predictions, each with its evidence.

Tier 1 above needs no training and no extra deps. Tier 2 is learned node2vec
embeddings (`kg_model.train_embeddings`, optional `gensim`) for similarity-based
scoring once you want a trained model. Topological signal, not validated causality.

## Antiviral repurposing

The first composite model built on the foundations above — finds host-directed
antiviral candidates for a respiratory virus, without a monolithic deep net.

`antiviral_repurposing {virus}` (Discovery Programs): a respiratory virus →
curated **host dependency / entry factors** (the human proteins it needs) →
**ChEMBL inhibitors** of those proteins as repurposing candidates → triaged
(RDKit + ADMET baselines) and ranked by **potency × target importance**, with a
polypharmacology bonus for hitting several host factors. Host-directed on
purpose: harder for the virus to mutate away than a direct-acting antiviral.

Known viruses: SARS-CoV-2, influenza, RSV, rhinovirus, parainfluenza. The
curated virus→host edges are written into the graph (`source=VirusHost`,
`ddd/sources/virhost.py`); a live VirHostNet/IntAct connector can extend them
later. Computational triage only — efficacy needs antiviral assays and lung PK.

## Knowledge graph viewer

The Lab's **Knowledge graph** panel renders PrimeKG subgraphs interactively,
in-app (no external Neo4j Browser): type an entity (disease, gene, drug) and
press **Visualize** for a force-directed graph coloured by node type, with the
centre node ringed and edges labelled by relation. Double-click any node to
expand its neighbours. Served by `/api/kg/subgraph` and the MCP `kg_subgraph`
tool; the vis-network library is vendored at `static/vendor/`, so it works
offline with no CDN.

## Knowledge sources · enrichment

The PrimeKG graph is a fixed 2023 snapshot. Two live public APIs enrich it
on demand — no bulk downloads, so it stays disk-safe — writing new nodes and
edges back into Neo4j tagged with their source, so provenance is always
visible.

- **Open Targets** (GraphQL): disease → target associations with the
  overall association score, and each target's known drugs. The target
  dossier ranks by this score when `enrich` is on (the default).
- **ChEMBL** (REST): a target's measured actives (pChEMBL ≥ 6) and molecule
  lookups, so a dossier shows how much real bioactivity data backs each top
  target.

From the Lab's **Knowledge sources** panel, the API
(`/api/sources/opentargets/disease`, `/opentargets/target_drugs`,
`/chembl/target`, `/chembl/molecule`) or MCP (`opentargets_disease`,
`opentargets_target_drugs`, `chembl_target_actives`, `chembl_molecule`).
`/api/kg/provenance` (MCP `kg_provenance`) reports edge counts by source, so
PrimeKG, Open Targets and ChEMBL contributions stay distinct. Enrichment is
best-effort: if an API is unreachable the dossier falls back to PrimeKG alone.

## Emulators

Rung 2 of the pharma.ai ladder: cheap physics-based predictors that *consume*
the knowledgebase instead of filling it.

**Docking** (`dock` armory tool, `Emulator` category). Give it a target (gene
symbol or UniProt accession) and one or more ligand SMILES; it fetches the
target's **AlphaFold** structure on demand (one small PDB, cached under
`data/armory/structures/`, no bulk downloads), prepares receptor and ligands,
runs **AutoDock Vina / smina** and returns binding affinities (kcal/mol, more
negative = tighter) with the poses saved under `data/armory/dock/`. The
knowledgebase feeds it directly: Open Targets picks the target, ChEMBL supplies
the measured actives to dock.

Engine and prep tools are **optional** and not in `requirements.txt` (they're
heavy and you're the one watching disk). The tool detects what's on the host
and, if something's missing, returns the exact install line instead of failing.

On **Apple Silicon** there's no pip wheel for `vina` (it tries to compile and
can't find Boost), so use the conda-forge build and point ddd at it:

```
brew install micromamba
micromamba create -y -n dock -c conda-forge vina openbabel
.venv/bin/pip install meeko
```

then in `~/Sites/ddd/.env`:

```
DDD_DOCK_VINA=micromamba run -n dock vina
DDD_DOCK_OBABEL=micromamba run -n dock obabel
```

(On Linux / Intel, `.venv/bin/pip install vina meeko` and `apt/brew` Open Babel
also work; the tool auto-detects the vina Python module or a `vina`/`smina` on
PATH. `DDD_DOCK_VINA` / `DDD_DOCK_OBABEL` override detection.)

Caveats, stated on every result: AlphaFold models are **apo** (no
ligand-shaped pocket) and the default box is a **blind** box over the whole
protein, so scores are a *relative triage signal*, not validated affinities.
Pass a pocket centre (`center="x,y,z"`) to dock into a known site.

## Knowledge graph (optional)

```
brew install neo4j
neo4j-admin dbms set-initial-password YOUR_PASSWORD
brew services start neo4j
```

Set `NEO4J_PASSWORD` in `~/Sites/ddd/.env`, then
`.venv/bin/python scripts/load_primekg.py --download` and `./serve.sh`.
PrimeKG: Chandak, Huang & Zitnik, *Scientific Data* 2023, doi:10.7910/DVN/IXA7BM.
