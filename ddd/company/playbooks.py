"""Playbooks: the work each department does. A playbook takes a context (for
logging) plus parameters and returns (summary_dict, markdown_lines)."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

from .. import kg, store
from ..armory import har as harmod
from ..armory import recon
from ..armory.paths import DIRS, stamp_name
from ..chem import describe
from ..models import BASELINES, ollama_models
from ..runner import predict, run as bench_run
from ..tasks import TASKS, PHASES, available_tasks
from . import charter as charter_mod
from . import db
from .paths import COMPANY_DIR, PUBLIC


# ---------------------------------------------------------------- Intelligence

def _har_prefix(url):
    return stamp_name(url, "har").rsplit("-", 2)[0]


def _hars_for(url):
    prefix = _har_prefix(url) + "-"
    files = [p for p in DIRS["har"].glob(prefix + "*.har") if not p.name.endswith(".slim.har")]
    return sorted(files, key=lambda p: p.name)


def _endpoint_key(e):
    return f"{e['method']} {e['url'].split('?')[0]}"


def _diff(old, new):
    oe = {_endpoint_key(e): e for e in old["data_endpoints"]}
    ne = {_endpoint_key(e): e for e in new["data_endpoints"]}
    changed = [k for k in ne.keys() & oe.keys()
               if json.dumps(ne[k].get("schema"), sort_keys=True) != json.dumps(oe[k].get("schema"), sort_keys=True)]
    ob = {b["url"].split("?")[0].rsplit("/", 1)[-1] for b in old["js_bundles"]}
    nb = {b["url"].split("?")[0].rsplit("/", 1)[-1] for b in new["js_bundles"]}
    return {"added": sorted(ne.keys() - oe.keys()), "removed": sorted(oe.keys() - ne.keys()),
            "schema_changed": sorted(changed), "bundles_changed": ob != nb,
            "requests": [old["entries"], new["entries"]]}


def intel_recon_one(ctx, url: str, run_rea: bool = True):
    ctx.log(f"Recon {url}")
    prev = _hars_for(url)
    res = recon.site_recon(url, explore=True, run_rea=run_rea)
    new_sum = harmod.summarize(res["har"])
    diff = _diff(harmod.summarize(prev[-1].name), new_sum) if prev else None
    ctx.log(f"  {res['entries']} requests, {res['data_endpoints']} data endpoints"
            + (f", changes: +{len(diff['added'])} −{len(diff['removed'])} ~{len(diff['schema_changed'])}" if diff else ", first capture"))
    return res, new_sum, diff


def intel_watchlist_sweep(ctx, run_rea: bool = True):
    c = charter_mod.load()
    lines = ["# Intelligence: watchlist sweep", "",
             "Public pages on the watchlist, captured and compared with the previous capture.", ""]
    sites, changed = [], 0
    for item in c["watchlist"]:
        url, label = item["url"], item.get("label") or item["url"]
        try:
            res, summ, diff = intel_recon_one(ctx, url, run_rea)
        except Exception as e:
            ctx.log(f"  failed: {e}")
            lines += [f"## {label}", "", f"Capture failed: {e}", ""]
            sites.append({"label": label, "error": str(e)[:200]})
            continue
        moved = bool(diff and (diff["added"] or diff["removed"] or diff["schema_changed"] or diff["bundles_changed"]))
        changed += moved
        lines += [f"## {label}", "", f"- URL: {url}", f"- Title: {summ.get('title')}",
                  f"- Requests: {res['entries']} · data endpoints: {res['data_endpoints']}",
                  f"- Recon report: `{res['report']}`"]
        if res.get("rea"):
            if "error" in res["rea"]:
                lines.append(f"- REA: **failed**: {res['rea']['error']}")
            else:
                lines.append(f"- REA evidence: `{res['rea']['output']}` ({res['rea'].get('records')} records)")
        if diff is None:
            lines.append("- First capture: baseline recorded.")
        else:
            lines.append("- **Changed since last capture**" if moved else "- No structural change since last capture.")
            for k in ("added", "removed", "schema_changed"):
                for ep in diff[k][:10]:
                    lines.append(f"  - {k.replace('_', ' ')}: `{ep}`")
            if diff["bundles_changed"]:
                lines.append("  - JS bundle names changed (front-end redeployed)")
        lines += [""] + [f"> {n}" for n in res["notes"]] + [""]
        sites.append({"label": label, "endpoints": res["data_endpoints"], "changed": moved, "report": res["report"]})
    summary = {"sites": len(sites), "changed": changed, "failed": sum("error" in s for s in sites), "detail": sites}
    return summary, lines


def intel_recon(ctx, url: str, run_rea: bool = True):
    res, summ, diff = intel_recon_one(ctx, url, run_rea)
    lines = [f"# Intelligence: recon of {summ.get('title') or url}", "", f"- URL: {url}",
             f"- Full recon report: `{res['report']}`", ""] + [f"- {n}" for n in res["notes"]]
    if diff:
        lines += ["", "## Change since last capture", "```json", json.dumps(diff, indent=1), "```"]
    return {"url": url, "endpoints": res["data_endpoints"], "report": res["report"], "diff": diff}, lines


# ---------------------------------------------------------------- Evaluation

def _best():
    return {(t, r["model"]): r["score"] for t, rows in store.leaderboard(TASKS).items() for r in rows}


def bench_full_sweep(ctx, include_llm: bool = False, llm_sample: int = 50):
    before = _best()
    models = list(BASELINES) + ([f"ollama:{m}" for m in ollama_models()] if include_llm else [])
    tasks = [t for t in available_tasks() if t["available"]]
    rows, fails = [], []
    for t in tasks:
        for m in models:
            ctx.log(f"{t['id']} × {m}")
            try:
                r = bench_run(t["id"], m, llm_sample if m.startswith("ollama:") else None)
            except Exception as e:
                ctx.log(f"  failed: {str(e)[:160]}")
                fails.append({"task": t["id"], "model": m, "error": str(e)[:200]})
                continue
            prev = before.get((t["id"], m))
            rows.append({"task": t["id"], "model": m, "metric": r["metric"], "score": r["score"],
                         "prev_best": prev, "hib": t["higher_is_better"], "n_test": r["n_scored"]})
    sb = store.scoreboard(TASKS)
    lines = ["# Evaluation: full benchmark sweep", "",
             f"Models: {', '.join(models)} · tasks attempted: {len(tasks)} · scaffold splits "
             "(OGB split for ogbg-molhiv). TDC sets are re-split here, so scores aren't comparable "
             "to TDC's own leaderboard.", ""]

    if sb["models"]:
        lines += ["## ddd score", "",
                  "Each task's models are min-max normalised to 0-100 (direction-aware), then averaged "
                  "per model across categories. Only tasks with 2+ scored models count, so a lone score "
                  "can't set the scale. This is a relative ranking among the models run here, not an "
                  "absolute quality score.", "",
                  "| Rank | Model | ddd score | " + " | ".join(sb["categories"]) + " | tasks |",
                  "|---|---|---|" + "---|" * len(sb["categories"]) + "---|"]
        for r in sb["models"]:
            cats = " | ".join(str(r["by_category"].get(c, "—")) for c in sb["categories"])
            lines.append(f"| {r['rank']} | {r['model']} | **{r['score']}** | {cats} | {r['tasks']} |")
        lines.append("")

    regress = 0
    by_phase = defaultdict(list)
    for r in rows:
        by_phase[TASKS[r["task"]].phase or "Other"].append(r)
    lines += ["## Results by drug-R&D phase", ""]
    for phase in PHASES + ["Other"]:
        if phase not in by_phase:
            continue
        lines += [f"### {phase}", "", "| Task | Model | Metric | Score | Previous best | n test |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(by_phase[phase], key=lambda r: (r["task"], r["model"])):
            flag = ""
            if r["prev_best"] is not None and r["score"] is not None:
                worse = r["score"] < r["prev_best"] if r["hib"] else r["score"] > r["prev_best"]
                if worse and abs(r["score"] - r["prev_best"]) > 0.01:
                    flag, regress = " ⚠", regress + 1
            lines.append(f"| {r['task']} | {r['model']} | {r['metric']} | {r['score']}{flag} | "
                         f"{r['prev_best'] if r['prev_best'] is not None else '—'} | {r['n_test']} |")
        lines.append("")
    if fails:
        lines += ["## Failures", ""] + [f"- {f['task']} × {f['model']}: {f['error']}" for f in fails] + [""]
    lines += ["Baselines are deterministic, so a ⚠ (worse than previous best by more than 0.01) "
              "means the data or code changed, not luck."]
    return {"runs": len(rows), "failures": len(fails), "regressions": regress,
            "models_ranked": len(sb["models"]), "top_model": sb["models"][0]["model"] if sb["models"] else None}, lines


# ---------------------------------------------------------------- Target biology

def bio_target_dossier(ctx, disease: str, top: int = 12, enrich: bool = True):
    st = kg.status()
    if not st.get("connected"):
        err = (st.get("error") or "").lower()
        if "unauthorized" in err or "authentication" in err:
            raise RuntimeError("Neo4j is running but ddd's credentials are wrong. Set NEO4J_PASSWORD "
                               "in ~/Sites/ddd/.env to your Neo4j password and restart (./serve.sh).")
        raise RuntimeError("Knowledge graph is offline. Start Neo4j and load PrimeKG (see README), "
                           f"then rerun. Detail: {st.get('error', 'no connection')[:160]}")
    if not st.get("nodes"):
        raise RuntimeError("Neo4j is connected but empty. Load PrimeKG: "
                           ".venv/bin/python scripts/load_primekg.py --download")
    name = kg.find_disease(disease)
    if not name:
        raise RuntimeError(f"No disease in PrimeKG matches '{disease}'. Try a broader name.")
    ctx.log(f"Disease node: {name}")
    rel = kg.relation_summary(name)
    targets = kg.disease_candidate_targets(name, top)
    drugs = kg.disease_drugs(name)

    # Enrich with Open Targets evidence scores (best-effort; needs internet).
    ot_scores, ot_note, ot_written = {}, "", 0
    if enrich:
        from ..sources import opentargets
        ctx.log("Fetching Open Targets association scores")
        try:
            r = opentargets.enrich_disease(disease, size=max(top, 25))
            if r.get("efo_id"):
                ot_scores = {t["symbol"]: t["score"] for t in r.get("targets", [])}
                ot_written = r.get("written", 0)
                ot_note = f"Open Targets {r['efo_id']} · {len(ot_scores)} scored targets · {ot_written} written to graph"
            else:
                ot_note = "Open Targets had no match for this disease"
        except Exception as e:
            ot_note = f"Open Targets unavailable ({str(e)[:80]})"
            ctx.log("  " + ot_note)

    # Union of PrimeKG candidates and Open Targets' top targets, ranked by OT score.
    genes = {t["gene"]: {"gene": t["gene"], "n_drugs": t.get("n_drugs", 0),
                         "n_phenotypes": t.get("n_phenotypes", 0)} for t in targets}
    for sym in ot_scores:
        genes.setdefault(sym, {"gene": sym, "n_drugs": 0, "n_phenotypes": 0})
    for g in genes.values():
        g["ot_score"] = ot_scores.get(g["gene"])
    ranked = sorted(genes.values(),
                    key=lambda g: (g["ot_score"] if g["ot_score"] is not None else -1, g["n_drugs"]),
                    reverse=True)[:top]

    # ChEMBL potent actives for the top few targets (best-effort).
    chembl_counts = {}
    if enrich:
        from ..sources import chembl
        for g in ranked[:5]:
            try:
                res = chembl.target_actives(g["gene"], limit=50)
                chembl_counts[g["gene"]] = res.get("count", 0) if res.get("found") else None
            except Exception:
                chembl_counts[g["gene"]] = None

    src = "PrimeKG" + (" + Open Targets" if ot_scores else "") + (" + ChEMBL" if chembl_counts else "")
    lines = [f"# Target dossier: {name}", "", f"Query: '{disease}' · sources: {src}", ""]
    if ot_note:
        lines += [f"_{ot_note}._", ""]
    lines += ["## What the graph knows", "", "| Relation | Linked type | Count |", "|---|---|---|"]
    lines += [f"| {r['relation']} | {r['target_type']} | {r['count']} |" for r in rel[:15]]
    lines += ["", "## Candidate targets", "",
              "Ranked by **Open Targets association score** (0-1, evidence-weighted) where available, then by "
              "PrimeKG druggability. OT score is the real target-ranking signal; PrimeKG drug count is a crowding "
              "proxy; ChEMBL actives counts potent measured compounds (pChEMBL≥6) for the top targets.", "",
              "| # | Gene | OT score | PrimeKG drugs | Phenotypes | ChEMBL actives |", "|---|---|---|---|---|---|"]
    for i, g in enumerate(ranked, 1):
        ot = f"{g['ot_score']:.3f}" if g["ot_score"] is not None else "—"
        ch = chembl_counts.get(g["gene"])
        chs = "—" if g["gene"] not in chembl_counts else ("n/a" if ch is None else str(ch))
        lines.append(f"| {i} | {g['gene']} | {ot} | {g['n_drugs']} | {g['n_phenotypes']} | {chs} |")
    lines += ["", "## Drugs already linked to the disease", ""]
    lines += [f"- {d['drug']} ({d['relation']})" for d in drugs] or ["None in the graph."]
    lines += ["", "_Open Targets scores are evidence-weighted associations, not proof of a causal, druggable target; "
              "all of this is for review, not efficacy._"]
    return {"disease": name, "targets": [g["gene"] for g in ranked], "linked_drugs": len(drugs),
            "ot_scored": len(ot_scores), "ot_written": ot_written}, lines


# ---------------------------------------------------------------- Chemistry

TRIAGE_TASKS = ["bbbp", "clintox", "bace", "hiv", "esol", "lipo"]


def chem_molecule_triage(ctx, smiles: str | list, label: str = ""):
    items = smiles if isinstance(smiles, list) else [s.strip() for s in str(smiles).replace(",", "\n").splitlines()]
    items = [s for s in items if s][:50]
    if not items:
        raise ValueError("Give at least one SMILES (newline or comma separated).")
    desc = [describe(s) for s in items]
    preds = defaultdict(dict)
    for t in TRIAGE_TASKS:
        ctx.log(f"Predicting {t} (trains the baseline on first use)")
        try:
            for s, p in zip(items, predict(t, items, "rf")):
                if p.get("valid"):
                    preds[s][t] = p["prediction"]
        except Exception as e:
            ctx.log(f"  {t} unavailable: {str(e)[:120]}")
    lines = [f"# Chemistry: molecule triage{' · ' + label if label else ''}", "",
             f"{len(items)} molecule(s). Predictions are random-forest baselines trained on public MoleculeNet sets "
             "with scaffold splits: research signals, not assay results.", "",
             "| # | SMILES | MW | logP | TPSA | QED | Ro5 viol. | BBB p | ClinTox p | BACE p | HIV p | logS | logD | Flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    flagged = 0
    for i, (s, d) in enumerate(zip(items, desc), 1):
        if not d.get("valid"):
            lines.append(f"| {i} | `{s}` | invalid SMILES |" + " |" * 11)
            flagged += 1
            continue
        p = preds[s]
        flags = []
        if d["lipinski_violations"] > 1:
            flags.append("Ro5")
        if d["qed"] < 0.3:
            flags.append("low QED")
        if p.get("clintox", 0) > 0.5:
            flags.append("tox risk")
        if p.get("esol") is not None and p["esol"] < -6:
            flags.append("poor solubility")
        flagged += bool(flags)
        f = lambda k: "—" if p.get(k) is None else f"{p[k]:.2f}"
        lines.append(f"| {i} | `{d['smiles']}` | {d['mol_weight']} | {d['logp']} | {d['tpsa']} | {d['qed']} | "
                     f"{d['lipinski_violations']} | {f('bbbp')} | {f('clintox')} | {f('bace')} | {f('hiv')} | "
                     f"{f('esol')} | {f('lipo')} | {', '.join(flags) or '—'} |")
    return {"molecules": len(items), "flagged": flagged}, lines


def generative_design(ctx, target: str = "", seeds: str = "", organism: str = "",
                      n: int = 12, generations: int = 20, dock: bool = False):
    """De-novo molecule design: evolve novel drug-like candidates for a target
    (our Chemistry42 analog). Seeds from the target's ChEMBL actives (or supplied
    SMILES), scored by QED + synthesizability + Ro5 + similarity."""
    from .. import generate as gen
    seed_list = None
    if seeds:
        seed_list = [s.strip() for s in str(seeds).replace(",", "\n").splitlines() if s.strip()]
    r = gen.generate(target=target, seeds=seed_list, organism=(organism or None),
                     n=min(int(n), 30), generations=min(int(generations), 40),
                     dock=bool(dock), progress=ctx.log)
    if not r.get("ok"):
        raise ValueError(r.get("error", "generation failed"))
    cands = r["candidates"]
    head = f"# Generative design{' · ' + target if target else ''}"
    lines = [head, "",
             f"Evolved **{r['unique_explored']}** unique molecules from **{r['seeds']}** seed(s) over "
             f"{r['generations']} generations; returning the top {len(cands)}. "
             + "; ".join(r.get("seed_meta", [])) + ".", "",
             "Reward-guided graph genetic algorithm over public chemistry (GB-GA style). "
             "Novelty = distance from seeds; QED/Ro5/SA are drug-likeness and synthesizability "
             "proxies, **not** validated activity.", ""]
    hdr = "| # | SMILES | Formula | Score | QED | SA | Sim→seed | Ro5 viol. | MW | logP |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if r.get("docked"):
        hdr = hdr[:-1] + " Dock kcal/mol |"
        sep = sep + "---|"
    lines += [hdr, sep]
    for c in cands:
        row = (f"| {c['rank']} | `{c['smiles']}` | {c['formula']} | {c['score']} | {c['qed']} | "
               f"{c['sa']} | {c['sim_to_seed']} | {c['ro5_violations']} | {c['mol_weight']} | {c['logp']} |")
        if r.get("docked"):
            row += f" {c.get('dock_kcal_mol', '—')} |"
        lines.append(row)
    lines += ["", "**How to read it.** Higher *Score* = better balance of drug-likeness (QED), "
              "synthesizability (SA, 1=easy), on-target similarity (Sim→seed), and Lipinski behaviour. "
              "*Sim→seed* near 1.0 means close to a known active; lower means more novel scaffold. "
              "Next step: run these through **Molecule triage** for ADMET baselines, or dock the top rows."]
    return {"candidates": len(cands), "explored": r["unique_explored"], "target": target or None}, lines


def target_intel(ctx, target: str = ""):
    """Readiness dossier: is this target already solved/drugged, or open space?"""
    from . import target_intel as ti
    if not target:
        raise ValueError("Give a target gene or UniProt accession.")
    return ti.dossier(ctx, target)


def identify_sequence(ctx, sequence: str = ""):
    """Identify a raw protein sequence (BLAST vs UniProtKB), then its dossier if known."""
    from . import target_intel as ti
    if not sequence or len(sequence.strip()) < 10:
        raise ValueError("Paste a protein sequence (≥10 residues).")
    return ti.identify(ctx, sequence)


def structure_watch(ctx, targets: str = ""):
    """Re-check a target watchlist for newly released structures / drugs and diff."""
    from . import target_intel as ti
    return ti.watch(ctx, targets)


def literature_discovery(ctx, disease: str = "", max_papers: int = 20,
                         max_targets: int = 6, max_compounds: int = 15, design: bool = False):
    """Read the literature for a condition and return evidence-backed, drug-like molecules."""
    from . import litdiscovery as ld
    return ld.discover(ctx, disease, max_papers=int(max_papers), max_targets=int(max_targets),
                       max_compounds=int(max_compounds), design=bool(design))


# ---------------------------------------------------------------- Publishing

_GH_CACHE = COMPANY_DIR / "github_cache.json"


def _repo_activity(repos, ctx):
    cache = json.loads(_GH_CACHE.read_text()) if _GH_CACHE.exists() else {}
    for repo in repos:
        hit = cache.get(repo)
        if hit and time.time() - hit.get("checked", 0) < 86400:
            continue
        try:
            r = httpx.get(f"https://api.github.com/repos/{repo}", timeout=10,
                          headers={"Accept": "application/vnd.github+json"})
            if r.status_code == 200:
                cache[repo] = {"pushed_at": r.json().get("pushed_at"), "stars": r.json().get("stargazers_count"),
                               "checked": time.time()}
            elif r.status_code == 404:
                cache[repo] = {"pushed_at": None, "missing": True, "checked": time.time()}
            else:
                ctx.log(f"GitHub API {r.status_code} for {repo}; keeping cached value")
                break
        except Exception as e:
            ctx.log(f"GitHub check skipped: {str(e)[:80]}")
            break
    _GH_CACHE.write_text(json.dumps(cache))
    return cache


def open_index(check_repos=False, ctx=None):
    from ..index_catalog import load_index
    entries = load_index()
    if check_repos:
        cache = _repo_activity(sorted({e["repo"] for e in entries if e.get("repo")}), ctx)
    else:
        cache = json.loads(_GH_CACHE.read_text()) if _GH_CACHE.exists() else {}
    for e in entries:
        e["activity"] = cache.get(e["repo"]) if e.get("repo") else None
    return entries


def _esc(s):
    return (str(s if s is not None else "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def export_public_site(index, board):
    c = charter_mod.load()
    sb = store.scoreboard(TASKS)
    sb_head = "".join(f"<th class=n>{_esc(cat)}</th>" for cat in sb["categories"])
    sb_rows = []
    for r in sb["models"]:
        cats = "".join(f"<td class=n>{r['by_category'].get(cat, '—')}</td>" for cat in sb["categories"])
        sb_rows.append(f"<tr><td class=n>{r['rank']}</td><td class=m>{_esc(r['model'])}</td>"
                       f"<td class=n><b>{r['score']}</b></td>{cats}<td class=n>{r['tasks']}</td></tr>")
    rows = []
    for tid, lst in board.items():
        t = TASKS[tid]
        for i, r in enumerate(lst, 1):
            rows.append(f"<tr><td>{_esc(tid)}</td><td>{_esc(t.phase or '—')}</td><td>{i}</td><td class=m>{_esc(r['model'])}</td>"
                        f"<td class=n>{r['score']:.4f}</td><td>{t.primary_metric} {'↑' if t.higher_is_better else '↓'}</td>"
                        f"<td class=n>{r['n_test']}</td></tr>")
    cards = []
    for e in index:
        act = e.get("activity") or {}
        when = (act.get("pushed_at") or "")[:10]
        repo = (f"<a href='https://github.com/{_esc(e['repo'])}'>{_esc(e['repo'])}</a>"
                + (f" · last push {when}" if when else "")) if e.get("repo") else "no repository"
        title = f"<a href='{_esc(e['url'])}'>{_esc(e['name'])}</a>" if e.get("url") else _esc(e["name"])
        cards.append(f"<article><h3>{title}</h3>"
                     f"<p class=cat>{_esc(e['category'])} · {repo}</p><p>{_esc(e['measures'])}</p>"
                     f"<p class=cav>{_esc(e['caveat'])}</p></article>")
    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>{_esc(c['name'])} · open bench</title>
<style>
:root{{--bg:#f5f4ef;--p:#fff;--ink:#1b1f1c;--mu:#5f6a63;--li:#dedbd1;--ac:#1f6f5c;--wa:#9a5b00;--ws:#fbf1df}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1311;--p:#161b18;--ink:#e7ece8;--mu:#94a199;--li:#2a322d;--ac:#5cc3a4;--wa:#e2a64a;--ws:#2c2416}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1080px;margin:auto;padding:32px 16px}} a{{color:var(--ac)}}
h1{{margin:0;font-size:28px}} .sub{{color:var(--mu)}} h2{{margin-top:36px;font-size:14px;letter-spacing:.08em;text-transform:uppercase;color:var(--mu)}}
.wrap{{overflow-x:auto;background:var(--p);border:1px solid var(--li);border-radius:10px}}
table{{border-collapse:collapse;width:100%;min-width:560px}} td,th{{padding:7px 10px;border-bottom:1px solid var(--li);text-align:left}}
th{{font-size:12px;color:var(--mu);font-weight:500}} .n{{text-align:right;font-family:ui-monospace,monospace}} .m{{font-family:ui-monospace,monospace}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
article{{background:var(--p);border:1px solid var(--li);border-radius:10px;padding:14px 16px}} article h3{{margin:0 0 4px;font-size:16px}}
.cat{{color:var(--mu);font-size:13px;margin:0 0 8px}} .cav{{background:var(--ws);color:var(--wa);font-size:13px;padding:6px 9px;border-radius:6px}}
footer{{color:var(--mu);font-size:13px;margin-top:36px}}
</style></head><body><main>
<h1>{_esc(c['name'])} · open bench</h1><p class=sub>{_esc(c['mission'])}</p>
<h2>ddd score</h2><p class=sub>Per-task scores normalised to 0-100 (direction-aware) and averaged per model across categories. A relative ranking among the models evaluated here, not an absolute quality score.</p>
<div class=wrap><table><tr><th class=n>Rank</th><th>Model</th><th class=n>ddd score</th>{sb_head}<th class=n>tasks</th></tr>
{''.join(sb_rows) or '<tr><td colspan=4>No results yet.</td></tr>'}</table></div>
<h2>Leaderboard by task</h2><div class=wrap><table><tr><th>Task</th><th>Phase</th><th>#</th><th>Model</th><th style="text-align:right">Score</th><th>Metric</th><th style="text-align:right">n test</th></tr>
{''.join(rows) or '<tr><td colspan=7>No results yet.</td></tr>'}</table></div>
<h2>Open benchmark index</h2><p class=sub>Every entry carries the caveat its own description leaves out.</p>
<div class=grid>{''.join(cards)}</div>
<footer>Generated {time.strftime('%Y-%m-%d %H:%M')} by the ddd Publishing department.</footer>
</main></body></html>"""
    (PUBLIC / "index.html").write_text(html)
    return "index.html"


def pub_weekly_digest(ctx, days: int = 7, check_repos: bool = True):
    since = time.time() - days * 86400
    recent = [r for r in db.runs(500) if (r["started"] or 0) >= since and r["playbook"] != "pub_weekly_digest"]
    board = store.leaderboard(TASKS)
    index = open_index(check_repos, ctx)
    page = export_public_site(index, board)
    ctx.log(f"Public site written ({len(index)} index entries, {sum(len(v) for v in board.values())} leaderboard rows)")
    by_dept = defaultdict(list)
    for r in recent:
        by_dept[r["department"]].append(r)
    lines = [f"# Weekly digest · {time.strftime('%Y-%m-%d')}", "",
             f"Window: last {days} days · {len(recent)} runs · "
             f"{sum(r['status'] == 'done' for r in recent)} done, {sum(r['status'] == 'error' for r in recent)} failed", ""]
    for dept, rs in by_dept.items():
        lines += [f"## {dept}", ""]
        for r in rs[:10]:
            s = r["summary"] if isinstance(r["summary"], dict) else {}
            brief = ", ".join(f"{k}: {v}" for k, v in s.items() if not isinstance(v, (list, dict)))[:160]
            lines.append(f"- {time.strftime('%a %H:%M', time.localtime(r['started']))} · {r['playbook']} · "
                         f"{r['status']}{' · ' + brief if brief else ''}{' · `' + r['report'] + '`' if r['report'] else ''}")
        lines.append("")
    lines += ["## Leaderboard leaders", ""]
    for tid, lst in board.items():
        if lst:
            lines.append(f"- {tid}: {lst[0]['model']} {lst[0]['metric']} {lst[0]['score']}")
    year_ago = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 365 * 86400))
    stale = [e["name"] for e in index if ((e.get("activity") or {}).get("pushed_at") or "9999") < year_ago]
    if stale:
        lines += ["", "## Index entries with no push in over a year", ""] + [f"- {n}" for n in stale]
    lines += ["", f"Public site: `/public/{page}`"]
    return {"runs_in_window": len(recent), "index_entries": len(index), "public_page": page}, lines


# ---------------------------------------------------------------- Antiviral repurposing

def _role_weight(role: str) -> float:
    r = (role or "").lower()
    if "protease" in r:
        return 1.0        # validated, druggable enzyme class
    if "receptor" in r:
        return 0.85
    if "cofactor" in r or "polymerase" in r:
        return 0.6
    return 0.55


def antiviral_repurposing(ctx, virus: str, top: int = 8, max_compounds: int = 20):
    """Host-directed antiviral repurposing for a respiratory virus.

    virus -> curated host dependency/entry factors -> ChEMBL inhibitors of those
    host proteins (repurposing candidates) -> triaged and ranked by potency x
    target importance, with a polypharmacology bonus for hitting several host
    factors. Host-directed: harder for the virus to escape than direct-acting.
    Not validated; needs antiviral assays and lung PK to confirm.
    """
    from ..sources import virhost, chembl
    from ..chem import describe
    from ..runner import predict

    seed = virhost.link_virus_hosts(virus)
    if not seed.get("found"):
        raise ValueError(f"Unknown virus '{virus}'. Known: {', '.join(virhost.VIRUSES)}")
    ctx.log(f"{seed['virus']} host factors: {', '.join(h['gene'] for h in seed['hosts'])}"
            + (f" ({seed['written']} written to graph)" if seed.get("written") else ""))

    targets, compounds = [], []
    for h in seed["hosts"]:
        w = _role_weight(h["role"])
        actives, n = [], 0
        ctx.log(f"ChEMBL inhibitors of {h['gene']}")
        try:
            res = chembl.target_actives(h["gene"], limit=50)
            actives = res.get("actives", []) or []
            n = res.get("count", len(actives))
        except Exception as e:
            ctx.log(f"  unavailable: {str(e)[:80]}")
        targets.append({**h, "weight": w, "actives": n})
        for a in actives:
            try:
                pc = float(a.get("pchembl") or 0)
            except (TypeError, ValueError):
                pc = 0.0
            if a.get("smiles"):
                compounds.append({"smiles": a["smiles"], "chembl_id": a.get("molecule_chembl_id"),
                                  "pchembl": pc, "target": h["gene"], "role": h["role"],
                                  "score": round(pc * w, 3)})

    # One row per compound: best single-target score, plus how many host targets it hits.
    agg = {}
    for c in compounds:
        k = c["chembl_id"] or c["smiles"]
        a = agg.get(k)
        if not a:
            agg[k] = dict(c, targets_hit={c["target"]})
        else:
            a["targets_hit"].add(c["target"])
            if c["score"] > a["score"]:
                a.update({"score": c["score"], "pchembl": c["pchembl"], "target": c["target"], "role": c["role"]})
    ranked = sorted(agg.values(), key=lambda c: (len(c["targets_hit"]), c["score"]), reverse=True)[:max_compounds]

    # Safety/property triage on the shortlist (RDKit + baselines; best-effort).
    smis = [c["smiles"] for c in ranked]
    desc = {s: describe(s) for s in smis}
    tox, sol = {}, {}
    if smis:
        try:
            for s, p in zip(smis, predict("clintox", smis, "rf")):
                if p.get("valid"):
                    tox[s] = p["prediction"]
        except Exception as e:
            ctx.log(f"  clintox baseline unavailable: {str(e)[:80]}")
        try:
            for s, p in zip(smis, predict("esol", smis, "rf")):
                if p.get("valid"):
                    sol[s] = p["prediction"]
        except Exception:
            pass

    trank = sorted(targets, key=lambda t: (t["weight"], t["actives"]), reverse=True)
    lines = [f"# Antiviral repurposing: {seed['virus']}", "",
             f"Host-directed strategy — rank human proteins the virus depends on, then the ChEMBL "
             f"inhibitors of those proteins as repurposing candidates. Sources: curated host factors "
             f"+ ChEMBL. Related disease in the graph: {seed['disease']}.", "",
             "## Host-directed targets", "",
             "Ranked by target class (entry proteases/receptors are the most tractable) and how many "
             "measured inhibitors ChEMBL already has.", "",
             "| # | Host factor | Role | Weight | ChEMBL inhibitors |", "|---|---|---|---|---|"]
    for i, t in enumerate(trank, 1):
        lines.append(f"| {i} | {t['gene']} | {t['role']} | {t['weight']:.2f} | {t['actives']} |")

    lines += ["", "## Candidate antiviral compounds", "",
              "Repurposing candidates: measured inhibitors (pChEMBL ≥ 6) of the host targets, scored by "
              "**potency × target weight**, with a bonus for hitting several host factors (polypharmacology). "
              "Properties from RDKit; tox/solubility are random-forest baselines — research signals, not assays.", "",
              "| # | ChEMBL ID | Best host target | pChEMBL | Hits | Score | MW | QED | ClinTox p | Flags |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(ranked, 1):
        d = desc.get(c["smiles"], {})
        flags = []
        if d.get("valid"):
            if d.get("lipinski_violations", 0) > 1:
                flags.append("Ro5")
            if d.get("qed", 1) < 0.3:
                flags.append("low QED")
        tp = tox.get(c["smiles"])
        if tp is not None and tp > 0.5:
            flags.append("tox risk")
        if sol.get(c["smiles"]) is not None and sol[c["smiles"]] < -6:
            flags.append("poor solubility")
        lines.append(
            f"| {i} | {c.get('chembl_id') or '—'} | {c['target']} | {c['pchembl']:.2f} | "
            f"{len(c['targets_hit'])} | {c['score']:.2f} | "
            f"{d.get('mol_weight', '—')} | {d.get('qed', '—')} | "
            f"{'—' if tp is None else f'{tp:.2f}'} | {', '.join(flags) or '—'} |")

    lines += ["", "_Host-directed candidates resist viral mutation better than direct-acting antivirals, "
              "but this is a computational triage only: efficacy needs antiviral assays (viral-titre "
              "reduction, selectivity index in cells) and lung pharmacokinetics. Not clinical advice._"]

    return {"virus": seed["virus"], "host_targets": len(targets),
            "candidates": len(ranked), "compounds_considered": len(agg),
            "top_target": trank[0]["gene"] if trank else None,
            "top_compound": ranked[0].get("chembl_id") if ranked else None}, lines


# ---------------------------------------------------------------- Antibacterial

def antibacterial_repurposing(ctx, bacterium: str, max_compounds: int = 20):
    """Antibacterial repurposing for a priority pathogen. Validated bacterial
    targets (cell wall, DNA gyrase, folate, fatty-acid synthesis) -> their ChEMBL
    inhibitors, triaged and ranked. Targets are bacterial proteins (resolved in
    ChEMBL by protein name), not in PrimeKG. Not validated; resistance-aware."""
    from ..sources import bughost
    seed = bughost.link_bacterium_targets(bacterium)
    if not seed.get("found"):
        raise ValueError(f"Unknown bacterium '{bacterium}'. Known: {', '.join(bughost.BACTERIA)}")
    ctx.log(f"{seed['bacterium']} targets: {', '.join(t['gene'] for t in seed['targets'])}")
    targets = [{"gene": t["gene"], "query": t["query"], "role": t["role"], "weight": 1.0,
                "organism": seed["bacterium"]} for t in seed["targets"]]
    ranked, desc, tox, sol, agg = _compounds_for_targets(ctx, targets, max_compounds)

    lines = [f"# Antibacterial repurposing: {seed['bacterium']}", "",
             "Validated bacterial targets → their ChEMBL inhibitors as repurposing candidates. "
             "Targets are bacterial proteins (resolved in ChEMBL by name), not in PrimeKG.", "",
             "## Bacterial targets", "", "| # | Target | Role | ChEMBL inhibitors |", "|---|---|---|---|"]
    for i, t in enumerate(targets, 1):
        lines.append(f"| {i} | {t['gene']} | {t['role']} | {t.get('actives', 0)} |")
    resolved = sum(1 for t in targets if t.get("actives"))
    if resolved < len(targets):
        lines += ["", f"_{len(targets) - resolved} target(s) returned no ChEMBL hits — the query name may need "
                  "tuning for this organism._"]
    lines += ["", "## Candidate antibacterial compounds", "",
              "Measured inhibitors (pChEMBL ≥ 6) of the targets, scored by potency, bonus for hitting several "
              "targets. Properties from RDKit; tox/solubility are baseline predictions — research signals, not assays.", "",
              "| # | ChEMBL ID | Best target | pChEMBL | Hits | Score | MW | QED | ClinTox p | Flags |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(ranked, 1):
        d = desc.get(c["smiles"], {})
        tp = tox.get(c["smiles"])
        flags = _compound_flags(d, tp, sol.get(c["smiles"]))
        lines.append(f"| {i} | {c.get('chembl_id') or '—'} | {c['target']} | {c['pchembl']:.2f} | "
                     f"{len(c['targets_hit'])} | {c['score']:.2f} | {d.get('mol_weight', '—')} | "
                     f"{d.get('qed', '—')} | {'—' if tp is None else f'{tp:.2f}'} | {', '.join(flags) or '—'} |")
    lines += ["", "_Computational triage only — efficacy needs MIC/time-kill assays and resistance testing. "
              "Not clinical advice._"]
    return {"bacterium": seed["bacterium"], "targets": len(targets), "targets_resolved": resolved,
            "candidates": len(ranked), "top_compound": ranked[0].get("chembl_id") if ranked else None}, lines


# ---------------------------------------------------------------- Repurposing (general)

def _compounds_for_targets(ctx, targets, max_compounds):
    """Shared compound layer: ChEMBL inhibitors of each target -> one row per
    compound (best score, targets hit) -> RDKit + baseline triage. `targets` is
    a list of dicts with 'gene' and 'weight'; each gets 'actives' filled in."""
    from ..sources import chembl
    from ..chem import describe
    from ..runner import predict
    compounds = []
    for t in targets:
        actives, n = [], 0
        ctx.log(f"ChEMBL inhibitors of {t['gene']}")
        try:
            res = chembl.target_actives(t.get("query") or t["gene"], limit=50,
                                        organism=t.get("organism"))
            actives = res.get("actives", []) or []
            n = res.get("count", len(actives))
        except Exception as e:
            ctx.log(f"  unavailable: {str(e)[:80]}")
        t["actives"] = n
        for a in actives:
            try:
                pc = float(a.get("pchembl") or 0)
            except (TypeError, ValueError):
                pc = 0.0
            if a.get("smiles"):
                compounds.append({"smiles": a["smiles"], "chembl_id": a.get("molecule_chembl_id"),
                                  "pchembl": pc, "target": t["gene"], "score": round(pc * float(t["weight"]), 3)})
    agg = {}
    for c in compounds:
        k = c["chembl_id"] or c["smiles"]
        a = agg.get(k)
        if not a:
            agg[k] = dict(c, targets_hit={c["target"]})
        else:
            a["targets_hit"].add(c["target"])
            if c["score"] > a["score"]:
                a.update({"score": c["score"], "pchembl": c["pchembl"], "target": c["target"]})
    ranked = sorted(agg.values(), key=lambda c: (len(c["targets_hit"]), c["score"]), reverse=True)[:max_compounds]
    smis = [c["smiles"] for c in ranked]
    desc = {s: describe(s) for s in smis}
    tox, sol = {}, {}
    if smis:
        try:
            for s, p in zip(smis, predict("clintox", smis, "rf")):
                if p.get("valid"):
                    tox[s] = p["prediction"]
        except Exception as e:
            ctx.log(f"  clintox baseline unavailable: {str(e)[:80]}")
        try:
            for s, p in zip(smis, predict("esol", smis, "rf")):
                if p.get("valid"):
                    sol[s] = p["prediction"]
        except Exception:
            pass
    return ranked, desc, tox, sol, agg


def _compound_flags(d, tp, sv):
    flags = []
    if d.get("valid"):
        if d.get("lipinski_violations", 0) > 1:
            flags.append("Ro5")
        if d.get("qed", 1) < 0.3:
            flags.append("low QED")
    if tp is not None and tp > 0.5:
        flags.append("tox risk")
    if sv is not None and sv < -6:
        flags.append("poor solubility")
    return flags


def disease_repurposing(ctx, disease: str, top: int = 10, max_compounds: int = 20, novel: bool = True):
    """General repurposing for a human disease (autoimmune, cancer, metabolic,
    neuro, ...). Targets come from the knowledge graph + Open Targets (and, with
    novel=True, a few link-prediction targets the graph implies). Then the shared
    compound layer ranks ChEMBL modulators by potency x association. Not validated."""
    st = kg.status()
    if not st.get("connected"):
        raise RuntimeError("Knowledge graph is offline. Start Neo4j and load PrimeKG (see README), then rerun. "
                           f"Detail: {st.get('error', 'no connection')[:160]}")
    if not st.get("nodes"):
        raise RuntimeError("Neo4j is connected but empty. Load PrimeKG: "
                           ".venv/bin/python scripts/load_primekg.py --download")
    name = kg.find_disease(disease)
    if not name:
        raise RuntimeError(f"No disease in PrimeKG matches '{disease}'. Try a broader name.")
    ctx.log(f"Disease node: {name}")

    ot_scores, ot_note = {}, ""
    from ..sources import opentargets
    ctx.log("Fetching Open Targets association scores")
    try:
        r = opentargets.enrich_disease(disease, size=max(top, 25))
        if r.get("efo_id"):
            ot_scores = {t["symbol"]: t["score"] for t in r.get("targets", [])}
            ot_note = f"Open Targets {r['efo_id']} · {len(ot_scores)} scored targets"
        else:
            ot_note = "Open Targets had no match for this disease"
    except Exception as e:
        ot_note = f"Open Targets unavailable ({str(e)[:80]})"
        ctx.log("  " + ot_note)

    genes = {}
    for t in kg.disease_candidate_targets(name, top):
        genes[t["gene"]] = {"gene": t["gene"], "ot_score": ot_scores.get(t["gene"]), "novel": False}
    for sym, sc in ot_scores.items():
        genes.setdefault(sym, {"gene": sym, "novel": False})["ot_score"] = sc
        genes[sym]["novel"] = genes[sym].get("novel", False)

    novel_added = 0
    if novel:
        try:
            from .. import kg_model
            pr = kg_model.predict_links(name, "gene", 12)
            for p in (pr.get("predictions") or [])[:6]:
                if p["name"] not in genes:
                    genes[p["name"]] = {"gene": p["name"], "ot_score": None, "novel": True, "aa": p.get("adamic_adar")}
                    novel_added += 1
        except Exception as e:
            ctx.log(f"  novel-target prediction skipped: {str(e)[:80]}")

    ranked_t = sorted(genes.values(),
                      key=lambda g: (g.get("ot_score") if g.get("ot_score") is not None else -1), reverse=True)[:top]
    targets = [{"gene": g["gene"], "weight": (g.get("ot_score") or 0.5),
                "ot_score": g.get("ot_score"), "novel": g.get("novel", False)} for g in ranked_t]

    ranked, desc, tox, sol, agg = _compounds_for_targets(ctx, targets, max_compounds)

    lines = [f"# Repurposing dossier: {name}", "",
             f"Query: '{disease}' · targets from PrimeKG + Open Targets"
             + (" + link prediction" if novel_added else "") + " · compounds from ChEMBL.", ""]
    if ot_note:
        lines += [f"_{ot_note}._", ""]
    lines += ["## Targets", "",
              "Ranked by Open Targets association (0–1). Rows marked **novel** came from link prediction — "
              "the graph implies them but hasn't been told. Weight (used to score compounds) is the OT score, "
              "or 0.5 for novel targets.", "",
              "| # | Target | OT score | Novel | ChEMBL modulators |", "|---|---|---|---|---|"]
    for i, t in enumerate(targets, 1):
        ot = f"{t['ot_score']:.3f}" if t["ot_score"] is not None else "—"
        lines.append(f"| {i} | {t['gene']} | {ot} | {'yes' if t['novel'] else ''} | {t.get('actives', 0)} |")

    lines += ["", "## Candidate compounds (repurposing)", "",
              "Measured modulators (pChEMBL ≥ 6) of the targets, scored by **potency × association**, bonus for "
              "hitting several targets. Properties from RDKit; tox/solubility are baseline predictions — research "
              "signals, not assays.", "",
              "| # | ChEMBL ID | Best target | pChEMBL | Hits | Score | MW | QED | ClinTox p | Flags |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(ranked, 1):
        d = desc.get(c["smiles"], {})
        tp = tox.get(c["smiles"])
        flags = _compound_flags(d, tp, sol.get(c["smiles"]))
        lines.append(f"| {i} | {c.get('chembl_id') or '—'} | {c['target']} | {c['pchembl']:.2f} | "
                     f"{len(c['targets_hit'])} | {c['score']:.2f} | {d.get('mol_weight', '—')} | "
                     f"{d.get('qed', '—')} | {'—' if tp is None else f'{tp:.2f}'} | {', '.join(flags) or '—'} |")
    lines += ["", "_Association and activity are not efficacy. Candidates need disease-relevant assays and "
              "safety/PK before they mean anything clinically. Not clinical advice._"]

    return {"disease": name, "targets": len(targets), "novel_targets": novel_added,
            "candidates": len(ranked), "top_target": targets[0]["gene"] if targets else None,
            "top_compound": ranked[0].get("chembl_id") if ranked else None}, lines


# ---------------------------------------------------------------- Omics interpretation

def deg_interpret(ctx, genes: str | list, disease: str = ""):
    """Interpret a DEG / gene set against the graph (diseases, pathways, druggable genes)."""
    from . import deg
    return deg.interpret(ctx, genes, disease)


# ---------------------------------------------------------------- Program Operations

def program_plan(ctx, disease: str, modality: str = "small molecule", lead: str = "", top: int = 8):
    """Pipeline-in-a-Person: turn a disease + modality into a runnable program
    plan (staged experiments, CRO tasks, outreach drafts, milestones, gates),
    grounded in ddd's target ranking where the graph is available."""
    from . import program_ops
    targets = []
    try:
        if kg.status().get("connected"):
            name = kg.find_disease(disease) or disease
            targets = [t["gene"] for t in kg.disease_candidate_targets(name, top)]
            if targets:
                ctx.log(f"Targets from graph: {', '.join(targets)}")
    except Exception as e:
        ctx.log(f"graph targets skipped: {str(e)[:80]}")
    if not targets:
        try:
            from ..sources import opentargets
            r = opentargets.enrich_disease(disease, size=top, write=False)
            targets = [t["symbol"] for t in r.get("targets", [])][:top]
            if targets:
                ctx.log(f"Targets from Open Targets: {', '.join(targets)}")
        except Exception as e:
            ctx.log(f"Open Targets skipped: {str(e)[:80]}")
    ctx.log(f"Drafting {modality} program plan for {disease}")
    return program_ops.build_plan(disease, modality, targets, lead)


# ---------------------------------------------------------------- Discovery programs

def discovery_program(ctx, disease: str, molecules: str | list = "", top_targets: int = 15):
    """Cross-department program: Scout -> Chemist -> Evaluator -> Curator (see programs.py)."""
    from . import programs
    mols = molecules if isinstance(molecules, list) else [m.strip() for m in str(molecules).replace(",", "\n").splitlines()]
    mols = [m for m in mols if m][:50]
    res = programs.run_program(disease, mols, top_targets,
                               on_step=lambda st: ctx.log(f"{st['role']}: {st['msg']}"))
    summary = {"disease": disease, "targets": len(res["targets"]),
               "molecules_screened": sum(1 for m in res["molecules"] if m.get("valid")),
               "graph": "offline" if res["kg_note"] else "online"}
    return summary, res["report"].rstrip("\n").splitlines()
