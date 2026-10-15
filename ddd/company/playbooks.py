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
from ..tasks import TASKS, available_tasks
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
    lines = ["# Evaluation: full benchmark sweep", "",
             f"Models: {', '.join(models)} · tasks: {len(tasks)} · scaffold splits (OGB split for ogbg-molhiv).", "",
             "| Task | Model | Metric | Score | Previous best | n test |", "|---|---|---|---|---|---|"]
    regress = 0
    for r in rows:
        flag = ""
        if r["prev_best"] is not None and r["score"] is not None:
            worse = r["score"] < r["prev_best"] if r["hib"] else r["score"] > r["prev_best"]
            if worse and abs(r["score"] - r["prev_best"]) > 0.01:
                flag, regress = " ⚠", regress + 1
        lines.append(f"| {r['task']} | {r['model']} | {r['metric']} | {r['score']}{flag} | "
                     f"{r['prev_best'] if r['prev_best'] is not None else '—'} | {r['n_test']} |")
    if fails:
        lines += ["", "## Failures", ""] + [f"- {f['task']} × {f['model']}: {f['error']}" for f in fails]
    lines += ["", "Baselines are deterministic, so a ⚠ (worse than previous best by more than 0.01) "
                  "means the data or code changed, not luck."]
    return {"runs": len(rows), "failures": len(fails), "regressions": regress}, lines


# ---------------------------------------------------------------- Target biology

def bio_target_dossier(ctx, disease: str, top: int = 12):
    st = kg.status()
    if not st.get("connected"):
        raise RuntimeError("Knowledge graph is offline. Start Neo4j and load PrimeKG (see README), then rerun.")
    name = kg.find_disease(disease)
    if not name:
        raise RuntimeError(f"No disease in PrimeKG matches '{disease}'. Try a broader name.")
    ctx.log(f"Disease node: {name}")
    rel = kg.relation_summary(name)
    targets = kg.disease_candidate_targets(name, top)
    drugs = kg.disease_drugs(name)
    lines = [f"# Target dossier: {name}", "", f"Query: '{disease}' · source: PrimeKG in Neo4j", "",
             "## What the graph knows", "", "| Relation | Linked type | Count |", "|---|---|---|"]
    lines += [f"| {r['relation']} | {r['target_type']} | {r['count']} |" for r in rel[:15]]
    lines += ["", "## Candidate targets", "",
              "Genes linked to the disease, ranked by how many drugs already touch them (a druggability proxy) "
              "and phenotype links. Novelty runs the other way: low drug counts are less validated but less crowded.", "",
              "| # | Gene | Drugs touching it | Phenotype links | Example drugs |", "|---|---|---|---|---|"]
    for i, t in enumerate(targets, 1):
        ex = ", ".join(kg.gene_drugs(t["gene"], 4))
        lines.append(f"| {i} | {t['gene']} | {t['n_drugs']} | {t['n_phenotypes']} | {ex or '—'} |")
    lines += ["", "## Drugs already linked to the disease", ""]
    lines += [f"- {d['drug']} ({d['relation']})" for d in drugs] or ["None in the graph."]
    lines += ["", "_Graph associations are hypotheses for review, not evidence of efficacy._"]
    return {"disease": name, "targets": [t["gene"] for t in targets], "linked_drugs": len(drugs)}, lines


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
    rows = []
    for tid, lst in board.items():
        t = TASKS[tid]
        for i, r in enumerate(lst, 1):
            rows.append(f"<tr><td>{_esc(tid)}</td><td>{i}</td><td class=m>{_esc(r['model'])}</td>"
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
<h2>Leaderboard</h2><div class=wrap><table><tr><th>Task</th><th>#</th><th>Model</th><th style="text-align:right">Score</th><th>Metric</th><th style="text-align:right">n test</th></tr>
{''.join(rows) or '<tr><td colspan=6>No results yet.</td></tr>'}</table></div>
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
