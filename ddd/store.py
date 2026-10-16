import json
import sqlite3
import time

from .config import DB_PATH

SCHEMA = """
create table if not exists results (
  id integer primary key autoincrement,
  ts real not null,
  task text not null,
  model text not null,
  split text not null,
  metric text not null,
  score real,
  n_train integer,
  n_test integer,
  extra text
);
"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    return c


def save(task, model, split, metric, score, n_train, n_test, extra):
    with _conn() as c:
        cur = c.execute(
            "insert into results(ts,task,model,split,metric,score,n_train,n_test,extra) values(?,?,?,?,?,?,?,?,?)",
            (time.time(), task, model, split, metric, score, n_train, n_test, json.dumps(extra)))
        return cur.lastrowid


def all_results(task=None):
    with _conn() as c:
        q = "select * from results"
        args = ()
        if task:
            q += " where task=?"
            args = (task,)
        rows = c.execute(q + " order by ts desc", args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["extra"] = json.loads(d["extra"] or "{}")
        out.append(d)
    return out


def leaderboard(tasks):
    """Best score per (task, model)."""
    rows = all_results()
    best = {}
    for r in rows:
        t = tasks.get(r["task"])
        if t is None or r["score"] is None:
            continue
        key = (r["task"], r["model"])
        hib = t.higher_is_better
        if key not in best or (r["score"] > best[key]["score"] if hib else r["score"] < best[key]["score"]):
            best[key] = r
    board = {}
    for (task_id, _), r in best.items():
        board.setdefault(task_id, []).append(r)
    for task_id, lst in board.items():
        hib = tasks[task_id].higher_is_better
        lst.sort(key=lambda r: r["score"], reverse=hib)
    return board


def _norm(score, lo, hi, hib):
    if hi == lo:
        return 50.0
    frac = (score - lo) / (hi - lo)
    return round(100 * (frac if hib else 1 - frac), 1)


def scoreboard(tasks):
    """DDDBench-style aggregate: each task's models min-max normalised to 0-100
    (direction-aware), then averaged per model over categories and overall.
    Only tasks with >=2 scored models contribute, so a lone score can't set the scale."""
    board = leaderboard(tasks)
    per = {}   # model -> {category -> [norm], "_all" -> [norm]}
    task_meta = {}
    for task_id, lst in board.items():
        scored = [r for r in lst if r["score"] is not None]
        if len(scored) < 2:
            continue
        t = tasks[task_id]
        vals = [r["score"] for r in scored]
        lo, hi = min(vals), max(vals)
        task_meta[task_id] = {"category": t.category or t.domain, "phase": t.phase,
                              "metric": t.primary_metric, "n_models": len(scored)}
        for r in scored:
            n = _norm(r["score"], lo, hi, t.higher_is_better)
            d = per.setdefault(r["model"], {"_all": []})
            d.setdefault(t.category or t.domain, []).append(n)
            d["_all"].append(n)
    cats = sorted({m["category"] for m in task_meta.values()})
    models = []
    for model, d in per.items():
        row = {"model": model, "score": round(sum(d["_all"]) / len(d["_all"]), 1),
               "tasks": len(d["_all"]), "by_category": {}}
        for c in cats:
            if c in d:
                row["by_category"][c] = round(sum(d[c]) / len(d[c]), 1)
        models.append(row)
    models.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(models, 1):
        r["rank"] = i
    return {"categories": cats, "models": models, "tasks_counted": len(task_meta)}
