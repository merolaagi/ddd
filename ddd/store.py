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
