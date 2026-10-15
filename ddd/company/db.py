"""Company state in SQLite: runs (every playbook execution) and schedules."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta

from ..config import DB_PATH

SCHEMA = """
create table if not exists runs (
  id integer primary key autoincrement,
  playbook text not null,
  department text not null,
  params text,
  trigger text,
  status text not null,
  started real,
  finished real,
  summary text,
  report text,
  log text default '',
  pid integer
);
create table if not exists schedules (
  id integer primary key autoincrement,
  playbook text not null,
  params text,
  every text not null,
  enabled integer not null default 1,
  last_run real,
  next_run real
);
"""


def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _row(r):
    if r is None:
        return None
    d = dict(r)
    if d.get("params"):
        d["params"] = json.loads(d["params"])
    if "summary" in d and d["summary"]:
        try:
            d["summary"] = json.loads(d["summary"])
        except Exception:
            pass
    return d


# ---------- runs ----------

def run_create(playbook, department, params, trigger):
    with conn() as c:
        cur = c.execute("insert into runs(playbook,department,params,trigger,status,started,pid) values(?,?,?,?,?,?,?)",
                        (playbook, department, json.dumps(params or {}), trigger, "running", time.time(), os.getpid()))
        return cur.lastrowid


def run_log(run_id, line):
    stamp = time.strftime("%H:%M:%S")
    with conn() as c:
        c.execute("update runs set log = coalesce(log,'') || ? where id=?", (f"[{stamp}] {line}\n", run_id))


def run_finish(run_id, status, summary=None, report=None):
    with conn() as c:
        c.execute("update runs set status=?, finished=?, summary=?, report=? where id=?",
                  (status, time.time(), json.dumps(summary) if summary is not None else None, report, run_id))


def run_get(run_id):
    with conn() as c:
        return _row(c.execute("select * from runs where id=?", (run_id,)).fetchone())


def runs(limit=50, playbook=None):
    with conn() as c:
        if playbook:
            rows = c.execute("select * from runs where playbook=? order by id desc limit ?", (playbook, limit))
        else:
            rows = c.execute("select * from runs order by id desc limit ?", (limit,))
        return [_row(r) for r in rows.fetchall()]


def mark_orphans():
    """Runs left 'running' by a server process that no longer exists can never finish."""
    with conn() as c:
        for r in c.execute("select id, pid from runs where status='running'").fetchall():
            alive = False
            if r["pid"] and r["pid"] != os.getpid():
                try:
                    os.kill(r["pid"], 0)
                    alive = True
                except OSError:
                    alive = False
            if not alive:
                c.execute("update runs set status='interrupted', finished=? where id=?", (time.time(), r["id"]))


# ---------- schedules ----------

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def next_time(every: str, after: float | None = None) -> float:
    """every: 'hourly' | 'daily HH:MM' | 'weekly mon HH:MM' (local time)."""
    now = datetime.fromtimestamp(after or time.time())
    parts = every.lower().split()
    if parts[0] == "hourly":
        return (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).timestamp()
    if parts[0] == "daily" and len(parts) == 2:
        h, m = map(int, parts[1].split(":"))
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= now:
            t += timedelta(days=1)
        return t.timestamp()
    if parts[0] == "weekly" and len(parts) == 3 and parts[1][:3] in DAYS:
        h, m = map(int, parts[2].split(":"))
        target = DAYS.index(parts[1][:3])
        t = now.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(days=(target - now.weekday()) % 7)
        if t <= now:
            t += timedelta(days=7)
        return t.timestamp()
    raise ValueError("every must be 'hourly', 'daily HH:MM' or 'weekly mon HH:MM'")


def schedule_add(playbook, params, every):
    nxt = next_time(every)
    with conn() as c:
        cur = c.execute("insert into schedules(playbook,params,every,enabled,next_run) values(?,?,?,1,?)",
                        (playbook, json.dumps(params or {}), every, nxt))
        return cur.lastrowid


def schedules():
    with conn() as c:
        return [_row(r) for r in c.execute("select * from schedules order by id").fetchall()]


def schedule_set(sid, enabled=None, delete=False):
    with conn() as c:
        if delete:
            c.execute("delete from schedules where id=?", (sid,))
        elif enabled is not None:
            c.execute("update schedules set enabled=? where id=?", (1 if enabled else 0, sid))


def due_schedules():
    now = time.time()
    with conn() as c:
        return [_row(r) for r in c.execute(
            "select * from schedules where enabled=1 and next_run<=?", (now,)).fetchall()]


def schedule_advance(sid, every):
    with conn() as c:
        c.execute("update schedules set last_run=?, next_run=? where id=?", (time.time(), next_time(every), sid))


def seed_default_schedules():
    """First start only: give the company a standing rhythm."""
    if schedules():
        return
    schedule_add("intel_watchlist_sweep", {"run_rea": True}, "weekly mon 06:00")
    schedule_add("bench_full_sweep", {}, "weekly sun 03:00")
    schedule_add("pub_weekly_digest", {}, "weekly mon 08:00")
