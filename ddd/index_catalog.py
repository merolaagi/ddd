"""Benchmark Index: an O3DC-style local catalogue of open benchmarks, each with
a known caveat. Seeded from the bench's own tasks plus well-known public
benchmarks. Entries live in data/benchmark_index.json so you can add your own.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_DIR
from .tasks import TASKS

INDEX_PATH = DATA_DIR / "benchmark_index.json"

_SEED_FILE = Path(__file__).resolve().parent / "company" / "index_seed.json"
SEED = json.loads(_SEED_FILE.read_text())


def _from_tasks():
    out = []
    for t in TASKS.values():
        out.append({
            "name": f"ddd/{t.id}", "category": "ddd open bench",
            "maintainer": "ddd bench (this install)", "repo": "merolaagi/ddd",
            "url": "", "measures": f"{t.name} — {t.primary_metric} {'↑' if t.higher_is_better else '↓'}, {t.split} split.",
            "caveat": t.caveat, "local_task": t.id,
        })
    return out


def load_index() -> list[dict]:
    custom = []
    if INDEX_PATH.exists():
        try:
            custom = json.loads(INDEX_PATH.read_text())
        except Exception:
            custom = []
    seen = set()
    merged = []
    for e in _from_tasks() + SEED + custom:
        key = e["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(e)
    return merged


def add_entry(entry: dict) -> dict:
    required = ("name", "category", "measures", "caveat")
    missing = [k for k in required if not entry.get(k)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    if (entry.get("caveat") or "").strip().lower() in ("", "none", "no known limitations"):
        raise ValueError("every entry needs a real known caveat")
    custom = json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else []
    custom.append({k: entry.get(k, "") for k in
                   ("name", "category", "maintainer", "repo", "url", "measures", "caveat")})
    INDEX_PATH.write_text(json.dumps(custom, indent=1))
    return {"added": entry["name"], "custom_entries": len(custom)}


def categories() -> dict:
    out = {}
    for e in load_index():
        out.setdefault(e["category"], 0)
        out[e["category"]] += 1
    return out
