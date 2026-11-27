"""Swarm supervisor — a self-running research company.

Worker agents (the department playbooks) grind on the focus diseases on their
own, and — when the queue has room — the supervisor goes *out into the world*:
a world scout reads recent literature, text-mines fresh diseases and targets out
of the papers, and feeds them into the queue, so the swarm keeps discovering new
things to work on instead of recycling a fixed list. A supervisor loop (`tick`)
keeps them busy, harvests their results, judges each against a tunable bar,
escalates what's worthy to the founder (an HQ inbox + email), and when a worker
comes up short it spawns a new specialist agent on the fly to attack the gap.
Driven by a schedule, so it is always working while the server is up (and
launchd keeps the server up).

Design: dispatch-and-harvest, non-blocking. Each tick harvests the runs that
finished since last time (score → escalate → spawn), tops the queue back up,
and dispatches a small batch of new runs. Workers run as ordinary tracked runs
(visible in HQ, each with its own report), bounded by the engine's 2-slot
semaphore, so the swarm never stampedes the one Mac.

Everything here is prioritisation over public data + reward-guided design — a
place to look, not a validated lead.
"""
from __future__ import annotations

import json
import os
import smtplib
import time
from email.message import EmailMessage

from . import charter as charter_mod
from . import db
from .paths import COMPANY_DIR

_STATE = COMPANY_DIR / "swarm.json"

_DEFAULT_CFG = {
    "enabled": True,
    "batch_per_tick": 1,        # new runs dispatched per tick
    "target_inflight": 3,       # keep this many tasks pending+running
    "max_dynamic_agents": 12,
    "escalate_leads": True,
    "escalate_findings": True,
    "email_alerts": True,
    "baseline_worker": "discovery_mission",
    "explore_world": True,      # go beyond focus_diseases: mine fresh topics from papers
    "world_batch": 3,           # fresh diseases to pick up per world scout
    "world_reserve": 1,         # in-flight slots always kept for world exploration
    "max_world_topics": 300,    # cap on remembered discovered topics
}


# ---------------------------------------------------------------- state
def _blank():
    return {"queue": [], "dynamic_agents": [], "escalations": [], "discovered": [],
            "world_cursor": 0,
            "stats": {"ticks": 0, "dispatched": 0, "harvested": 0, "leads": 0,
                      "findings": 0, "agents_spawned": 0, "errors": 0, "last_tick": None,
                      "world_scouts": 0, "world_topics": 0},
            "seq": 0}


def _load():
    if _STATE.exists():
        try:
            s = json.loads(_STATE.read_text())
            for k, v in _blank().items():
                s.setdefault(k, v)
            return s
        except Exception:
            pass
    return _blank()


def _save(s):
    # keep history bounded
    done = [t for t in s["queue"] if t["status"] in ("done", "error")]
    live = [t for t in s["queue"] if t["status"] in ("pending", "running")]
    s["queue"] = live + done[-60:]
    s["escalations"] = s["escalations"][-120:]
    # keep unworked topics, trim the worked tail
    unworked = [d for d in s.get("discovered", []) if not d.get("worked")]
    worked = [d for d in s.get("discovered", []) if d.get("worked")]
    s["discovered"] = (unworked + worked[-240:])[-300:]
    try:
        _STATE.write_text(json.dumps(s, indent=1))
    except Exception:
        pass


def _cfg():
    c = dict(_DEFAULT_CFG)
    c.update((charter_mod.load().get("swarm") or {}))
    return c


def _nid(s, prefix):
    s["seq"] += 1
    return f"{prefix}{s['seq']}"


# ---------------------------------------------------------------- scoring
def _score(playbook: str, summary: dict) -> list:
    """What, if anything, in this result is worth the founder's attention."""
    s = summary or {}
    out = []
    if playbook == "discovery_mission":
        pv = s.get("pathway_validated") or []
        pv_note = (f" Pathway rule engine validated {', '.join(pv)} (known modulators on record)." if pv else "")
        if s.get("useful") and s.get("lead_target"):
            out.append({"tier": "lead",
                        "title": f"Lead: {s['lead_target']} for {s.get('disease','?')}",
                        "detail": f"The mission cleared the bar on {s['lead_target']} "
                                  f"(tried {s.get('targets_tried','?')} target(s))." + pv_note})
        elif pv:
            out.append({"tier": "finding",
                        "title": f"Known-modulator anchor for {s.get('disease','?')}: {', '.join(pv[:4])}",
                        "detail": f"No clean de-novo lead yet, but the pathway rule engine flags "
                                  f"{len(pv)} enzyme(s) with approved modulators — a repurposing anchor."})
        elif s.get("targets_tried"):
            out.append({"tier": "finding",
                        "title": f"No clean lead yet for {s.get('disease','?')}",
                        "detail": f"Tried {s['targets_tried']} target(s); none cleared the bar."})
    elif playbook == "literature_discovery":
        tg = s.get("targets") or []
        if s.get("druglike", 0) > 0 or len(tg) >= 3:
            out.append({"tier": "finding",
                        "title": f"Targets for {s.get('disease','?')}: {', '.join(tg[:4])}",
                        "detail": f"{s.get('papers_read','?')} papers read; "
                                  f"{s.get('druglike',0)} clean drug-like candidate(s)."})
    elif playbook == "target_intel":
        cr = s.get("crowding")
        if cr in ("Open space", "Lightly explored"):
            out.append({"tier": "finding",
                        "title": f"{s.get('target','?')}: {cr}",
                        "detail": f"{s.get('pdb',0)} PDB structure(s), {s.get('drugs',0)} known drug(s) — "
                                  "room to work."})
    elif playbook in ("disease_repurposing", "antiviral_repurposing", "antibacterial_repurposing"):
        if s.get("candidates"):
            pe = s.get("pathway_enzymes") or []
            pe_note = (f" Pathway rule engine anchors: {', '.join(pe[:4])}." if pe else "")
            out.append({"tier": "finding",
                        "title": f"Repurposing hits for {s.get('disease') or s.get('virus') or s.get('bacterium','?')}",
                        "detail": f"{s.get('candidates')} ranked candidate(s); top target {s.get('top_target','?')}."
                                  + pe_note})
    elif playbook == "structure_watch":
        if s.get("changes"):
            out.append({"tier": "finding",
                        "title": f"{s['changes']} target(s) changed",
                        "detail": "New experimental structures or drugs released on the watchlist."})
    return out


# ---------------------------------------------------------------- deficiency → new agent
def _deficiency(playbook: str, summary: dict, existing_foci: set) -> dict | None:
    """When a worker comes up short, define a specialist agent to attack the gap.
    Returns an agent spec, or None. Deduped by (playbook, focus) upstream."""
    s = summary or {}
    if playbook == "discovery_mission" and not s.get("useful"):
        disease = s.get("disease")
        pv = s.get("pathway_validated") or []
        if disease:
            # Strongest signal: the pathway rule engine already flags enzymes with
            # approved modulators for this disease — spawn a pathway-repurposer to
            # mine those known mechanisms first.
            if pv and ("disease_repurposing", disease) not in existing_foci:
                return {"name": f"pathway-repurposer-{_slug(disease)}", "focus": disease,
                        "playbook": "disease_repurposing", "params": {"disease": disease},
                        "reason": f"pathway rule engine flags known modulators for {disease} "
                                  f"({', '.join(pv[:3])}); mine those mechanisms for repurposing."}
            # the mission found targets but no clean lead — widen the search with a
            # deeper literature read + repurposing net on the same disease.
            if ("disease_repurposing", disease) not in existing_foci:
                return {"name": f"repurposer-{_slug(disease)}", "focus": disease,
                        "playbook": "disease_repurposing", "params": {"disease": disease},
                        "reason": f"discovery mission found no clean lead for {disease}; "
                                  "widen to all known modulators (repurposing)."}
            if ("literature_discovery", disease) not in existing_foci:
                return {"name": f"deep-reader-{_slug(disease)}", "focus": disease,
                        "playbook": "literature_discovery",
                        "params": {"disease": disease, "max_papers": 25, "max_targets": 8, "design": True},
                        "reason": f"deepen the literature read for {disease} (more papers, design analogs)."}
    if playbook == "literature_discovery" and not (s.get("targets")):
        disease = s.get("disease")
        if disease and ("bio_target_dossier", disease) not in existing_foci:
            return {"name": f"graph-scout-{_slug(disease)}", "focus": disease,
                    "playbook": "bio_target_dossier", "params": {"disease": disease},
                    "reason": f"literature yielded no targets for {disease}; fall back to the graph."}
    return None


def _slug(x):
    return "".join(ch for ch in (x or "").lower() if ch.isalnum() or ch == " ").strip().replace(" ", "-")[:28]


# ---------------------------------------------------------------- email
def _email(esc: dict, cfg: dict, ctx=None):
    if not cfg.get("email_alerts"):
        return
    host = os.environ.get("DDD_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("DDD_SMTP_PORT", "587"))
    user = os.environ.get("DDD_SMTP_USER", "").strip()
    pw = os.environ.get("DDD_SMTP_PASS", "").strip()
    to = os.environ.get("DDD_ALERT_TO", user).strip()
    if not (user and pw and to):
        if ctx:
            ctx.log("email: not configured (set DDD_SMTP_USER/PASS/ALERT_TO in .env) — inbox only")
        return
    try:
        m = EmailMessage()
        m["Subject"] = f"[ddd {esc['tier']}] {esc['title']}"
        m["From"] = user
        m["To"] = to
        m.set_content(f"{esc['title']}\n\n{esc['detail']}\n\n— ddd swarm supervisor\n"
                      f"(prioritisation over public data, not a validated lead)")
        with smtplib.SMTP(host, port, timeout=20) as srv:
            srv.starttls()
            srv.login(user, pw)
            srv.send_message(m)
        if ctx:
            ctx.log(f"email sent to {to}")
    except Exception as e:
        if ctx:
            ctx.log(f"email failed: {str(e)[:80]}")


def _escalate(s, esc, task, report, cfg, ctx):
    tier = esc["tier"]
    if tier == "lead" and not cfg.get("escalate_leads"):
        return
    if tier == "finding" and not cfg.get("escalate_findings"):
        return
    rec = {"id": _nid(s, "e"), "tier": tier, "title": esc["title"], "detail": esc["detail"],
           "playbook": task["playbook"], "report": report, "created": time.time(), "read": False}
    s["escalations"].append(rec)
    s["stats"]["leads" if tier == "lead" else "findings"] += 1
    if ctx:
        ctx.log(f"★ escalate [{tier}] {esc['title']}")
    _email(rec, cfg, ctx)


# ---------------------------------------------------------------- queue helpers
def _active_foci(s):
    return {(t["playbook"], (t["params"] or {}).get("disease") or (t["params"] or {}).get("target"))
            for t in s["queue"] if t["status"] in ("pending", "running")}


def _enqueue(s, playbook, params, origin, label):
    s["queue"].append({"id": _nid(s, "t"), "playbook": playbook, "params": params,
                       "origin": origin, "label": label, "status": "pending",
                       "run_id": None, "created": time.time(), "finished": None})


def _spawn(s, spec, cfg, ctx):
    if len([a for a in s["dynamic_agents"]]) >= cfg["max_dynamic_agents"]:
        return False
    if any(a["playbook"] == spec["playbook"] and a["focus"] == spec["focus"] for a in s["dynamic_agents"]):
        return False
    spec = dict(spec, created=time.time())
    s["dynamic_agents"].append(spec)
    s["stats"]["agents_spawned"] += 1
    _enqueue(s, spec["playbook"], spec["params"], spec["name"], f"{spec['name']}: {spec['reason']}")
    if ctx:
        ctx.log(f"✦ spawned agent {spec['name']} — {spec['reason']}")
    return True


# ---------------------------------------------------------------- world scout
def _scout(s, cfg, ctx):
    """Go out into recent literature and pick up fresh diseases + targets to work on.
    Appends new topics to s['discovered']; returns the list of (kind, name) added."""
    from . import litdiscovery as ld
    cur = int(s.get("world_cursor", 0))
    known = {d["name"].lower() for d in s.get("discovered", [])}
    known |= {str(x).lower() for x in charter_mod.load().get("focus_diseases", [])}
    res = ld.scout_world(cursor=cur, want=int(cfg.get("world_batch", 3)), exclude=known)
    s["world_cursor"] = cur + 1
    s["stats"]["world_scouts"] = s["stats"].get("world_scouts", 0) + 1
    added = []
    for nm in res.get("diseases", []):
        s["discovered"].append({"name": nm, "kind": "disease", "worked": False,
                                "source": res.get("query"), "added": time.time()})
        added.append(("disease", nm))
    for nm in (res.get("targets") or [])[:2]:
        s["discovered"].append({"name": nm, "kind": "target", "worked": False,
                                "source": res.get("query"), "added": time.time()})
        added.append(("target", nm))
    s["stats"]["world_topics"] = s["stats"].get("world_topics", 0) + len(added)
    if ctx:
        if added:
            ctx.log("🌐 world scout — picked up " + ", ".join(f"{k}:{n}" for k, n in added)
                    + f" (from {res.get('papers', 0)} recent papers)")
        else:
            ctx.log(f"🌐 world scout — no new topics this pass ({res.get('papers', 0)} papers read)")
    return added


# ---------------------------------------------------------------- the heartbeat
def tick(ctx, max_dispatch: int | None = None):
    cfg = _cfg()
    s = _load()
    if not cfg.get("enabled", True):
        ctx.log("swarm disabled (charter.swarm.enabled=false)")
        _save(s)
        return {"enabled": False}, ["# Swarm tick", "", "Swarm is disabled."]
    batch = int(max_dispatch if max_dispatch is not None else cfg["batch_per_tick"])
    harvested = dispatched = escalated = spawned = 0

    # 1) HARVEST finished runs
    for t in s["queue"]:
        if t["status"] != "running" or not t.get("run_id"):
            continue
        row = db.run_get(t["run_id"])
        if not row or row.get("status") == "running":
            continue
        t["status"] = row.get("status", "error")
        t["finished"] = time.time()
        harvested += 1
        if t["status"] == "error":
            s["stats"]["errors"] += 1
            ctx.log(f"✗ {t['playbook']} ({t.get('label','')[:40]}) errored")
            continue
        try:
            summary = json.loads(row.get("summary") or "{}")
        except Exception:
            summary = {}
        before = len(s["escalations"])
        for esc in _score(t["playbook"], summary):
            _escalate(s, esc, t, row.get("report"), cfg, ctx)
        escalated += len(s["escalations"]) - before
        spec = _deficiency(t["playbook"], summary,
                           {(a["playbook"], a["focus"]) for a in s["dynamic_agents"]})
        if spec and _spawn(s, spec, cfg, ctx):
            spawned += 1
    s["stats"]["harvested"] += harvested

    # 2) SEED work so the swarm never idles
    foci = _active_foci(s)
    inflight = len([t for t in s["queue"] if t["status"] in ("pending", "running")])
    bw = cfg["baseline_worker"]
    target = cfg["target_inflight"]
    explore = cfg.get("explore_world", True)
    # reserve a slot (or more) for the world so a full focus list can't starve the
    # scout; never take the last focus slot away when there are focus diseases.
    reserve = min(int(cfg.get("world_reserve", 1)), max(0, target - 1)) if explore else 0
    focus_cap = target - reserve

    # 2a) keep the charter's focus diseases covered (up to the focus budget)
    if inflight < focus_cap:
        for disease in charter_mod.load().get("focus_diseases", []):
            if inflight >= focus_cap:
                break
            if (bw, disease) in foci:
                continue
            _enqueue(s, bw, {"disease": disease}, "baseline", f"scan {disease}")
            foci.add((bw, disease))
            inflight += 1

    # 2b) go beyond the fixed list — work the topics scouted from the world's papers
    if explore and inflight < target:
        pending_topics = [d for d in s["discovered"] if not d.get("worked")]
        if not pending_topics:
            try:
                _scout(s, cfg, ctx)
                pending_topics = [d for d in s["discovered"] if not d.get("worked")]
            except Exception as e:
                ctx.log(f"🌐 world scout skipped: {str(e)[:80]}")
        for d in pending_topics:
            if inflight >= target:
                break
            if d["kind"] == "target":
                pb, params = "target_intel", {"target": d["name"]}
            else:
                pb, params = bw, {"disease": d["name"]}
            d["worked"] = True
            if (pb, d["name"]) in foci:
                continue
            _enqueue(s, pb, params, "world", f"world · {d['kind']}: {d['name']}")
            foci.add((pb, d["name"]))
            inflight += 1

    # 3) DISPATCH a small batch of pending tasks as tracked runs
    from . import engine
    for t in s["queue"]:
        if dispatched >= batch:
            break
        if t["status"] != "pending":
            continue
        try:
            rid = engine.start(t["playbook"], t["params"], trigger="swarm")
            t["status"] = "running"
            t["run_id"] = rid
            dispatched += 1
            s["stats"]["dispatched"] += 1
            ctx.log(f"▶ dispatch {t['playbook']} — {t.get('label','')[:50]} (run r{rid})")
        except Exception as e:
            t["status"] = "error"
            s["stats"]["errors"] += 1
            ctx.log(f"✗ could not dispatch {t['playbook']}: {str(e)[:70]}")

    s["stats"]["ticks"] += 1
    s["stats"]["last_tick"] = time.time()
    _save(s)

    pend = len([t for t in s["queue"] if t["status"] == "pending"])
    run = len([t for t in s["queue"] if t["status"] == "running"])
    unworked_topics = len([d for d in s["discovered"] if not d.get("worked")])
    lines = ["# Swarm tick", "",
             f"Harvested **{harvested}**, dispatched **{dispatched}**, escalated **{escalated}**, "
             f"spawned **{spawned}** new agent(s). In flight: {run} running, {pend} pending, "
             f"{len(s['dynamic_agents'])} dynamic agent(s). World topics known: "
             f"{len(s['discovered'])} ({unworked_topics} not yet worked).", ""]
    world_now = [t for t in s["queue"] if t.get("origin") == "world" and t["status"] in ("running", "pending")]
    if world_now:
        lines += ["## From the world (literature-scouted)", ""]
        for t in world_now[:8]:
            lines.append(f"- {t.get('label','')}")
        lines.append("")
    unread = [e for e in s["escalations"] if not e["read"]]
    if unread:
        lines += ["## Unread escalations", ""]
        for e in unread[-10:][::-1]:
            lines.append(f"- **[{e['tier']}]** {e['title']} — {e['detail']}")
    summary = {"harvested": harvested, "dispatched": dispatched, "escalated": escalated,
               "spawned": spawned, "running": run, "pending": pend,
               "dynamic_agents": len(s["dynamic_agents"]), "unread": len(unread),
               "world_topics": len(s["discovered"]), "world_unworked": unworked_topics}
    return summary, lines


# ---------------------------------------------------------------- read API (UI/MCP)
def status():
    s = _load()
    cfg = _cfg()
    q = s["queue"]
    return {
        "enabled": cfg.get("enabled", True),
        "config": cfg,
        "inflight": {"running": len([t for t in q if t["status"] == "running"]),
                     "pending": len([t for t in q if t["status"] == "pending"]),
                     "done": len([t for t in q if t["status"] == "done"]),
                     "error": len([t for t in q if t["status"] == "error"])},
        "running": [{"playbook": t["playbook"], "label": t.get("label"), "run_id": t.get("run_id")}
                    for t in q if t["status"] == "running"],
        "pending": [{"playbook": t["playbook"], "label": t.get("label")}
                    for t in q if t["status"] == "pending"][:20],
        "dynamic_agents": s["dynamic_agents"],
        "stats": s["stats"],
        "escalations": sorted(s["escalations"], key=lambda e: (e["read"], -e["created"]))[:40],
        "unread": len([e for e in s["escalations"] if not e["read"]]),
        "discovered": sorted(s.get("discovered", []), key=lambda d: -(d.get("added") or 0))[:40],
        "world": {"known": len(s.get("discovered", [])),
                  "unworked": len([d for d in s.get("discovered", []) if not d.get("worked")]),
                  "scouts": s["stats"].get("world_scouts", 0)},
    }


def _last_log(run_id):
    if not run_id:
        return None
    try:
        row = db.run_get(run_id)
        log = (row or {}).get("log") or ""
        lines = [l for l in log.splitlines() if l.strip()]
        if lines:
            # strip the "[HH:MM:SS] " stamp
            l = lines[-1]
            return l.split("] ", 1)[1] if l.startswith("[") and "] " in l else l
    except Exception:
        return None
    return None


def live():
    """Graph + progress + activity feed for the real-time swarm overlay."""
    s = _load()
    cfg = _cfg()
    nodes = [{"id": "sup", "kind": "supervisor", "label": "Supervisor",
              "sub": "Overseer", "progress": None}]
    edges = []
    agent_ids = {}
    for a in s["dynamic_agents"]:
        nid = "agent:" + a["name"]
        agent_ids[a["name"]] = nid
        nodes.append({"id": nid, "kind": "agent", "label": a["name"],
                      "sub": a["playbook"], "reason": a.get("reason"), "progress": None})
        edges.append({"from": "sup", "to": nid, "kind": "spawn"})
    for t in s["queue"]:
        if t["status"] not in ("running", "pending"):
            continue
        nid = t["id"]
        nodes.append({"id": nid, "kind": t["status"], "playbook": t["playbook"],
                      "label": t["playbook"].replace("_", " "),
                      "sub": t.get("label"),
                      "progress": _last_log(t.get("run_id")) if t["status"] == "running" else None,
                      "run_id": t.get("run_id")})
        src = agent_ids.get(t.get("origin"), "sup")
        edges.append({"from": src, "to": nid, "kind": "dispatch"})

    # activity feed — chronological events the swarm produced
    acts = []
    for t in s["queue"]:
        if t.get("created"):
            acts.append({"t": t["created"], "icon": "▶",
                         "text": f"dispatch {t['playbook']} · {t.get('label','')[:48]}"})
        if t.get("finished") and t["status"] == "done":
            acts.append({"t": t["finished"], "icon": "✓", "text": f"done {t['playbook']}"})
        elif t.get("finished") and t["status"] == "error":
            acts.append({"t": t["finished"], "icon": "✗", "text": f"error {t['playbook']}"})
    for a in s["dynamic_agents"]:
        acts.append({"t": a.get("created", 0), "icon": "✦",
                     "text": f"spawned {a['name']} — {a.get('reason','')[:60]}"})
    for d in s.get("discovered", []):
        acts.append({"t": d.get("added", 0), "icon": "🌐",
                     "text": f"picked up {d['kind']} from the world: {d['name']}"})
    for e in s["escalations"]:
        acts.append({"t": e["created"], "icon": "★" if e["tier"] == "lead" else "•",
                     "text": f"[{e['tier']}] {e['title']}"})
    acts.sort(key=lambda x: x["t"] or 0, reverse=True)

    return {
        "enabled": cfg.get("enabled", True),
        "stats": s["stats"],
        "counts": {"running": len([n for n in nodes if n["kind"] == "running"]),
                   "pending": len([n for n in nodes if n["kind"] == "pending"]),
                   "agents": len(s["dynamic_agents"])},
        "nodes": nodes,
        "edges": edges,
        "activity": acts[:40],
        "escalations": sorted(s["escalations"], key=lambda e: (e["read"], -e["created"]))[:12],
        "unread": len([e for e in s["escalations"] if not e["read"]]),
    }


def escalations(unread_only=False, limit=50):
    s = _load()
    es = [e for e in s["escalations"] if (not unread_only or not e["read"])]
    return sorted(es, key=lambda e: -e["created"])[:limit]


def mark_read(ids=None):
    s = _load()
    n = 0
    for e in s["escalations"]:
        if (ids is None or e["id"] in ids) and not e["read"]:
            e["read"] = True
            n += 1
    _save(s)
    return n


def configure(**kw):
    cur = (charter_mod.load().get("swarm") or {})
    cur = dict(_DEFAULT_CFG, **cur)
    for k, v in kw.items():
        if k in _DEFAULT_CFG and v is not None:
            cur[k] = v
    charter_mod.save({"swarm": cur})
    return cur
