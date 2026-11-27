"""Mission Control — the concierge / overall helper for ddd.

The Analyst *does* things (runs playbooks). Mission Control *explains and guides*:
what the app can do, what the agents and the world scout are doing right now, what
turned up, and what you should do next. Rules-based and instant (no model API); it
reads the same live state the rest of the company writes. Think "Ask vix / mission
control", but for the drug-discovery bench.
"""
from __future__ import annotations

import re
import time

# plain-language capability map: (name, what it does, the phrase to type)
CAPABILITIES = [
    ("Discovery mission", "agents work in tandem (read literature → check target readiness → design molecules) and loop until a candidate clears the bar", "Find me a drug for pancreatic cancer"),
    ("Repurposing", "find existing drugs/compounds to repurpose for a disease, virus or bacterium", "Repurposing candidates for rheumatoid arthritis"),
    ("Target intelligence", "is a target open space or already crowded? structure + known drugs + verdict", "Is TYK2 worth pursuing?"),
    ("Generative design", "invent novel, drug-like molecules for a target (graph genetic algorithm)", "Design novel molecules for EGFR"),
    ("Program plan", "stage a small-molecule or antibody program end to end, with CROs and gates", "Program plan for LAMA2 muscular dystrophy"),
    ("Literature discovery", "read recent papers and surface the targets and chemicals they point to", "Read the research on ALS"),
    ("Novel target prediction", "ask the knowledge graph for targets nobody has drugged yet", "Novel targets for lung cancer"),
    ("Hypothesis check", "is A linked to B? shared neighbours, shortest path, link-prediction score", "Is TERT linked to idiopathic pulmonary fibrosis?"),
    ("Protein structure", "fetch the AlphaFold model + error map for a target", "Show the structure of KRAS"),
    ("Virtual cell", "build the living cell for a disease — targets in their real compartments", "Build a cell for psoriasis"),
    ("Pathway ingredients", "the druggable enzymes and their known modulators woven in from the pathway rule engine", "What enzymes are druggable?"),
    ("Benchmark", "score every model on every task and rank them", "Run the benchmark"),
]

# —— intent detection (meta/guide only; science actions stay with the Analyst) ——
_HELP = re.compile(r"\bwhat can (you|this|ddd|the app|it)\b|\bwhat (is|are) (ddd|this app|this)\b|"
                   r"\bwhat do you do\b|\bwho are you\b|\bexplain (ddd|the app|this|yourself)\b|"
                   r"\bwhat('?s| is) possible\b|\bcapabilit", re.I)
_HOWTO = re.compile(r"\bhow (do i|to) (use|start|get started|navigate|run|work|operate|drive)\b|"
                    r"\bhow does (this|ddd|the app|it) work\b|\bget(ting)? started\b|"
                    r"\bwhere (do i|to) (start|begin)\b", re.I)
_STATUS = re.compile(r"\bstatus\b|\bmission control\b|\bwhat('?s| is| are)?\s*(happening|going on|running|up)\b|"
                     r"\bwhat (are|is) (the )?(agents?|swarm|supervisor|company) (doing|up to|working on)\b|"
                     r"\boverview\b|\bdashboard\b|\bwhat have the agents\b", re.I)
_WORLD = re.compile(r"\bworld scout\b|\bout (in|into) the world\b|\bwhat('?s| have you)?\s*(discover\w*|found|picked up)\b|"
                    r"\bfrom (the )?(world|literature|papers?)\b|\bnew diseases?\b", re.I)
_INBOX = re.compile(r"\bescalations?\b|\bmy (leads?|inbox|findings?)\b|\bany (leads?|news|findings?)\b|\bwhat('?s)? in my inbox\b", re.I)
_NEXT = re.compile(r"\bwhat should i do\b|\bwhat('?s| is)?\s*next\b|\bguide me\b|\brecommend\b|\bwhat now\b|\bwhere to focus\b", re.I)


def matches(ql: str) -> bool:
    return bool(_HELP.search(ql) or _HOWTO.search(ql) or _STATUS.search(ql)
                or _WORLD.search(ql) or _INBOX.search(ql) or _NEXT.search(ql))


def _ago(ts) -> str:
    if not ts:
        return "—"
    d = max(0, int(time.time() - ts))
    if d < 60:
        return f"{d}s ago"
    if d < 3600:
        return f"{d // 60}m ago"
    if d < 86400:
        return f"{d // 3600}h ago"
    return f"{d // 86400}d ago"


def _swarm():
    try:
        from . import swarm
        return swarm.status()
    except Exception:
        return None


def _recent_runs(limit=6):
    try:
        from .db import runs
        return runs(limit)
    except Exception:
        return []


def _act(label, ask):
    return {"label": label, "ask": ask}


# —————————————————————————————————————————————— answers
def _help_answer():
    rows = ["| What | It does | Just ask |", "|---|---|---|"]
    for name, blurb, ex in CAPABILITIES:
        rows.append(f"| **{name}** | {blurb} | *{ex}* |")
    say = ("## What ddd can do\n\n"
           "I'm your mission control — ask me in plain language and I'll run the right tool, "
           "or tell you what the agents are up to. Here's the menu:\n\n"
           + "\n".join(rows) +
           "\n\nUnderneath, a **swarm** works on its own around the clock: it covers the charter's "
           "focus diseases, and when it has room it sends a **world scout** into recent literature to "
           "pick up fresh diseases and targets to chase. It escalates anything promising to your inbox.\n\n"
           "Ask me *\"what's happening right now?\"* to see it live, or *\"what should I do next?\"* and "
           "I'll point you at the highest-value move.")
    return {"kind": "guide", "say": say,
            "actions": [_act("What's happening right now?", "What's happening right now?"),
                        _act("What has the world scout found?", "What has the world scout found?"),
                        _act("What should I do next?", "What should I do next?"),
                        _act("Find me a drug for pancreatic cancer", "Find me a drug for pancreatic cancer")]}


def _howto_answer(q):
    say = ("## Getting around\n\n"
           "Three places:\n\n"
           "- **This chat (Mission Control)** — ask for anything in plain language. I run the tool and show the result.\n"
           "- **The [Lab](/lab)** — the bench: tasks, models, leaderboard, the knowledge graph, the virtual cell and protein structures.\n"
           "- **HQ (/)** — the company view: departments, runs, reports, and the live swarm overlay.\n\n"
           "You don't have to start anything by hand — the swarm is already running. Good first moves: "
           "*\"what's happening right now?\"*, *\"find me a drug for &lt;disease&gt;\"*, or *\"is &lt;gene&gt; worth pursuing?\"*.")
    return {"kind": "guide", "say": say,
            "actions": [_act("What can this app do?", "What can this app do?"),
                        _act("What's happening right now?", "What's happening right now?"),
                        _act("What should I do next?", "What should I do next?")]}


def _status_answer():
    s = _swarm()
    runs = _recent_runs(6)
    lines = ["## Mission control — live", ""]
    if s:
        world = s.get("world") or {}
        inflight = s.get("inflight") or {}
        lines += [f"**Swarm:** {'on' if s.get('enabled') else 'off'} · "
                  f"{inflight.get('running', 0)} running, {inflight.get('pending', 0)} pending · "
                  f"{len(s.get('dynamic_agents', []))} dynamic agent(s) · "
                  f"**{s.get('unread', 0)} unread** in your inbox.",
                  f"**World scout:** {world.get('known', 0)} topics picked up from papers "
                  f"({world.get('unworked', 0)} not yet worked), {world.get('scouts', 0)} scout(s) so far.", ""]
    else:
        lines += ["_Swarm state unavailable right now._", ""]
    if runs:
        lines += ["**Recent runs:**", "", "| Playbook | Status | When |", "|---|---|---|"]
        for r in runs:
            lines.append(f"| {r.get('playbook', '?')} | {r.get('status', '?')} | {_ago(r.get('started'))} |")
    else:
        lines += ["_No runs yet._"]
    actions = [_act("What has the world scout found?", "What has the world scout found?"),
               _act("What should I do next?", "What should I do next?"),
               _act("Run a swarm tick now", "Run a swarm tick now")]
    if s and s.get("unread"):
        actions.insert(0, _act(f"Review my {s['unread']} lead(s)", "What's in my inbox?"))
    return {"kind": "guide", "say": "\n".join(lines), "actions": actions}


def _world_answer():
    s = _swarm()
    disc = (s or {}).get("discovered") or []
    lines = ["## The world scout", "",
             "When the queue has room, the swarm reads **recent Europe PMC papers**, text-mines fresh "
             "diseases and targets out of them, and queues them up — so it keeps finding new things to "
             "work on instead of recycling the focus list. You'll see it in the **swarm overlay** on HQ "
             "(🌐 lines) and in each tick report under *\"From the world\"*.", ""]
    if disc:
        lines += ["**Picked up so far:**", "", "| Topic | Kind | Worked? | When |", "|---|---|---|---|"]
        for d in disc[:20]:
            lines.append(f"| {d.get('name', '?')} | {d.get('kind', '?')} | "
                         f"{'yes' if d.get('worked') else 'queued'} | {_ago(d.get('added'))} |")
    else:
        lines += ["_Nothing picked up yet._ The scout runs when the swarm has a free slot; "
                  "run a tick to kick it, or let the schedule do it."]
    return {"kind": "guide", "say": "\n".join(lines),
            "actions": [_act("Run a swarm tick now", "Run a swarm tick now"),
                        _act("What's happening right now?", "What's happening right now?")]}


def _inbox_answer():
    s = _swarm()
    esc = (s or {}).get("escalations") or []
    unread = [e for e in esc if not e.get("read")]
    show = unread or esc
    lines = ["## Your inbox", ""]
    if not show:
        lines += ["Nothing escalated yet. The swarm only pings you when something clears the bar "
                  "(a lead) or is worth a look (a finding)."]
    else:
        head = "Unread" if unread else "Recent (all read)"
        lines += [f"**{head} — {len(show)}:**", ""]
        for e in show[:12]:
            tier = (e.get("tier") or "note").upper()
            lines.append(f"- **[{tier}]** {e.get('title', '')}" + (f" — {e.get('detail', '')}" if e.get('detail') else ""))
    return {"kind": "guide", "say": "\n".join(lines),
            "actions": [_act("What should I do next?", "What should I do next?"),
                        _act("What's happening right now?", "What's happening right now?")]}


def _next_answer():
    s = _swarm()
    recs, actions = [], []
    if s:
        world = s.get("world") or {}
        if s.get("unread"):
            recs.append(f"**Review your inbox** — the swarm escalated **{s['unread']}** item(s) worth a look.")
            actions.append(_act("Open my inbox", "What's in my inbox?"))
        if world.get("unworked"):
            recs.append(f"**Let the world topics run** — {world['unworked']} literature-scouted topic(s) are "
                        f"queued but not worked yet. A swarm tick will start them.")
            actions.append(_act("Run a swarm tick now", "Run a swarm tick now"))
        if not s.get("enabled"):
            recs.append("**Turn the swarm on** — it's currently off, so nothing is running autonomously.")
        if not s.get("running") and not s.get("pending"):
            recs.append("**Start something** — nothing is in flight. Kick off a discovery mission on a disease you care about.")
            actions.append(_act("Find me a drug for pancreatic cancer", "Find me a drug for pancreatic cancer"))
    if not recs:
        recs.append("Everything's ticking over. Point me at a disease or target and I'll dig in — "
                    "or ask *\"what's happening right now?\"* to watch the swarm work.")
        actions.append(_act("What's happening right now?", "What's happening right now?"))
    say = "## What I'd do next\n\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))
    # de-dup actions by ask
    seen, uniq = set(), []
    for a in actions:
        if a["ask"] not in seen:
            seen.add(a["ask"]); uniq.append(a)
    return {"kind": "guide", "say": say, "actions": uniq[:4]}


def guide(q: str) -> dict:
    ql = (q or "").lower()
    # order: specific intents first, generic help last
    if _WORLD.search(ql):
        return _world_answer()
    if _INBOX.search(ql):
        return _inbox_answer()
    if _NEXT.search(ql):
        return _next_answer()
    if _STATUS.search(ql):
        return _status_answer()
    if _HOWTO.search(ql):
        return _howto_answer(q)
    return _help_answer()
