"""site_recon: capture -> summarise -> (REA) -> markdown report."""
from __future__ import annotations

import json
import time

from . import har as harmod
from . import rea as reamod
from .paths import DIRS


def _schema_block(s, limit=1800):
    t = json.dumps(s, indent=1)
    return t if len(t) <= limit else t[:limit] + "\n…"


def site_recon(url: str, explore: bool = True, run_rea: bool = True) -> dict:
    cap = harmod.capture(url, explore=explore)
    summ = harmod.summarize(cap["har"])
    rea_res = reamod.inspect(cap["har"]) if run_rea else None

    notes = []
    if summ["data_endpoints"]:
        notes.append(f"{len(summ['data_endpoints'])} first-party JSON endpoint(s) carry the page data; "
                     "their schemas below are the shape to mirror.")
    embedded = [b for b in summ["js_bundles"] if b.get("embedded_data_hint")]
    if embedded:
        notes.append("Data appears bundled inside JS (" + ", ".join(b["url"].split("/")[-1][:50] for b in embedded[:3]) +
                     "); use har_payload on it to read the records.")
    if not summ["data_endpoints"] and not embedded:
        notes.append("No obvious data payload: the page may be static, precomputed into HTML, or need interaction "
                     "the auto-explore did not trigger.")
    if any("api.github.com" in c["url"] for c in summ["third_party_calls"]):
        notes.append("The client calls the GitHub API directly (live repo checks in the browser).")

    lines = [f"# Recon: {summ.get('title') or url}", "",
             f"- URL: {url}", f"- Captured: {time.strftime('%Y-%m-%d %H:%M')}",
             f"- HAR: `{cap['har']}` ({cap['entries']} requests, {cap['clicks']} auto-clicks)",
             f"- By kind: {summ['by_kind']}", "", "## Read-out", ""]
    lines += [f"- {n}" for n in notes]
    lines += ["", "## First-party data endpoints", ""]
    for d in summ["data_endpoints"][:12]:
        lines += [f"### {d['method']} {d['url']}", f"status {d['status']} · {d['bytes']} bytes", "",
                  "```json", _schema_block(d.get("schema")), "```", ""]
    if not summ["data_endpoints"]:
        lines += ["None found.", ""]
    lines += ["## JS bundles (largest first)", ""]
    for b in summ["js_bundles"][:10]:
        lines.append(f"- {b['url']} · {b['bytes']} bytes · {b.get('json_like_keys', 0)} JSON-like keys"
                     + (" · **likely embedded data**" if b.get("embedded_data_hint") else ""))
    lines += ["", "## Third-party calls", ""]
    lines += [f"- {c['method']} {c['url']} ({c['status']})" for c in summ["third_party_calls"][:20]] or ["None."]
    lines += ["", "## REA", ""]
    if rea_res is None:
        lines.append("Skipped.")
    elif "error" in rea_res:
        lines.append(f"Not run: {rea_res['error']}")
    else:
        lines.append(f"`{rea_res['command']}` → exit {rea_res['exit_code']}, "
                     f"{rea_res.get('records')} records, output `{rea_res['output']}`")
    if summ["failed_requests"]:
        lines += ["", "## Failed requests", ""] + [f"- {f['url']} ({f['error']})" for f in summ["failed_requests"]]
    if cap["errors"]:
        lines += ["", "## Capture warnings", ""] + [f"- {e}" for e in cap["errors"]]

    report = DIRS["reports"] / (cap["har"].rsplit(".", 1)[0] + ".md")
    report.write_text("\n".join(lines) + "\n")
    return {"report": report.name, "har": cap["har"], "entries": cap["entries"],
            "data_endpoints": len(summ["data_endpoints"]), "notes": notes,
            "rea": rea_res}
