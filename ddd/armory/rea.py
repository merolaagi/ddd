"""Bridge to REA (github.com/morluto/rea) for HAR evidence extraction."""
from __future__ import annotations

import json
import shutil
import subprocess

from .paths import DIRS, resolve


def _cmd():
    if shutil.which("rea"):
        return ["rea"]
    if shutil.which("npx"):
        return ["npx", "-y", "rea-agents@latest"]
    return None


def available() -> dict:
    c = _cmd()
    return {"available": bool(c), "command": " ".join(c) if c else None}


def _slim(path):
    """REA caps input at 32 MiB: drop image/font/css/media bodies into a sibling copy."""
    data = json.loads(path.read_text())
    for e in data.get("log", {}).get("entries", []):
        c = e.get("response", {}).get("content", {})
        mime = (c.get("mimeType") or "").lower()
        if mime.startswith(("image/", "font/", "video/", "audio/")) or "css" in mime or "woff" in mime:
            c.pop("text", None)
            c.pop("encoding", None)
    out = path.with_name(path.stem + ".slim.har")
    out.write_text(json.dumps(data))
    return out


def inspect(har: str) -> dict:
    path = resolve("har", har)
    base = _cmd()
    if not base:
        return {"error": "REA needs Node.js 22+ (brew install node); then it runs via npx rea-agents."}
    target = _slim(path) if path.stat().st_size > 30 * 1024 * 1024 else path
    cmd = base + ["inspect-web-network-capture", str(target.resolve()), "har", "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"error": "REA timed out after 10 minutes"}
    out = DIRS["rea"] / (path.stem + ".rea.json")
    out.write_text(r.stdout or "")
    result = {"command": " ".join(cmd[:-3]) + " <har> har --json", "exit_code": r.returncode,
              "output": out.name}
    if r.returncode != 0:
        result["stderr"] = (r.stderr or "")[-800:]
    try:
        doc = json.loads(r.stdout)
        nr = doc.get("normalized_result") or doc
        result["records"] = nr.get("total_records", len(nr.get("records") or []))
        result["confidence"] = doc.get("confidence")
        result["limitations"] = (nr.get("limitations") or doc.get("limitations") or [])[:4]
    except Exception:
        result["note"] = "REA output was not JSON; see stderr and the saved output file."
    return result
