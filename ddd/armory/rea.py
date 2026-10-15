"""Bridge to REA (github.com/morluto/rea) for HAR evidence extraction."""
from __future__ import annotations

import ipaddress
import json
import shutil
import subprocess

from .paths import DIRS, resolve

MAX_BYTES = 30 * 1024 * 1024


def _cmd():
    if shutil.which("rea"):
        return ["rea"]
    if shutil.which("npx"):
        return ["npx", "-y", "rea-agents@latest"]
    return None


def available() -> dict:
    c = _cmd()
    return {"available": bool(c), "command": " ".join(c) if c else None}


def _clean_ip(value):
    """REA's HAR schema rejects '[ipv6]' and ''. Return a plain address, or None to drop the field."""
    if not isinstance(value, str):
        return None
    v = value.strip().strip("[]")
    try:
        return str(ipaddress.ip_address(v.split("%")[0]))
    except ValueError:
        return None


def _prepare(path):
    """Write a REA-ready copy into the rea folder: normalise serverIPAddress, and
    drop media/font/css bodies when the capture is over REA's 32 MiB input cap."""
    data = json.loads(path.read_text())
    big = path.stat().st_size > MAX_BYTES
    fixed = 0
    for e in data.get("log", {}).get("entries", []):
        if "serverIPAddress" in e:
            ip = _clean_ip(e["serverIPAddress"])
            if ip != e["serverIPAddress"]:
                fixed += 1
            if ip is None:
                e.pop("serverIPAddress")
            else:
                e["serverIPAddress"] = ip
        if big:
            c = e.get("response", {}).get("content", {})
            mime = (c.get("mimeType") or "").lower()
            if mime.startswith(("image/", "font/", "video/", "audio/")) or "css" in mime or "woff" in mime:
                c.pop("text", None)
                c.pop("encoding", None)
    out = DIRS["rea"] / (path.stem + ".rea-input.har")
    out.write_text(json.dumps(data))
    return out, {"ip_fields_normalised": fixed, "media_bodies_dropped": big}


def inspect(har: str) -> dict:
    path = resolve("har", har)
    base = _cmd()
    if not base:
        return {"error": "REA needs Node.js 22+ (brew install node); then it runs via npx rea-agents."}
    target, prep = _prepare(path)
    cmd = base + ["inspect-web-network-capture", str(target.resolve()), "har", "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"error": "REA timed out after 10 minutes"}
    finally:
        target.unlink(missing_ok=True)
    out = DIRS["rea"] / (path.stem + ".rea.json")
    out.write_text(r.stdout or "")
    result = {"command": " ".join(base) + " inspect-web-network-capture <har> har --json",
              "exit_code": r.returncode, "output": out.name, "prepared": prep}
    try:
        doc = json.loads(r.stdout)
    except Exception:
        doc = None
    if r.returncode != 0:
        if isinstance(doc, dict) and doc.get("message"):
            issues = "; ".join(f"{'/'.join(map(str, i.get('path', [])))[-60:]}: {i.get('message', '')[:120]}"
                               for i in (doc.get("details") or {}).get("issues", [])[:3])
            result["error"] = f"REA rejected the capture: {doc['message']}" + (f" ({issues})" if issues else "")
        else:
            result["error"] = f"REA exited {r.returncode}: {(r.stderr or r.stdout or '')[-300:]}"
        return result
    if not isinstance(doc, dict):
        result["error"] = "REA output was not JSON; see the saved output file."
        return result
    nr = doc.get("normalized_result") or doc
    result["records"] = nr.get("total_records", len(nr.get("records") or []))
    result["confidence"] = doc.get("confidence")
    result["limitations"] = (nr.get("limitations") or doc.get("limitations") or [])[:4]
    return result
