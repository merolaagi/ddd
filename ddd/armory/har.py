"""HAR capture (Playwright) and HAR analysis."""
from __future__ import annotations

import base64
import json
import re
from collections import Counter
from urllib.parse import urlparse

from .paths import DIRS, resolve, stamp_name

SKIP_TEXT = re.compile(r"submit|send|join|sign|log ?in|register|subscribe|delete|remove|buy|pay|confirm|logout|export", re.I)
SENSITIVE_HEADERS = {"cookie", "set-cookie", "authorization", "proxy-authorization", "x-api-key"}
KEY_PATTERN = re.compile(r'"[A-Za-z_][A-Za-z0-9_]{1,40}"\s*:')


def _redact(path):
    data = json.loads(path.read_text())
    n = 0
    for e in data.get("log", {}).get("entries", []):
        for side in ("request", "response"):
            for h in e.get(side, {}).get("headers", []):
                if h.get("name", "").lower() in SENSITIVE_HEADERS:
                    h["value"] = "[redacted]"
                    n += 1
            if e.get(side, {}).get("cookies"):
                e[side]["cookies"] = []
    path.write_text(json.dumps(data))
    return n


def capture(url: str, explore: bool = True, max_clicks: int = 15, scrolls: int = 6) -> dict:
    """Open URL in headless Chromium, optionally scroll and click tab-like controls,
    and save the full network log (bodies embedded) as HAR."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http(s)://")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError("Playwright missing: .venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium") from e

    out = DIRS["har"] / stamp_name(url, "har")
    origin = urlparse(url).hostname
    clicks, clicked_labels, errors = 0, [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(record_har_path=str(out), record_har_content="embed",
                                  viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
        except Exception as e:
            errors.append(f"goto: {str(e)[:160]}")
        for _ in range(max(0, int(scrolls))):
            page.mouse.wheel(0, 1400)
            page.wait_for_timeout(600)
        if explore:
            seen = set()
            for el in page.query_selector_all("button, [role=tab], [role=button], summary, [data-tab], .tab"):
                if clicks >= int(max_clicks):
                    break
                try:
                    if not el.is_visible():
                        continue
                    label = (el.inner_text() or el.get_attribute("aria-label") or "").strip()[:60]
                    if not label or label in seen or SKIP_TEXT.search(label):
                        continue
                    seen.add(label)
                    el.click(timeout=2500)
                    page.wait_for_timeout(900)
                    page.keyboard.press("Escape")
                    clicks += 1
                    clicked_labels.append(label)
                    if urlparse(page.url).hostname != origin:
                        page.go_back(wait_until="networkidle", timeout=20000)
                except Exception as e:
                    errors.append(f"click {label!r}: {str(e)[:100]}")
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
        title = page.title()
        ctx.close()
        browser.close()
    redacted = _redact(out)
    entries = len(json.loads(out.read_text())["log"]["entries"])
    return {"har": out.name, "url": url, "title": title, "entries": entries,
            "clicks": clicks, "clicked": clicked_labels, "redacted_headers": redacted,
            "errors": errors[:10]}


# ---------- analysis ----------

def _site(host: str) -> str:
    parts = (host or "").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _body(entry) -> str | None:
    c = entry.get("response", {}).get("content", {})
    text = c.get("text")
    if text is None:
        return None
    if c.get("encoding") == "base64":
        try:
            return base64.b64decode(text).decode("utf-8", "replace")
        except Exception:
            return None
    return text


def _kind(mime: str, url: str) -> str:
    m = (mime or "").lower()
    if "json" in m:
        return "json"
    if "javascript" in m or url.split("?")[0].endswith((".js", ".mjs")):
        return "js"
    if "html" in m:
        return "html"
    if "css" in m:
        return "css"
    if m.startswith("image/"):
        return "image"
    if "font" in m or url.split("?")[0].endswith((".woff", ".woff2", ".ttf")):
        return "font"
    if "csv" in m or "text/plain" in m:
        return "text"
    return "other"


def schema(obj, depth=0, max_depth=4):
    """Compact structural schema of a JSON value."""
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, (int, float)):
        return "number"
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return "str"
    if depth >= max_depth:
        return "list" if isinstance(obj, list) else "object"
    if isinstance(obj, dict):
        return {k: schema(v, depth + 1, max_depth) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        if not obj:
            return []
        return {"__list__": len(obj), "item": schema(obj[0], depth + 1, max_depth)}
    return type(obj).__name__


def _load(har_name):
    path = resolve("har", har_name)
    return path, json.loads(path.read_text())


def summarize(har: str) -> dict:
    path, data = _load(har)
    entries = data.get("log", {}).get("entries", [])
    pages = data.get("log", {}).get("pages", [])
    page_url = entries[0]["request"]["url"] if entries else ""
    home = _site(urlparse(page_url).hostname)
    kinds, hosts = Counter(), Counter()
    data_endpoints, bundles, third_party_calls, failed = [], [], [], []
    for e in entries:
        url = e["request"]["url"]
        host = urlparse(url).hostname or ""
        hosts[host] += 1
        mime = e.get("response", {}).get("content", {}).get("mimeType", "")
        if (e.get("response", {}).get("status") or 0) <= 0:
            kinds["failed"] += 1
            failed.append({"url": url[:300], "error": e["response"].get("_failureText")})
            continue
        k = _kind(mime, url)
        kinds[k] += 1
        body = _body(e)
        size = e.get("response", {}).get("content", {}).get("size", 0) or (len(body) if body else 0)
        first_party = _site(host) == home
        rec = {"method": e["request"]["method"], "url": url[:300], "status": e["response"]["status"],
               "kind": k, "bytes": size, "first_party": first_party}
        if k in ("json", "text", "other") and body:
            try:
                obj = json.loads(body)
                rec["schema"] = schema(obj)
                rec["kind"] = "json"
                (data_endpoints if first_party else third_party_calls).append(rec)
                continue
            except Exception:
                pass
        if k == "js" and body:
            keys = len(KEY_PATTERN.findall(body))
            rec["json_like_keys"] = keys
            rec["embedded_data_hint"] = keys > 200
            bundles.append(rec)
        elif not first_party and k not in ("image", "font", "css"):
            third_party_calls.append(rec)
    bundles.sort(key=lambda r: r["bytes"], reverse=True)
    return {
        "har": path.name, "page": page_url, "title": pages[0].get("title") if pages else None,
        "entries": len(entries), "by_kind": dict(kinds), "hosts": dict(hosts.most_common(20)),
        "data_endpoints": data_endpoints[:40],
        "js_bundles": bundles[:15],
        "third_party_calls": third_party_calls[:30],
        "failed_requests": failed[:20],
    }


def payload(har: str, match: str, max_chars: int = 20000) -> dict:
    _, data = _load(har)
    for i, e in enumerate(data.get("log", {}).get("entries", [])):
        if match in e["request"]["url"]:
            body = _body(e) or ""
            return {"ordinal": i, "url": e["request"]["url"], "status": e["response"]["status"],
                    "mime": e["response"].get("content", {}).get("mimeType"),
                    "length": len(body), "truncated": len(body) > max_chars,
                    "body": body[: int(max_chars)]}
    return {"error": f"no entry whose URL contains {match!r}"}
