"""App settings stored on the server, so keys can be entered in the app instead of the environment.

Stored in data/settings.json with owner-only permissions and never sent back to the browser in full.
A value set here takes precedence over the matching environment variable; clearing it falls back to the
environment again.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = Path(os.environ.get("FML_SETTINGS", ROOT / "data" / "settings.json"))

FIELDS = {
    "anthropic_api_key": {"env": "ANTHROPIC_API_KEY", "secret": True, "default": ""},
    "anthropic_model": {"env": "FML_MODEL", "secret": False, "default": "claude-sonnet-5"},
    "neo4j_uri": {"env": "NEO4J_URI", "secret": False, "default": ""},
    "neo4j_user": {"env": "NEO4J_USER", "secret": False, "default": "neo4j"},
    "neo4j_password": {"env": "NEO4J_PASSWORD", "secret": True, "default": ""},
    "neo4j_database": {"env": "NEO4J_DATABASE", "secret": False, "default": "neo4j"},
}
MODELS = ["claude-sonnet-5", "claude-opus-5-5", "claude-fable-5-1", "claude-haiku-4-5-20251001"]


def _read():
    try:
        return json.loads(FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get(name):
    spec = FIELDS[name]
    stored = _read().get(name)
    if stored:
        return stored
    return os.environ.get(spec["env"], spec["default"])


def source(name):
    if _read().get(name):
        return "app"
    if os.environ.get(FIELDS[name]["env"]):
        return "environment"
    return "default"


def mask(value):
    if not value:
        return ""
    if len(value) <= 10:
        return "•" * len(value)
    return value[:7] + "•" * 6 + value[-4:]


def view():
    out = {}
    for name, spec in FIELDS.items():
        val = get(name)
        out[name] = {"value": mask(val) if spec["secret"] else val, "set": bool(val), "source": source(name),
                     "secret": spec["secret"], "env": spec["env"]}
    out["models"] = MODELS
    return out


def update(changes):
    """Apply changes. A secret sent as an empty string is left alone; send clear=True in the list to remove it."""
    data = _read()
    for name, val in (changes.get("values") or {}).items():
        if name not in FIELDS:
            continue
        val = (val or "").strip()
        if FIELDS[name]["secret"] and not val:
            continue
        if val:
            data[name] = val
        else:
            data.pop(name, None)
    for name in changes.get("clear") or []:
        data.pop(name, None)
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(FILE)
    return view()
