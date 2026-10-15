import re
import time
from pathlib import Path
from urllib.parse import urlparse

from ..config import DATA_DIR

ARMORY = DATA_DIR / "armory"
DIRS = {"har": ARMORY / "har", "rea": ARMORY / "rea", "reports": ARMORY / "reports"}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def stamp_name(url: str, ext: str) -> str:
    host = urlparse(url).hostname or "page"
    path = urlparse(url).path.strip("/").replace("/", "_")[:40]
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", f"{host}{'_' + path if path else ''}")
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}.{ext}"


def resolve(kind: str, name: str) -> Path:
    if kind not in DIRS:
        raise ValueError(f"unknown artifact kind {kind}")
    name = Path(name).name
    if not _SAFE.match(name):
        raise ValueError("bad file name")
    p = DIRS[kind] / name
    if not p.exists():
        raise FileNotFoundError(f"{kind}/{name} not found")
    return p


def list_artifacts():
    out = {}
    for kind, d in DIRS.items():
        files = sorted(d.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        out[kind] = [{"name": p.name, "bytes": p.stat().st_size,
                      "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
                     for p in files if p.is_file()]
    return out
