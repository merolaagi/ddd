import re
import time

from ..config import DATA_DIR

COMPANY_DIR = DATA_DIR / "company"
REPORTS = COMPANY_DIR / "reports"
PUBLIC = COMPANY_DIR / "public"
for _d in (COMPANY_DIR, REPORTS, PUBLIC):
    _d.mkdir(parents=True, exist_ok=True)

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def report_path(playbook: str, run_id: int):
    return REPORTS / f"{time.strftime('%Y%m%d-%H%M')}-{playbook}-r{run_id}.md"


def safe_report(name: str):
    if not _SAFE.match(name or ""):
        raise ValueError("bad report name")
    p = REPORTS / name
    if not p.exists():
        raise FileNotFoundError(name)
    return p
