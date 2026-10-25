"""Tools Armory: on-demand tools the ddd app pulls up when needed.

Each tool is a plain function plus metadata, so the web UI, the JSON API and
the MCP server all expose the same set. Add a tool by registering it in TOOLS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import dock, har, rea, recon
from .paths import list_artifacts


@dataclass
class Tool:
    name: str
    title: str
    category: str
    description: str
    params: dict = field(default_factory=dict)   # name -> {"type", "default", "help"}
    fn: Callable = None
    slow: bool = False

    def meta(self):
        return {k: v for k, v in self.__dict__.items() if k != "fn"}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("site_recon", "Site recon", "Recon",
         "Capture a public page as HAR (headless Chromium, optional auto-explore), "
         "summarise its endpoints and data payloads, optionally run REA on the capture, "
         "and write a markdown report.",
         {"url": {"type": "str", "help": "public page URL"},
          "explore": {"type": "bool", "default": True, "help": "click tabs/buttons to trigger lazy data"},
          "run_rea": {"type": "bool", "default": True, "help": "also run REA inspect-web-network-capture"}},
         recon.site_recon, slow=True),
    Tool("har_capture", "HAR capture", "Capture",
         "Load a public URL in headless Chromium and save everything it fetched as a HAR "
         "(cookies and auth headers redacted).",
         {"url": {"type": "str"},
          "explore": {"type": "bool", "default": True},
          "max_clicks": {"type": "int", "default": 15},
          "scrolls": {"type": "int", "default": 6}},
         har.capture, slow=True),
    Tool("har_summary", "HAR summary", "Analyse",
         "Summarise a saved HAR: hosts, first-party data endpoints with inferred JSON "
         "schemas, JS bundles and embedded-data hints.",
         {"har": {"type": "str", "help": "file name from the artifacts list"}},
         har.summarize),
    Tool("har_payload", "HAR payload", "Analyse",
         "Return the response body of the first HAR entry whose URL contains a substring.",
         {"har": {"type": "str"}, "match": {"type": "str"},
          "max_chars": {"type": "int", "default": 20000}},
         har.payload),
    Tool("rea_inspect", "REA inspect", "Analyse",
         "Run REA's inspect-web-network-capture on a saved HAR (needs Node 22+/npx).",
         {"har": {"type": "str"}},
         rea.inspect, slow=True),
    Tool("dock", "Dock ligands", "Emulator",
         "Dock ligands (SMILES) into a target with AutoDock Vina / smina. Fetches the "
         "target's AlphaFold structure on demand, preps receptor + ligands and returns "
         "binding affinities (kcal/mol; more negative = tighter). Blind box over the whole "
         "protein unless a pocket centre is given. Needs a docking engine installed on the host.",
         {"target": {"type": "str", "help": "gene symbol (e.g. PDGFRA) or UniProt accession"},
          "smiles": {"type": "text", "help": "one or more SMILES, newline- or comma-separated"},
          "exhaustiveness": {"type": "int", "default": 8, "help": "Vina search effort"},
          "n_poses": {"type": "int", "default": 9},
          "center": {"type": "str", "default": "", "help": "optional pocket centre 'x,y,z'"},
          "size": {"type": "float", "default": 0.0, "help": "cubic box side (A) with an explicit centre"}},
         dock.dock, slow=True),
    Tool("artifacts", "Artifacts", "Library",
         "List captured HARs, REA outputs and recon reports.",
         {}, lambda: list_artifacts()),
]}

PRESETS = [
    {"label": "DDDBench", "url": "https://dddbench.insilico.com/"},
    {"label": "Virtual Aging Cell · Chorus v3", "url": "https://virtualcell.insilico.com/chorus_v3.html"},
    {"label": "O3DC index", "url": "https://o3dc.org/"},
]


def run_tool(name: str, args: dict | None = None):
    if name not in TOOLS:
        raise KeyError(f"unknown tool '{name}'. Known: {', '.join(TOOLS)}")
    t = TOOLS[name]
    args = dict(args or {})
    for p, spec in t.params.items():
        if p not in args and "default" in spec:
            args[p] = spec["default"]
    allowed = set(t.params)
    return t.fn(**{k: v for k, v in args.items() if k in allowed})
