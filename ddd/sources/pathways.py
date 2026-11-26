"""Pathway ingredients from Fate Memory Lab — woven into ddd's drug research.

FML's pathway library is a curated enzyme -> reaction -> deficiency-disease ->
known-drugs map (the `enzyme` field is the gene symbol, e.g. XO, DHFR, TYMS).
ddd uses it as an *ingredient*, not a separate page: when a target is one of
these enzymes, ddd pulls in its pathway, what it converts, the disease its
deficiency causes, and the drugs that already hit it (with mechanism). An
optional knock-down simulation gives mechanistic target validation.

Pure local data + compute (no network). Degrades to None if FML isn't present.
"""
from __future__ import annotations

_INDEX = None


def _index():
    """enzyme gene symbol (upper) -> list of (pathway, reaction)."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    idx = {}
    try:
        from ..fml.pathways import PATHWAYS
        for p in PATHWAYS.values():
            for r in p["reactions"]:
                enz = (r.get("enzyme") or "").strip().upper()
                if enz:
                    idx.setdefault(enz, []).append((p, r))
    except Exception:
        idx = {}
    _INDEX = idx
    return idx


def _reaction_arrow(r):
    subs = " + ".join(r.get("substrates", {}).keys())
    prods = " + ".join(r.get("products", {}).keys())
    return f"{subs} → {prods}" if (subs or prods) else ""


def for_target(symbol: str) -> dict | None:
    """Pathway + known-drug context for a target enzyme (by gene symbol)."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None
    hits = _index().get(symbol)
    if not hits:
        return None
    contexts = []
    for p, r in hits:
        contexts.append({
            "pathway": p["name"], "pathway_id": p["id"],
            "reaction": r["name"], "reaction_id": r["id"], "enzyme": r["enzyme"],
            "converts": _reaction_arrow(r),
            "deficiency": r.get("deficiency"),
            "drugs": r.get("drugs", []),
            "regulators": r.get("regulators", []),
        })
    druggable = any(c["drugs"] for c in contexts)
    return {"target": symbol, "found": True, "druggable": druggable, "contexts": contexts,
            "source": "Fate Memory Lab pathway library"}


def for_disease(disease: str) -> dict | None:
    """FML pathways whose deficiency matches a disease → their druggable enzymes."""
    d = (disease or "").strip().lower()
    if not d:
        return None
    rows = []
    try:
        from ..fml.pathways import PATHWAYS
    except Exception:
        return None
    for p in PATHWAYS.values():
        for r in p["reactions"]:
            def_ = (r.get("deficiency") or "").lower()
            if def_ and (d in def_ or any(w in def_ for w in d.split() if len(w) > 4)):
                rows.append({"pathway": p["name"], "enzyme": r["enzyme"], "reaction": r["name"],
                             "deficiency": r.get("deficiency"), "drugs": r.get("drugs", [])})
    return {"disease": disease, "hits": rows} if rows else None


def knockdown(symbol: str) -> dict | None:
    """Mechanistic validation ingredient: simulate knocking the enzyme down in its
    pathway and report the main downstream change (FML what_if)."""
    symbol = (symbol or "").strip().upper()
    hits = _index().get(symbol)
    if not hits:
        return None
    p, r = hits[0]
    try:
        from ..fml.pathways import what_if
        res = what_if(p, r["id"])
        return {"enzyme": symbol, "pathway": p["name"], "label": res.get("label"),
                "changes": (res.get("changes") or [])[:5], "settled": res.get("settled")}
    except Exception:
        return None
