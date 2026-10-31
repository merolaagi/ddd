"""Literature grounding via Europe PMC (free, no key).

Gives ddd the "interpret in biological context" layer — real papers for a
disease, gene or question, ranked by citations. Used by the analyst and can be
cited into dossiers. On-demand, nothing stored; a Chroma vector store can cache
and embed these later for RAG, but the live search already grounds answers.
"""
from __future__ import annotations

from collections import defaultdict

import httpx

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ANNO = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"


def search(query: str, limit: int = 8, sort: str = "CITED desc") -> dict:
    r = httpx.get(BASE, params={"query": query, "format": "json",
                                "pageSize": min(max(limit, 1), 25),
                                "resultType": "lite", "sort": sort},
                  timeout=30, follow_redirects=True)
    r.raise_for_status()
    res = (r.json().get("resultList") or {}).get("result", [])
    papers = []
    for a in res[:limit]:
        src, pid, doi = a.get("source"), a.get("id"), a.get("doi")
        url = (f"https://europepmc.org/article/{src}/{pid}" if src and pid
               else (f"https://doi.org/{doi}" if doi else None))
        papers.append({
            "source": src,
            "id": pid,
            "title": a.get("title"),
            "authors": a.get("authorString"),
            "year": a.get("pubYear"),
            "journal": a.get("journalTitle") or "",
            "pmid": a.get("pmid"),
            "doi": doi,
            "cited": a.get("citedByCount"),
            "url": url,
        })
    return {"query": query, "count": len(papers), "papers": papers}


# Europe PMC text-mining annotation types we read
_TYPES = ("Gene_Proteins", "Chemicals")


def annotate(papers: list, types: tuple = _TYPES) -> dict:
    """Text-mined entities per paper, from Europe PMC's annotations API — the
    'reading' step. Aggregates across papers: for each entity, how many papers
    mention it (evidence strength) and total mentions. Returns {gene_proteins,
    chemicals} as ranked lists of {name, papers, mentions, ids}."""
    ided = [(p.get("source"), p.get("id")) for p in papers if p.get("source") and p.get("id")]
    agg = {t: defaultdict(lambda: {"papers": set(), "mentions": 0}) for t in types}
    for i in range(0, len(ided), 8):              # API takes up to 8 ids per call
        batch = ided[i:i + 8]
        params = [("format", "JSON")]
        params += [("articleIds", f"{s}:{pid}") for s, pid in batch]
        for t in types:
            params.append(("type", t))
        try:
            r = httpx.get(ANNO, params=params, timeout=40, follow_redirects=True)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue
        for art in (data or []):
            art_id = f"{art.get('source')}:{art.get('extId')}"
            for ann in (art.get("annotations") or []):
                typ = ann.get("type")
                if typ not in agg:
                    continue
                tags = ann.get("tags") or []
                name = (tags[0].get("name") if tags else None) or ann.get("exact")
                if not name:
                    continue
                name = name.strip()
                rec = agg[typ][name]
                rec["papers"].add(art_id)
                rec["mentions"] += 1
                if tags and tags[0].get("uri"):
                    rec["uri"] = tags[0]["uri"]

    def rank(d):
        rows = [{"name": k, "papers": len(v["papers"]), "mentions": v["mentions"],
                 "uri": v.get("uri")} for k, v in d.items()]
        return sorted(rows, key=lambda x: (x["papers"], x["mentions"]), reverse=True)

    return {"gene_proteins": rank(agg.get("Gene_Proteins", {})),
            "chemicals": rank(agg.get("Chemicals", {})),
            "papers_read": len(ided)}
