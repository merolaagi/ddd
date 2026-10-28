"""Literature grounding via Europe PMC (free, no key).

Gives ddd the "interpret in biological context" layer — real papers for a
disease, gene or question, ranked by citations. Used by the analyst and can be
cited into dossiers. On-demand, nothing stored; a Chroma vector store can cache
and embed these later for RAG, but the live search already grounds answers.
"""
from __future__ import annotations

import httpx

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


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
