"""Sequence identification via EBI BLAST — "what protein did we just find?".

Given a raw protein sequence (one ddd discovered or designed), run blastp against
UniProtKB and return the top matches: accession, description, % identity, e-value.
That tells us whether the sequence is already a known protein (so we can pull its
structure and known drugs) or looks novel.

EBI Job Dispatcher REST (public, no key). It is asynchronous: submit -> poll ->
fetch. A generic contact address is sent as the service requires one; the user's
own address is never used. Best-effort: if the service is slow or down, the caller
treats the sequence as unidentified and moves on.
"""
from __future__ import annotations

import time

import httpx

BASE = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"
CONTACT = "ddd-bench@example.com"   # generic; EBI requires a contact, not the user's
_TIMEOUT = 30


def _clean(seq: str) -> str:
    # strip FASTA header and whitespace
    lines = [l.strip() for l in str(seq).splitlines() if l.strip() and not l.startswith(">")]
    return "".join(lines).upper().replace(" ", "")


def identify(sequence: str, max_wait: int = 120, progress=None) -> dict:
    log = progress or (lambda s: None)
    seq = _clean(sequence)
    if len(seq) < 10:
        return {"ok": False, "error": "sequence too short (need ≥10 residues)"}
    data = {"email": CONTACT, "program": "blastp", "stype": "protein",
            "database": "uniprotkb", "sequence": seq}
    try:
        r = httpx.post(f"{BASE}/run", data=data, timeout=_TIMEOUT)
        r.raise_for_status()
        job = r.text.strip()
    except Exception as e:
        return {"ok": False, "error": f"BLAST submit failed: {str(e)[:120]}"}
    log(f"BLAST job {job} submitted; waiting for results")
    deadline = time.time() + max_wait
    status = "RUNNING"
    while time.time() < deadline:
        time.sleep(5)
        try:
            s = httpx.get(f"{BASE}/status/{job}", timeout=_TIMEOUT).text.strip()
        except Exception:
            continue
        status = s
        if s in ("FINISHED", "ERROR", "FAILURE", "NOT_FOUND"):
            break
    if status != "FINISHED":
        return {"ok": False, "error": f"BLAST did not finish in time (status {status})", "job": job}
    try:
        res = httpx.get(f"{BASE}/result/{job}/json", timeout=_TIMEOUT).json()
    except Exception as e:
        return {"ok": False, "error": f"BLAST result fetch failed: {str(e)[:120]}", "job": job}
    hits = []
    for h in (res.get("hits") or [])[:10]:
        hsp = (h.get("hit_hsps") or [{}])[0]
        aln_len = hsp.get("hsp_align_len") or 0
        ident = hsp.get("hsp_identity") or 0
        pct = round(100 * ident / aln_len, 1) if aln_len else None
        hits.append({
            "accession": h.get("hit_acc"),
            "id": h.get("hit_id"),
            "description": (h.get("hit_desc") or "")[:140],
            "identity_pct": pct,
            "evalue": hsp.get("hsp_expect"),
            "score": hsp.get("hsp_bit_score"),
            "length": h.get("hit_len"),
        })
    top = hits[0] if hits else None
    known = bool(top and (top.get("identity_pct") or 0) >= 98)
    return {"ok": True, "query_length": len(seq), "job": job, "hits": hits,
            "top": top, "known": known,
            "verdict": ("exact/near-exact match to a known protein" if known
                        else ("similar to known proteins (not exact)" if hits else "no significant match — looks novel")),
            "note": "blastp vs UniProtKB via EBI. ≥98% identity over the alignment is treated as a known protein."}
