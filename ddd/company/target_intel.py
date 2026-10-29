"""Target Intelligence — "is this target worth ddd's effort, or already taken?".

Ties together everything ddd can learn about a target before committing design
effort to it:
  identity (UniProt)  →  is its structure solved? (AlphaFold + experimental PDB)
                      →  how drugged / crowded is it? (ChEMBL actives + Open Targets drugs)
                      →  one verdict: open space / emerging / crowded.

Three entry points:
  dossier(target)        – the readiness check for a known target (gene/UniProt).
  identify(sequence)     – a raw sequence ddd found -> what protein is it (BLAST),
                           then the dossier for the match.
  watch(targets, state)  – re-check a watchlist for newly released structures/drugs
                           and diff against the last snapshot (for scheduled runs).
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import COMPANY_DIR

_WATCH_STATE = COMPANY_DIR / "structure_watch.json"


# ---------------------------------------------------------------- gather
def _gather(target: str, ctx=None) -> dict:
    log = (ctx.log if ctx else (lambda s: None))
    out = {"target": target, "gene": None, "uniprot": None,
           "alphafold": None, "pdb": None, "chembl": None, "known_drugs": None}
    # identity
    try:
        from ..armory.dock import resolve_uniprot
        info = resolve_uniprot(target)
        out["uniprot"] = info["accession"]
        out["gene"] = info.get("symbol") or target
        out["protein"] = info.get("protein")
    except Exception as e:
        log(f"identity: couldn't resolve to UniProt ({str(e)[:70]})")
    # AlphaFold (predicted structure)
    if out["uniprot"]:
        try:
            from ..sources import structure
            s = structure.fetch(out["uniprot"])
            if s.get("ok"):
                out["alphafold"] = {"mean_plddt": s["mean_plddt"], "length": s["length"],
                                    "bands": s.get("plddt_bands"), "source": s.get("source")}
                log(f"AlphaFold: mean pLDDT {s['mean_plddt']} over {s['length']} residues")
        except Exception as e:
            log(f"AlphaFold: unavailable ({str(e)[:70]})")
    # experimental structures (PDB)
    if out["uniprot"]:
        try:
            from ..sources import pdb
            p = pdb.structures_for_uniprot(out["uniprot"])
            if p.get("ok"):
                out["pdb"] = p
                log(f"PDB: {p['count']} experimental structure(s)"
                    + (f", best {p['best_resolution']} Å" if p.get('best_resolution') else "")
                    + (", ligand-bound" if p.get("ligand_bound") else ""))
        except Exception as e:
            log(f"PDB: unavailable ({str(e)[:70]})")
    # ChEMBL actives (chemical matter)
    gene = out["gene"] or target
    try:
        from ..sources import chembl
        res = chembl.target_actives(gene, limit=1)
        out["chembl"] = {"actives": res.get("count", 0)}
        log(f"ChEMBL: {out['chembl']['actives']} active compound(s)")
    except Exception as e:
        log(f"ChEMBL: unavailable ({str(e)[:70]})")
    # Open Targets known drugs (clinical maturity)
    try:
        from ..sources import opentargets
        kd = opentargets.target_known_drugs(gene, size=10)
        if kd.get("found", True) is not False:
            rows = kd.get("rows") or kd.get("drugs") or []
            phases = [r.get("phase") for r in rows if r.get("phase") is not None]
            names = []
            for r in rows:
                d = r.get("drug") or {}
                nm = d.get("name") if isinstance(d, dict) else r.get("drug")
                if nm and nm not in names:
                    names.append(nm)
            out["known_drugs"] = {"count": kd.get("count", len(rows)),
                                  "max_phase": (max(phases) if phases else None),
                                  "examples": names[:5]}
            log(f"Open Targets: {out['known_drugs']['count']} known drug(s), max phase "
                f"{out['known_drugs']['max_phase']}")
    except Exception as e:
        log(f"Open Targets: unavailable ({str(e)[:70]})")
    return out


# ---------------------------------------------------------------- verdict
def _verdict(g: dict) -> dict:
    af, pdb, ch, kd = g.get("alphafold"), g.get("pdb"), g.get("chembl"), g.get("known_drugs")
    pdb_n = (pdb or {}).get("count", 0)
    ligand_bound = (pdb or {}).get("ligand_bound", False)
    actives = (ch or {}).get("actives", 0)
    max_phase = (kd or {}).get("max_phase")
    n_drugs = (kd or {}).get("count", 0)

    # structural readiness
    if pdb_n > 0 and ligand_bound:
        struct = "solved experimentally, with ligand-bound structures (pocket characterised)"
    elif pdb_n > 0:
        struct = "solved experimentally (apo); no ligand-bound entry seen in the top set"
    elif af:
        struct = f"no experimental structure; AlphaFold model available (mean pLDDT {af.get('mean_plddt')})"
    else:
        struct = "no experimental or predicted structure found"

    # crowdedness
    try:
        mp = float(max_phase) if max_phase is not None else 0
    except (TypeError, ValueError):
        mp = 0
    if mp >= 4 or (n_drugs and mp >= 3):
        bucket = "Crowded / mature"
        why = f"approved or late-clinical drugs exist ({n_drugs} known, max phase {max_phase})."
        move = "Only pursue with a clear differentiator (new mechanism, selectivity, modality)."
    elif mp >= 1 or actives >= 200:
        bucket = "Active / emerging"
        why = f"clinical or heavy preclinical interest ({n_drugs} drugs, {actives} ChEMBL actives)."
        move = "Fast-following is possible; look for an unaddressed sub-indication or a better profile."
    elif actives >= 1 or pdb_n > 0:
        bucket = "Lightly explored"
        why = f"some chemical matter ({actives} actives) but no clinical drug yet."
        move = "Good design target — build on the known actives as seeds."
    else:
        bucket = "Open space"
        why = "no known drugs and little/no chemical matter in ChEMBL."
        move = "High-risk, high-reward: ddd's generative design loop earns its keep here."

    return {"structure": struct, "crowding": bucket, "why": why, "recommendation": move,
            "druggable_pocket_known": ligand_bound}


def _snapshot(g: dict) -> dict:
    return {"uniprot": g.get("uniprot"),
            "af": bool(g.get("alphafold")),
            "pdb_count": (g.get("pdb") or {}).get("count", 0),
            "pdb_newest": (g.get("pdb") or {}).get("newest"),
            "drugs": (g.get("known_drugs") or {}).get("count", 0),
            "max_phase": (g.get("known_drugs") or {}).get("max_phase")}


# ---------------------------------------------------------------- dossier
def dossier(ctx, target: str):
    g = _gather(target, ctx)
    v = _verdict(g)
    name = g.get("gene") or target
    lines = [f"# Target intelligence: {name}", ""]
    if g.get("uniprot"):
        lines.append(f"**Identity** · {g.get('protein') or name} · UniProt `{g['uniprot']}`"
                     + (f" · {g['alphafold']['length']} residues" if g.get("alphafold") else ""))
    else:
        lines.append(f"**Identity** · couldn't resolve `{target}` to a human UniProt entry "
                     "(viral/bacterial targets may not resolve here).")
    lines += ["", "## Verdict", "",
              f"- **Crowding: {v['crowding']}** — {v['why']}",
              f"- **Structure:** {v['structure']}",
              f"- **Druggable pocket known:** {'yes' if v['druggable_pocket_known'] else 'not evident'}",
              f"- **Recommended move:** {v['recommendation']}", "",
              "## Evidence", "", "| Signal | Finding |", "|---|---|"]
    af = g.get("alphafold")
    lines.append(f"| AlphaFold model | {'yes · mean pLDDT ' + str(af['mean_plddt']) if af else 'not found'} |")
    pdb = g.get("pdb")
    if pdb:
        lines.append(f"| Experimental structures (PDB) | {pdb['count']}"
                     + (f" · best {pdb['best_resolution']} Å" if pdb.get('best_resolution') else "")
                     + (f" · newest {pdb['newest']}" if pdb.get('newest') else "")
                     + (" · ligand-bound" if pdb.get('ligand_bound') else "") + " |")
    else:
        lines.append("| Experimental structures (PDB) | none found / unavailable |")
    ch = g.get("chembl")
    lines.append(f"| ChEMBL active compounds | {ch['actives'] if ch else 'unavailable'} |")
    kd = g.get("known_drugs")
    if kd:
        ex = (" · e.g. " + ", ".join(kd["examples"])) if kd.get("examples") else ""
        lines.append(f"| Known drugs (Open Targets) | {kd['count']} · max phase {kd['max_phase']}{ex} |")
    else:
        lines.append("| Known drugs (Open Targets) | none found / unavailable |")
    if pdb and pdb.get("entries"):
        lines += ["", "## Recent experimental structures", "", "| PDB | Method | Res (Å) | Ligands | Released |", "|---|---|---|---|---|"]
        for e in pdb["entries"][:6]:
            lines.append(f"| {e['id']} | {e.get('method') or '—'} | {e.get('resolution') or '—'} | "
                         f"{e.get('ligands') or 0} | {e.get('released') or '—'} |")
    lines += ["", "_Signals are from public AlphaFold DB, RCSB PDB, ChEMBL and Open Targets. "
              "The verdict is a prioritisation heuristic, not a guarantee of novelty or success._"]
    summary = {"target": name, "uniprot": g.get("uniprot"), "crowding": v["crowding"],
               "pdb": (pdb or {}).get("count", 0), "drugs": (kd or {}).get("count", 0),
               "structure_known": bool(af or (pdb or {}).get("count"))}
    return summary, lines


# ---------------------------------------------------------------- identify sequence
def identify(ctx, sequence: str):
    from ..sources import blast
    ctx.log("Running blastp vs UniProtKB (asynchronous; may take up to ~2 min)")
    r = blast.identify(sequence, progress=ctx.log)
    name = "sequence"
    lines = [f"# Sequence identification", ""]
    if not r.get("ok"):
        lines += [f"- Could not identify: {r.get('error')}",
                  "- Treat as **novel / unverified** until BLAST succeeds."]
        return {"identified": False, "error": r.get("error")}, lines
    top = r.get("top")
    lines += [f"- Query length: {r['query_length']} residues",
              f"- **{r['verdict']}**", ""]
    if r.get("hits"):
        lines += ["| Hit | Description | Identity | E-value |", "|---|---|---|---|"]
        for h in r["hits"][:6]:
            lines.append(f"| {h.get('accession') or h.get('id')} | {h.get('description')} | "
                         f"{h.get('identity_pct')}% | {h.get('evalue')} |")
    summary = {"identified": bool(r.get("known")), "top": (top or {}).get("accession")}
    # if it's a known protein, chain straight into its readiness dossier
    if r.get("known") and top and top.get("accession"):
        ctx.log(f"Match {top['accession']} ≥98% identity — pulling its target dossier")
        acc = top["accession"]
        dsum, dlines = dossier(ctx, acc)
        lines += ["", "---", ""] + dlines
        summary["dossier"] = dsum
    else:
        lines += ["", "_No near-exact match — ddd would treat this as a novel sequence. "
                  "Fold it (ESMFold) and characterise de novo rather than looking it up._"]
    return summary, lines


# ---------------------------------------------------------------- watch
def watch(ctx, targets, _state_path: Path = _WATCH_STATE):
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.replace(",", "\n").splitlines() if t.strip()]
    targets = [t for t in targets if t][:30]
    if not targets:
        raise ValueError("Give at least one target (gene or UniProt), comma/newline separated.")
    try:
        prev = json.loads(_state_path.read_text()) if _state_path.exists() else {}
    except Exception:
        prev = {}
    now, changes = {}, []
    for t in targets:
        ctx.log(f"Checking {t}")
        g = _gather(t, ctx)
        snap = _snapshot(g)
        key = snap.get("uniprot") or t
        now[key] = snap
        old = prev.get(key)
        if old:
            d = []
            if snap["pdb_count"] > old.get("pdb_count", 0):
                d.append(f"+{snap['pdb_count'] - old.get('pdb_count',0)} PDB structure(s) (newest {snap['pdb_newest']})")
            if not old.get("af") and snap["af"]:
                d.append("AlphaFold model now available")
            if snap["drugs"] > old.get("drugs", 0):
                d.append(f"+{snap['drugs'] - old.get('drugs',0)} known drug(s)")
            if snap.get("max_phase") and snap["max_phase"] != old.get("max_phase"):
                d.append(f"max clinical phase {old.get('max_phase')} → {snap['max_phase']}")
            if d:
                changes.append((t, key, d))
        else:
            changes.append((t, key, ["first snapshot — baseline recorded"]))
    try:
        _state_path.write_text(json.dumps(now, indent=1))
    except Exception as e:
        ctx.log(f"couldn't persist watch state: {str(e)[:70]}")
    lines = [f"# Structure watch — {len(targets)} target(s)", ""]
    real = [c for c in changes if c[2] != ["first snapshot — baseline recorded"]]
    if real:
        lines += ["## Changes since last check", ""]
        for t, key, d in real:
            lines.append(f"- **{t}** (`{key}`): " + "; ".join(d))
    else:
        lines += ["No new structures or drugs since the last check." if prev
                  else "Baseline recorded for all targets; changes will show on the next run.", ""]
    lines += ["", "## Current snapshot", "", "| Target | PDB | AlphaFold | Known drugs | Max phase |", "|---|---|---|---|---|"]
    for t in targets:
        key = None
        for k, s in now.items():
            if s.get("uniprot") == t or k == t:
                key = k; break
        s = now.get(key, {})
        lines.append(f"| {t} | {s.get('pdb_count',0)} | {'yes' if s.get('af') else 'no'} | "
                     f"{s.get('drugs',0)} | {s.get('max_phase') if s.get('max_phase') is not None else '—'} |")
    return {"targets": len(targets), "changes": len(real)}, lines
