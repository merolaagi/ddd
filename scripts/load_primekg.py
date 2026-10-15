"""Load PrimeKG (Chandak, Huang & Zitnik, 2023) into Neo4j.

Usage:
  python scripts/load_primekg.py --download
  python scripts/load_primekg.py --csv /path/to/kg.csv [--limit 500000]

PrimeKG lists each undirected edge in both directions; only one copy is kept.
"""
import argparse
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddd.config import DATA_DIR  # noqa: E402
from ddd.kg import TYPE_LABEL, driver  # noqa: E402

PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
BATCH = 5000


def download(dest: Path):
    print(f"Downloading PrimeKG kg.csv (~900 MB) to {dest} ...")
    with httpx.stream("GET", PRIMEKG_URL, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%}", end="", flush=True)
    print()


def run_batches(session, cypher, rows, label):
    t = time.time()
    for i in range(0, len(rows), BATCH):
        session.run(cypher, rows=rows[i:i + BATCH])
        print(f"\r  {label}: {min(i + BATCH, len(rows)):,}/{len(rows):,}", end="", flush=True)
    print(f"  ({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only load the first N edges (for a quick test)")
    ap.add_argument("--wipe", action="store_true", help="delete existing Entity nodes first")
    a = ap.parse_args()

    path = a.csv or DATA_DIR / "primekg_kg.csv"
    if a.download and not path.exists():
        download(path)
    if not path.exists():
        sys.exit(f"{path} not found. Use --download or --csv.")

    print("Reading CSV ...")
    df = pd.read_csv(path, low_memory=False, nrows=a.limit or None)
    df = df[df["x_index"] < df["y_index"]]
    print(f"  {len(df):,} unique edges")

    with driver().session() as s:
        if a.wipe:
            print("Wiping existing graph ...")
            while s.run("match (n:Entity) with n limit 10000 detach delete n return count(*) as c").single()["c"]:
                pass
        s.run("create constraint entity_idx if not exists for (n:Entity) require n.idx is unique")
        s.run("create index entity_name if not exists for (n:Entity) on (n.name)")

        nodes = pd.concat([
            df[["x_index", "x_id", "x_type", "x_name", "x_source"]].set_axis(["idx", "id", "type", "name", "source"], axis=1),
            df[["y_index", "y_id", "y_type", "y_name", "y_source"]].set_axis(["idx", "id", "type", "name", "source"], axis=1),
        ]).drop_duplicates("idx")
        nodes["id"] = nodes["id"].astype(str)
        for t, grp in nodes.groupby("type"):
            label = TYPE_LABEL.get(t, "Other")
            rows = grp.to_dict("records")
            run_batches(s, f"""unwind $rows as r
                merge (n:Entity {{idx: r.idx}})
                set n:{label}, n.id = r.id, n.type = r.type, n.name = r.name, n.source = r.source""",
                        rows, f"nodes[{t}]")

        edges = df[["x_index", "y_index", "relation", "display_relation"]].rename(
            columns={"x_index": "a", "y_index": "b", "display_relation": "display"}).to_dict("records")
        run_batches(s, """unwind $rows as r
            match (a:Entity {idx: r.a}) match (b:Entity {idx: r.b})
            create (a)-[:REL {relation: r.relation, display: r.display}]->(b)""", edges, "edges")
    print("Done.")


if __name__ == "__main__":
    main()
