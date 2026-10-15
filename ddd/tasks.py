"""Task registry. Each task = public dataset + split rule + primary metric.

v0 uses MoleculeNet CSVs (DeepChem mirror) with deterministic Bemis-Murcko
scaffold splits, plus OGB's ogbg-molhiv with its official split when the
optional `ogb` package is installed.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, asdict

import httpx
import numpy as np
import pandas as pd

from .config import DATA_DIR

DC = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"


@dataclass
class Task:
    id: str
    name: str
    domain: str
    kind: str            # "cls" or "reg"
    url: str | None
    smiles_col: str
    label_col: str
    split: str           # "scaffold" or "ogb"
    question: str        # natural-language prompt used for LLM adapters
    caveat: str          # O3DC-style known limitation

    @property
    def primary_metric(self) -> str:
        return "auroc" if self.kind == "cls" else "rmse"

    @property
    def higher_is_better(self) -> bool:
        return self.kind == "cls"

    def to_dict(self):
        d = asdict(self)
        d["primary_metric"] = self.primary_metric
        d["higher_is_better"] = self.higher_is_better
        return d


TASKS: dict[str, Task] = {t.id: t for t in [
    Task("bbbp", "Blood-brain barrier penetration", "ADMET", "cls", DC + "BBBP.csv",
         "smiles", "p_np", "scaffold",
         "Does this molecule penetrate the blood-brain barrier?",
         "Labels mix sources and thresholds; some SMILES are invalid or duplicated."),
    Task("esol", "Aqueous solubility (ESOL)", "ADMET", "reg", DC + "delaney-processed.csv",
         "smiles", "measured log solubility in mols per litre", "scaffold",
         "Predict log aqueous solubility (mol/L).",
         "Only ~1.1k molecules; small and chemically narrow."),
    Task("lipo", "Lipophilicity (logD7.4)", "ADMET", "reg", DC + "Lipophilicity.csv",
         "smiles", "exp", "scaffold",
         "Predict octanol/water logD at pH 7.4.",
         "Single-assay AstraZeneca set; limited chemotype diversity."),
    Task("clintox", "Clinical-trial toxicity failure", "Clinical", "cls", DC + "clintox.csv.gz",
         "smiles", "CT_TOX", "scaffold",
         "Did this drug fail clinical trials for toxicity reasons?",
         "Very imbalanced (~8% positive); FDA-approved vs failed is confounded by era."),
    Task("bace", "BACE-1 inhibition", "Binding", "cls", DC + "bace.csv",
         "mol", "Class", "scaffold",
         "Is this molecule an active BACE-1 inhibitor (IC50 below 100 nM)?",
         "Single target, congeneric series; scaffold leakage is common."),
    Task("hiv", "HIV replication inhibition", "Disease biology", "cls", DC + "HIV.csv",
         "smiles", "HIV_active", "scaffold",
         "Does this molecule inhibit HIV replication?",
         "~3.5% actives; 'confirmed moderately active' merged with active."),
    Task("ogbg-molhiv", "HIV (OGB official split)", "Disease biology", "cls", None,
         "smiles", "HIV_active", "ogb",
         "Does this molecule inhibit HIV replication?",
         "Same source as HIV; included to compare against the public OGB leaderboard."),
]}


def _cache_path(task: Task):
    return DATA_DIR / f"{task.id}.csv"


def load_frame(task: Task) -> pd.DataFrame:
    """Return DataFrame with columns smiles, y."""
    p = _cache_path(task)
    if task.split == "ogb":
        return _load_ogb(task)
    if not p.exists():
        r = httpx.get(task.url, timeout=120, follow_redirects=True)
        r.raise_for_status()
        raw = r.content
        if task.url.endswith(".gz"):
            raw = gzip.decompress(raw)
        df = pd.read_csv(io.BytesIO(raw))
        df = df[[task.smiles_col, task.label_col]].rename(
            columns={task.smiles_col: "smiles", task.label_col: "y"})
        df = df.dropna()
        df.to_csv(p, index=False)
    return pd.read_csv(p)


def _load_ogb(task: Task) -> pd.DataFrame:
    try:
        from ogb.graphproppred import GraphPropPredDataset
    except ImportError as e:
        raise RuntimeError("ogbg-molhiv needs the optional 'ogb' package: pip install ogb") from e
    root = DATA_DIR / "ogb"
    ds = GraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    split = ds.get_idx_split()
    mapping = root / "ogbg_molhiv" / "mapping" / "mol.csv.gz"
    m = pd.read_csv(mapping)
    df = pd.DataFrame({"smiles": m["smiles"], "y": m["HIV_active"]})
    df["split"] = "train"
    df.loc[split["valid"], "split"] = "valid"
    df.loc[split["test"], "split"] = "test"
    return df


def task_or_404(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Known: {', '.join(TASKS)}")
    return TASKS[task_id]


def available_tasks():
    out = []
    for t in TASKS.values():
        d = t.to_dict()
        d["cached"] = _cache_path(t).exists() or (t.split == "ogb" and (DATA_DIR / "ogb").exists())
        if t.split == "ogb":
            try:
                import ogb  # noqa: F401
                d["available"] = True
            except ImportError:
                d["available"] = False
        else:
            d["available"] = True
        out.append(d)
    return out


def class_balance(y) -> float:
    y = np.asarray(y)
    return float(y.mean()) if len(y) else 0.0
