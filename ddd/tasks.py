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
    category: str = ""   # Disease biology | Binding & activity | ADME | Safety | Clinical
    phase: str = ""      # one of PHASES
    source: str = "MoleculeNet"
    metric: str = ""     # override: auroc | rmse | mae | spearman
    sep: str = ","

    @property
    def primary_metric(self) -> str:
        if self.metric:
            return self.metric
        return "auroc" if self.kind == "cls" else "rmse"

    @property
    def higher_is_better(self) -> bool:
        return self.primary_metric in ("auroc", "auprc", "spearman", "r2")

    def to_dict(self):
        d = asdict(self)
        d["primary_metric"] = self.primary_metric
        d["higher_is_better"] = self.higher_is_better
        return d


PHASES = ["Target discovery", "Hit identification", "Hit-to-lead", "Lead optimisation: ADME",
          "Lead optimisation: safety", "Clinical development"]
CATEGORIES = ["Disease biology", "Binding & activity", "ADME", "Safety", "Clinical"]

DV = "https://dataverse.harvard.edu/api/access/datafile/"


def _tdc(tid, dv_id, name, kind, category, phase, question, caveat, metric=""):
    return Task(tid, name, category, kind, DV + str(dv_id), "Drug", "Y", "scaffold", question, caveat,
                category=category, phase=phase, source="TDC", metric=metric, sep="\t")


_SCAFFOLD_NOTE = "Re-split here with a deterministic scaffold split, so scores aren't comparable to TDC's 5-seed leaderboard."

TASKS: dict[str, Task] = {t.id: t for t in [
    Task("bbbp", "Blood-brain barrier penetration", "ADMET", "cls", DC + "BBBP.csv",
         "smiles", "p_np", "scaffold",
         "Does this molecule penetrate the blood-brain barrier?",
         "Labels mix sources and thresholds; some SMILES are invalid or duplicated.",
         category="ADME", phase="Lead optimisation: ADME"),
    Task("esol", "Aqueous solubility (ESOL)", "ADMET", "reg", DC + "delaney-processed.csv",
         "smiles", "measured log solubility in mols per litre", "scaffold",
         "Predict log aqueous solubility (mol/L).",
         "Only ~1.1k molecules; small and chemically narrow.",
         category="ADME", phase="Lead optimisation: ADME"),
    Task("lipo", "Lipophilicity (logD7.4)", "ADMET", "reg", DC + "Lipophilicity.csv",
         "smiles", "exp", "scaffold",
         "Predict octanol/water logD at pH 7.4.",
         "Single-assay AstraZeneca set; limited chemotype diversity.",
         category="ADME", phase="Lead optimisation: ADME"),
    Task("clintox", "Clinical-trial toxicity failure", "Clinical", "cls", DC + "clintox.csv.gz",
         "smiles", "CT_TOX", "scaffold",
         "Did this drug fail clinical trials for toxicity reasons?",
         "Very imbalanced (~8% positive); FDA-approved vs failed is confounded by era.",
         category="Clinical", phase="Clinical development"),
    Task("bace", "BACE-1 inhibition", "Binding", "cls", DC + "bace.csv",
         "mol", "Class", "scaffold",
         "Is this molecule an active BACE-1 inhibitor (IC50 below 100 nM)?",
         "Single target, congeneric series; scaffold leakage is common.",
         category="Binding & activity", phase="Hit-to-lead"),
    Task("hiv", "HIV replication inhibition", "Disease biology", "cls", DC + "HIV.csv",
         "smiles", "HIV_active", "scaffold",
         "Does this molecule inhibit HIV replication?",
         "~3.5% actives; 'confirmed moderately active' merged with active.",
         category="Disease biology", phase="Hit identification"),
    Task("ogbg-molhiv", "HIV (OGB official split)", "Disease biology", "cls", None,
         "smiles", "HIV_active", "ogb",
         "Does this molecule inhibit HIV replication?",
         "Same source as HIV; included to compare against the public OGB leaderboard.",
         category="Disease biology", phase="Hit identification"),

    _tdc("tdc_caco2", 4259569, "Caco-2 permeability (Wang)", "reg", "ADME", "Lead optimisation: ADME",
         "Predict Caco-2 apparent permeability (log Papp, cm/s).",
         "~900 molecules from one lab assay. " + _SCAFFOLD_NOTE, metric="mae"),
    _tdc("tdc_solubility", 4259610, "Aqueous solubility (AqSolDB)", "reg", "ADME", "Lead optimisation: ADME",
         "Predict log aqueous solubility (log mol/L).",
         "Aggregates 9 sources with differing protocols. " + _SCAFFOLD_NOTE, metric="mae"),
    _tdc("tdc_bbb", 4259566, "Blood-brain barrier (Martins)", "cls", "ADME", "Lead optimisation: ADME",
         "Does this molecule cross the blood-brain barrier?",
         "Class-imbalanced toward BBB+ . " + _SCAFFOLD_NOTE),
    _tdc("tdc_bioav", 4259567, "Oral bioavailability (Ma)", "cls", "ADME", "Lead optimisation: ADME",
         "Is this molecule orally bioavailable (F >= 20%)?",
         "Only ~640 molecules; noisy human endpoint. " + _SCAFFOLD_NOTE),
    _tdc("tdc_pgp", 4259597, "P-glycoprotein inhibition (Broccatelli)", "cls", "ADME", "Lead optimisation: ADME",
         "Does this molecule inhibit P-glycoprotein?",
         "Literature-aggregated actives/inactives. " + _SCAFFOLD_NOTE),
    _tdc("tdc_ppbr", 6413140, "Plasma protein binding (AZ)", "reg", "ADME", "Lead optimisation: ADME",
         "Predict percent plasma protein binding.",
         "Species/assay variation; bounded 0-100 skews error. " + _SCAFFOLD_NOTE, metric="mae"),
    _tdc("tdc_cyp3a4", 4259582, "CYP3A4 inhibition (Veith)", "cls", "ADME", "Lead optimisation: ADME",
         "Does this molecule inhibit CYP3A4?",
         "PubChem qHTS; actives/inactives thresholded. " + _SCAFFOLD_NOTE),
    _tdc("tdc_cyp2d6", 4259580, "CYP2D6 inhibition (Veith)", "cls", "ADME", "Lead optimisation: ADME",
         "Does this molecule inhibit CYP2D6?",
         "qHTS assay; strong class imbalance. " + _SCAFFOLD_NOTE),
    _tdc("tdc_cyp2c9", 4259577, "CYP2C9 inhibition (Veith)", "cls", "ADME", "Lead optimisation: ADME",
         "Does this molecule inhibit CYP2C9?",
         "qHTS assay; strong class imbalance. " + _SCAFFOLD_NOTE),
    _tdc("tdc_herg", 4259588, "hERG blockade", "cls", "Safety", "Lead optimisation: safety",
         "Does this molecule block the hERG channel (cardiotoxicity risk)?",
         "Small, heterogeneous actives; threshold choice matters. " + _SCAFFOLD_NOTE),
    _tdc("tdc_ames", 4259564, "Ames mutagenicity", "cls", "Safety", "Lead optimisation: safety",
         "Is this molecule mutagenic in the Ames test?",
         "Strain/protocol variation across sources. " + _SCAFFOLD_NOTE),
    _tdc("tdc_dili", 4259585, "Drug-induced liver injury", "cls", "Safety", "Lead optimisation: safety",
         "Is this drug associated with liver injury?",
         "Only ~475 drugs; label definitions are debated. " + _SCAFFOLD_NOTE),
    _tdc("tdc_halflife", 4266799, "Half-life (Obach)", "reg", "ADME", "Lead optimisation: ADME",
         "Predict drug half-life (hours).",
         "~670 drugs spanning orders of magnitude. " + _SCAFFOLD_NOTE, metric="spearman"),
    _tdc("tdc_clearance_hep", 4266187, "Hepatocyte clearance (AZ)", "reg", "ADME", "Lead optimisation: ADME",
         "Predict intrinsic hepatocyte clearance.",
         "Assay noise is high at the extremes. " + _SCAFFOLD_NOTE, metric="spearman"),
]}


def _cache_path(task: Task):
    return DATA_DIR / f"{task.id}.csv"


def load_frame(task: Task) -> pd.DataFrame:
    """Return DataFrame with columns smiles, y."""
    p = _cache_path(task)
    if task.split == "ogb":
        return _load_ogb(task)
    if not p.exists():
        r = httpx.get(task.url, timeout=180, follow_redirects=True)
        r.raise_for_status()
        raw = r.content
        if task.url.endswith(".gz"):
            raw = gzip.decompress(raw)
        df = pd.read_csv(io.BytesIO(raw), sep=task.sep)
        cols = {c.lower(): c for c in df.columns}
        sc = cols.get(task.smiles_col.lower(), task.smiles_col)
        lc = cols.get(task.label_col.lower(), task.label_col)
        df = df[[sc, lc]].rename(columns={sc: "smiles", lc: "y"})
        df = df.dropna().drop_duplicates("smiles")
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
