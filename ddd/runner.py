"""Run one (task, model) evaluation end to end and record it."""
from __future__ import annotations

import math
import time

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from . import store
from .chem import featurize, scaffold_split
from .config import MODEL_DIR
from .models import BASELINES, fit_predict_sklearn, llm_predict
from .tasks import load_frame, task_or_404


def metrics(kind, y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if kind == "cls":
        if len(np.unique(y)) < 2:
            return {"auroc": None, "auprc": None, "n_scored": int(len(y))}
        return {"auroc": round(float(roc_auc_score(y, p)), 4),
                "auprc": round(float(average_precision_score(y, p)), 4),
                "n_scored": int(len(y))}
    return {"rmse": round(math.sqrt(mean_squared_error(y, p)), 4),
            "mae": round(float(mean_absolute_error(y, p)), 4),
            "r2": round(float(r2_score(y, p)), 4),
            "n_scored": int(len(y))}


def get_split(task, df):
    if task.split == "ogb":
        tr = np.where(df["split"] == "train")[0]
        va = np.where(df["split"] == "valid")[0]
        te = np.where(df["split"] == "test")[0]
        return tr, va, te
    return scaffold_split(df["smiles"].tolist())


def run(task_id: str, model: str, sample: int | None = None, on_progress=None) -> dict:
    t0 = time.time()
    task = task_or_404(task_id)
    df = load_frame(task).reset_index(drop=True)
    tr, va, te = get_split(task, df)
    test = df.iloc[te]
    note = ""

    if model in BASELINES:
        X, keep = featurize(df["smiles"].tolist())
        idx_map = -np.ones(len(df), dtype=int)
        idx_map[keep] = np.arange(keep.sum())
        tr_k = idx_map[tr][idx_map[tr] >= 0]
        te_k_mask = idx_map[te] >= 0
        te_k = idx_map[te][te_k_mask]
        y = df["y"].to_numpy()
        y_k = y[keep]
        fitted, preds = fit_predict_sklearn(model, task.kind, X[tr_k], y_k[tr_k], X[te_k])
        y_test = y_k[te_k]
        joblib.dump(fitted, MODEL_DIR / f"{task.id}__{model}.joblib")
        n_train = len(tr_k)
    elif model.startswith("ollama:"):
        name = model.split(":", 1)[1]
        n = min(sample or 100, len(test))
        test = test.sample(n=n, random_state=0) if n < len(test) else test
        preds = llm_predict(name, task, test["smiles"].tolist(), on_progress)
        y_test = test["y"].to_numpy()
        n_train = 0
        note = f"zero-shot on {n} sampled test molecules"
    else:
        raise ValueError(f"unknown model '{model}'")

    m = metrics(task.kind, y_test, preds)
    score = m.get(task.primary_metric)
    extra = {**m, "seconds": round(time.time() - t0, 1), "note": note}
    rid = store.save(task.id, model, task.split, task.primary_metric, score,
                     int(n_train), int(len(y_test)), extra)
    return {"id": rid, "task": task.id, "model": model, "split": task.split,
            "metric": task.primary_metric, "score": score, **extra}


def predict(task_id: str, smiles: list[str], model: str = "rf") -> list[dict]:
    """Predict with a baseline trained by a previous run (trains one if missing)."""
    task = task_or_404(task_id)
    path = MODEL_DIR / f"{task.id}__{model}.joblib"
    if not path.exists():
        run(task_id, model)
    est = joblib.load(path)
    X, keep = featurize(smiles)
    out, j = [], 0
    for s, k in zip(smiles, keep):
        if not k:
            out.append({"smiles": s, "valid": False})
            continue
        x = X[j:j + 1]
        j += 1
        v = float(est.predict_proba(x)[0, 1]) if task.kind == "cls" else float(est.predict(x)[0])
        out.append({"smiles": s, "valid": True, "prediction": round(v, 4),
                    "meaning": "probability of yes" if task.kind == "cls" else task.label_col})
    return out
