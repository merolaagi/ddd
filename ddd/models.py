"""Model adapters. Every adapter answers: given train (smiles, y) and test smiles,
return predictions (probabilities for cls, values for reg).

- rf, logreg      : local fingerprint baselines (free, seconds)
- ollama:<model>  : zero-shot local LLM via Ollama (free, slower)
"""
from __future__ import annotations

import json
import re

import httpx
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from .chem import featurize, describe
from .config import OLLAMA_URL

BASELINES = ("rf", "logreg")


def make_sklearn(name: str, kind: str):
    if name == "rf":
        if kind == "cls":
            return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0,
                                          class_weight="balanced_subsample", min_samples_leaf=1)
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    if name == "logreg":
        if kind == "cls":
            return LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
        return Ridge(alpha=10.0)
    raise ValueError(f"unknown baseline {name}")


def fit_predict_sklearn(name, kind, X_train, y_train, X_test):
    m = make_sklearn(name, kind)
    m.fit(X_train, y_train)
    if kind == "cls":
        return m, m.predict_proba(X_test)[:, 1]
    return m, m.predict(X_test)


def ollama_models() -> list[str]:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")


def _ask_ollama(model: str, prompt: str) -> str:
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0},
    }, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")


def _parse_value(text: str):
    try:
        obj = json.loads(text)
        for k in ("value", "probability", "answer", "prediction"):
            if k in obj:
                return float(obj[k])
    except Exception:
        pass
    m = _NUM.search(text or "")
    return float(m.group()) if m else None


def llm_predict(model: str, task, smiles_list, on_progress=None):
    preds = []
    for i, s in enumerate(smiles_list):
        d = describe(s)
        facts = {k: d[k] for k in ("formula", "mol_weight", "logp", "tpsa", "hbd", "hba") if k in d}
        if task.kind == "cls":
            ask = ('Reply with JSON {"value": p} where p is your probability (0 to 1) that the answer is yes.')
        else:
            ask = 'Reply with JSON {"value": x} where x is your numeric prediction.'
        prompt = (f"You are a medicinal chemist.\nQuestion: {task.question}\n"
                  f"Molecule SMILES: {s}\nComputed properties: {json.dumps(facts)}\n{ask}")
        v = None
        try:
            v = _parse_value(_ask_ollama(model, prompt))
        except Exception:
            v = None
        if v is None:
            v = 0.5 if task.kind == "cls" else float("nan")
        if task.kind == "cls":
            v = min(max(v, 0.0), 1.0)
        preds.append(v)
        if on_progress:
            on_progress(i + 1, len(smiles_list))
    return np.array(preds, dtype=float)


def available_models():
    return list(BASELINES) + [f"ollama:{m}" for m in ollama_models()]


__all__ = ["featurize", "fit_predict_sklearn", "llm_predict", "available_models", "BASELINES"]
