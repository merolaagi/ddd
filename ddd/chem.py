"""RDKit featurization, scaffold split and molecule description."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def mol(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def featurize(smiles_list):
    """Morgan fingerprints. Returns (X, keep_mask)."""
    fps, keep = [], []
    for s in smiles_list:
        m = mol(s)
        if m is None:
            keep.append(False)
            continue
        fps.append(_GEN.GetFingerprintAsNumPy(m).astype(np.uint8))
        keep.append(True)
    X = np.vstack(fps) if fps else np.zeros((0, 2048), dtype=np.uint8)
    return X, np.array(keep, dtype=bool)


def scaffold(smiles: str) -> str:
    m = mol(smiles)
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles_list, frac=(0.8, 0.1, 0.1)):
    """Deterministic Bemis-Murcko split: biggest scaffold groups go to train,
    rare scaffolds end up in valid/test, so test chemotypes are unseen."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles_list):
        groups[scaffold(s)].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles_list)
    cut_train, cut_valid = frac[0] * n, (frac[0] + frac[1]) * n
    train, valid, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= cut_train:
            train += g
        elif len(train) + len(valid) + len(g) <= cut_valid:
            valid += g
        else:
            test += g
    return np.array(train), np.array(valid), np.array(test)


def describe(smiles: str) -> dict:
    m = mol(smiles)
    if m is None:
        return {"valid": False, "smiles": smiles}
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    hbd = rdMolDescriptors.CalcNumHBD(m)
    hba = rdMolDescriptors.CalcNumHBA(m)
    tpsa = rdMolDescriptors.CalcTPSA(m)
    rot = rdMolDescriptors.CalcNumRotatableBonds(m)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "valid": True,
        "smiles": Chem.MolToSmiles(m),
        "formula": rdMolDescriptors.CalcMolFormula(m),
        "mol_weight": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot,
        "rings": rdMolDescriptors.CalcNumRings(m),
        "qed": round(QED.qed(m), 3),
        "lipinski_violations": violations,
        "scaffold": scaffold(smiles),
    }
