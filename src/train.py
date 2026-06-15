# src/train.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data import PROC_DIR, TARGET, NUM_COLS, CAT_COLS  # for consistency
from src.features import build_preprocessor

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _load_train_val():
    X_train = pd.read_csv(PROC_DIR / "X_train.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").squeeze("columns")
    X_val = pd.read_csv(PROC_DIR / "X_val.csv")
    y_val = pd.read_csv(PROC_DIR / "y_val.csv").squeeze("columns")
    return X_train, y_train, X_val, y_val

def _pick_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=200, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced"
        )
    raise ValueError("Unknown model. Use 'logreg' or 'rf'.")

def train(model: str = "logreg") -> str:
    X_train, y_train, X_val, y_val = _load_train_val()
    pre = build_preprocessor()
    clf = _pick_model(model)

    pipe = Pipeline([("pre", pre), ("model", clf)])

    # Cross-validated ROC-AUC on train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC ({model}): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # Fit & validate
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Validation ROC-AUC: {val_auc:.3f}")

    run_id = f"{model}_seed42"
    joblib.dump(pipe, MODELS_DIR / f"heart_model_{run_id}.pkl")
    (REPORTS_DIR / f"metrics_{run_id}.json").write_text(
        json.dumps(
            {"cv_auc_mean": float(cv_auc.mean()),
             "cv_auc_std": float(cv_auc.std()),
             "val_auc": float(val_auc)},
            indent=2
        )
    )
    print(f"💾 Saved: models/heart_model_{run_id}.pkl")
    print(f"📝 Saved: reports/metrics_{run_id}.json")
    return run_id

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    args = ap.parse_args()
    train(args.model)
