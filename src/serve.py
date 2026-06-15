from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Default paths (override when importing if needed)
MODELS_DIR = Path("models")

# These should match Notebook 09 definitions
FEATURES_ORDER = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
DTYPES = {
    "age": float, "sex": int, "cp": int, "trestbps": float, "chol": float,
    "fbs": int, "restecg": int, "thalach": float, "exang": int,
    "oldpeak": float, "slope": int, "ca": float, "thal": int
}

def load_model(run_id: str) -> object:
    path = MODELS_DIR / f"heart_model_{run_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def validate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    df = df[FEATURES_ORDER].copy()
    for c, t in DTYPES.items():
        df[c] = df[c].astype(t)
    return df

def predict_proba(model_pipe, df: pd.DataFrame) -> np.ndarray:
    dfv = validate_input_frame(df)
    return model_pipe.predict_proba(dfv)[:, 1]

def predict_label(model_pipe, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    p = predict_proba(model_pipe, df)
    return (p >= threshold).astype(int)
