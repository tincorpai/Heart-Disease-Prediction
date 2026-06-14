from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Optional
import pandas as pd
import joblib
from pathlib import Path
import json

from src.serve import FEATURES_ORDER, DTYPES, predict_proba, predict_label, load_model

app = FastAPI(title="Heart Disease Risk API", version="1.0")

# Load best model by default at startup (from pointer file)
REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")
BEST_PATH   = REPORTS_DIR / "best_model.json"

if BEST_PATH.exists():
    best = json.loads(BEST_PATH.read_text())
    RUN_ID = best["best_run_id"]
else:
    # fallback: latest pkl
    candidates = sorted(MODELS_DIR.glob("heart_model_*.pkl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No model artifacts found.")
    RUN_ID = candidates[0].stem.replace("heart_model_", "")

MODEL = load_model(RUN_ID)

class Instance(BaseModel):
    # features must be present (dict-like)
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int

class BatchRequest(BaseModel):
    instances: List[Instance]
    threshold: Optional[float] = 0.5

@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID, "features": FEATURES_ORDER}

@app.post("/predict_proba")
def predict_proba_endpoint(payload: BatchRequest):
    df = pd.DataFrame([i.dict() for i in payload.instances])
    try:
        p = predict_proba(MODEL, df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"run_id": RUN_ID, "probabilities": p.tolist()}

@app.post("/predict")
def predict_endpoint(payload: BatchRequest):
    df = pd.DataFrame([i.dict() for i in payload.instances])
    try:
        yhat = predict_label(MODEL, df, threshold=payload.threshold or 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"run_id": RUN_ID, "threshold": payload.threshold or 0.5, "predictions": yhat.tolist()}
