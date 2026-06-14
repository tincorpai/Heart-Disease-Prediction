from pathlib import Path
import sys
import os
PROJECT_ROOT = "/content/drive/MyDrive/Dr. Taiwo famuyiwa - Data Science & Biostatistics Portfolio/Machine Learning Projects/Heart-Disease-Prediction"
sys.path.append(PROJECT_ROOT)


from src.serve import load_model, FEATURES_ORDER, predict_proba, predict_label
import streamlit as st
import pandas as pd
import json
# Adjust this path to your real project root


st.set_page_config(page_title="Heart Disease Risk Demo", layout="centered")
st.title("❤️ Heart Disease Risk — Demo")

REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")
BEST_PATH   = REPORTS_DIR / "best_model.json"

if BEST_PATH.exists():
    best = json.loads(BEST_PATH.read_text())
    RUN_ID = best["best_run_id"]
else:
    candidates = sorted(MODELS_DIR.glob("heart_model_*.pkl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        st.error("No model artifacts found.")
        st.stop()
    RUN_ID = candidates[0].stem.replace("heart_model_", "")

model = load_model(RUN_ID)
st.caption(f"Using model run: {RUN_ID}")

# UI inputs
cols = st.columns(2)
inputs = {}
defaults = {"age":57,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":1,"ca":0.0,"thal":2}
for i, feat in enumerate(FEATURES_ORDER):
    with cols[i % 2]:
        if feat in ["sex","cp","fbs","restecg","exang","slope","thal"]:
            inputs[feat] = st.number_input(feat, value=int(defaults.get(feat, 0)))
        else:
            inputs[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)))

thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    proba = predict_proba(model, df)[0]
    yhat  = int(proba >= thr)
    st.metric("Predicted probability", f"{proba:.3f}")
    st.metric("Predicted label", "Disease" if yhat==1 else "No Disease")
    st.caption("Adjust the threshold to explore sensitivity/specificity tradeoffs.")
