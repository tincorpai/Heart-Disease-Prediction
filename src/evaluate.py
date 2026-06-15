"""
evaluate.py
------------
Evaluate trained models for Heart Disease Prediction.
Generates ROC, Precision-Recall, Confusion Matrix, and Calibration plots.

Usage (from terminal or Colab):
    python src/evaluate.py --run_id logreg_seed42
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.calibration import CalibrationDisplay

# Directories
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
PROC_DIR = Path("data/processed")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------

def load_splits():
    """Load test split for evaluation."""
    X_test = pd.read_csv(PROC_DIR / "X_test.csv")
    y_test = pd.read_csv(PROC_DIR / "y_test.csv").squeeze()
    return X_test, y_test


def evaluate(run_id: str):
    """
    Evaluate a trained model using stored pickle artifact.
    Args:
        run_id: Model identifier (e.g., 'logreg_seed42' or 'rf_seed42').
    """
    model_path = MODELS_DIR / f"heart_model_{run_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load model and test data
    model = joblib.load(model_path)
    X_test, y_test = load_splits()

    # Predict probabilities and labels
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # -----------------------------------------------------------------
    #  Metrics
    # -----------------------------------------------------------------
    roc_auc = auc(*roc_curve(y_test, proba)[:2][::-1])
    pr_auc = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    metrics_dict = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(report["accuracy"]),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1_score": float(report["1"]["f1-score"]),
    }

    # Save metrics
    metrics_path = REPORTS_DIR / f"test_metrics_{run_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")

    # -----------------------------------------------------------------
    #  Plots
    # -----------------------------------------------------------------
    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.savefig(REPORTS_DIR / f"roc_curve_{run_id}.png", bbox_inches="tight")
    plt.close()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(REPORTS_DIR / f"pr_curve_{run_id}.png", bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{run_id}.png", bbox_inches="tight")
    plt.close()

    # Calibration Curve
    plt.figure()
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10)
    plt.title("Calibration Curve")
    plt.savefig(REPORTS_DIR / f"calibration_curve_{run_id}.png", bbox_inches="tight")
    plt.close()

    print("📊 Evaluation plots saved in 'reports/' folder.")

    # -----------------------------------------------------------------
    #  Summary Output
    # -----------------------------------------------------------------
    print("=== Evaluation Summary ===")
    for k, v in metrics_dict.items():
        print(f"{k:<15}: {v:.3f}")

    return metrics_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained Heart Disease model.")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g., logreg_seed42")
    args = parser.parse_args()

    evaluate(args.run_id)
