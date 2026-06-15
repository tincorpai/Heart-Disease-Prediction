

"""
src/data.py
-----------
Handles loading, cleaning, and splitting the Heart Disease dataset
without internet download dependency.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
RAW_CSV = Path("data/raw/data.csv")
PROC_DIR = Path("data/processed")

# Expected columns for validation
EXPECTED_COLS = {
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
}

# -----------------------------------------------------------
# Helper function for headerless UCI datasets
# -----------------------------------------------------------
def fix_headerless_uci_dataset(csv_path="data/raw/data.csv"):
    """
    Fixes a UCI Heart Disease dataset that has no headers
    and multi-class 'num' column (0–4) as target.
    Converts 'num' into binary 'target' and saves it back.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Read without header
    df = pd.read_csv(csv_path, header=None)

    # Assign proper columns
    df.columns = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","num"
    ]

    # Convert num→target and drop num
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)

    # Save back
    df.to_csv(csv_path, index=False)
    print(f"✅ Fixed headerless dataset saved to: {csv_path}")
    return df


# -----------------------------------------------------------
# Main function to load dataset
# -----------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Loads and validates the Heart Disease dataset.
    Automatically fixes headerless version if detected.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"❌ Dataset not found at {RAW_CSV}. "
            "Please upload heart.csv to 'data/raw/'."
        )

    # Try reading normally
    try:
        df = pd.read_csv(RAW_CSV)
    except Exception:
        df = pd.read_csv(RAW_CSV, header=None)

    # Detect headerless dataset (columns are numeric)
    if all(str(c).isdigit() for c in df.columns):
        print("⚠️ Detected headerless dataset — applying fix.")
        df = fix_headerless_uci_dataset(str(RAW_CSV))

    # Verify correct columns
    missing = EXPECTED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Clean duplicates and shuffle
    df = df.drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)

    print("✅ Dataset loaded successfully.")
    return df


# -----------------------------------------------------------
# Split data into train/validation/test
# -----------------------------------------------------------
def split_save(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=42):
    """
    Split the dataset into train, validation, and test sets
    and save them to 'data/processed/'.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split train+val vs test
    X_trv, X_test, y_trv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Split train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trv, y_trv, test_size=val_size, stratify=y_trv, random_state=seed
    )

    # Save splits
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROC_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROC_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROC_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROC_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROC_DIR / "y_test.csv", index=False)

    print("✅ Train/Validation/Test splits saved in data/processed/")


if __name__ == "__main__":
    df = load_data()
    split_save(df)
