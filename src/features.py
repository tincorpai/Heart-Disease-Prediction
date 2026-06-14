"""
features.py
------------
Feature engineering module for Heart Disease Prediction.
Contains reusable functions for derived feature creation,
feature preprocessing, and feature name extraction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer


# ----------------------------------------------------------------
# 1️⃣ Derived Feature Creation
# ----------------------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically meaningful derived features.

    hr_reserve  = thalach / (220 - age)
    bp_per_age  = trestbps / age
    chol_per_age = chol / age
    age_bin     = 5 quantile bins of age
    """
    out = df.copy()

    # Avoid division by zero
    out["age"] = out["age"].clip(lower=1)
    out["trestbps"] = out["trestbps"].clip(lower=1)
    out["chol"] = out["chol"].clip(lower=1)
    out["thalach"] = out["thalach"].clip(lower=1)

    # 1️⃣ Heart Rate Reserve
    max_hr_pred = 220 - out["age"]
    out["hr_reserve"] = (out["thalach"] / max_hr_pred).clip(0, 3)

    # 2️⃣ BP per Age
    out["bp_per_age"] = (out["trestbps"] / out["age"]).clip(0, 5)

    # 3️⃣ Chol per Age
    out["chol_per_age"] = (out["chol"] / out["age"]).clip(0, 10)

    # 4️⃣ Age Bin (quantile-based)
    kbinner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    out["age_bin"] = kbinner.fit_transform(out[["age"]]).astype(int)

    return out


# ----------------------------------------------------------------
# 2️⃣ Feature Name Extraction Helper
# ----------------------------------------------------------------
def get_feature_names(preproc: Pipeline, num_cols: list[str], cat_cols: list[str],
                      engineered_num: list[str], engineered_cat: list[str]) -> list[str]:
    """
    Extracts human-readable feature names from a fitted preprocessing pipeline.
    Works with numeric + one-hot encoded categorical features.
    """
    derived_numeric = num_cols + engineered_num
    derived_categor = cat_cols + engineered_cat

    # Numeric pass-through features
    num_names = derived_numeric

    # Get one-hot feature names for categorical
    ohe = preproc.named_steps["cols"].named_transformers_["cat"].named_steps["onehot"]
    ohe_out = list(ohe.get_feature_names_out(derived_categor))

    return num_names + ohe_out


# ----------------------------------------------------------------
# 3️⃣ Preprocessor Builder (optional utility)
# ----------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols):
    """
    Build the full preprocessing pipeline with numeric/categorical transformations
    and derived feature generation.
    """
    ENGINEERED_NUM = ["hr_reserve", "bp_per_age", "chol_per_age"]
    ENGINEERED_CAT = ["age_bin"]

    num_cols_all = num_cols + ENGINEERED_NUM
    cat_cols_all = cat_cols + ENGINEERED_CAT

    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    deriver = FunctionTransformer(add_derived_features, feature_names_out="one-to-one", validate=False)

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols_all),
        ("cat", cat_pipe, cat_cols_all)
    ])

    full_pipeline = Pipeline([
        ("derive", deriver),
        ("cols", preprocessor)
    ])

    return full_pipeline
