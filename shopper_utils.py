"""
Shared helpers for the Online Shoppers Purchasing Intention project.

Both the training script (train_models.py) and the Streamlit app (app.py)
import from here, so the column handling stays identical at train time and
at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "Revenue"

# Continuous / count features -> log1p + standard scaling
NUMERIC_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Label-like features (integer codes or strings) -> one-hot encoding
CATEGORICAL_COLS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]

# Already 0/1 -> passed through unchanged
BINARY_COLS = ["Weekend"]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS

# Display name -> file name inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}


def _to_binary(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE, True/False, 1/0, yes/no ... to integer 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS).astype(int)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes of a raw dataframe (raw UCI file or uploaded test CSV)."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for col in BINARY_COLS:
        if col in out.columns:
            out[col] = _to_binary(out[col])
    if TARGET_COL in out.columns:
        out[TARGET_COL] = _to_binary(out[TARGET_COL])
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def missing_columns(df: pd.DataFrame, need_target: bool = True) -> list:
    required = FEATURE_COLS + ([TARGET_COL] if need_target else [])
    return [c for c in required if c not in df.columns]


def positive_scores(model, X) -> np.ndarray:
    """Score for the positive class, used for AUC.

    Uses predict_proba when the estimator has it, otherwise falls back to
    decision_function (our SVC is trained without probability calibration).
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(y_true, y_pred, y_score) -> dict:
    """The six metrics required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
