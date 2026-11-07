"""
Train and evaluate five classifiers on the UCI Online Shoppers Purchasing
Intention dataset.

Run:
    python train_models.py

Outputs:
    data/train_data.csv        80 % stratified training split
    test_data.csv              20 % held-out test split (upload this in the app)
    model/<name>.joblib        full sklearn Pipeline (preprocessing + classifier)
    model/metrics.json         test-set metrics + best hyper-parameters
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shopper_utils import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MODEL_FILES,
    NUMERIC_COLS,
    TARGET_COL,
    clean_frame,
    evaluate,
    positive_scores,
)

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
SEED = 2611  # fixed seed so the split and results are reproducible
TEST_SIZE = 0.20


def build_preprocessor() -> ColumnTransformer:
    # Page counts, durations and PageValues are heavily right-skewed,
    # so they are log-compressed before scaling.
    numeric_pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )


# classifier + small hyper-parameter grid (tuned with 5-fold CV on F1,
# because the positive class is only ~15 % of sessions)
CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"clf__max_depth": [4, 6, 8, 12, None], "clf__min_samples_leaf": [1, 10, 30]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    ),
    "Support Vector Machine": (
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.01]},
    ),
}


def main() -> None:
    df = clean_frame(pd.read_csv(RAW_CSV))
    print(f"Loaded {df.shape[0]} rows x {len(FEATURE_COLS)} features")
    print("Class balance:", df[TARGET_COL].value_counts(normalize=True).round(3).to_dict())

    train_df, test_df = train_test_split(
        df[FEATURE_COLS + [TARGET_COL]],
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=SEED,
    )
    train_df.to_csv(ROOT / "data" / "train_data.csv", index=False)
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report = {"dataset": "UCI Online Shoppers Purchasing Intention", "seed": SEED,
              "n_train": len(train_df), "n_test": len(test_df), "models": {}}

    for name, (clf, grid) in CANDIDATES.items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        scores = evaluate(y_test, y_pred, positive_scores(best, X_test))
        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])

        report["models"][name] = {
            "metrics": {k: round(float(v), 4) for k, v in scores.items()},
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": round(float(search.best_score_), 4),
        }
        print(f"{name:<24} {time.time() - start:5.1f}s  "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    with open(MODEL_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    table = pd.DataFrame({m: r["metrics"] for m, r in report["models"].items()}).T
    print("\n" + table.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
