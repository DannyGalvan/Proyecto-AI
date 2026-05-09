import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from xgboost import XGBClassifier

from preprocess import load_processed, preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../data/models")

# Two models for comparison:
#   - Logistic Regression: linear baseline, fast, interpretable
#   - XGBoost: gradient boosting, handles non-linearities, state-of-the-art for tabular fraud detection
MODELS = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=None,  # set dynamically based on class ratio
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

CV_SCORERS = {
    "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score),
    "precision": "precision",
    "recall": "recall",
}


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos


def cross_validate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> dict:
    print(f"\n--- Cross-validating: {name} ({n_splits}-fold StratifiedKFold) ---")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    t0 = time.time()
    results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=CV_SCORERS, n_jobs=1
    )
    elapsed = time.time() - t0

    summary = {metric: results[f"test_{metric}"].mean() for metric in CV_SCORERS}
    summary["cv_time_s"] = round(elapsed, 1)

    for metric, val in summary.items():
        print(f"  {metric:<15}: {val:.4f}")
    return summary


def train_final(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    print(f"\n--- Training final model: {name} ---")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def run_training(sample_size: int | None = None, use_cached: bool = True) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        if not use_cached:
            raise FileNotFoundError
        X_train, X_test, y_train, y_test = load_processed()[:4]
        print("Loaded preprocessed data from cache.")
    except (FileNotFoundError, Exception):
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, _ = preprocess(sample_size=sample_size)

    # Set XGBoost scale_pos_weight dynamically to handle class imbalance
    spw = compute_scale_pos_weight(y_train)
    print(f"\nClass imbalance ratio (neg/pos): {spw:.1f}x")
    MODELS["xgboost"].set_params(scale_pos_weight=spw)

    cv_results = {}
    trained_models = {}

    for name, model in MODELS.items():
        cv_results[name] = cross_validate_model(name, model, X_train, y_train)
        trained_models[name] = train_final(name, model, X_train, y_train)
        joblib.dump(trained_models[name], os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"  Saved to data/models/{name}.pkl")

    # Summary table
    print("\n=== Cross-Validation Summary ===")
    summary_df = pd.DataFrame(cv_results).T
    print(summary_df.to_string())

    return {"cv_results": cv_results, "models": trained_models,
            "X_test": X_test, "y_test": y_test}


if __name__ == "__main__":
    run_training(sample_size=200_000, use_cached=False)
