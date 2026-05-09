import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../data/models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "../../data/reports")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":     round(average_precision_score(y_true, y_prob), 4),
    }


def print_report(name: str, metrics: dict, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))


def plot_confusion_matrix(y_true, y_pred, model_name: str, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
    )
    ax.set_title(f"Confusion Matrix\n{model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")


def plot_roc_curves(results: dict, ax):
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc = data["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_feature_importance(model, feature_names: list, ax, top_n: int = 15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        ax.text(0.5, 0.5, "No feature importance available", ha="center")
        return

    idx = np.argsort(importances)[-top_n:]
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")


def evaluate_all(threshold: float = 0.5) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet")).squeeze()

    results = {}
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No trained models found. Run train.py first.")

    for mf in sorted(model_files):
        name = mf.replace(".pkl", "")
        model = joblib.load(os.path.join(MODELS_DIR, mf))

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_report(name, metrics, y_test, y_pred)

        results[name] = {"model": model, "y_true": y_test, "y_pred": y_pred,
                         "y_prob": y_prob, "metrics": metrics}

    # --- Comparison table ---
    print("\n=== FINAL MODEL COMPARISON ===")
    comparison = pd.DataFrame({n: d["metrics"] for n, d in results.items()}).T
    print(comparison.to_string())
    comparison.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    # --- Plots ---
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(7 * (n_models + 1), 12))

    # Confusion matrices (top row)
    for i, (name, data) in enumerate(results.items()):
        plot_confusion_matrix(data["y_true"], data["y_pred"], name, axes[0, i])

    # ROC curves (top right)
    plot_roc_curves(results, axes[0, n_models])

    # Feature importances (bottom row)
    feature_names = list(X_test.columns)
    for i, (name, data) in enumerate(results.items()):
        plot_feature_importance(data["model"], feature_names, axes[1, i])

    # Error analysis: false negatives are more costly than false positives in fraud
    print("\n=== ERROR ANALYSIS ===")
    for name, data in results.items():
        cm = confusion_matrix(data["y_true"], data["y_pred"])
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name}:")
        print(f"  True Positives  (caught fraud):     {tp:,}")
        print(f"  False Negatives (missed fraud):     {fn:,}  ← critical: undetected fraud")
        print(f"  False Positives (false alarm):      {fp:,}  ← cost: user friction")
        print(f"  True Negatives  (correct normal):   {tn:,}")

    # Hide unused subplot
    for j in range(n_models, axes[1].shape[0]):
        axes[1, j].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(REPORTS_DIR, "evaluation_plots.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nPlots saved to {fig_path}")

    # Save metrics as JSON
    json_results = {n: d["metrics"] for n, d in results.items()}
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    return results


if __name__ == "__main__":
    evaluate_all()
