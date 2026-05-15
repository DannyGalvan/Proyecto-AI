"""Entrenamiento del modelo Deep Learning de semana 4.

Reutiliza los datos preprocesados de semana 3 y guarda artefactos comparables
con el pipeline ML: modelo, metricas, curvas y ejemplos de aciertos/errores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dl.model import DLHyperParams, build_model
from src.ml.preprocess import load_processed, preprocess

MODELS_DIR = ROOT_DIR / "data" / "models"
REPORTS_DIR = ROOT_DIR / "data" / "reports"


def ensure_processed_data(sample_size: int | None, use_cached: bool):
    try:
        if not use_cached:
            raise FileNotFoundError
        return load_processed()
    except (FileNotFoundError, OSError):
        print("No se encontraron datos procesados; ejecutando preprocess de semana 3.")
        return preprocess(sample_size=sample_size)


def compute_class_weights(y_train: pd.Series) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int).to_numpy(),
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def evaluate_predictions(y_true, y_prob, threshold: float) -> tuple[dict, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    return (
        {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
            "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        },
        y_pred,
    )


def save_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    history[["loss", "val_loss"]].plot(ax=axes[0])
    axes[0].set_title("Loss de entrenamiento y validacion")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].grid(True, alpha=0.3)

    metric_cols = [c for c in ["auc", "val_auc", "recall", "val_recall"] if c in history]
    history[metric_cols].plot(ax=axes[1])
    axes[1].set_title("Metricas principales")
    axes[1].set_xlabel("Epoca")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusion - Deep Learning")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], ["Normal", "Fraude"])
    ax.set_yticks([0, 1], ["Normal", "Fraude"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_examples(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_per_group: int = 8,
) -> None:
    examples = X_test.copy()
    examples["actual"] = y_true.to_numpy()
    examples["predicted"] = y_pred
    examples["fraud_probability"] = y_prob
    examples["result_type"] = np.select(
        [
            (examples["actual"] == 1) & (examples["predicted"] == 1),
            (examples["actual"] == 0) & (examples["predicted"] == 0),
            (examples["actual"] == 0) & (examples["predicted"] == 1),
            (examples["actual"] == 1) & (examples["predicted"] == 0),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )

    selected = (
        examples.sort_values("fraud_probability", ascending=False)
        .groupby("result_type", group_keys=False)
        .head(n_per_group)
    )
    selected.to_csv(output_path, index=False)


def append_dl_to_comparison(metrics: dict, reports_dir: Path) -> None:
    comparison_path = reports_dir / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path, index_col=0)
    else:
        comparison = pd.DataFrame()

    comparison.loc["deep_learning_mlp", list(metrics.keys())] = list(metrics.values())
    comparison.to_csv(comparison_path)


def train_deep_learning(
    epochs: int = 50,
    batch_size: int = 256,
    sample_size: int | None = 200_000,
    validation_split: float = 0.2,
    threshold: float = 0.5,
    use_cached: bool = True,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler = ensure_processed_data(sample_size, use_cached)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model = build_model(X_train.shape[1], DLHyperParams())
    class_weights = compute_class_weights(y_train)

    checkpoint_path = MODELS_DIR / "dl_best_model.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("\n--- Entrenando Deep Learning MLP ---")
    print(f"Features: {X_train.shape[1]} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Class weights: {class_weights}")

    history_obj = model.fit(
        X_train.astype("float32"),
        y_train.astype("float32"),
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    history = pd.DataFrame(history_obj.history)
    history.index.name = "epoch"
    history_path = REPORTS_DIR / "dl_training_history.csv"
    history.to_csv(history_path)
    save_training_curves(history, REPORTS_DIR / "dl_training_curves.png")

    y_prob = model.predict(X_test.astype("float32"), batch_size=batch_size).ravel()
    metrics, y_pred = evaluate_predictions(y_test, y_prob, threshold)

    metrics_path = REPORTS_DIR / "dl_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix_plot(y_test, y_pred, REPORTS_DIR / "dl_confusion_matrix.png")
    save_prediction_examples(
        X_test,
        y_test,
        y_pred,
        y_prob,
        REPORTS_DIR / "dl_prediction_examples.csv",
    )
    append_dl_to_comparison(metrics, REPORTS_DIR)

    final_model_path = MODELS_DIR / "dl_final_model.keras"
    model.save(final_model_path)
    joblib.dump(scaler, MODELS_DIR / "scaler_dl.pkl")

    print("\n=== Deep Learning Metrics ===")
    for key, value in metrics.items():
        print(f"{key:<16}: {value}")
    print(f"\nModelo final: {final_model_path}")
    print(f"Mejor checkpoint: {checkpoint_path}")
    print(f"Reportes: {REPORTS_DIR}")

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "model_path": str(final_model_path),
        "reports_dir": str(REPORTS_DIR),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo DL de fraude.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Reprocesa datos desde data/raw en vez de usar data/processed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train_deep_learning(
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            validation_split=args.validation_split,
            threshold=args.threshold,
            use_cached=not args.no_cache,
        )
    except FileNotFoundError as exc:
        print("\nNo se pudo iniciar el entrenamiento DL.")
        print(exc)
        print("\nPasos esperados:")
        print("1. Crea data/raw/ en la raiz del proyecto.")
        print("2. Coloca ahi el CSV descargado desde Kaggle.")
        print("3. Ejecuta: python src/ml/preprocess.py")
        print("4. Ejecuta: python src/dl/train_dl.py")
        sys.exit(1)
