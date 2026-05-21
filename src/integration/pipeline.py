
"""Pipeline de integración: datos -> ML/DL -> NLP -> resultado final."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import joblib


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.ml.preprocess import load_processed, load_raw, preprocess
from src.nlp.nlp_component import NLPComponent
from src.search_csp.agent import FraudDetectionAgent
from src.config.settings import (
	DATASET_PATH,
	MODELS_DIR,
	REPORTS_DIR,
	ML_SAMPLE_SIZE,
	PREDICTION_THRESHOLD,
)


def load_data_for_inference() -> tuple[pd.DataFrame, pd.Series]:
	"""Carga datos procesados; si no existen, los genera desde data/financial_data.csv."""
	try:
		_, X_test, _, y_test, _ = load_processed()
		return X_test, y_test
	except FileNotFoundError:
		_, X_test, _, y_test, _ = preprocess(sample_size=ML_SAMPLE_SIZE, save=True)
		return X_test, y_test


def run_ml_branch(X_test: pd.DataFrame, nlp: NLPComponent, models_dir: Path) -> dict:
	"""Ejecuta rama ML (XGBoost) y genera explicación + resumen NLP."""
	ml_path = models_dir / "xgboost.pkl"
	if not ml_path.exists():
		raise FileNotFoundError(
			f"No se encontró el modelo ML en {ml_path}. "
			"Entrena primero con: python src/ml/train.py"
		)

	ml_model = joblib.load(ml_path)
	y_prob = ml_model.predict_proba(X_test)[:, 1]
	y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
	idx_fraude = np.where(y_pred == 1)[0]

	explanation = "No se detectaron fraudes en el set de prueba."
	if len(idx_fraude) > 0:
		explanation = nlp.explain_prediction(X_test.iloc[idx_fraude[0]], ml_model)

	summary = nlp.summarize_batch(X_test, y_pred, y_prob)
	return {
		"model": "xgboost",
		"predicted_fraud": int((y_pred == 1).sum()),
		"avg_risk": float(np.mean(y_prob)),
		"explanation": explanation,
		"summary": summary,
	}


def run_dl_branch(X_test: pd.DataFrame, nlp: NLPComponent, models_dir: Path, reports_dir: Path) -> dict:
	"""Ejecuta rama DL (modelo keras o fallback a reporte DL real) y genera resumen NLP."""
	dl_model_path = models_dir / "dl_final_model.keras"
	dl_examples_path = reports_dir / "dl_prediction_examples.csv"

	y_prob = None
	y_pred = None
	explanation = "No se detectaron fraudes en el set de prueba (DL)."

	if dl_model_path.exists():
		try:
			import tensorflow as tf  # type: ignore

			dl_model = tf.keras.models.load_model(dl_model_path)
			y_prob = dl_model.predict(X_test.values.astype("float32"), verbose=0).ravel()
			y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
		except ModuleNotFoundError:
			pass

	if y_prob is None or y_pred is None:
		if not dl_examples_path.exists():
			raise FileNotFoundError(
				f"No hay modelo DL ejecutable ni reporte {dl_examples_path}. "
				"Ejecuta: python src/dl/train_dl.py"
			)
		examples = pd.read_csv(dl_examples_path)
		y_prob = examples["fraud_probability"].to_numpy()
		y_pred = examples["predicted"].to_numpy().astype(int)

		feature_cols = [
			c
			for c in examples.columns
			if c not in ["actual", "predicted", "fraud_probability", "result_type"]
		]
		X_ref = examples[feature_cols]
	else:
		X_ref = X_test

	idx_fraude = np.where(y_pred == 1)[0]
	if len(idx_fraude) > 0:
		top_features = X_ref.iloc[idx_fraude[0]].abs().sort_values(ascending=False).head(3)
		explanation = "\n".join([f"- {f}: valor={v:.2f}" for f, v in top_features.items()])

	summary = nlp.summarize_batch(X_ref, y_pred, y_prob)
	return {
		"model": "deep_learning_mlp",
		"predicted_fraud": int((y_pred == 1).sum()),
		"avg_risk": float(np.mean(y_prob)),
		"explanation": explanation,
		"summary": summary,
	}


def run_astar_branch(sample_size: int = 5000) -> dict:
	"""Ejecuta A* (Módulo A) sobre una muestra de transacciones raw para
	identificar la transacción fraudulenta de mayor prioridad heurística."""
	try:
		df_raw = load_raw(sample_size=sample_size)
	except FileNotFoundError as exc:
		return {
			"executed": False,
			"reason": str(exc),
		}

	transactions = df_raw.to_dict("records")
	agent = FraudDetectionAgent(transactions)
	result = agent.search()

	if result is None:
		return {
			"executed": True,
			"sample_size": int(len(transactions)),
			"fraud_found": False,
			"top_fraud_transaction": None,
		}

	# Sanitiza tipos numpy a tipos nativos JSON-serializables
	clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in result.items()}
	return {
		"executed": True,
		"sample_size": int(len(transactions)),
		"fraud_found": True,
		"top_fraud_transaction": clean,
	}


def main() -> None:
	models_dir = MODELS_DIR
	reports_dir = REPORTS_DIR
	output_path = reports_dir / "integration_summary.json"

	X_test, y_test = load_data_for_inference()
	_ = y_test  # reservado para extensiones de evaluación cruzada

	nlp = NLPComponent(feature_names=X_test.columns.tolist())
	ml_result = run_ml_branch(X_test, nlp, models_dir)
	dl_result = run_dl_branch(X_test, nlp, models_dir, reports_dir)

	print("\n[Módulo A] Ejecutando A* sobre muestra de transacciones raw...")
	astar_result = run_astar_branch(sample_size=5000)
	if astar_result.get("fraud_found"):
		tx = astar_result["top_fraud_transaction"]
		print(
			f"  A* halló fraude prioritario: type={tx.get('type')} amount={tx.get('amount')} "
			f"nameOrig={tx.get('nameOrig')}"
		)
	elif astar_result.get("executed"):
		print("  A*: no se encontraron fraudes en la muestra.")
	else:
		print(f"  A* no se pudo ejecutar: {astar_result.get('reason')}")

	final_result = {
		"dataset": str(DATASET_PATH),
		"test_size": int(len(X_test)),
		"ml": ml_result,
		"dl": dl_result,
		"module_a_astar": astar_result,
		"conclusion": (
			"Pipeline integrado ejecutado: datos reales -> ML/DL -> NLP + A* -> resultado final."
		),
	}

	reports_dir.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(final_result, f, indent=2, ensure_ascii=False)

	print("\n=== PIPELINE INTEGRADO COMPLETO ===")
	print(final_result["conclusion"])
	print(f"\n[ML] Fraudes predichos: {ml_result['predicted_fraud']} | Riesgo medio: {ml_result['avg_risk']:.2%}")
	print("[ML] Ejemplo de explicación:")
	print(ml_result["explanation"])
	print(f"\n[DL] Fraudes predichos: {dl_result['predicted_fraud']} | Riesgo medio: {dl_result['avg_risk']:.2%}")
	print("[DL] Ejemplo de explicación:")
	print(dl_result["explanation"])
	print(f"\nResultado guardado en: {output_path}")


if __name__ == "__main__":
	main()
