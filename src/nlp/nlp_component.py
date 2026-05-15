
"""
nlp_component.py
Componente de procesamiento de lenguaje natural para el proyecto.
Enfocado en explicabilidad y resumen de riesgo en transacciones bancarias.
"""

import numpy as np
import pandas as pd

class NLPComponent:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def explain_prediction(self, transaction: pd.Series, model, top_n: int = 3) -> str:
        """
        Genera una explicación textual de por qué una transacción fue clasificada como fraude o no.
        Usa la importancia de los features (coeficientes o importancias del modelo).
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return "No se puede explicar la predicción para este modelo."

        # Seleccionar los features más influyentes para esta transacción
        values = transaction.values
        if self.feature_names is not None:
            features = self.feature_names
        else:
            features = transaction.index.tolist()
        scores = np.abs(values * importances)
        top_idx = np.argsort(scores)[-top_n:][::-1]
        top_features = [(features[i], values[i], importances[i]) for i in top_idx]

        explicacion = "\n".join([
            f"- {nombre}: valor={valor:.2f}, importancia={imp:.2f}"
            for nombre, valor, imp in top_features
        ])
        return (
            f"Explicación de la predicción:\n"
            f"Los siguientes atributos influyeron más en la decisión del modelo:\n"
            f"{explicacion}"
        )

    def summarize_batch(self, df: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray) -> str:
        """
        Genera un resumen textual de un lote de transacciones clasificadas.
        """
        n = len(df)
        n_fraude = (y_pred == 1).sum()
        n_normal = n - n_fraude
        riesgo_medio = np.mean(y_prob)
        resumen = (
            f"Se analizaron {n} transacciones.\n"
            f"- {n_fraude} fueron clasificadas como FRAUDE.\n"
            f"- {n_normal} como normales.\n"
            f"El riesgo promedio estimado fue de {riesgo_medio:.2%}."
        )
        return resumen
