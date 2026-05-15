"""Arquitectura Deep Learning para deteccion de fraude financiero.

El dataset de semana 3 ya entrega features tabulares escaladas y one-hot encoded,
por lo que se usa una red densa pequena en lugar de CNN/RNN. La regularizacion se
centra en dropout y L2 para reducir overfitting sobre patrones de fraude muy
marcados en una clase minoritaria.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DLHyperParams:
    """Hiperparametros principales de la red neuronal."""

    hidden_units: tuple[int, int, int] = (128, 64, 32)
    dropout_rates: tuple[float, float, float] = (0.35, 0.25, 0.15)
    learning_rate: float = 1e-3
    l2_strength: float = 1e-4


def build_model(input_dim: int, params: DLHyperParams | None = None):
    """Construye y compila un MLP binario en TensorFlow/Keras.

    Args:
        input_dim: numero de columnas/features del pipeline preprocesado.
        params: hiperparametros de arquitectura y optimizacion.

    Returns:
        Modelo Keras compilado con binary crossentropy, Adam y metricas utiles
        para clases desbalanceadas.
    """
    if input_dim <= 0:
        raise ValueError("input_dim debe ser mayor que 0")

    params = params or DLHyperParams()

    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.metrics import AUC, Precision, Recall
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    layers = [Input(shape=(input_dim,), name="fraud_features")]
    for idx, (units, dropout) in enumerate(
        zip(params.hidden_units, params.dropout_rates), start=1
    ):
        layers.extend(
            [
                Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(params.l2_strength),
                    name=f"dense_{idx}",
                ),
                BatchNormalization(name=f"batch_norm_{idx}"),
                Dropout(dropout, name=f"dropout_{idx}"),
            ]
        )

    layers.append(Dense(1, activation="sigmoid", name="fraud_probability"))

    model = Sequential(layers, name="fraud_mlp")
    model.compile(
        optimizer=Adam(learning_rate=params.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )
    return model
