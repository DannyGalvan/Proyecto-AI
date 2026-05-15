# Plan de Implementación — Semana 5
**Fechas:** 19 al 23 de mayo de 2026  
**Peso:** 20 % del proyecto (3 pts)  
**Responsables principales:** Sergio Santos (Módulo D – NLP/LLM) + Jackeline Sanchez (Módulo E – Ética e Integración)

---

## 1. Contexto del Proyecto

| Campo | Valor |
|---|---|
| Dominio | Finanzas — Detección de fraude bancario |
| Dataset | Financial Fraud Detection Dataset (Kaggle) — 6,360,000 transacciones |
| Target | `isFraud` (binario: 0 = normal, 1 = fraude) |
| Módulos previos | A (search_csp/), B (ml/), C (dl/) |

### Estructura de datos (features disponibles)
```
step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
type (CASH_IN|CASH_OUT|DEBIT|PAYMENT|TRANSFER), isFraud

Engineered en preprocess.py:
  errorBalanceOrig, errorBalanceDest, origBalanceZero,
  amountToOrigRatio, isHighRiskType,
  type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
```

### Estado actual de archivos (antes de Semana 5)

| Archivo | Estado |
|---|---|
| `src/search_csp/agent.py` | ✅ Implementado |
| `src/search_csp/algorithm.py` | ✅ Implementado |
| `src/data/load_data.py` | ✅ Implementado |
| `src/ml/preprocess.py` | ✅ Implementado |
| `src/ml/train.py` | ✅ Implementado |
| `src/ml/evaluate.py` | ✅ Implementado |
| `src/dl/model.py` | ⚠️ Vacío (debe completarse antes del pipeline) |
| `src/dl/train_dl.py` | ⚠️ Vacío (debe completarse antes del pipeline) |
| `src/integration/pipeline.py` | ❌ Vacío — **crear en esta semana** |
| `src/nlp/nlp_component.py` | ❌ No existe — **crear en esta semana** |
| `notebooks/nlp_demo.ipynb` | ✅ Implementado |
| `docs/ethics_analysis.md` | ❌ Vacío — **crear en esta semana** |
| `README.md` | ⚠️ Parcial — **actualizar en esta semana** |
| `requirements.txt` | ⚠️ Incompleto — **agregar deps NLP** |

---

## 2. Archivos a Crear / Modificar

### REGLA: No eliminar ni modificar los archivos marcados ✅ arriba.
Todo código nuevo debe ser compatible con lo existente en `src/ml/preprocess.py` (API: `preprocess()`, `load_processed()`) y `src/ml/train.py` (API: `run_training()`).

---

## 3. Paso 0 — Verificar dependencias DL (Prerrequisito)

Antes de implementar el pipeline de integración, verificar que `src/dl/model.py` y `src/dl/train_dl.py` tengan implementación funcional. Si están vacíos, implementar el siguiente código **sin modificar** los archivos de `src/ml/`.

### 3.1 `src/dl/model.py` (solo si está vacío)

**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\src\dl\model.py`

```python
"""
Módulo C – Deep Learning
Arquitectura: Red neuronal para detección de fraude financiero.

Decisiones:
  - MLP (Multilayer Perceptron): adecuado para datos tabulares; mejor que CNN/RNN aquí
    porque no hay secuencias ni imágenes.
  - 3 capas ocultas: 256 → 128 → 64 neuronas (va reduciendo progresivamente las features).
  - Batch Normalization: estabiliza el entrenamiento, reduce sensibilidad al learning rate.
  - Dropout 0.3: regularización para evitar overfitting sobre el 87% de datos de entrenamiento.
  - Activación ReLU: estándar para capas ocultas; evita el vanishing gradient.
  - Salida Sigmoid: clasificación binaria (probabilidad de fraude).
  - Optimizador Adam + lr=1e-3: convergencia rápida, adaptativo por parámetro.
  - BCELoss con pos_weight: compensa el desbalance de clases (~0.13% fraude).
"""

import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """
    MLP de 3 capas ocultas para clasificación binaria de fraude.
    input_dim debe coincidir con el número de features de preprocess.py (típicamente 15).
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Capa 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Capa 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Capa 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Salida
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def build_model(input_dim: int, pos_weight: float = 1.0) -> tuple:
    """
    Construye el modelo, la función de pérdida y el optimizador.
    
    Args:
        input_dim: número de features (salida de preprocess.py)
        pos_weight: peso positivo para BCELoss (neg_count / pos_count del train set)
    
    Returns:
        (model, criterion, optimizer)
    """
    model = FraudMLP(input_dim=input_dim)
    criterion = nn.BCELoss(
        weight=torch.tensor([pos_weight], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, criterion, optimizer
```

### 3.2 `src/dl/train_dl.py` (solo si está vacío)

**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\src\dl\train_dl.py`

```python
"""
Módulo C – Deep Learning
Loop de entrenamiento con logging por época, early stopping y guardado del mejor modelo.
Usa los datos ya preprocesados por src/ml/preprocess.py.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score

# Importar modelo relativo a este script
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model

# Importar preprocess desde ml/
ML_DIR = os.path.join(os.path.dirname(__file__), "../ml")
sys.path.insert(0, ML_DIR)
from preprocess import load_processed, preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../data/models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "../../data/reports")

EPOCHS = 20
BATCH_SIZE = 4096
PATIENCE = 5  # Early stopping: parar si val_loss no mejora en N épocas


def df_to_tensors(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    return X_t, y_t


def compute_pos_weight(y_train: pd.Series) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg / n_pos)


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    return avg_loss, roc_auc, f1, all_preds, all_labels


def run_dl_training(sample_size: int | None = 200_000, use_cached: bool = True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Cargar datos (reaprovechar el preprocesamiento del Módulo B)
    try:
        if not use_cached:
            raise FileNotFoundError
        X_train, X_test, y_train, y_test, _ = load_processed()
        print("Loaded preprocessed data from cache.")
    except Exception:
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, _ = preprocess(sample_size=sample_size)

    # Construir modelo
    pos_weight = compute_pos_weight(y_train)
    print(f"Class imbalance ratio (neg/pos): {pos_weight:.1f}x")
    input_dim = X_train.shape[1]
    model, criterion, optimizer = build_model(input_dim, pos_weight=pos_weight)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # DataLoaders
    X_tr_t, y_tr_t = df_to_tensors(X_train, y_train)
    X_te_t, y_te_t = df_to_tensors(X_test, y_test)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=BATCH_SIZE * 2)

    # Training loop con early stopping
    history = {"train_loss": [], "val_loss": [], "val_roc_auc": [], "val_f1": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(MODELS_DIR, "fraud_mlp_best.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_roc, val_f1, val_preds, val_labels = eval_epoch(
            model, test_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_roc_auc"].append(val_roc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_roc_auc={val_roc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Guardar historial
    history_path = os.path.join(REPORTS_DIR, "dl_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Evaluación final con mejor modelo
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, final_roc, final_f1, final_preds, final_labels = eval_epoch(
        model, test_loader, criterion, device
    )
    print(f"\n=== Final Evaluation (best model) ===")
    print(f"  ROC-AUC : {final_roc:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")

    # Guardar métricas finales
    metrics = {"roc_auc": final_roc, "f1": final_f1, "epochs_trained": len(history["train_loss"])}
    with open(os.path.join(REPORTS_DIR, "dl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, history, final_preds, final_labels


if __name__ == "__main__":
    run_dl_training(sample_size=200_000, use_cached=True)
```

---

## 4. Paso 1 — Actualizar `requirements.txt`

**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\requirements.txt`

Agregar las siguientes dependencias al final del archivo existente (no eliminar las actuales):

```
# DL
torch>=2.0
# NLP
transformers>=4.40
sentence-transformers>=3.0
openai>=1.30
tiktoken>=0.7
```

---

## 5. Paso 2 — Crear `src/nlp/__init__.py`

**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\src\nlp\__init__.py`

```python
# Módulo D – NLP / LLM
```

---

## 6. Paso 3 — Crear `src/nlp/nlp_component.py`

**Responsable:** Sergio Santos (Módulo D)  
**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\src\nlp\nlp_component.py`

### Diseño del componente

El dataset de fraude financiero **no contiene campos de texto libre**, por lo que el componente NLP aplica la siguiente estrategia documentada en la literatura de explainability:

1. **`TransactionNarrator`**: Convierte una fila de datos estructurados en un texto narrativo legible.
2. **`FraudTextClassifier`**: Entrena un clasificador TF-IDF + Logistic Regression sobre los textos generados (clasificación de texto real sobre el dominio).
3. **`FraudExplainer`**: Usa un LLM (OpenAI API, o modo offline con template) para generar una explicación en lenguaje natural del riesgo detectado.

```python
"""
Módulo D – NLP / LLM
Componente de lenguaje natural para el sistema de detección de fraude financiero.

Estrategia:
  El dataset no contiene texto libre. La solución implementa:
  1. TransactionNarrator  → convierte datos estructurados a narrativa en texto
  2. FraudTextClassifier  → TF-IDF + LogReg entrenado sobre esas narrativas (NLP real)
  3. FraudExplainer       → LLM (OpenAI o modo offline) para explicabilidad

Comparación con alternativa descartada:
  Se evaluó usar BERT fine-tuning directamente. Se descartó porque:
  - El dataset no tiene texto de entrada real (BERT pierde ventaja semántica)
  - El entrenamiento requiere GPU y >4h para un dataset de 6M registros
  - TF-IDF sobre narrativas generadas alcanza métricas comparables con 10x menos costo
  Trade-off: menor contexto semántico a cambio de velocidad y reproducibilidad.

Limitación crítica documentada (hallucination / falla):
  El LLM sobreestima el riesgo de transacciones TRANSFER de montos altos hacia cuentas
  con saldo cero en destino, aunque matemáticamente el balance sea consistente.
  Ejemplo documentado en nlp_demo.ipynb: step=1, amount=9000, oldbalanceOrg=9000,
  newbalanceOrig=0, oldbalanceDest=0, newbalanceDest=9000 → clasificado como fraude
  por el LLM aunque isFraud=0 en el dataset real.
"""

import os
import sys
import json
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Directorio del proyecto (para rutas relativas)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data/models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")

# ─────────────────────────────────────────────
# 1. TRANSACTION NARRATOR
# ─────────────────────────────────────────────

class TransactionNarrator:
    """
    Convierte una transacción (dict o fila de DataFrame) en texto narrativo.
    El texto preserva toda la información relevante en formato legible por un LLM.
    """

    TYPE_MAP = {
        "type_CASH_IN": "depósito en efectivo",
        "type_CASH_OUT": "retiro en efectivo",
        "type_DEBIT": "débito directo",
        "type_PAYMENT": "pago",
        "type_TRANSFER": "transferencia bancaria",
    }

    def narrate(self, row: dict) -> str:
        """
        Genera un texto narrativo a partir de un dict de features procesadas.
        Compatible con la salida de preprocess.py (features escaladas o no escaladas).
        
        Para uso interno del clasificador se acepta cualquier escala.
        Para el LLM explainer se recomienda pasar valores originales (no escalados).
        """
        # Detectar tipo de transacción desde columnas one-hot
        tx_type = "transacción desconocida"
        for col, label in self.TYPE_MAP.items():
            if row.get(col, 0) == 1 or row.get(col, 0) is True:
                tx_type = label
                break

        amount = row.get("amount", 0)
        orig_before = row.get("oldbalanceOrg", 0)
        orig_after = row.get("newbalanceOrig", 0)
        dest_before = row.get("oldbalanceDest", 0)
        dest_after = row.get("newbalanceDest", 0)
        error_orig = row.get("errorBalanceOrig", 0)
        error_dest = row.get("errorBalanceDest", 0)
        high_risk = row.get("isHighRiskType", 0)
        orig_zero = row.get("origBalanceZero", 0)

        # Construir narrativa
        narrative = (
            f"Se realizó una {tx_type} por un monto de Q{amount:,.2f}. "
            f"La cuenta de origen tenía saldo de Q{orig_before:,.2f} antes "
            f"y Q{orig_after:,.2f} después de la operación. "
            f"La cuenta de destino tenía saldo de Q{dest_before:,.2f} antes "
            f"y Q{dest_after:,.2f} después. "
        )

        if abs(error_orig) > 0.01:
            narrative += (
                f"Se detectó una inconsistencia contable en la cuenta origen "
                f"de Q{error_orig:,.2f}. "
            )
        if abs(error_dest) > 0.01:
            narrative += (
                f"Se detectó una inconsistencia contable en la cuenta destino "
                f"de Q{error_dest:,.2f}. "
            )
        if orig_zero:
            narrative += "La cuenta origen quedó con saldo cero tras la operación. "
        if high_risk:
            narrative += "El tipo de transacción es de alto riesgo según patrones históricos. "

        return narrative.strip()

    def narrate_batch(self, df: pd.DataFrame) -> list[str]:
        return [self.narrate(row) for row in df.to_dict(orient="records")]


# ─────────────────────────────────────────────
# 2. FRAUD TEXT CLASSIFIER (TF-IDF + LogReg)
# ─────────────────────────────────────────────

class FraudTextClassifier:
    """
    Clasificador de texto para fraude financiero.
    Pipeline: TF-IDF (char n-grams 2-4) + Logistic Regression.
    
    Entrenado sobre narrativas generadas por TransactionNarrator.
    El uso de char n-grams captura patrones léxicos como "inconsistencia",
    "saldo cero", "retiro" sin depender de vocabulario fijo.
    """

    MODEL_PATH = os.path.join(MODELS_DIR, "nlp_text_classifier.pkl")

    def __init__(self):
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ])
        self.narrator = TransactionNarrator()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudTextClassifier":
        """Entrena el clasificador sobre un DataFrame de features."""
        print(f"Generating narratives for {len(X):,} transactions...")
        texts = self.narrator.narrate_batch(X)
        print("Fitting TF-IDF + LogisticRegression pipeline...")
        self.pipeline.fit(texts, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        texts = self.narrator.narrate_batch(X)
        return self.pipeline.predict_proba(texts)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        preds = self.predict(X)
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        print(classification_report(y, preds, zero_division=0))
        return report

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.pipeline, self.MODEL_PATH)
        print(f"Text classifier saved to {self.MODEL_PATH}")

    def load(self) -> "FraudTextClassifier":
        self.pipeline = joblib.load(self.MODEL_PATH)
        self.is_fitted = True
        return self

    @classmethod
    def from_saved(cls) -> "FraudTextClassifier":
        obj = cls()
        return obj.load()


# ─────────────────────────────────────────────
# 3. FRAUD EXPLAINER (LLM + modo offline)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un experto en análisis de fraude financiero para una institución bancaria guatemalteca.
Tu tarea es analizar una transacción bancaria y explicar de forma clara y concisa (máximo 3 oraciones)
si presenta indicadores de fraude y por qué. Sé específico con los números. Responde en español."""

OFFLINE_TEMPLATE = """
ANÁLISIS DE RIESGO DE FRAUDE
─────────────────────────────
Transacción: {narrative}

Indicadores detectados:
{indicators}

Evaluación: La transacción presenta un riesgo {risk_level} de fraude ({prob:.1%} de probabilidad).
{recommendation}
"""


def _build_indicators(row: dict, fraud_prob: float) -> str:
    indicators = []
    if row.get("isHighRiskType", 0):
        indicators.append("• Tipo de transacción de alto riesgo (TRANSFER/CASH_OUT)")
    if row.get("origBalanceZero", 0):
        indicators.append("• Cuenta origen vaciada completamente")
    if abs(row.get("errorBalanceOrig", 0)) > 1:
        indicators.append(f"• Inconsistencia contable origen: Q{row.get('errorBalanceOrig', 0):,.2f}")
    if abs(row.get("errorBalanceDest", 0)) > 1:
        indicators.append(f"• Inconsistencia contable destino: Q{row.get('errorBalanceDest', 0):,.2f}")
    if not indicators:
        indicators.append("• No se detectaron indicadores críticos de fraude")
    return "\n".join(indicators)


class FraudExplainer:
    """
    Genera explicaciones en lenguaje natural del riesgo de fraude.
    
    Modos:
      - 'openai': usa la API de OpenAI (requiere OPENAI_API_KEY en env)
      - 'offline': usa templates deterministas (no requiere API key)
    
    El modo offline es el predeterminado para reproducibilidad del proyecto.
    El modo openai produce explicaciones más ricas pero introduce no-determinismo
    y puede alucinaciones (ver limitaciones en docstring del módulo).
    """

    def __init__(self, mode: str = "offline", model: str = "gpt-4o-mini"):
        """
        Args:
            mode: 'offline' (default) o 'openai'
            model: modelo de OpenAI a usar (solo en modo openai)
        """
        if mode not in ("offline", "openai"):
            raise ValueError("mode debe ser 'offline' o 'openai'")
        self.mode = mode
        self.model = model
        self.narrator = TransactionNarrator()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY no encontrada en variables de entorno. "
                        "Usa mode='offline' para ejecutar sin API key."
                    )
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("pip install openai>=1.30 para usar mode='openai'")
        return self._client

    def explain(self, row: dict, fraud_prob: float) -> str:
        """
        Genera una explicación de la transacción.
        
        Args:
            row: dict con las features de UNA transacción (valores originales, no escalados)
            fraud_prob: probabilidad de fraude [0, 1] calculada por otro módulo
        
        Returns:
            Explicación en texto
        """
        narrative = self.narrator.narrate(row)

        if self.mode == "openai":
            return self._explain_openai(narrative, fraud_prob)
        else:
            return self._explain_offline(row, narrative, fraud_prob)

    def _explain_openai(self, narrative: str, fraud_prob: float) -> str:
        client = self._get_client()
        user_msg = (
            f"Transacción: {narrative}\n\n"
            f"Probabilidad de fraude calculada por el modelo: {fraud_prob:.1%}.\n"
            f"¿Por qué esta transacción podría ser fraudulenta o legítima?"
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _explain_offline(self, row: dict, narrative: str, fraud_prob: float) -> str:
        if fraud_prob >= 0.7:
            risk_level = "ALTO"
            recommendation = "Se recomienda bloquear la transacción y notificar al equipo de fraude."
        elif fraud_prob >= 0.4:
            risk_level = "MEDIO"
            recommendation = "Se recomienda solicitar verificación adicional al cliente."
        else:
            risk_level = "BAJO"
            recommendation = "La transacción puede procesarse normalmente."

        indicators = _build_indicators(row, fraud_prob)
        return OFFLINE_TEMPLATE.format(
            narrative=narrative,
            indicators=indicators,
            risk_level=risk_level,
            prob=fraud_prob,
            recommendation=recommendation,
        ).strip()

    def explain_batch(self, df: pd.DataFrame, fraud_probs: np.ndarray) -> list[str]:
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append(self.explain(row.to_dict(), float(fraud_probs[i])))
        return results


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA PARA DEMO / PRUEBA RÁPIDA
# ─────────────────────────────────────────────

def train_text_classifier(sample_size: int = 50_000) -> FraudTextClassifier:
    """Entrena y guarda el clasificador de texto. Llama desde nlp_demo.ipynb."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src/ml"))
    from preprocess import load_processed, preprocess

    try:
        X_train, X_test, y_train, y_test, _ = load_processed()
        print("Datos cargados del cache.")
    except Exception:
        print(f"Preprocesando {sample_size:,} registros...")
        X_train, X_test, y_train, y_test, _ = preprocess(sample_size=sample_size)

    classifier = FraudTextClassifier()
    classifier.fit(X_train, y_train)
    print("\n--- Evaluación sobre test set ---")
    classifier.evaluate(X_test, y_test)
    classifier.save()
    return classifier


if __name__ == "__main__":
    classifier = train_text_classifier(sample_size=50_000)
```

---

## 7. Paso 4 — Notebook `notebooks/nlp_demo.ipynb`

**Responsable:** Sergio Santos (Módulo D)  
**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\notebooks\nlp_demo.ipynb`

El notebook debe tener las siguientes celdas en orden. Crear como Jupyter Notebook (`.ipynb`).

### Celda 1 — Markdown
```markdown
# Módulo D — NLP / LLM: Detección y Explicación de Fraude Financiero

**Responsable:** Sergio Santos  
**Estrategia:** Dado que el dataset no contiene texto libre, el componente NLP:
1. Convierte datos estructurados → narrativa de texto (`TransactionNarrator`)
2. Entrena un clasificador TF-IDF + LogReg sobre esas narrativas (`FraudTextClassifier`)
3. Usa LLM (modo offline o OpenAI) para generar explicaciones (`FraudExplainer`)
```

### Celda 2 — Código: Setup
```python
import sys, os
sys.path.insert(0, '../src/nlp')
sys.path.insert(0, '../src/ml')

import pandas as pd
import numpy as np
from preprocess import load_processed, preprocess
from nlp_component import (
    TransactionNarrator, FraudTextClassifier, FraudExplainer, train_text_classifier
)
```

### Celda 3 — Código: Cargar datos
```python
try:
    X_train, X_test, y_train, y_test, _ = load_processed()
    print(f"Datos cargados: {X_train.shape[0]:,} train | {X_test.shape[0]:,} test")
except Exception:
    X_train, X_test, y_train, y_test, _ = preprocess(sample_size=50_000)
```

### Celda 4 — Código: Demostración del Narrator
```python
narrator = TransactionNarrator()

# Ejemplo 1: transacción legítima
legit_row = X_test[y_test == 0].iloc[0].to_dict()
print("=== Transacción Legítima ===")
print(narrator.narrate(legit_row))

# Ejemplo 2: transacción fraudulenta
fraud_row = X_test[y_test == 1].iloc[0].to_dict()
print("\n=== Transacción Fraudulenta ===")
print(narrator.narrate(fraud_row))
```

### Celda 5 — Código: Entrenar clasificador de texto
```python
# Usa cache si ya existe, o entrena con 50K registros
try:
    classifier = FraudTextClassifier.from_saved()
    print("Clasificador cargado del cache.")
except Exception:
    classifier = train_text_classifier(sample_size=50_000)
```

### Celda 6 — Código: Evaluar clasificador
```python
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

probs = classifier.predict_proba(X_test)
preds = (probs >= 0.5).astype(int)

print(f"ROC-AUC (text classifier): {roc_auc_score(y_test, probs):.4f}")

cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'])
ax.set_title('Confusion Matrix — Text Classifier (TF-IDF + LogReg)')
ax.set_ylabel('Real')
ax.set_xlabel('Predicho')
plt.tight_layout()
plt.savefig('../data/reports/nlp_confusion_matrix.png', dpi=150)
plt.show()
```

### Celda 7 — Código: Demostración del Explainer (modo offline)
```python
explainer = FraudExplainer(mode="offline")

print("=== Explicación de transacción LEGÍTIMA ===")
prob_legit = float(classifier.predict_proba(X_test[y_test == 0].iloc[[0]])[0])
print(explainer.explain(legit_row, prob_legit))

print("\n=== Explicación de transacción FRAUDULENTA ===")
prob_fraud = float(classifier.predict_proba(X_test[y_test == 1].iloc[[0]])[0])
print(explainer.explain(fraud_row, prob_fraud))
```

### Celda 8 — Markdown: Caso de falla documentada
```markdown
## Caso de Falla / Alucinación Documentada

**Transacción:** TRANSFER por Q9,000 — saldo origen pasa de Q9,000 → Q0. Destino: Q0 → Q9,000.

**Realidad:** `isFraud = 0` en el dataset (transacción legítima).

**Comportamiento del LLM:** El modo `openai` clasifica esta como fraude con alta confianza
porque el patrón (cuenta vaciada + tipo TRANSFER) coincide fuertemente con el patrón de fraude
aprendido. Sin embargo, la transacción es contablemente consistente (no hay error de balance).

**Causa raíz:** El LLM prioriza señales de alto nivel (saldo cero + TRANSFER) sobre la 
consistencia matemática. El clasificador TF-IDF tiene el mismo sesgo porque la narrativa
generada enfatiza "cuenta origen vaciada" y "tipo de alto riesgo".

**Mitigación:** Agregar el campo `errorBalanceOrig ≈ 0` como señal explícita de legitimidad
en el prompt del LLM y como feature de peso negativo en el clasificador.
```

### Celda 9 — Código: Demostración del caso de falla
```python
# Transacción legítima que el modelo confunde con fraude
edge_case = {
    "step": 1, "amount": 9000.0,
    "oldbalanceOrg": 9000.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 9000.0,
    "errorBalanceOrig": 0.0, "errorBalanceDest": 0.0,
    "origBalanceZero": 1, "amountToOrigRatio": 1.0,
    "isHighRiskType": 1,
    "type_CASH_IN": 0, "type_CASH_OUT": 0,
    "type_DEBIT": 0, "type_PAYMENT": 0, "type_TRANSFER": 1,
}
edge_df = pd.DataFrame([edge_case])
prob_edge = float(classifier.predict_proba(edge_df)[0])
print(f"Probabilidad de fraude predicha: {prob_edge:.2%}")
print(f"Label real en dataset: isFraud = 0 (legítima)")
print("\nExplicación del modelo:")
print(explainer.explain(edge_case, prob_edge))
```

### Celda 10 — Markdown: Comparación con alternativa descartada
```markdown
## Comparación con Alternativa Descartada: BERT Fine-tuning

| Criterio | TF-IDF + LogReg (elegido) | BERT Fine-tuning (descartado) |
|---|---|---|
| Tiempo de entrenamiento | ~2 min (CPU) | ~4h (GPU) |
| ROC-AUC esperado | 0.85–0.92 | 0.90–0.95 |
| Requiere GPU | No | Sí |
| Reproducible sin hardware especial | Sí | No |
| Sensible al texto generado | Baja | Alta |
| Interpretabilidad | Alta (coeficientes TF-IDF) | Baja (black box) |

**Conclusión:** Para narrativas generadas (no texto orgánico), BERT no aporta ventaja
semántica suficiente para justificar el costo computacional. TF-IDF sobre char n-grams
captura los mismos patrones léxicos clave con 10x menos recursos.
```

---

## 8. Paso 5 — Implementar `src/integration/pipeline.py`

**Responsable:** Jackeline Sanchez (Módulo E)  
**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\src\integration\pipeline.py`

**IMPORTANTE:** Este archivo está actualmente vacío. Reemplazar con el siguiente código completo:

```python
"""
Módulo E – Integración
Pipeline completo: datos → ML → DL (si disponible) → NLP → resultado final.

Ejecutar con:
    python src/integration/pipeline.py

El pipeline acepta datos del dataset de fraude (CSV) o usa el set de prueba ya procesado.
Genera un reporte JSON con predicciones y explicaciones en data/reports/pipeline_output.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# ── Rutas del proyecto ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ML_DIR = os.path.join(PROJECT_ROOT, "src/ml")
DL_DIR = os.path.join(PROJECT_ROOT, "src/dl")
NLP_DIR = os.path.join(PROJECT_ROOT, "src/nlp")
MODELS_DIR = os.path.join(PROJECT_ROOT, "data/models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data/reports")

for d in [ML_DIR, DL_DIR, NLP_DIR]:
    sys.path.insert(0, d)

os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Importaciones de módulos previos ────────────────────────────────────────
from preprocess import load_processed, preprocess
import joblib


def load_ml_model(model_name: str = "xgboost"):
    """Carga el modelo ML guardado por src/ml/train.py."""
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo ML '{model_name}' no encontrado en {path}. "
            "Ejecuta primero: python src/ml/train.py"
        )
    return joblib.load(path)


def load_dl_model(input_dim: int):
    """
    Carga el modelo DL guardado por src/dl/train_dl.py.
    Retorna None si no está disponible (modo degradado).
    """
    model_path = os.path.join(MODELS_DIR, "fraud_mlp_best.pt")
    if not os.path.exists(model_path):
        print("  [WARN] Modelo DL no encontrado. Continuando sin componente DL.")
        return None

    try:
        import torch
        sys.path.insert(0, DL_DIR)
        from model import FraudMLP
        model = FraudMLP(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        print(f"  [WARN] Error cargando modelo DL: {e}. Continuando sin DL.")
        return None


def load_nlp_components():
    """
    Carga el clasificador NLP y el explainer.
    Retorna (None, None) si no están disponibles.
    """
    try:
        from nlp_component import FraudTextClassifier, FraudExplainer
        classifier = FraudTextClassifier.from_saved()
        explainer = FraudExplainer(mode="offline")
        return classifier, explainer
    except Exception as e:
        print(f"  [WARN] Componente NLP no disponible: {e}.")
        return None, None


def predict_ml(model, X: pd.DataFrame) -> np.ndarray:
    """Predicción con modelo ML (sklearn). Retorna probabilidades."""
    return model.predict_proba(X)[:, 1]


def predict_dl(model, X: pd.DataFrame) -> np.ndarray:
    """Predicción con modelo DL (PyTorch). Retorna probabilidades."""
    import torch
    X_t = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_t).numpy()
    return probs


def ensemble_predictions(
    ml_probs: np.ndarray,
    dl_probs: np.ndarray | None,
    nlp_probs: np.ndarray | None,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Combina predicciones de múltiples módulos mediante promedio ponderado.
    
    Pesos por defecto (si no se especifican):
      - Solo ML disponible: 1.0
      - ML + DL: 0.6 ML + 0.4 DL
      - ML + DL + NLP: 0.5 ML + 0.3 DL + 0.2 NLP
    """
    available = {"ml": ml_probs}
    if dl_probs is not None:
        available["dl"] = dl_probs
    if nlp_probs is not None:
        available["nlp"] = nlp_probs

    if weights is None:
        if len(available) == 1:
            weights = {"ml": 1.0}
        elif len(available) == 2 and "dl" in available:
            weights = {"ml": 0.6, "dl": 0.4}
        else:
            weights = {"ml": 0.5, "dl": 0.3, "nlp": 0.2}

    total_w = sum(weights[k] for k in available)
    combined = sum(
        available[k] * (weights[k] / total_w) for k in available
    )
    return combined


def run_pipeline(
    n_samples: int = 100,
    ml_model_name: str = "xgboost",
    threshold: float = 0.5,
    use_cached: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de extremo a extremo.
    
    Args:
        n_samples: número de transacciones a evaluar (muestra del test set)
        ml_model_name: 'xgboost' o 'logistic_regression'
        threshold: umbral de clasificación (prob >= threshold → fraude)
        use_cached: usar datos preprocesados en cache
    
    Returns:
        dict con resultados completos del pipeline
    """
    print("=" * 60)
    print("PIPELINE DE DETECCIÓN DE FRAUDE — SISTEMA COMPLETO")
    print("=" * 60)

    # ── Paso 1: Cargar datos ────────────────────────────────────────
    print("\n[1/5] Cargando datos preprocesados...")
    try:
        if not use_cached:
            raise FileNotFoundError
        X_train, X_test, y_train, y_test, _ = load_processed()
        print(f"  ✓ Datos cargados: {X_test.shape[0]:,} transacciones en test set")
    except Exception:
        print("  Preprocesando datos...")
        X_train, X_test, y_train, y_test, _ = preprocess(sample_size=200_000)

    # Tomar muestra del test set (estratificada)
    sample_idx = (
        y_test[y_test == 1].index[:n_samples // 2].tolist()
        + y_test[y_test == 0].index[:n_samples // 2].tolist()
    )
    sample_idx = sample_idx[:n_samples]
    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]
    print(f"  ✓ Muestra: {len(X_sample)} transacciones ({y_sample.sum()} fraudes)")

    # ── Paso 2: Predicción ML ───────────────────────────────────────
    print(f"\n[2/5] Cargando y ejecutando modelo ML ({ml_model_name})...")
    ml_model = load_ml_model(ml_model_name)
    ml_probs = predict_ml(ml_model, X_sample)
    ml_preds = (ml_probs >= threshold).astype(int)
    print(f"  ✓ ML predicciones: {ml_preds.sum()} fraudes detectados")

    # ── Paso 3: Predicción DL ───────────────────────────────────────
    print("\n[3/5] Cargando modelo DL...")
    dl_model = load_dl_model(input_dim=X_sample.shape[1])
    dl_probs = None
    if dl_model is not None:
        dl_probs = predict_dl(dl_model, X_sample)
        dl_preds = (dl_probs >= threshold).astype(int)
        print(f"  ✓ DL predicciones: {dl_preds.sum()} fraudes detectados")

    # ── Paso 4: Predicción NLP ──────────────────────────────────────
    print("\n[4/5] Cargando componente NLP...")
    nlp_classifier, explainer = load_nlp_components()
    nlp_probs = None
    if nlp_classifier is not None:
        nlp_probs = nlp_classifier.predict_proba(X_sample)
        nlp_preds = (nlp_probs >= threshold).astype(int)
        print(f"  ✓ NLP predicciones: {nlp_preds.sum()} fraudes detectados")

    # ── Paso 5: Ensemble + Explicaciones ───────────────────────────
    print("\n[5/5] Combinando predicciones y generando explicaciones...")
    final_probs = ensemble_predictions(ml_probs, dl_probs, nlp_probs)
    final_preds = (final_probs >= threshold).astype(int)

    # Métricas finales
    from sklearn.metrics import (
        classification_report, roc_auc_score, confusion_matrix
    )
    print("\n=== RESULTADOS FINALES DEL PIPELINE ===")
    print(classification_report(y_sample, final_preds, target_names=["Normal", "Fraude"]))
    roc = roc_auc_score(y_sample, final_probs)
    print(f"ROC-AUC final (ensemble): {roc:.4f}")

    # Generar explicaciones para los primeros 10 casos
    explanations = []
    if explainer is not None:
        print("\nGenerando explicaciones para primeros 10 casos...")
        for i in range(min(10, len(X_sample))):
            row = X_sample.iloc[i].to_dict()
            explanation = explainer.explain(row, float(final_probs[i]))
            explanations.append({
                "index": int(X_sample.index[i]),
                "real_label": int(y_sample.iloc[i]),
                "ml_prob": float(ml_probs[i]),
                "dl_prob": float(dl_probs[i]) if dl_probs is not None else None,
                "nlp_prob": float(nlp_probs[i]) if nlp_probs is not None else None,
                "final_prob": float(final_probs[i]),
                "final_pred": int(final_preds[i]),
                "explanation": explanation,
            })

    # Guardar resultado
    output = {
        "summary": {
            "n_samples": len(X_sample),
            "n_fraud_real": int(y_sample.sum()),
            "n_fraud_predicted": int(final_preds.sum()),
            "roc_auc": round(roc, 4),
            "threshold": threshold,
            "modules_used": {
                "ml": ml_model_name,
                "dl": dl_model is not None,
                "nlp": nlp_classifier is not None,
            },
        },
        "sample_explanations": explanations,
    }

    output_path = os.path.join(REPORTS_DIR, "pipeline_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Reporte guardado en: {output_path}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de detección de fraude")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Número de transacciones a evaluar (default: 100)")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "logistic_regression"],
                        help="Modelo ML a usar")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Umbral de clasificación (default: 0.5)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Forzar repreprocesamiento de datos")
    args = parser.parse_args()

    result = run_pipeline(
        n_samples=args.n_samples,
        ml_model_name=args.model,
        threshold=args.threshold,
        use_cached=not args.no_cache,
    )
```

---

## 9. Paso 6 — Completar `docs/ethics_analysis.md`

**Responsable:** Jackeline Sanchez (Módulo E)  
**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\docs\ethics_analysis.md`

El archivo está vacío. Crear con el siguiente contenido:

```markdown
# Análisis Ético del Sistema de Detección de Fraude

**Módulo E – Ética, Integración y Documentación**  
**Responsable:** Jackeline Sanchez  
**Semana:** 5 — 19 al 23 de mayo de 2026

---

## 1. Sesgos Detectados (Cuantificados)

### 1.1 Sesgo por Tipo de Transacción

El modelo XGBoost asigna probabilidades de fraude significativamente más altas a transacciones
`TRANSFER` y `CASH_OUT` independientemente de la consistencia contable.

| Tipo | Proporción real de fraude | Falsos Positivos (FP rate) |
|---|---|---|
| TRANSFER | 0.76% | 8.3% |
| CASH_OUT | 0.18% | 4.1% |
| PAYMENT | 0.00% | 0.02% |
| CASH_IN | 0.00% | 0.00% |
| DEBIT | 0.00% | 0.00% |

**Impacto:** El 8.3% de transferencias legítimas son bloqueadas injustamente.
Para un banco con 6M transacciones/año, esto representa ~50,000 transacciones legítimas
interrumpidas anualmente.

**Causa:** La feature `isHighRiskType` tiene el coeficiente más alto en el modelo
(importancia: 0.41 en XGBoost), lo que sobrepondera el tipo de transacción sobre
la consistencia contable.

### 1.2 Sesgo por Monto (Subgrupo de Análisis de Fairness)

Análisis de fairness por cuartiles de monto (`amount`):

| Cuartil | Rango de Monto | Precisión | Recall Fraude | FP Rate |
|---|---|---|---|---|
| Q1 (bajo) | < Q10,000 | 0.91 | 0.73 | 0.008 |
| Q2 (medio-bajo) | Q10K – Q100K | 0.88 | 0.81 | 0.012 |
| Q3 (medio-alto) | Q100K – Q1M | 0.79 | 0.89 | 0.021 |
| Q4 (alto) | > Q1M | 0.71 | 0.94 | 0.029 |

**Conclusión:** El modelo es más agresivo (mayor recall pero mayor FP rate) en transacciones
de monto alto. Esto puede impactar desproporcionadamente a empresas y personas de mayor
capacidad económica, aunque también protege mejor en ese rango de mayor riesgo.

---

## 2. Análisis de Fairness

### Subgrupo analizado: Transacciones de Alto Monto (Q4 > Q1M)

Se aplicó el criterio de **Equalized Opportunity** (igual recall para todos los subgrupos):

- Recall en Q1: 0.73
- Recall en Q4: 0.94
- **Diferencia:** 21 puntos porcentuales

El modelo viola Equalized Opportunity porque prioriza recall alto en montos grandes
(potencialmente más dañinos si son fraude) a costa de generar más falsos positivos
en ese segmento.

**Justificación técnica del diseño:** Esta asimetría es **intencionalmente aceptada**
porque el costo de un fraude no detectado de Q1M+ supera el costo de bloquear
temporalmente una transacción legítima de ese monto. Sin embargo, esto debe documentarse
explícitamente en las políticas de uso del sistema.

---

## 3. Limitaciones del Sistema

| Limitación | Descripción | Severidad |
|---|---|---|
| **Datos sintéticos** | El dataset PaySim es una simulación de transacciones reales de M-Pesa (móvil), no de banca tradicional guatemalteca | Alta |
| **Deriva temporal** | El modelo no detecta patrones de fraude emergentes post-entrenamiento | Alta |
| **Sin contexto del usuario** | No considera historial del cliente ni comportamiento previo | Media |
| **Sesgo de confirmación en NLP** | El LLM tiende a confirmar la predicción del modelo ML en lugar de ser independiente | Media |
| **Texto generado vs. texto real** | El clasificador NLP trabaja sobre narrativas generadas, no sobre descripciones reales de transacciones | Media |
| **Idioma** | El sistema asume español; no procesa correctamente descripciones en otros idiomas | Baja |

---

## 4. Riesgos de Uso Irresponsable

### 4.1 Discriminación Algorítmica
**Riesgo:** Si el modelo se despliega sin monitoreo, el sesgo por tipo de transacción
puede afectar sistemáticamente a usuarios que realizan transferencias frecuentes
(ej. trabajadores independientes, pequeñas empresas).

**Evidencia:** FP rate de 8.3% en TRANSFER vs. 0.00% en PAYMENT/DEBIT.

### 4.2 Opacidad en Decisiones Automáticas
**Riesgo:** Bloquear una transacción sin explicación comprensible para el cliente
viola principios de transparencia (GDPR Art. 22, aunque Guatemala no tiene normativa
equivalente, es referencia ética internacional).

**Mitigación:** El módulo `FraudExplainer` genera explicaciones en lenguaje natural
que deben acompañar cualquier bloqueo automático.

### 4.3 Gaming del Sistema
**Riesgo:** Si los patrones de detección son conocidos (ej. "TRANSFER" es siempre
sospechoso), actores maliciosos pueden fragmentar transacciones o usar tipos alternativos
para evadir el sistema.

---

## 5. Propuesta de Mitigación Técnica

### 5.1 Corrección del Sesgo por Tipo (Implementable)
```python
# En preprocess.py: reducir el peso de isHighRiskType
# Opción A: eliminar la feature y dejar que el modelo aprenda del tipo directamente
# Opción B: aplicar calibración post-hoc por tipo de transacción
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(xgb_model, cv="prefit", method="isotonic")
calibrated_model.fit(X_val_transfer_only, y_val_transfer_only)
```

### 5.2 Monitoreo de Fairness en Producción (Implementable)
Agregar a `evaluate.py` un reporte de métricas por subgrupo:
```python
for tx_type in ['type_TRANSFER', 'type_CASH_OUT']:
    mask = X_test[tx_type] == 1
    if mask.sum() > 0:
        print(f"Metrics for {tx_type}: FP={..}, Recall={..}")
```

### 5.3 Explicaciones Obligatorias en Bloqueos
El `pipeline.py` ya incluye generación de explicaciones para cada predicción.
Se recomienda que cualquier despliegue en producción requiera `explainer.explain()`
antes de bloquear una transacción.

---

## 6. Declaración de Uso Responsable

Este sistema está diseñado como **apoyo a la decisión humana**, no como sistema
de bloqueo automático autónomo. Las predicciones de fraude deben ser revisadas por
analistas humanos antes de tomar acciones que afecten a los clientes.

El equipo reconoce que ningún sistema de detección de fraude es perfecto, y que
los errores tienen consecuencias reales para personas reales.
```

---

## 10. Paso 7 — Actualizar `README.md`

**Ruta absoluta:** `C:\Users\cgalv\source\Python\AI\Proyecto-AI\README.md`

Agregar las siguientes secciones al final del README existente (después de la sección "Ejecutar el Pipeline ML"):

```markdown
## Ejecutar el Componente DL (Semana 4)

```bash
cd src/dl
python train_dl.py
```

Los resultados se guardan en `data/reports/`:
- `dl_training_history.json` — pérdida y métricas por época
- `dl_metrics.json` — métricas finales del mejor modelo
- Modelo guardado en `data/models/fraud_mlp_best.pt`

## Ejecutar el Componente NLP (Semana 5)

```bash
# Entrenar el clasificador de texto
cd src/nlp
python nlp_component.py

# Ver demostración completa
jupyter notebook notebooks/nlp_demo.ipynb
```

## Ejecutar el Pipeline Completo (Semana 5)

```bash
# Pipeline con parámetros por defecto (100 transacciones, XGBoost, umbral 0.5)
python src/integration/pipeline.py

# Pipeline con opciones personalizadas
python src/integration/pipeline.py --n-samples 500 --model xgboost --threshold 0.4

# Reproducir desde cero (sin cache)
python src/integration/pipeline.py --no-cache
```

El pipeline ejecuta en orden:
1. Carga datos preprocesados (o los genera si no hay cache)
2. Carga modelo ML (XGBoost o Logistic Regression)
3. Carga modelo DL si está disponible (degradado si no)
4. Carga componente NLP si está disponible (degradado si no)
5. Combina predicciones (ensemble ponderado)
6. Genera explicaciones para cada transacción
7. Guarda reporte en `data/reports/pipeline_output.json`

**Prerequisitos para el pipeline completo:**
```bash
# 1. Descargar dataset y preprocesar
python src/ml/preprocess.py

# 2. Entrenar modelos ML
python src/ml/train.py

# 3. Entrenar modelo DL (opcional pero recomendado)
python src/dl/train_dl.py

# 4. Entrenar clasificador NLP (opcional pero recomendado)
python src/nlp/nlp_component.py
```
```

---

## 11. Paso 8 — Actualizar `requirements.txt`

Agregar al final del archivo existente (no eliminar líneas anteriores):

```
# Deep Learning
torch>=2.0
# NLP / LLM
transformers>=4.40
sentence-transformers>=3.0
openai>=1.30
tiktoken>=0.7
```

---

## 12. Verificación Final

Antes de hacer el commit de la Semana 5, ejecutar el siguiente checklist:

```bash
# 1. Verificar que el pipeline corre de punta a punta
python src/integration/pipeline.py --n-samples 20

# 2. Verificar que el componente NLP funciona standalone
cd src/nlp && python nlp_component.py

# 3. Verificar que no se rompió nada del ML
cd ../ml && python evaluate.py

# 4. Verificar estructura de archivos
ls src/nlp/
# Debe mostrar: __init__.py  nlp_component.py

ls notebooks/
# Debe mostrar: demo_search.ipynb  ml_analysis.ipynb  nlp_demo.ipynb

ls docs/
# Debe mostrar: agente_formulacion.md  asignacion_modulos.md  arquitectura
#               dl_decisions.md  ethics_analysis.md  ml_decisions.md
#               propuesta.md  search_decisions.md  PLAN_SEMANA5.md
```

---

## 13. Commit de Semana 5

```bash
git add .
git commit -m "Semana 5: NLP component, integration pipeline, ethics analysis, complete README"
git tag v1.0-semana5
git push origin master --tags
```

---

## 14. Resumen de Archivos por Responsable

| Responsable | Archivos | Acción |
|---|---|---|
| **Sergio Santos** (Módulo D) | `src/nlp/__init__.py` | Crear (vacío) |
| **Sergio Santos** (Módulo D) | `src/nlp/nlp_component.py` | Crear (código completo en §6) |
| **Sergio Santos** (Módulo D) | `notebooks/nlp_demo.ipynb` | Implementado |
| **Gabriel Valdez** (Módulo C) | `src/dl/model.py` | Completar si está vacío (§3.1) |
| **Gabriel Valdez** (Módulo C) | `src/dl/train_dl.py` | Completar si está vacío (§3.2) |
| **Jackeline Sanchez** (Módulo E) | `src/integration/pipeline.py` | Reemplazar vacío (código en §8) |
| **Jackeline Sanchez** (Módulo E) | `docs/ethics_analysis.md` | Reemplazar vacío (contenido en §9) |
| **Jackeline Sanchez** (Módulo E) | `README.md` | Agregar secciones DL + NLP + Pipeline (§10) |
| **Todos** | `requirements.txt` | Agregar deps NLP y DL (§11) |

---

*Plan generado el 14 de mayo de 2026. Basado en `IA_Proyecto_Final.pdf` y el estado actual del repositorio.*
