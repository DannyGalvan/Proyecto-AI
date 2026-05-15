# Decisiones del Modelo Deep Learning

**Modulo:** C - Deep Learning  
**Responsable:** Gabriel Valdez  
**Dominio:** Deteccion de fraude financiero  
**Base de integracion:** Pipeline ML de semana 3 (`src/ml/preprocess.py`)

---

## 1. Objetivo

El objetivo de semana 4 es entrenar una red neuronal que use los mismos datos
preprocesados de semana 3 y permita comparar su rendimiento contra los modelos
clasicos ya evaluados: Logistic Regression y XGBoost.

El problema es clasificacion binaria:

- `0`: transaccion normal
- `1`: transaccion fraudulenta

Como el fraude es una clase extremadamente minoritaria, la evaluacion prioriza
`recall`, `F1`, `ROC-AUC` y `Average Precision`, no solo `accuracy`.

---

## 2. Arquitectura Elegida

Se implemento un perceptron multicapa (MLP) en TensorFlow/Keras:

| Bloque | Configuracion | Justificacion |
|---|---|---|
| Entrada | `input_dim = numero de features procesadas` | Reutiliza las features tabulares de semana 3 |
| Capa 1 | Dense 128 + ReLU | Captura combinaciones no lineales iniciales |
| Regularizacion 1 | BatchNorm + Dropout 0.35 | Reduce sensibilidad a escala y overfitting |
| Capa 2 | Dense 64 + ReLU | Resume interacciones entre variables de saldo, monto y tipo |
| Regularizacion 2 | BatchNorm + Dropout 0.25 | Mantiene generalizacion en clase minoritaria |
| Capa 3 | Dense 32 + ReLU | Compacta la representacion antes de salida |
| Regularizacion 3 | BatchNorm + Dropout 0.15 | Regularizacion menor cerca de la salida |
| Salida | Dense 1 + Sigmoid | Probabilidad de fraude |

### Por que MLP

El dataset es tabular, no secuencial ni de imagen. Por eso una CNN o RNN no
aportaria una inductive bias adecuada. Un MLP es suficiente para modelar
interacciones no lineales entre:

- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `errorBalanceOrig`
- `amountToOrigRatio`
- `isHighRiskType`
- dummies de tipo de transaccion

---

## 3. Hiperparametros

| Hiperparametro | Valor | Razon |
|---|---:|---|
| Optimizador | Adam | Buen punto de partida para redes densas tabulares |
| Learning rate | `0.001` | Convergencia estable sin pasos agresivos |
| Loss | Binary crossentropy | Clasificacion binaria probabilistica |
| Batch size | `256` | Balance entre estabilidad y velocidad |
| Epocas maximas | `50` | Limite superior; early stopping decide el corte real |
| L2 | `1e-4` | Penaliza pesos grandes y reduce overfitting |
| Dropout | `0.35`, `0.25`, `0.15` | Regularizacion decreciente por profundidad |
| Monitor | `val_auc` | Mas informativo que accuracy en clases desbalanceadas |
| Early stopping | paciencia `8` | Detiene cuando no mejora validacion |

---

## 4. Manejo del Desbalanceo

El dataset tiene muchos mas casos normales que fraudes. Para evitar que la red
aprenda a predecir siempre "normal", `train_dl.py` calcula `class_weight` con
`sklearn.utils.class_weight.compute_class_weight`.

Esto aumenta el costo de equivocarse en fraudes durante el entrenamiento sin
duplicar datos ni crear ejemplos sinteticos.

---

## 5. Integracion con Semana 3

El entrenamiento de DL carga los datos desde:

- `data/processed/X_train.parquet`
- `data/processed/X_test.parquet`
- `data/processed/y_train.parquet`
- `data/processed/y_test.parquet`
- `data/processed/scaler.pkl`

Si esos archivos no existen, el script ejecuta el preprocesamiento de semana 3.
Esto mantiene una comparacion justa porque Logistic Regression, XGBoost y el MLP
usan las mismas features y la misma particion train/test.

---

## 6. Artefactos Generados

Al ejecutar:

```bash
python src/dl/train_dl.py
```

se generan:

| Archivo | Descripcion |
|---|---|
| `data/models/dl_best_model.keras` | Mejor checkpoint segun `val_auc` |
| `data/models/dl_final_model.keras` | Modelo final con mejores pesos restaurados |
| `data/models/scaler_dl.pkl` | Scaler usado por el pipeline de semana 3 |
| `data/reports/dl_training_history.csv` | Loss y metricas por epoca |
| `data/reports/dl_training_curves.png` | Curvas de entrenamiento/validacion |
| `data/reports/dl_confusion_matrix.png` | Matriz de confusion del MLP |
| `data/reports/dl_metrics.json` | Accuracy, precision, recall, F1, ROC-AUC y AP |
| `data/reports/dl_prediction_examples.csv` | Ejemplos de aciertos y errores |
| `data/reports/model_comparison.csv` | Comparacion con ML incluyendo el MLP |

---

## 7. Comparacion con ML

La expectativa tecnica es que XGBoost siga siendo un competidor fuerte porque
los arboles gradient boosting suelen rendir muy bien en datos tabulares. El MLP
puede capturar no linealidades, pero normalmente necesita mas datos, ajuste de
hiperparametros y regularizacion cuidadosa para superar a modelos de arboles.

La comparacion debe hacerse con `model_comparison.csv`, observando sobre todo:

- `recall`: cuantos fraudes reales detecta.
- `precision`: cuantas alertas son realmente fraude.
- `f1`: balance entre precision y recall.
- `avg_precision`: robusta para clases desbalanceadas.
- falsos negativos en la matriz de confusion.

En fraude financiero, un falso negativo suele ser mas grave que un falso
positivo, porque implica dejar pasar una transaccion fraudulenta.

---

## 8. Limitaciones

1. **Datos tabulares:** El MLP no tiene una ventaja estructural tan fuerte como
   XGBoost en este tipo de dataset.
2. **Desbalance extremo:** Aunque se usan pesos por clase, la red puede ser
   sensible al umbral de decision.
3. **Costo computacional:** Entrenar DL es mas lento que Logistic Regression y
   puede requerir GPU para iteraciones grandes.
4. **Generalizacion temporal:** El dataset representa una ventana especifica; en
   produccion se debe monitorear drift y reentrenar.
5. **Umbral fijo:** `0.5` es un punto de partida. Para despliegue real se
   optimizaria por costo esperado de fraude y friccion al cliente.

---

## 9. Defensa Oral

Puntos clave para explicar:

- Se eligio MLP porque el problema es tabular y binario.
- Se reutilizo exactamente el pipeline de semana 3 para comparacion justa.
- Se agrego regularizacion: dropout, L2, batch normalization y early stopping.
- Se usaron pesos por clase para atender el desbalance.
- La metrica critica es recall/F1, no accuracy.
- XGBoost puede superar al MLP por ser muy fuerte en datos tabulares; eso no
  invalida el modulo DL, sino que muestra un trade-off realista.
