# Decisiones del Pipeline de Machine Learning

**Módulo:** B — Pipeline ML  
**Responsable:** Sergio Santos  
**Dominio:** Detección de Fraude Financiero  
**Dataset:** Financial Fraud Detection — 6,360,000 transacciones bancarias

---

## 1. Preprocesamiento

### 1.1 Columnas eliminadas

| Columna | Razón |
|---|---|
| `nameOrig`, `nameDest` | Identificadores de cuenta — alta cardinalidad, sin valor predictivo generalizable |
| `isFlaggedFraud` | Columna derivada del propio dataset (data leakage): el banco la pone cuando ya sospecha fraude, lo que contaminaría al modelo |

### 1.2 Ingeniería de Features

Se crearon cuatro features derivadas que capturan el patrón matemático del fraude:

| Feature | Fórmula | Justificación |
|---|---|---|
| `errorBalanceOrig` | `newbalanceOrig + amount - oldbalanceOrg` | En fraudes, los saldos no cuadran aritméticamente — este error es una señal fuerte |
| `errorBalanceDest` | `oldbalanceDest + amount - newbalanceDest` | Detecta inconsistencias en el saldo del destinatario |
| `origBalanceZero` | `newbalanceOrig == 0` → 1/0 | Los fraudes vacían la cuenta de origen completamente |
| `amountToOrigRatio` | `amount / (oldbalanceOrg + 1)` | Ratio del monto respecto al saldo disponible — fraudes tienden a usar todo el saldo |
| `isHighRiskType` | `type in {TRANSFER, CASH_OUT}` | En el dataset, **solo TRANSFER y CASH_OUT** contienen fraudes; el resto tiene 0% de fraude |

### 1.3 Manejo de Nulos

El dataset PaySim es sintético y está completo (no tiene valores faltantes reales). Aun así, se aplica `df.fillna(0)` como medida defensiva por dos razones:
1. **Robustez ante datos futuros:** en producción, transacciones reales pueden llegar con campos incompletos.
2. **Después de la ingeniería de features:** divisiones como `amount / (oldbalanceOrg + 1)` evitan NaN/inf, pero el `+1` mantiene la semántica de "saldo prácticamente cero" sin generar nulos.

### 1.4 Tratamiento de Outliers

El dataset tiene una distribución extremadamente sesgada en `amount` y los saldos: la mayoría de transacciones son pequeñas pero existen montos de hasta ~92 millones. Esto afecta directamente a Logistic Regression (sensible a outliers) y al StandardScaler (la media/desviación se distorsionan).

**Estrategia elegida — Winsorización al percentil 99.5:**
Se aplica clipping (`np.clip`) sobre las columnas `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` al cuantil 99.5 del conjunto de entrenamiento. Los valores extremos se truncan al límite superior en lugar de eliminarse.

**Por qué winsorización y no eliminación:**
- Eliminar transacciones grandes **eliminaría fraudes reales** (los fraudes tienden a usar montos altos)
- La winsorización preserva la información de que "esta transacción es grande" sin permitir que un outlier mueva la media del scaler
- Los percentiles se calculan **sólo en train** para evitar data leakage

**Alternativa descartada — log-transform:** Se evaluó aplicar `log1p` a los montos. Funciona bien para LR pero distorsiona el espacio de features para XGBoost, que ya maneja distribuciones sesgadas internamente con su histograma de splits.

### 1.5 Manejo del Desbalanceo de Clases

El dataset completo tiene ~0.13% de fraudes — desbalanceo extremo de ~770:1 sobre los 6.36M registros originales.

**Nota sobre la muestra de entrenamiento:** Con `sample_size=200_000` se conservan los 8,213 fraudes completos y se submuestrean los normales a ~192K, resultando en un ratio efectivo de **~23:1** durante el entrenamiento. Esto NO altera el problema — el modelo aprende sobre la firma del fraude, no sobre la frecuencia base.

**Estrategia elegida:** Se usó `class_weight='balanced'` en Logistic Regression y `scale_pos_weight` (ratio neg/pos calculado dinámicamente sobre el train set) en XGBoost. Esto pesa los errores sobre la clase minoritaria sin modificar la distribución de datos.

**Alternativa descartada — SMOTE:** Se evaluó SMOTE (oversampling sintético) pero fue descartado porque:
1. Con 6M registros, SMOTE sobre 8,000 fraudes genera ruido artificial
2. El costo computacional es prohibitivo a esta escala
3. `scale_pos_weight` en XGBoost logra el mismo efecto de forma implícita y eficiente

### 1.6 Normalización

Se aplicó `StandardScaler` a las features numéricas continuas. El scaler se ajustó **únicamente sobre el conjunto de entrenamiento** para evitar data leakage hacia el conjunto de test.

Las features binarias (`origBalanceZero`, `isHighRiskType`, dummies de `type`) no se normalizan.

---

## 2. Modelos Elegidos

### Modelo 1: Logistic Regression (baseline)

**Por qué se eligió como baseline:**
- Modelo lineal interpretable — los coeficientes muestran directamente qué features importan y en qué dirección
- Rápido de entrenar incluso en 6M registros
- Sirve como cota inferior de rendimiento: si XGBoost no supera LR claramente, el problema puede estar en el preprocesamiento

**Limitaciones:**
- Asume relaciones lineales entre features e `isFraud`
- No captura interacciones entre variables (p.ej., que `TRANSFER` + `errorBalanceOrig > 0` sea particularmente sospechoso)
- Tiende a menor recall en clases muy desbalanceadas incluso con `class_weight='balanced'`

### Modelo 2: XGBoost (modelo principal)

**Por qué XGBoost sobre Random Forest:**

| Criterio | Random Forest | XGBoost |
|---|---|---|
| Velocidad en 6M rows | Lento (O(n·d·T)) | Más rápido (histograma aproximado) |
| Manejo de desbalanceo | `class_weight` | `scale_pos_weight` nativo |
| Regularización | Ninguna explícita | L1 + L2 + `min_child_weight` |
| Interpretabilidad | Feature importance SHAP | Feature importance SHAP |
| Performance en fraude tabular | Bueno | Estado del arte en tabular data |

**Alternativa descartada — LightGBM:** LightGBM es igualmente válido y más rápido aún, pero XGBoost tiene mayor documentación y reproducibilidad en el contexto académico. Se usó XGBoost con `tree_method='hist'` para eficiencia.

**Hiperparámetros elegidos:**
- `n_estimators=300`: suficiente para converger sin sobreajuste
- `learning_rate=0.05`: conservador, mejores generalizaciones que 0.1
- `max_depth=6`: captura interacciones hasta 6 niveles sin overfitting excesivo
- `scale_pos_weight`: calculado dinámicamente como `n_negatives / n_positives`

---

## 3. Validación

**Estrategia:** StratifiedKFold con 5 folds.  
Se usó `stratify=True` para mantener la misma proporción de fraudes (~0.13%) en cada fold. Sin estratificación, algunos folds podrían tener 0 fraudes por azar.

**Métricas reportadas:**

| Métrica | Por qué se usa en este dominio |
|---|---|
| **F1** | Balance entre precision y recall — la métrica principal en fraude |
| **ROC-AUC** | Mide la capacidad discriminativa independiente del umbral |
| **Precision-Recall AUC** | Más informativa que ROC cuando las clases están muy desbalanceadas |
| **Recall** | Prioritario: los fraudes no detectados (FN) tienen costo económico directo |
| **Precision** | Secundario: demasiados falsos positivos generan fricción al usuario |
| **Accuracy** | Se reporta pero NO se usa como criterio de selección — es engañosa con desbalanceo |

---

## 4. Análisis de Errores

### Por qué el Recall importa más que la Precisión

Un **Falso Negativo** (fraude no detectado) significa que el dinero del cliente fue robado y el banco asume la pérdida.

Un **Falso Positivo** (transacción normal bloqueada) molesta al cliente pero se resuelve con una llamada.

**El costo asimétrico de los errores favorece maximizar el Recall**, aunque eso sacrifique algo de Precisión. El umbral de decisión (0.5 por defecto) puede ajustarse para mover ese balance según la política de riesgo del banco.

### Comparación cuantitativa entre modelos

Resultados sobre 40,000 transacciones del test set (muestra estratificada de 200K, con outlier clipping al P99.5):

| Métrica | Logistic Regression | XGBoost |
|---|---|---|
| Accuracy | 0.9640 | **0.9996** |
| Precision | 0.5342 | **0.9945** |
| Recall | 0.9732 | **0.9951** |
| F1 | 0.6898 | **0.9948** |
| ROC-AUC | 0.9946 | **0.9999** |
| Avg. Precision | 0.9351 | **0.9987** |
| Fraudes detectados (TP) | 1,599 / 1,643 | **1,635 / 1,643** |
| Fraudes perdidos (FN) | 44 | **8** |
| Falsas alarmas (FP) | 1,394 | **9** |

**XGBoost supera a Logistic Regression en todas las métricas.** La diferencia más importante en este dominio es en los Falsos Negativos: LR deja pasar 44 fraudes vs 8 de XGBoost. Cada FN representa un fraude real no bloqueado — costo económico directo para el banco y el cliente.

La baja Precision de LR (0.53) muestra que el modelo lineal no puede separar limpiamente fraudes de normales: por cada fraude detectado, genera ~1,394 falsas alarmas, lo que haría el sistema inoperable en producción.

XGBoost captura las interacciones no lineales entre `errorBalanceOrig`, `isHighRiskType` y `amountToOrigRatio` que son la firma matemática del fraude en este dataset.

### Curvas ROC

Las curvas ROC comparadas de ambos modelos se generan en `data/reports/roc_curves.png` (visualización detallada, una curva por modelo + baseline aleatorio).

![Curvas ROC](../data/reports/roc_curves.png)

**Lectura de la gráfica:**
- **XGBoost (AUC=0.9999):** la curva se pega prácticamente a la esquina superior izquierda — el modelo logra altísimo recall sin sacrificar precisión.
- **Logistic Regression (AUC=0.9944):** la curva también es alta pero requiere mover el umbral hacia abajo para detectar fraudes, lo que dispara los falsos positivos.
- **Curva Precisión-Recall** (`data/reports/precision_recall_curve.png`): más informativa que ROC con clases desbalanceadas, confirma la superioridad de XGBoost (AP=0.9986 vs 0.9314).

### Análisis de Feature Importance

Las 3 features más importantes según XGBoost (ver `data/reports/feature_importance.png`):

1. **`errorBalanceOrig`** — el desajuste matemático entre saldos es la firma más fuerte del fraude
2. **`isHighRiskType`** — confirmar que la transacción es TRANSFER o CASH_OUT casi duplica la confianza de la detección
3. **`amount`** — los fraudes tienden a montos significativamente más altos que las transacciones normales

Esto valida la **ingeniería de features**: 2 de las 3 variables más predictivas son features derivadas que NO existían en el dataset original.

---

## 5. Interfaz con Otros Módulos

El Módulo B no opera de forma aislada. Estos son los artifacts que produce y quién los consume:

| Artifact generado | Ubicación | Consumido por |
|---|---|---|
| `X_train.parquet`, `X_test.parquet` | `data/processed/` | **Módulo C (DL)** reutiliza el preprocesamiento exacto |
| `y_train.parquet`, `y_test.parquet` | `data/processed/` | **Módulo C (DL)** entrena sobre las mismas etiquetas |
| `scaler.pkl` | `data/processed/` | Reutilizable para inferencia online |
| `xgboost.pkl` | `data/models/` | **Módulo D (NLP)** lo carga para explicar predicciones; **Módulo E (pipeline)** lo invoca en el flujo integrado |
| `logistic_regression.pkl` | `data/models/` | Disponible como modelo alternativo |
| `model_comparison.csv` | `data/reports/` | **Módulo C (DL)** agrega su fila para comparación end-to-end |
| `metrics.json`, `evaluation_plots.png` | `data/reports/` | Documentación grupal y defensa oral |

**Por qué importa esto:** el rendimiento del Módulo C (Deep Learning) depende directamente de la calidad del preprocesamiento del Módulo B. Si el `errorBalanceOrig` no estuviera bien calculado, el MLP del Módulo C entrenaría sobre features pobres y se vería superado por XGBoost trivialmente.

---

## 6. Limitaciones del Pipeline

1. **Sesgo temporal:** El dataset cubre 30 días (744 horas). Los modelos entrenados aquí pueden degradarse frente a nuevas tácticas de fraude que emerjan después.
2. **Distribución de entrenamiento:** Se usó un sample estratificado de 200K registros para velocidad de iteración. El rendimiento en los 6.36M completos puede diferir.
3. **Sin datos de contexto:** El dataset no incluye información geográfica, dispositivo, historial de usuario ni IP — variables que los sistemas de fraude en producción sí incorporan.
4. **Umbrales fijos:** El umbral de 0.5 es un punto de partida. En producción se usaría un umbral optimizado por costo (p.ej., esperanza matemática de pérdida).
