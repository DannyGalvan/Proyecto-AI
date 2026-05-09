# Decisiones del Pipeline de Machine Learning

**Módulo:** B — Pipeline ML  
**Responsable:** Daniel Galvan  
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

### 1.3 Manejo del Desbalanceo de Clases

El dataset tiene ~0.13% de fraudes — un desbalanceo extremo de ~770:1.

**Estrategia elegida:** Se usó `class_weight='balanced'` en Logistic Regression y `scale_pos_weight` (ratio neg/pos) en XGBoost. Esto pesa los errores sobre la clase minoritaria sin modificar la distribución de datos.

**Alternativa descartada — SMOTE:** Se evaluó SMOTE (oversampling sintético) pero fue descartado porque:
1. Con 6M registros, SMOTE sobre 8,000 fraudes genera ruido artificial
2. El costo computacional es prohibitivo a esta escala
3. `scale_pos_weight` en XGBoost logra el mismo efecto de forma implícita y eficiente

### 1.4 Normalización

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

Resultados sobre 40,000 transacciones del test set (muestra estratificada de 200K):

| Métrica | Logistic Regression | XGBoost |
|---|---|---|
| Accuracy | 0.9667 | **0.9996** |
| Precision | 0.5543 | **0.9945** |
| Recall | 0.9726 | **0.9951** |
| F1 | 0.7061 | **0.9948** |
| ROC-AUC | 0.9944 | **0.9999** |
| Avg. Precision | 0.9314 | **0.9986** |
| Fraudes detectados (TP) | 1,598 / 1,643 | **1,635 / 1,643** |
| Fraudes perdidos (FN) | 45 | **8** |
| Falsas alarmas (FP) | 1,285 | **9** |

**XGBoost supera a Logistic Regression en todas las métricas.** La diferencia más importante en este dominio es en los Falsos Negativos: LR deja pasar 45 fraudes vs 8 de XGBoost. Cada FN representa un fraude real no bloqueado — costo económico directo para el banco y el cliente.

La alta Precision de LR (0.55) muestra que el modelo lineal no puede separar limpiamente fraudes de normales: por cada fraude detectado, genera 1,285 falsas alarmas, lo que haría el sistema inoperable en producción.

XGBoost captura las interacciones no lineales entre `errorBalanceOrig`, `isHighRiskType` y `amountToOrigRatio` que son la firma matemática del fraude en este dataset.

---

## 5. Limitaciones del Pipeline

1. **Sesgo temporal:** El dataset cubre 30 días (744 horas). Los modelos entrenados aquí pueden degradarse frente a nuevas tácticas de fraude que emerjan después.
2. **Distribución de entrenamiento:** Se usó un sample estratificado de 200K registros para velocidad de iteración. El rendimiento en los 6.36M completos puede diferir.
3. **Sin datos de contexto:** El dataset no incluye información geográfica, dispositivo, historial de usuario ni IP — variables que los sistemas de fraude en producción sí incorporan.
4. **Umbrales fijos:** El umbral de 0.5 es un punto de partida. En producción se usaría un umbral optimizado por costo (p.ej., esperanza matemática de pérdida).
