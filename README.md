# Sistema de IA para Detección de Fraude Financiero

**Curso:** Inteligencia Artificial – 045  
**Universidad Mariano Gálvez de Guatemala**  
**Dominio:** Finanzas — Detección de fraude en transacciones bancarias

## Equipo

| Integrante | Módulo |
|---|---|
| Jackeline Sanchez | A — Agente y Búsqueda / CSP + E — Ética e Integración |
| Daniel Galvan | B — Pipeline ML |
| Gabriel Valdez | C — Deep Learning |
| Sergio Santos | D — NLP / LLM |

## Dataset

**Fuente:** [Financial Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)  
**Tamaño:** 6,360,000 transacciones bancarias  
**Target:** `isFraud` (binario: 0 = normal, 1 = fraude)

## Indice de Documentacion

Para navegar la documentacion completa desde un solo lugar:

- `docs/INDICE_DOCUMENTACION.md`

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración Centralizada (.env)

Todas las rutas y parámetros globales ahora se controlan desde un solo archivo: `.env`.

```bash
# 1. Copiar plantilla
copy .env.example .env

# 2. Editar variables si tu entorno cambia
# DATASET_PATH=data/financial_data.csv
# PROCESSED_DIR=data/processed
# MODELS_DIR=data/models
# REPORTS_DIR=data/reports
# ML_SAMPLE_SIZE=200000
# PREDICTION_THRESHOLD=0.5
```

Los scripts en `src/` consumen esta configuración automáticamente.

### Descargar el dataset

**Opción 1 — Kaggle CLI (recomendado):**
```bash
# Configurar credenciales: ir a kaggle.com → Account → API → Create Token
# Colocar kaggle.json en ~/.kaggle/
pip install kaggle
kaggle datasets download -d sriharshaeedala/financial-fraud-detection-dataset
unzip financial-fraud-detection-dataset.zip -d data/
# Renombrar/mover el CSV principal a:
# data/financial_data.csv
```

**Opción 2 — Manual:**  
Descargar desde el enlace de arriba y colocar el CSV en `data/financial_data.csv`.

## Estructura del Proyecto

```
Proyecto-AI/
├── src/
│   ├── search_csp/       # Módulo A: Agente y búsqueda
│   │   ├── agent.py
│   │   └── algorithm.py
│   ├── data/
│   │   └── load_data.py
│   ├── ml/               # Módulo B: Pipeline ML
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── dl/               # Módulo C: Deep Learning
│   │   ├── model.py
│   │   └── train_dl.py
│   ├── nlp/              # Módulo D: NLP / LLM
│   │   ├── nlp_component.py
│   └── integration/
│       └── pipeline.py
├── notebooks/
│   ├── demo_search.ipynb
│   ├── ml_analysis.ipynb
│   ├── dl_analysis.ipynb
│   └── nlp_demo.ipynb
├── data/
│   ├── financial_data.csv # Dataset principal
│   ├── processed/        # Datos preprocesados (generados)
│   ├── models/           # Modelos entrenados (generados)
│   └── reports/          # Métricas y gráficas (generados)
├── docs/
│   ├── propuesta.md
│   ├── asignacion_modulos.md
│   ├── agente_formulacion.md
│   ├── search_decisions.md
│   ├── ml_decisions.md
│   ├── dl_decisions.md
│   ├── ethics_analysis.md
│   ├── checklist_defensa_modulos.md
│   ├── PLAN_SEMANA5.md
│   └── INDICE_DOCUMENTACION.md
└── requirements.txt
```

## Ejecutar el Pipeline ML (Semana 3)

```bash
# 1. Preprocesar datos (sample de 200K para demo; quitar sample_size para usar todo)
cd src/ml
python preprocess.py

# 2. Entrenar modelos (Logistic Regression vs XGBoost)
python train.py

# 3. Evaluar y generar reportes
python evaluate.py

# 4. Ver análisis completo en el notebook
cd ../../
jupyter notebook notebooks/ml_analysis.ipynb
```

## Ejecutar el Pipeline Completo (Semana 5)

```bash
# 1. (Opcional) Reprocesar y entrenar ML
python src/ml/preprocess.py
python src/ml/train.py

# 2. (Opcional) Entrenar DL
python src/dl/train_dl.py

# 3. Ejecutar pipeline integrado (datos -> ML/DL -> NLP)
python src/integration/pipeline.py

# 4. Ver demo NLP con ejemplos reales del dominio
jupyter notebook notebooks/nlp_demo.ipynb

# 5. Revisar análisis ético
type docs/ethics_analysis.md
```

Salida principal del pipeline integrado:
- `data/reports/integration_summary.json` (resumen final ML + DL + NLP)

Los resultados se guardan automáticamente en `data/reports/`:
- `model_comparison.csv` — tabla de métricas
- `dl_metrics.json` — métricas del modelo de Deep Learning
- `dl_training_curves.png` — curvas de entrenamiento DL
- `dl_prediction_examples.csv` — ejemplos reales de aciertos/errores DL
- `integration_summary.json` — resultado final del flujo integrado
