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

## Instalación

```bash
pip install -r requirements.txt
```

# Instalar dependencias NLP (si no están incluidas):
python -m textblob.download_corpora

### Descargar el dataset

**Opción 1 — Kaggle CLI (recomendado):**
```bash
# Configurar credenciales: ir a kaggle.com → Account → API → Create Token
# Colocar kaggle.json en ~/.kaggle/
pip install kaggle
kaggle datasets download -d sriharshaeedala/financial-fraud-detection-dataset
unzip financial-fraud-detection-dataset.zip -d data/raw/
```

**Opción 2 — Manual:**  
Descargar desde el enlace de arriba y colocar el CSV en `data/raw/`.

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
│   │   └── nlp_demo.ipynb
│   └── integration/
│       └── pipeline.py
├── notebooks/
│   ├── demo_search.ipynb
│   └── ml_analysis.ipynb
├── data/
│   ├── raw/              # Dataset CSV aquí
│   ├── processed/        # Datos preprocesados (generados)
│   ├── models/           # Modelos entrenados (generados)
│   └── reports/          # Métricas y gráficas (generados)
├── docs/
│   ├── propuesta.md
│   ├── asignacion_modulos.md
│   ├── agente_formulacion.md
│   ├── search_decisions.md
│   ├── ml_decisions.md
│   └── dl_decisions.md
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
# 1. Ejecutar el pipeline de integración (ML/DL + NLP):
cd src/integration
python pipeline.py

# 2. Demostración del componente NLP:
cd ../nlp
jupyter notebook nlp_demo.ipynb

# 3. Análisis ético:
cd ../../docs
# Editar y revisar ethics_analysis.md
```

Los resultados se guardan automáticamente en `data/reports/`:
- `model_comparison.csv` — tabla de métricas
- `roc_curves.png` — curvas ROC comparadas
- `confusion_matrices.png` — matrices de confusión
- `feature_importance.png` — importancia de variables
