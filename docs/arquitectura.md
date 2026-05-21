# Arquitectura del Sistema

**Proyecto:** Sistema de IA para Detección de Fraude Financiero  
**Dataset:** PaySim — 6,362,620 transacciones bancarias

---

## Diagrama de Arquitectura General

```mermaid
flowchart TD
    DS[(PaySim Dataset\n6.36M transacciones\nCSV)]

    subgraph A ["Módulo A — Agente y Búsqueda / CSP\n(Jackeline Sanchez)"]
        A1[FraudDetectionAgent\nagent.py]
        A2[A* Search\nalgorithm.py]
        A3([demo_search.ipynb])
        A1 --> A2 --> A3
    end

    subgraph B ["Módulo B — Pipeline ML\n(Sergio Santos)"]
        B1[preprocess.py\nlimpieza · features · escalado]
        B2[train.py\nLogistic Regression · XGBoost]
        B3[evaluate.py\nprecision · recall · F1 · ROC-AUC]
        B1 --> B2 --> B3
    end

    subgraph C ["Módulo C — Deep Learning\n(Gabriel Valdez)"]
        C1[model.py\nMLP 128→64→32 + Dropout + BN]
        C2[train_dl.py\nentrenamiento · early stopping]
        C1 --> C2
    end

    subgraph D ["Módulo D — NLP / LLM\n(Daniel Galvan)"]
        D1[nlp_component.py\nexplicabilidad por feature importance]
    end

    subgraph E ["Módulo E — Integración y Ética\n(Jackeline Sanchez)"]
        E1[pipeline.py\ndatos → ML/DL → NLP → resultado]
        E2[(integration_summary.json)]
        E1 --> E2
    end

    DS --> A1
    DS --> B1
    B1 -->|X_train · X_test\n.parquet| C1
    B2 -->|xgboost.pkl| D1
    C2 -->|dl_final_model.keras| E1
    B3 --> E1
    D1 --> E1
    B2 -->|xgboost.pkl| E1
```

---

## Flujo de Datos End-to-End

```mermaid
sequenceDiagram
    participant CSV as PaySim CSV
    participant B as Módulo B (ML)
    participant C as Módulo C (DL)
    participant D as Módulo D (NLP)
    participant E as Módulo E (Pipeline)
    participant A as Módulo A (A*)

    CSV->>B: 6.36M filas raw
    B->>B: feature engineering + escalado
    B->>C: X_train, X_test (parquet)
    B->>E: xgboost.pkl + métricas
    C->>E: dl_final_model.keras + dl_metrics.json
    B->>D: xgboost.pkl (feature importances)
    D->>E: explicaciones en texto
    E->>E: ensemble ML + DL + NLP
    E-->>E: integration_summary.json

    CSV->>A: muestra de transacciones
    A->>A: A* prioriza por heurística h(n)
    A-->>A: transacción fraudulenta encontrada
```

---

## Dependencias entre Artefactos

```mermaid
graph LR
    CSV[(financial_data.csv)] --> PRE[preprocess.py]
    PRE --> XTR[X_train.parquet]
    PRE --> XTE[X_test.parquet]
    PRE --> SCL[scaler.pkl]

    XTR --> TRAIN[train.py]
    XTE --> TRAIN
    TRAIN --> XGB[xgboost.pkl]
    TRAIN --> LR[logistic_regression.pkl]

    XTR --> DL[train_dl.py]
    XTE --> DL
    SCL --> DL
    DL --> DLMOD[dl_final_model.keras]
    DL --> DLMET[dl_metrics.json]
    DL --> DLEX[dl_prediction_examples.csv]

    XGB --> NLP[nlp_component.py]
    XTE --> NLP

    XGB --> PIPE[pipeline.py]
    DLMOD --> PIPE
    NLP --> PIPE
    PIPE --> OUT[(integration_summary.json)]
```
