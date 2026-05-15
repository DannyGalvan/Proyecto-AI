# Checklist de Defensa por Módulo

## Objetivo de este documento
Este checklist sirve para preparar la defensa oral individual y grupal del proyecto final, asegurando que cada módulo pueda explicar:
- Qué hace técnicamente.
- Qué archivos lo implementan.
- Qué datos recibe y qué salida entrega.
- Cómo se integra con el siguiente módulo.

---

## Módulo A - Agente y Búsqueda / CSP

### Archivos clave
- src/search_csp/agent.py
- src/search_csp/algorithm.py
- notebooks/demo_search.ipynb
- docs/agente_formulacion.md
- docs/search_decisions.md

### Checklist técnico
- [ ] Explica estado inicial, estado meta, acciones y función de evaluación.
- [ ] Explica por qué se eligió el algoritmo de búsqueda/CSP implementado.
- [ ] Muestra un caso real del dominio y cómo el agente decide.
- [ ] Justifica complejidad y limitaciones del algoritmo.
- [ ] Compara con al menos una alternativa descartada.

### Preguntas de defensa esperadas
- ¿Por qué este algoritmo y no otro?
- ¿Qué pasa si el espacio de estados crece mucho?
- ¿Qué parte del dominio se beneficia de esta búsqueda?

---

## Módulo B - Pipeline ML

### Archivos clave
- src/data/load_data.py
- src/ml/preprocess.py
- src/ml/train.py
- src/ml/evaluate.py
- notebooks/ml_analysis.ipynb
- docs/ml_decisions.md

### Checklist técnico
- [ ] Explica cómo se carga el dataset real desde data/financial_data.csv.
- [ ] Explica limpieza, feature engineering, encoding y normalización.
- [ ] Explica por qué se entrenaron al menos dos modelos.
- [ ] Muestra métricas relevantes: precision, recall, F1, ROC-AUC.
- [ ] Explica análisis de errores (FP/FN) y trade-offs.
- [ ] Identifica el modelo final elegido y por qué.

### Preguntas de defensa esperadas
- ¿Cuál fue el criterio para seleccionar el modelo ganador?
- ¿Qué errores son más caros en este dominio: FP o FN?
- ¿Qué feature aporta más y por qué?

---

## Módulo C - Deep Learning

### Archivos clave
- src/dl/model.py
- src/dl/train_dl.py
- notebooks/dl_analysis.ipynb
- docs/dl_decisions.md

### Checklist técnico
- [ ] Explica arquitectura de red y decisión de hiperparámetros.
- [ ] Muestra curvas de entrenamiento/validación.
- [ ] Explica estrategia contra overfitting.
- [ ] Muestra comparación cuantitativa contra ML.
- [ ] Explica ejemplos correctos e incorrectos (dl_prediction_examples.csv).

### Preguntas de defensa esperadas
- ¿Por qué esta arquitectura y no una más simple?
- ¿Qué muestra la curva de entrenamiento sobre generalización?
- ¿Cuándo conviene usar DL sobre ML en este proyecto?

---

## Módulo D - NLP / LLM

### Archivos clave
- src/nlp/nlp_component.py
- notebooks/nlp_demo.ipynb

### Checklist técnico
- [ ] Explica cómo NLP convierte predicciones en texto explicable.
- [ ] Muestra explain_prediction con transacciones reales del test.
- [ ] Muestra summarize_batch con resultados reales del lote.
- [ ] Muestra bloque de explicabilidad con FN/FP reales en nlp_demo.ipynb.
- [ ] Explica limitaciones del componente (no causalidad, dependencia del modelo base).

### Preguntas de defensa esperadas
- ¿Qué valor aporta NLP al negocio además de una probabilidad?
- ¿Cómo se detecta una explicación potencialmente engañosa?
- ¿Qué limitación crítica tiene el componente NLP?

---

## Módulo E - Ética, Integración y Documentación

### Archivos clave
- src/integration/pipeline.py
- docs/ethics_analysis.md
- README.md
- data/reports/integration_summary.json

### Checklist técnico
- [ ] Ejecuta pipeline completo con un solo comando.
- [ ] Muestra resultado final integrado ML + DL + NLP.
- [ ] Presenta análisis ético con evidencia cuantitativa real.
- [ ] Explica fairness por subgrupos y riesgos de uso irresponsable.
- [ ] Presenta mitigaciones aplicables técnicamente.
- [ ] Verifica README reproducible desde cero.

### Preguntas de defensa esperadas
- ¿Qué falla primero si rompes un módulo del pipeline?
- ¿Qué controles de seguridad/ética aplican antes de producción?
- ¿Cómo se monitorea drift y fairness en el tiempo?

---

## Cómo se comunican y complementan los módulos

Flujo funcional completo:
1. Módulo B carga y transforma datos reales.
2. Módulo B entrena modelos ML y guarda artefactos en data/models.
3. Módulo C entrena modelo DL y guarda métricas y ejemplos en data/reports.
4. Módulo D toma salidas de ML/DL y genera explicaciones y resúmenes legibles.
5. Módulo E integra todo en src/integration/pipeline.py y produce resultado final consolidado.
6. Módulo A aporta razonamiento clásico para decisiones/planificación complementarias al pipeline predictivo.

Relación de dependencia entre módulos:
- B y C dependen de datos consistentes de entrada.
- D depende de predicciones de B/C para explicar riesgo.
- E depende de todos para demostrar sistema de extremo a extremo.
- A complementa la toma de decisión con lógica/búsqueda donde aplique.

---

## Explicación simple de qué hace la solución y cómo funciona

Qué hace:
- Detecta riesgo de fraude en transacciones bancarias usando datos reales.
- Combina modelos predictivos (ML y DL) con un componente NLP que explica resultados.

Cómo funciona:
1. Lee el dataset real desde data/financial_data.csv.
2. Limpia y transforma los datos para crear features útiles.
3. Evalúa modelos ML y DL para estimar probabilidad de fraude.
4. Convierte esas predicciones en explicaciones entendibles para análisis humano.
5. Entrega un resumen final integrado en data/reports/integration_summary.json.

Por qué es una solución de IA de extremo a extremo:
- Usa razonamiento clásico (A), aprendizaje supervisado (B), deep learning (C), NLP moderno (D) e integración ética/técnica (E) sobre un mismo problema real.

---

## Checklist rápido pre-defensa (grupo completo)
- [ ] El pipeline corre en limpio con un comando y genera integration_summary.json.
- [ ] Cada propietario presenta solo su módulo.
- [ ] Cada integrante puede explicar el módulo anterior y siguiente al suyo.
- [ ] README permite reproducir desde cero sin pasos ocultos.
- [ ] ethics_analysis.md tiene evidencia real, no texto genérico.
- [ ] Notebooks muestran resultados reales y no simulaciones.
