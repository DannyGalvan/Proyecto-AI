# Propuesta y Fundamentos

## Dominio Elegido
**Dominio**: Finanzas  
**Problema**: Detección de fraudes financieros en transacciones bancarias. El objetivo es identificar transacciones fraudulentas realizadas por agentes maliciosos que buscan agotar las cuentas de los clientes a través de transferencias y retiros en efectivo.

## Justificación del Dataset
El dataset seleccionado contiene transacciones bancarias de una institución financiera, con datos relevantes para la detección de fraudes. Este dataset tiene más de 6 millones de registros, lo que permite entrenar modelos robustos para identificar transacciones inusuales.

**Características del dataset**:
- **URL del Dataset**: [Dataset](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset?resource=download)
- **Tamaño**: 6,360,000 registros
- **Columnas clave**:
  - **`step`**: Representa un paso en la simulación, con cada paso equivalente a una hora. El total abarca 744 pasos, lo que equivale a 30 días.
  - **`type`**: Tipo de transacción. Los valores posibles son:
    - `CASH_IN`
    - `CASH_OUT`
    - `DEBIT`
    - `PAYMENT`
    - `TRANSFER`
  - **`amount`**: Monto de la transacción en la moneda local. Los montos varían desde transacciones muy pequeñas hasta millones.
  - **`nameOrig`**: Identificación del cliente que inicia la transacción.
  - **`oldbalanceOrg`**: El saldo del cliente antes de la transacción.
  - **`newbalanceOrig`**: El saldo del cliente después de la transacción.
  - **`nameDest`**: Identificación del cliente receptor de la transacción.
  - **`oldbalanceDest`**: El saldo del destinatario antes de la transacción.
  - **`newbalanceDest`**: El saldo del destinatario después de la transacción.
  - **`isFraud`**: Indicador binario que marca si la transacción es fraudulenta (1) o no (0).
  - **`isFlaggedFraud`**: Marca las transacciones grandes, por encima de los 200,000, consideradas sospechosas o fraudulentas.

## Razonamiento del Uso de IA
La detección de fraudes financieros es un desafío complejo debido a las diferentes tácticas que utilizan los agentes fraudulentos. El enfoque propuesto involucra el uso de varios enfoques de IA, como el aprendizaje automático supervisado, aprendizaje profundo y procesamiento de lenguaje natural (NLP) para analizar y clasificar transacciones sospechosas. Esto incluirá:
- **Machine Learning (ML)**: Para clasificar las transacciones como fraudulentas o no fraudulentas basándose en características clave como el monto, el tipo de transacción y los saldos involucrados.
- **Deep Learning (DL)**: Para aprender patrones más complejos en las transacciones, como las relaciones no lineales entre diferentes atributos.
- **NLP/LLM**: Para procesar cualquier texto o descripción relacionada con las transacciones (si existe) y detectar posibles fraudes en mensajes o comentarios asociados a las transacciones.

## Módulos Involucrados
- **Módulo A**: Agente y Búsqueda / CSP
- **Módulo B**: Pipeline ML
- **Módulo C**: Deep Learning
- **Módulo D**: NLP / LLM
- **Módulo E**: Ética, Integración y Documentación