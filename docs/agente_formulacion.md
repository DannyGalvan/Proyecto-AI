# Formulación del Problema como Agente

## Estado Inicial
El agente inicia con un conjunto de transacciones financieras que están registradas en la base de datos de la institución financiera. Cada transacción incluye información como el monto, la ubicación, el tipo de transacción y una etiqueta que indica si la transacción es legítima o fraudulenta.

## Estado Meta
El objetivo del agente es identificar todas las transacciones fraudulentas. El estado meta es un conjunto de transacciones que están correctamente clasificadas como fraudulentas o no fraudulentas.

## Acciones
El agente ejecuta los siguientes pasos:
1. **Recolección de datos**: Extraer las transacciones de la base de datos.
2. **Preprocesamiento**: Limpiar y normalizar los datos para la posterior clasificación.
3. **Clasificación**: Ejecutar un algoritmo de clasificación para predecir si una transacción es fraudulenta o no.
4. **Evaluación**: Evaluar el desempeño del modelo utilizando métricas de precisión, recall y F1.

## Función de Evaluación
La función de evaluación del agente mide cuán bien el modelo predice las transacciones fraudulentas:
- **Precisión**: Proporción de transacciones clasificadas como fraudulentas que realmente son fraudulentas.
- **Recall**: Proporción de transacciones fraudulentas que son correctamente identificadas.
- **F1 Score**: Media armónica de la precisión y recall.

## Justificación del Algoritmo
Se utilizará un algoritmo de búsqueda como **A*** para optimizar la búsqueda de transacciones fraudulentas. A* es adecuado para este problema debido a su capacidad para encontrar rutas eficientes mediante una heurística que guía la búsqueda hacia la meta (transacciones fraudulentas). La heurística podría basarse en características como la cantidad de transacciones inusuales de un cliente o el monto total de transacciones en un periodo corto de tiempo.