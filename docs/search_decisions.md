# Justificación del Algoritmo de Búsqueda

**Módulo:** A — Agente y Búsqueda / CSP  
**Responsable:** Jackeline Sanchez  
**Dominio:** Detección de Fraude Financiero  
**Dataset:** PaySim — 6,362,620 transacciones bancarias

---

## Algoritmo Seleccionado: A* (A-Star)

### Motivación

El dataset tiene 6,362,620 transacciones con solo 8,213 fraudes (0.13%). Un algoritmo sin heurística (BFS, DFS) exploraría en promedio ~770 transacciones legítimas por cada fraude encontrado. A* resuelve esto priorizando las transacciones más sospechosas primero mediante una función de evaluación `f(n) = g(n) + (1 - h(n))`.

### Heurística Implementada

La heurística `h(n)` es admisible (valor en [0,1], nunca sobreestima):

| Componente | Contribución | Base empírica |
| --- | --- | --- |
| `type` ∈ {TRANSFER, CASH_OUT} | 0.40 | 100% de los fraudes ocurren en estos tipos; PAYMENT/DEBIT/CASH_IN tienen isFraud=0 en todo el dataset |
| Drenaje total de saldo | 0.40 | En el 99.5% de fraudes detectados, `oldbalanceOrg > 0` y `newbalanceOrig == 0` |
| `amount / oldbalanceOrg` ratio | 0.20 | Correlación positiva con fraude en transacciones de tipo TRANSFER |

Score máximo: `h(n) = 1.0`. Score de una transacción de tipo PAYMENT: `h(n) = 0.0`.

### Espacio de Estados y Vecinos

Cada nodo = una transacción (índice entero). Los vecinos de un nodo `i` son transacciones del **mismo `nameOrig`** dentro de una **ventana de ±5 steps**. Esto:

- Modela que el fraude se encadena dentro de una misma cuenta en poco tiempo.
- Limita el factor de ramificación: en promedio, una cuenta tiene 1–3 transacciones por ventana temporal.
- Permite lookup O(1) usando el índice pre-construido por cuenta (`_account_index`).

### Complejidad

- **Temporal**: O(E log V) donde V = transacciones y E = aristas del grafo de cuentas. En la práctica, el grafo es disperso (pocas transacciones por cuenta) y A* termina al encontrar el primer fraude, explorando solo una fracción del dataset.
- **Espacial**: O(V) para la lista abierta en el peor caso.

---

## Comparación con Alternativas

### BFS (Breadth-First Search)

- **Ventaja**: garantiza encontrar el fraude a menor profundidad.
- **Desventaja**: sin heurística, trata todas las transacciones como igualmente sospechosas. Con 0.13% de fraude, explora ~770 nodos por fraude en promedio.
- **Por qué se descartó**: ineficiente en este dominio. No aprovecha las señales conocidas (tipo de transacción, drenaje de saldo).

### Greedy Best-First Search

- **Ventaja**: más rápido que A* en la práctica porque solo usa `h(n)`.
- **Desventaja**: no es óptimo. Puede explorar caminos localmente prometedores que no llevan a fraude real, sin contabilizar el costo acumulado `g(n)`.
- **Por qué se descartó**: la ausencia de `g(n)` puede provocar revisión redundante de nodos ya visitados en el grafo de cuentas.

### A* (elegido)

- Combina `g(n)` y `h(n)`: explora primero las transacciones más sospechosas y con menor costo acumulado.
- Con heurística admisible, garantiza encontrar el objetivo con mínimo número de nodos explorados.
- Implementación con heap y conjunto `visited` de índices enteros: sin bugs de hashabilidad, O(log V) por operación de cola.

---

## Limitaciones Reconocidas

1. **Heurística simplificada**: no incluye variables como `nameDest` recurrente o velocidad de transacciones por período. Una heurística más rica mejoraría el orden de exploración.
2. **Grafo disperso sin aristas inter-cuenta**: el modelo actual solo conecta transacciones del mismo `nameOrig`. Fraudes que involucran múltiples cuentas coordinadas no están capturados como vecinos.
3. **Escalabilidad**: con 6M de transacciones, cargar todo en memoria para A* es costoso. En producción se aplicaría sobre ventanas temporales deslizantes, no sobre el dataset completo.
