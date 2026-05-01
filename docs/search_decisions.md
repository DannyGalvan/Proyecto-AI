# Justificación del Algoritmo de Búsqueda / CSP
 
## Elección del Algoritmo
 
### Algoritmo Seleccionado: A* (A-Star)
 
El algoritmo A* es un algoritmo de búsqueda informada utilizado para encontrar el camino más corto entre un nodo de inicio y un nodo objetivo. Elegimos A* debido a su eficiencia en la exploración de espacios de búsqueda grandes, combinando la búsqueda de menor costo (BFS) y la heurística (Greedy Best-First Search). Esto permite encontrar la solución de manera óptima mientras se mantiene eficiente en cuanto a tiempo de ejecución.
 
### Comparación con Alternativas
 
- **BFS (Breadth-First Search)**: Aunque BFS garantiza encontrar la solución más corta, no es eficiente en términos de tiempo, ya que explora todos los nodos posibles sin tener en cuenta la distancia al objetivo. Esto puede hacer que sea muy lento cuando se trata de un espacio de búsqueda grande.
- **Algoritmo de Dijkstra**: Dijkstra también encuentra el camino más corto, pero no utiliza ninguna heurística, lo que lo hace menos eficiente que A*. Sin embargo, es útil cuando no se tiene una buena estimación heurística de la distancia al objetivo.
 
### Complejidad
 
- **Complejidad temporal de A***: O(b^d), donde *b* es el factor de ramificación y *d* es la profundidad del árbol de búsqueda. A* es más eficiente que otros algoritmos como BFS porque utiliza la heurística para priorizar la exploración de nodos más prometedores.
- **Heurística**: La función heurística **h(n)** utilizada para A* se basa en características como el monto de la transacción, la frecuencia de las transacciones inusuales del cliente y la ubicación. Estas características permiten que el algoritmo enfoque su búsqueda en las transacciones más sospechosas.
 
### Decisiones de Diseño
 
- **Espacio de Estados**: El espacio de estados está compuesto por las transacciones financieras registradas en la base de datos. Cada estado es una transacción que debe ser evaluada por su riesgo de fraude.
- **Función de Evaluación**: La función de evaluación **f(n) = g(n) + h(n)** se calcula considerando el costo acumulado hasta la transacción actual **g(n)** (que en este caso podría estar relacionado con características como el monto o frecuencia de las transacciones) y la estimación heurística **h(n)** (que mide cuán sospechosa es la transacción en función de factores como el monto o el tipo de transacción).
 
## Limitaciones del Algoritmo
 
Aunque A* es un algoritmo eficiente, tiene limitaciones, especialmente cuando no se dispone de una heurística adecuada. Si la heurística no es precisa, el algoritmo puede volverse ineficiente, ya que podría explorar muchas transacciones irrelevantes. Además, A* no es adecuado para problemas en los que no se puede definir una heurística confiable.