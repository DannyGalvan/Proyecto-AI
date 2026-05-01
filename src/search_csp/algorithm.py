import heapq

def heuristic(transaction):
    """
    Heurística para el algoritmo A*. En este caso, la heurística podría basarse en el monto de la transacción,
    la frecuencia de transacciones del cliente y otras características sospechosas.
    """
    # La heurística puede ajustarse dependiendo de los datos disponibles.
    return transaction['amount'] * 0.1  # Ejemplo: más alto es más sospechoso

def a_star_search(transactions):
    """
    Algoritmo A* para la detección de fraudes en las transacciones.
    """
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(transactions[0]), 0, transactions[0]))  # (f(n), g(n), transacción)
    closed_list = set()

    while open_list:
        current_f, current_g, current_transaction = heapq.heappop(open_list)

        if current_transaction['isFraud'] == 1:
            return current_transaction  # Se encontró una transacción fraudulenta

        closed_list.add(current_transaction)

        for neighbor in get_neighbors(current_transaction, transactions):
            if neighbor not in closed_list:
                cost = current_g + 1
                heapq.heappush(open_list, (cost + heuristic(neighbor), cost, neighbor))

    return None

def get_neighbors(current_transaction, transactions):
    """
    Simula los vecinos de una transacción, que en este caso serían las transacciones cercanas en el espacio de estados.
    """
    neighbors = []
    for transaction in transactions:
        if transaction != current_transaction:
            neighbors.append(transaction)
    return neighbors