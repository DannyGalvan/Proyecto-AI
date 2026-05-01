class Agent:
    def __init__(self, transactions):
        self.transactions = transactions  # Lista de transacciones
        self.visited = set()  # Nodos visitados
        self.path = []  # Ruta desde el inicio hasta el objetivo

    def is_goal(self, transaction):
        # Verifica si la transacción es fraudulenta (es el estado meta)
        return transaction['isFraud'] == 1

    def get_neighbors(self, current_transaction):
        # Obtiene las transacciones cercanas (simula la expansión de nodos en el espacio de estados)
        neighbors = []
        for transaction in self.transactions:
            if transaction != current_transaction:
                neighbors.append(transaction)
        return neighbors

    def search(self):
        # Algoritmo A* para buscar transacciones fraudulentas
        open_list = [(self.transactions[0], 0)]  # (transacción, costo acumulado)
        while open_list:
            current_transaction, current_cost = open_list.pop(0)

            # Verificar si hemos alcanzado el objetivo
            if self.is_goal(current_transaction):
                return self.path

            # Expansión de vecinos
            for neighbor in self.get_neighbors(current_transaction):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    open_list.append((neighbor, current_cost + 1))  # Aumenta el costo de la transacción
                    self.path.append(neighbor)

        return None  # Si no se encuentra ninguna transacción fraudulenta