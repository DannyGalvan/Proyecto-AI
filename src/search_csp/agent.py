from src.search_csp.algorithm import a_star_search


class FraudDetectionAgent:
    """
    Models fraud detection as a graph search problem.

    State space: each transaction is a node identified by its list index.
    Initial state: the full set of unreviewed transactions in the open list.
    Goal state: any transaction where isFraud == 1.
    Actions: expand to related transactions from the same origin account
             within a ±5-step time window.
    Evaluation function: f(n) = g(n) + (1 - h(n)), where h is the
             domain heuristic from algorithm.py (fraud risk score in [0,1]).
    """

    def __init__(self, transactions: list[dict]):
        self.transactions = transactions
        self._account_index: dict[str, list[int]] = {}
        self._build_account_index()

    def _build_account_index(self) -> None:
        """Pre-index transactions by origin account for O(1) neighbor lookup."""
        for i, tx in enumerate(self.transactions):
            key = tx.get("nameOrig", "")
            self._account_index.setdefault(key, []).append(i)

    def is_goal(self, transaction: dict) -> bool:
        return transaction.get("isFraud") == 1

    def get_neighbors(self, idx: int) -> list[int]:
        """
        Neighbors = transactions from the same origin account within ±5 steps.
        This restricts branching to account-level activity patterns,
        reflecting the domain reality that fraud chains within one account.
        """
        tx = self.transactions[idx]
        origin = tx.get("nameOrig", "")
        step = tx.get("step", 0)
        return [
            i
            for i in self._account_index.get(origin, [])
            if i != idx and abs(self.transactions[i].get("step", 0) - step) <= 5
        ]

    def search(self) -> dict | None:
        """
        Run A* to find the highest-priority fraudulent transaction.
        Returns the transaction dict or None if no fraud found.
        """
        return a_star_search(self.transactions, self.get_neighbors)
