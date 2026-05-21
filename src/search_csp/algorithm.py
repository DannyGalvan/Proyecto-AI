import heapq

FRAUD_TYPES = {"TRANSFER", "CASH_OUT"}


def heuristic(tx: dict) -> float:
    """
    Admissible fraud risk score in [0, 1]. Higher = more suspicious.

    Domain rules derived from dataset analysis:
    - Fraud only occurs in TRANSFER and CASH_OUT transaction types.
    - Complete balance drain (oldbalanceOrg > 0, newbalanceOrig == 0) is the
      strongest single indicator: it means the account was fully emptied.
    - Amount-to-balance ratio captures partial drains.

    The score is capped at 1.0 to remain admissible (never overestimates).
    """
    score = 0.0

    if tx.get("type") in FRAUD_TYPES:
        score += 0.4

    old_bal = tx.get("oldbalanceOrg", 0)
    new_bal = tx.get("newbalanceOrig", 0)
    if old_bal > 0 and new_bal == 0:
        score += 0.4  # complete balance drain

    if old_bal > 0:
        score += 0.2 * min(tx.get("amount", 0) / old_bal, 1.0)

    return min(score, 1.0)


def a_star_search(
    transactions: list[dict],
    get_neighbors_fn,
) -> dict | None:
    """
    A* search over the transaction graph.

    Priority function: f(n) = g(n) + (1 - h(n))
    Subtracting h from 1 converts a high-risk score into a low cost,
    so heapq (min-heap) naturally explores the most suspicious nodes first.

    Args:
        transactions: list of transaction dicts.
        get_neighbors_fn: callable(idx: int) -> list[int] of neighbor indices.

    Returns the first fraudulent transaction found, or None.
    """
    open_list: list[tuple] = []
    counter = 0
    visited: set[int] = set()

    for i, tx in enumerate(transactions):
        f = 1.0 - heuristic(tx)
        heapq.heappush(open_list, (f, 0, counter, i))
        counter += 1

    while open_list:
        f, g, _, idx = heapq.heappop(open_list)

        if idx in visited:
            continue
        visited.add(idx)

        if transactions[idx].get("isFraud") == 1:
            return transactions[idx]

        for neighbor_idx in get_neighbors_fn(idx):
            if neighbor_idx not in visited:
                new_g = g + 1
                new_f = new_g + (1.0 - heuristic(transactions[neighbor_idx]))
                counter += 1
                heapq.heappush(open_list, (new_f, new_g, counter, neighbor_idx))

    return None
