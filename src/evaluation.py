from typing import Iterable, List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Precision@K.

    Args:
        retrieved_ids: Ranked list of retrieved document or chunk ids.
        relevant_ids: Set of relevant document or chunk ids.
        k: Cutoff rank.

    Returns:
        Precision at k.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved_ids: Ranked list of retrieved document or chunk ids.
        relevant_ids: Set of relevant document or chunk ids.
        k: Cutoff rank.

    Returns:
        Recall at k.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)

    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    ranked_lists: Iterable[List[str]],
    relevant_ids_list: Iterable[Set[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank across multiple queries.

    Args:
        ranked_lists: Ranked retrieved ids for each query.
        relevant_ids_list: Relevant ids for each query.

    Returns:
        Mean reciprocal rank.
    """
    reciprocal_ranks = []

    for retrieved_ids, relevant_ids in zip(ranked_lists, relevant_ids_list):
        rank = 0

        for index, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant_ids:
                rank = index
                break

        reciprocal_ranks.append(0.0 if rank == 0 else 1.0 / rank)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
