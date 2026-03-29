"""
Deterministic table-level metrics for schema retriever evaluation.

No LLM calls — pure set arithmetic over table name sets.
These run as part of Level 1 evaluation (fast, free).
"""


def table_recall(required: set[str], retrieved: set[str]) -> float:
    """Fraction of required tables present in the retrieved (FK-expanded) set.

    A recall of 1.0 means every table needed to answer the query was retrieved.
    A recall of 0.0 means none of the required tables were found.
    """
    if not required:
        return 1.0
    return len(required & retrieved) / len(required)


def table_precision(
    required: set[str], optional: set[str], retrieved: set[str]
) -> float:
    """Fraction of retrieved tables that are either required or optional.

    Measures noise in retrieval — tables that are neither required nor
    acceptable FK-expansion collateral count against precision.
    """
    if not retrieved:
        return 1.0
    acceptable = required | optional
    return len(retrieved & acceptable) / len(retrieved)


def tables_missed(required: set[str], retrieved: set[str]) -> set[str]:
    """Required tables absent from the retrieved set (recall failures)."""
    return required - retrieved


def tables_hallucinated(
    required: set[str], optional: set[str], retrieved: set[str]
) -> set[str]:
    """Retrieved tables that are neither required nor acceptable optional tables."""
    acceptable = required | optional
    return retrieved - acceptable
