"""
DatasetBuilder: calls RAG functions directly to build retrieval results.

Bypasses the schema_retriever node intentionally — the node doesn't expose
per-doc page_content or pre-FK-expansion table sets needed for evaluation.

One DB call (schema_extractor) is cached for the entire evaluation run.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document

from evaluation.ground_truth import GroundTruthEntry, GROUND_TRUTH
from schema_rag_pipeline import (
    init_embeddings,
    schema_extractor,
    smart_retrieve,
    fk_expand,
)


@dataclass
class RetrievalResult:
    query_id: str
    query: str
    # Raw documents from smart_retrieve() — before FK expansion
    retrieved_docs: list[Document]
    # Table names directly retrieved (pre-FK-expand)
    retrieved_tables: set[str]
    # Table names after FK graph walk (what the node ultimately uses)
    fk_expanded_tables: set[str]
    # doc.page_content per retrieved doc — used as Ragas retrieved_contexts
    # (business description + sample queries, NOT DDL)
    page_contents: list[str]


class DatasetBuilder:
    """Build retrieval results for a list of GroundTruthEntry records.

    Usage:
        builder = DatasetBuilder()
        results = builder.build(GROUND_TRUTH)   # dict[query_id, RetrievalResult]
    """

    def __init__(self) -> None:
        self._schema_info: dict | None = None

    def _get_schema(self) -> dict:
        """Return cached schema_extractor() result (one DB call per session)."""
        if self._schema_info is None:
            self._schema_info = schema_extractor()
        return self._schema_info

    def build(
        self, entries: list[GroundTruthEntry] | None = None
    ) -> dict[str, RetrievalResult]:
        """Run retrieval for each entry and return a query_id → RetrievalResult map.

        Args:
            entries: Ground truth entries to evaluate. Defaults to all 30 from GROUND_TRUTH.

        Returns:
            dict mapping query_id to its RetrievalResult.
        """
        if entries is None:
            entries = GROUND_TRUTH

        # Ensure embeddings are loaded before any retrieval
        init_embeddings()

        schema = self._get_schema()
        results: dict[str, RetrievalResult] = {}

        for entry in entries:
            docs = smart_retrieve(entry.query, schema)
            expanded = fk_expand(docs, schema)

            results[entry.query_id] = RetrievalResult(
                query_id=entry.query_id,
                query=entry.query,
                retrieved_docs=docs,
                retrieved_tables={doc.metadata["table_name"] for doc in docs},
                fk_expanded_tables=expanded,
                page_contents=[doc.page_content for doc in docs],
            )

        return results
