"""
Integration tests for the schema retriever.

Requires: live PostgreSQL DB + ChromaDB (populated via schema_rag_pipeline.py).

Level 1 — Table Name Recall (deterministic, no LLM cost):
  @pytest.mark.integration
  Checks that all required tables for each ground truth query appear in the
  FK-expanded retrieval set. Pass threshold: recall >= 0.85 per query.

Level 2 — Context Relevancy (Ragas LLM-judge):
  @pytest.mark.integration @pytest.mark.llm_eval
  Checks aggregate Ragas context_precision and context_recall scores.
  Requires OPENAI_API_KEY.

Run Level 1 only (fast, no API cost):
    pytest tests/integration/ -m "integration and not llm_eval" -v

Run everything:
    pytest tests/integration/ -m integration -v
"""

import pytest

from evaluation.dataset_builder import DatasetBuilder, RetrievalResult
from evaluation.ground_truth import GROUND_TRUTH, GroundTruthEntry
from evaluation.ragas_evaluator import run_retrieval_only_evaluation
from evaluation.metrics_config import THRESHOLDS
from evaluation.table_metrics import table_recall, tables_missed


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped fixtures — one DB call + one ChromaDB call per pytest session
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def retrieval_results() -> dict[str, RetrievalResult]:
    """Build retrieval results for all 30 ground truth queries once per session."""
    return DatasetBuilder().build(GROUND_TRUTH)


@pytest.fixture(scope="session")
def ragas_scores(retrieval_results) -> dict:
    """Run Ragas evaluation once and cache scores for both Level 2 tests."""
    return run_retrieval_only_evaluation(
        retrieval_results=retrieval_results,
        ground_truth=GROUND_TRUTH,
        save_results=True,
        run_ragas=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Deterministic Table Name Recall
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel1TableRecall:
    """Level 1: every required table must appear in the FK-expanded retrieval set."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "entry",
        GROUND_TRUTH,
        ids=lambda e: e.query_id,
    )
    def test_required_tables_recalled(
        self, entry: GroundTruthEntry, retrieval_results: dict[str, RetrievalResult]
    ):
        """Each required table must be in the FK-expanded set (recall >= 0.85)."""
        result = retrieval_results[entry.query_id]
        recall = table_recall(entry.required_tables, result.fk_expanded_tables)
        missed = tables_missed(entry.required_tables, result.fk_expanded_tables)

        assert recall >= 0.85, (
            f"[{entry.query_id}] '{entry.query}'\n"
            f"  recall={recall:.2f}  missed={sorted(missed)}\n"
            f"  retrieved (direct): {sorted(result.retrieved_tables)}\n"
            f"  fk_expanded:        {sorted(result.fk_expanded_tables)}\n"
            f"  required:           {sorted(entry.required_tables)}"
        )

    @pytest.mark.integration
    def test_overall_mean_recall(
        self, retrieval_results: dict[str, RetrievalResult]
    ):
        """Mean recall across all 30 queries must be >= 0.85."""
        recalls = [
            table_recall(e.required_tables, retrieval_results[e.query_id].fk_expanded_tables)
            for e in GROUND_TRUTH
        ]
        mean = sum(recalls) / len(recalls)

        failing = [
            (e.query_id, r)
            for e, r in zip(GROUND_TRUTH, recalls)
            if r < 0.85
        ]
        assert mean >= 0.85, (
            f"Mean recall={mean:.4f} below 0.85. "
            f"Failing queries: {failing}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Ragas Context Relevancy
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel2ContextRelevancy:
    """Level 2: Ragas LLM-judge checks that retrieved page_content is relevant."""

    @pytest.mark.integration
    @pytest.mark.llm_eval
    def test_context_precision_above_threshold(self, ragas_scores: dict):
        """LLMContextPrecisionWithoutReference must be >= 0.80."""
        score = ragas_scores["context_precision"]
        threshold = THRESHOLDS["context_precision"]
        assert score >= threshold, (
            f"context_precision={score:.4f} below threshold {threshold}. "
            "Retrieved contexts contain too much irrelevant schema noise."
        )

    @pytest.mark.integration
    @pytest.mark.llm_eval
    def test_context_recall_above_threshold(self, ragas_scores: dict):
        """LLMContextRecall must be >= 0.85."""
        score = ragas_scores["context_recall"]
        threshold = THRESHOLDS["context_recall"]
        assert score >= threshold, (
            f"context_recall={score:.4f} below threshold {threshold}. "
            "Retrieved contexts are missing information needed to answer the queries."
        )
