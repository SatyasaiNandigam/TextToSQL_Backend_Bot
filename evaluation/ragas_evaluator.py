"""
Ragas evaluation runner for the schema retriever node.

Entry point: run_retrieval_only_evaluation()

Combines:
- Level 1: deterministic table recall / precision (no LLM cost)
- Level 2: Ragas LLM-judge context metrics (requires OPENAI_API_KEY)

Results are saved to evaluation/results/retrieval_<timestamp>.json.

Run standalone:
    python -c "from evaluation.ragas_evaluator import run_retrieval_only_evaluation; run_retrieval_only_evaluation()"
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from evaluation.dataset_builder import DatasetBuilder, RetrievalResult
from evaluation.ground_truth import GroundTruthEntry, GROUND_TRUTH, get_ground_truth
from evaluation.metrics_config import get_metrics, THRESHOLDS
from evaluation.table_metrics import (
    table_recall,
    table_precision,
    tables_missed,
    tables_hallucinated,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Deterministic table metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_table_metrics(
    ground_truth: list[GroundTruthEntry],
    retrieval_results: dict[str, RetrievalResult],
) -> dict:
    """Compute deterministic Level 1 metrics for all entries."""
    per_query = []
    for entry in ground_truth:
        result = retrieval_results[entry.query_id]
        recall = table_recall(entry.required_tables, result.fk_expanded_tables)
        precision = table_precision(
            entry.required_tables, entry.optional_tables, result.fk_expanded_tables
        )
        per_query.append(
            {
                "query_id": entry.query_id,
                "category": entry.category,
                "query": entry.query,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "retrieved_tables": sorted(result.retrieved_tables),
                "fk_expanded_tables": sorted(result.fk_expanded_tables),
                "required_tables": sorted(entry.required_tables),
                "missed": sorted(tables_missed(entry.required_tables, result.fk_expanded_tables)),
                "hallucinated": sorted(
                    tables_hallucinated(
                        entry.required_tables, entry.optional_tables, result.fk_expanded_tables
                    )
                ),
            }
        )

    recalls = [r["recall"] for r in per_query]
    precisions = [r["precision"] for r in per_query]
    mean_recall = sum(recalls) / len(recalls)
    mean_precision = sum(precisions) / len(precisions)

    return {
        "mean_recall": round(mean_recall, 4),
        "mean_precision": round(mean_precision, 4),
        "recall_pass": mean_recall >= THRESHOLDS["context_recall"],
        "per_query": per_query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Ragas LLM-judge context metrics
# ─────────────────────────────────────────────────────────────────────────────

async def _score_all_samples(
    ground_truth: list[GroundTruthEntry],
    retrieval_results: dict[str, RetrievalResult],
) -> dict:
    """Run ContextPrecision + ContextRecall concurrently over all samples.

    Ragas 0.4.x API:
      - Use metric.ascore(user_input, retrieved_contexts, response, reference)
      - asyncio.gather() runs all samples in parallel

    Field mapping (from EVALUATION_PLAN.md):
      user_input         → entry.query
      retrieved_contexts → result.page_contents  (doc.page_content, NOT DDL)
      response           → entry.reference       (gold answer used as proxy)
      reference          → entry.reference       (for ContextRecall)
    """
    precision_metric, recall_metric = get_metrics()

    precision_tasks = []
    recall_tasks = []

    for entry in ground_truth:
        result = retrieval_results[entry.query_id]
        contexts = result.page_contents

        precision_tasks.append(
            precision_metric.ascore(
                user_input=entry.query,
                retrieved_contexts=contexts,
                response=entry.reference,
            )
        )
        recall_tasks.append(
            recall_metric.ascore(
                user_input=entry.query,
                retrieved_contexts=contexts,
                reference=entry.reference,
            )
        )

    print(f"  Scoring {len(precision_tasks)} samples with ContextPrecision + ContextRecall...")
    precision_results, recall_results = await asyncio.gather(
        asyncio.gather(*precision_tasks),
        asyncio.gather(*recall_tasks),
    )

    per_query_ragas = []
    for entry, p, r in zip(ground_truth, precision_results, recall_results):
        per_query_ragas.append(
            {
                "query_id": entry.query_id,
                "context_precision": round(float(p.value), 4),
                "context_recall": round(float(r.value), 4),
            }
        )

    mean_precision = sum(x["context_precision"] for x in per_query_ragas) / len(per_query_ragas)
    mean_recall = sum(x["context_recall"] for x in per_query_ragas) / len(per_query_ragas)

    return {
        "context_precision": round(mean_precision, 4),
        "context_recall": round(mean_recall, 4),
        "context_precision_pass": mean_precision >= THRESHOLDS["context_precision"],
        "context_recall_pass": mean_recall >= THRESHOLDS["context_recall"],
        "per_query": per_query_ragas,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_only_evaluation(
    retrieval_results: dict[str, RetrievalResult] | None = None,
    ground_truth: list[GroundTruthEntry] | None = None,
    save_results: bool = True,
    run_ragas: bool = True,
) -> dict:
    """Run full retrieval evaluation (Level 1 + Level 2).

    Args:
        retrieval_results: Pre-built results dict from DatasetBuilder.build().
                           If None, DatasetBuilder is called automatically (hits DB + ChromaDB).
        ground_truth:      List of GroundTruthEntry records. Defaults to all 30.
        save_results:      Whether to write JSON to evaluation/results/.
        run_ragas:         Whether to run Level 2 Ragas LLM-judge metrics.
                           Set False to skip LLM calls (Level 1 only, no OPENAI_API_KEY needed).

    Returns:
        dict with keys:
          "table_metrics"          — Level 1 deterministic results
          "context_precision"      — mean Ragas score (if run_ragas=True)
          "context_recall"         — mean Ragas score (if run_ragas=True)
          "context_precision_pass" — bool (if run_ragas=True)
          "context_recall_pass"    — bool (if run_ragas=True)
    """
    if ground_truth is None:
        ground_truth = get_ground_truth()

    if retrieval_results is None:
        print("Building retrieval results (DB + ChromaDB call)...")
        retrieval_results = DatasetBuilder().build(ground_truth)

    # ── Level 1: deterministic table metrics ─────────────────────────────────
    print("\n=== Level 1: Table Recall / Precision ===")
    table_metrics = _compute_table_metrics(ground_truth, retrieval_results)
    print(f"  Mean Recall:    {table_metrics['mean_recall']:.4f}  (threshold ≥ {THRESHOLDS['context_recall']})")
    print(f"  Mean Precision: {table_metrics['mean_precision']:.4f}")

    failed = [r for r in table_metrics["per_query"] if r["recall"] < THRESHOLDS["context_recall"]]
    if failed:
        print(f"\n  ⚠ {len(failed)} queries below recall threshold:")
        for r in failed:
            print(f"    [{r['query_id']}] recall={r['recall']:.2f}  missed={r['missed']}")
    else:
        print("  ✓ All queries meet recall threshold")

    output: dict = {"table_metrics": table_metrics}

    # ── Level 2: Ragas LLM-judge metrics ─────────────────────────────────────
    if run_ragas:
        print("\n=== Level 2: Ragas Context Metrics ===")
        ragas_scores = asyncio.run(_score_all_samples(ground_truth, retrieval_results))

        print(f"  context_precision: {ragas_scores['context_precision']:.4f}  (threshold ≥ {THRESHOLDS['context_precision']})")
        print(f"  context_recall:    {ragas_scores['context_recall']:.4f}  (threshold ≥ {THRESHOLDS['context_recall']})")

        prec_status = "✓" if ragas_scores["context_precision_pass"] else "✗"
        rec_status = "✓" if ragas_scores["context_recall_pass"] else "✗"
        print(f"  {prec_status} precision  {rec_status} recall")

        output.update(ragas_scores)

    # ── Save results ──────────────────────────────────────────────────────────
    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"retrieval_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved → {out_path}")

    return output


if __name__ == "__main__":
    run_retrieval_only_evaluation()
