import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.knowledge_base import build_documents
from backend.data.processing import split_documents
from backend.types import RagDocument
from evals.rag_eval import (
    EVAL_CASES,
    build_eval_retrieve_nodes,
    extract_context_ids,
    print_eval_dataset_overview,
)
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate context_ids while preserving first-seen order."""
    unique_values: list[str] = []

    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    return unique_values


async def evaluate_single_case_with_ragas(
    retrieve_node_name: str,
    retrieve_docs_node: Callable[[str], list[RagDocument]],
    case: dict,
    precision_metric: IDBasedContextPrecision,
    recall_metric: IDBasedContextRecall,
) -> dict:
    """Evaluate retrieval results for a single question using RAGAS."""
    question = case["question"]
    reference_context_ids = case["reference_context_ids"]
    retrieved_docs = retrieve_docs_node(question)

    retrieved_context_ids = deduplicate_preserve_order(
        extract_context_ids(retrieved_docs)
    )

    sample = SingleTurnSample(
        user_input=question,
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids,
    )

    precision_score = await precision_metric.single_turn_ascore(sample)
    recall_score = await recall_metric.single_turn_ascore(sample)

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "retrieve_node_name": retrieve_node_name,
        "question": question,
        "reference_context_ids": reference_context_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "ragas_context_precision": precision_score,
        "ragas_context_recall": recall_score,
    }


async def evaluate_retrieve_node_with_ragas(
    retrieve_node_name: str,
    retrieve_docs_node: Callable[[str], list[RagDocument]],
    eval_cases: list[dict],
) -> list[dict]:
    """Evaluate a single retrieve_docs node function using RAGAS."""
    case_results: list[dict] = []
    precision_metric = IDBasedContextPrecision()
    recall_metric = IDBasedContextRecall()

    for case in eval_cases:
        case_results.append(
            await evaluate_single_case_with_ragas(
                retrieve_node_name,
                retrieve_docs_node,
                case,
                precision_metric,
                recall_metric,
            )
        )

    return case_results


def summarize_ragas_results(case_results: list[dict]) -> dict:
    """Aggregate per-case RAGAS results into average scores."""
    total_cases = len(case_results)

    if total_cases == 0:
        return {
            "total_cases": 0,
            "ragas_context_precision": 0.0,
            "ragas_context_recall": 0.0,
        }

    return {
        "total_cases": total_cases,
        "ragas_context_precision": (
            sum(item["ragas_context_precision"] for item in case_results) / total_cases
        ),
        "ragas_context_recall": (
            sum(item["ragas_context_recall"] for item in case_results) / total_cases
        ),
    }


def summarize_ragas_results_by_category(case_results: list[dict]) -> dict[str, dict]:
    """Aggregate RAGAS results grouped by question category."""
    grouped_results: dict[str, list[dict]] = {}

    for case_result in case_results:
        category = case_result["category"]
        grouped_results.setdefault(category, []).append(case_result)

    return {
        category: summarize_ragas_results(group_case_results)
        for category, group_case_results in grouped_results.items()
    }


def print_case_results(case_results: list[dict]) -> None:
    """Print per-case RAGAS results."""
    for case_result in case_results:
        print("-" * 80)
        print(f"case_id: {case_result['case_id']}")
        print(f"retrieve node: {case_result['retrieve_node_name']}")
        print(f"category: {case_result['category']}")
        print(f"question: {case_result['question']}")
        print(f"reference context_ids: {case_result['reference_context_ids']}")
        print(f"retrieved context_ids: {case_result['retrieved_context_ids']}")
        print(f"RAGAS Context Precision: {case_result['ragas_context_precision']:.4f}")
        print(f"RAGAS Context Recall: {case_result['ragas_context_recall']:.4f}")


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Print aggregated RAGAS results for every retrieve node."""
    print("\n" + "=" * 80)
    print("RAGAS retrieval evaluation summary")
    print("=" * 80)

    for retrieve_node_name, case_results in all_results.items():
        summary = summarize_ragas_results(case_results)
        category_summary = summarize_ragas_results_by_category(case_results)

        print(f"retrieve node: {retrieve_node_name}")
        print(f"  total cases: {summary['total_cases']}")
        print(f"  RAGAS Context Precision: {summary['ragas_context_precision']:.4f}")
        print(f"  RAGAS Context Recall: {summary['ragas_context_recall']:.4f}")
        print("  per-category results:")

        for category, category_result in category_summary.items():
            print(
                "    "
                f"- {category}: "
                f"Context Precision={category_result['ragas_context_precision']:.4f}, "
                f"Context Recall={category_result['ragas_context_recall']:.4f}"
            )

        print()


async def main() -> None:
    """Run the RAGAS-based retrieval evaluation."""
    print_eval_dataset_overview()

    documents = build_documents()
    split_documents_list = split_documents(documents)
    retrieve_nodes = build_eval_retrieve_nodes(split_documents_list)

    all_results: dict[str, list[dict]] = {}

    for retrieve_node_name, retrieve_docs_node in retrieve_nodes.items():
        print("\n" + "=" * 80)
        print(f"Starting RAGAS evaluation: {retrieve_node_name}")
        print("=" * 80)

        case_results = await evaluate_retrieve_node_with_ragas(
            retrieve_node_name,
            retrieve_docs_node,
            EVAL_CASES,
        )
        all_results[retrieve_node_name] = case_results
        print_case_results(case_results)

    print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
