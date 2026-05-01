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
    """对 context_id 去重，但保留第一次出现的顺序。"""
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
    """使用 RAGAS 评测单个问题的检索结果。"""
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
    """使用 RAGAS 评测某一个 retrieve_docs 节点函数。"""
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
    """把逐题 RAGAS 结果汇总成平均分。"""
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
    """按问题类别汇总 RAGAS 结果。"""
    grouped_results: dict[str, list[dict]] = {}

    for case_result in case_results:
        category = case_result["category"]
        grouped_results.setdefault(category, []).append(case_result)

    return {
        category: summarize_ragas_results(group_case_results)
        for category, group_case_results in grouped_results.items()
    }


def print_case_results(case_results: list[dict]) -> None:
    """打印逐题 RAGAS 结果。"""
    for case_result in case_results:
        print("-" * 80)
        print(f"案例编号: {case_result['case_id']}")
        print(f"检索节点: {case_result['retrieve_node_name']}")
        print(f"类别: {case_result['category']}")
        print(f"问题: {case_result['question']}")
        print(f"标准 context_id: {case_result['reference_context_ids']}")
        print(f"召回 context_id: {case_result['retrieved_context_ids']}")
        print(f"RAGAS Context Precision: {case_result['ragas_context_precision']:.4f}")
        print(f"RAGAS Context Recall: {case_result['ragas_context_recall']:.4f}")


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """打印所有检索节点的 RAGAS 汇总结果。"""
    print("\n" + "=" * 80)
    print("RAGAS 检索评测汇总")
    print("=" * 80)

    for retrieve_node_name, case_results in all_results.items():
        summary = summarize_ragas_results(case_results)
        category_summary = summarize_ragas_results_by_category(case_results)

        print(f"检索节点: {retrieve_node_name}")
        print(f"  题目总数: {summary['total_cases']}")
        print(f"  RAGAS Context Precision: {summary['ragas_context_precision']:.4f}")
        print(f"  RAGAS Context Recall: {summary['ragas_context_recall']:.4f}")
        print("  分类结果:")

        for category, category_result in category_summary.items():
            print(
                "    "
                f"- {category}: "
                f"Context Precision={category_result['ragas_context_precision']:.4f}, "
                f"Context Recall={category_result['ragas_context_recall']:.4f}"
            )

        print()


async def main() -> None:
    """执行基于 RAGAS 的公开数据集检索评测。"""
    print_eval_dataset_overview()

    documents = build_documents()
    split_documents_list = split_documents(documents)
    retrieve_nodes = build_eval_retrieve_nodes(split_documents_list)

    all_results: dict[str, list[dict]] = {}

    for retrieve_node_name, retrieve_docs_node in retrieve_nodes.items():
        print("\n" + "=" * 80)
        print(f"开始 RAGAS 评测: {retrieve_node_name}")
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
