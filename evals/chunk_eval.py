import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# 这里单独把项目根目录加入 sys.path。
# 原因和其他评测脚本一样：这个脚本通常会被直接运行。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHUNK_PROFILES, DEFAULT_CHUNK_PROFILE_NAME
from backend.data.knowledge_base import build_documents
from backend.data.processing import split_documents
from backend.rag.retrievers import (
    build_bm25_retriever,
    build_evaluation_vectorstore,
    build_hybrid_retriever,
    build_vector_retriever,
)
from evals.rag_eval import EVAL_CASES, print_eval_dataset_overview
from evals.ragas_retrieval_eval import (
    evaluate_retriever_with_ragas,
    summarize_ragas_results,
    summarize_ragas_results_by_category,
)


# 任务 4 的重点仍然是“只比较 chunk 策略”。
# 所以这里固定使用同一种检索器，只让 chunk 配置变化。
TARGET_RETRIEVER_NAME = "hybrid"


def build_target_retriever_for_profile(split_documents_list, profile_name: str):
    """根据 chunk 策略构建本次评测要使用的检索器。"""
    # 每个 profile 都使用独立的 collection_name。
    # 这样可以避免多个实验之间互相污染临时向量库。
    collection_name = f"chunk_eval_{profile_name}_{uuid4().hex}"

    vectorstore = build_evaluation_vectorstore(
        split_documents_list,
        collection_name=collection_name,
    )

    if TARGET_RETRIEVER_NAME == "vector":
        return build_vector_retriever(vectorstore)

    if TARGET_RETRIEVER_NAME == "bm25":
        return build_bm25_retriever(split_documents_list)

    if TARGET_RETRIEVER_NAME == "hybrid":
        return build_hybrid_retriever(split_documents_list, vectorstore)

    raise ValueError(
        f"未知的检索器名称: {TARGET_RETRIEVER_NAME}。"
        "可选值有: vector、bm25、hybrid"
    )


def print_chunk_profile_overview() -> None:
    """打印当前要比较的 chunk 策略。"""
    print("=" * 80)
    print("Chunk 策略对比说明")
    print("=" * 80)
    print(f"当前固定检索器: {TARGET_RETRIEVER_NAME}")
    print(f"默认在线策略: {DEFAULT_CHUNK_PROFILE_NAME}")
    print("本次对比的 chunk 配置:")

    for profile_name, profile in CHUNK_PROFILES.items():
        print(f"- {profile_name}")
        print(f"  chunk_size: {profile['chunk_size']}")
        print(f"  chunk_overlap: {profile['chunk_overlap']}")
        print(f"  description: {profile['description']}")


def print_profile_results(
    profile_name: str,
    split_document_count: int,
    case_results: list[dict],
) -> None:
    """打印某一种 chunk 策略的 RAGAS 结果。"""
    summary = summarize_ragas_results(case_results)
    category_summary = summarize_ragas_results_by_category(case_results)

    print("\n" + "=" * 80)
    print(f"Chunk 策略: {profile_name}")
    print("=" * 80)
    print(f"切分后片段数: {split_document_count}")
    print(f"题目总数: {summary['total_cases']}")
    print(f"RAGAS Context Precision: {summary['ragas_context_precision']:.4f}")
    print(f"RAGAS Context Recall: {summary['ragas_context_recall']:.4f}")
    print("分类结果:")

    for category, category_result in category_summary.items():
        print(
            "  "
            f"- {category}: "
            f"Context Precision={category_result['ragas_context_precision']:.4f}, "
            f"Context Recall={category_result['ragas_context_recall']:.4f}"
        )


def print_final_profile_comparison(all_profile_results: dict[str, dict]) -> None:
    """打印所有 chunk 策略的最终对比结果。"""
    print("\n" + "=" * 80)
    print("Chunk 策略最终对比")
    print("=" * 80)

    for profile_name, result in all_profile_results.items():
        print(f"策略: {profile_name}")
        print(f"  切分后片段数: {result['split_document_count']}")
        print(
            "  "
            f"RAGAS Context Precision: "
            f"{result['summary']['ragas_context_precision']:.4f}"
        )
        print(
            "  "
            f"RAGAS Context Recall: "
            f"{result['summary']['ragas_context_recall']:.4f}"
        )
        print()


async def main() -> None:
    """执行 chunk 策略对比评测。"""
    # 第一步：说明评测集构成。
    print_eval_dataset_overview()
    print()

    # 第二步：说明本次 chunk 对比规则。
    print_chunk_profile_overview()

    # 第三步：准备原始文档。
    documents = build_documents()

    # 第四步：在同一批文档上分别套用不同 chunk 策略。
    all_profile_results: dict[str, dict] = {}

    for profile_name in CHUNK_PROFILES:
        split_documents_list = split_documents(documents, profile_name=profile_name)
        retriever = build_target_retriever_for_profile(
            split_documents_list,
            profile_name,
        )
        case_results = await evaluate_retriever_with_ragas(
            TARGET_RETRIEVER_NAME,
            retriever,
            EVAL_CASES,
        )
        summary = summarize_ragas_results(case_results)

        all_profile_results[profile_name] = {
            "split_document_count": len(split_documents_list),
            "summary": summary,
            "case_results": case_results,
        }

        print_profile_results(
            profile_name,
            len(split_documents_list),
            case_results,
        )

    # 第五步：打印最终对比结果。
    print_final_profile_comparison(all_profile_results)


if __name__ == "__main__":
    asyncio.run(main())
