import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import BEDROCK_EMBEDDING_MODEL_ID, MODEL_PROVIDER, VECTOR_BACKEND

EXECUTION_MODE = "langgraph"
from backend.data.knowledge_base import build_documents, build_eval_cases
from backend.data.processing import split_documents
from backend.rag.retrievers import (
    build_bm25_retriever,
    build_hybrid_retriever,
    get_vector_document_count,
    build_vector_retriever,
    load_vectorstore,
)
from backend.types import RagDocument


EVAL_CASES = build_eval_cases()


def ensure_vectorstore_ready():
    vectorstore = load_vectorstore()

    if get_vector_document_count(vectorstore) == 0:
        raise RuntimeError(
            "No usable vector index was found. Please run `python build_index.py` first to build the index."
        )

    return vectorstore


def build_eval_retrieve_nodes(
    split_documents_list,
) -> dict[str, Callable[[str], list[RagDocument]]]:
    vectorstore = ensure_vectorstore_ready()
    vector_retriever = build_vector_retriever(vectorstore)
    keyword_retriever = build_bm25_retriever(split_documents_list)
    hybrid_retriever = build_hybrid_retriever(split_documents_list, vectorstore)

    return {
        "vector_retrieve": vector_retriever.invoke,
        "keyword_retrieve": keyword_retriever.invoke,
        "fuse_retrieve": hybrid_retriever.invoke,
    }


def extract_context_ids(retrieved_docs) -> list[str]:
    context_ids: list[str] = []

    for doc in retrieved_docs:
        context_id = doc.metadata.get("context_id")

        if isinstance(context_id, str):
            context_ids.append(context_id)

    return context_ids


def count_relevant_in_top_k(
    context_ids: list[str],
    reference_context_ids: list[str],
    k: int,
) -> int:
    relevant_count = 0

    for context_id in context_ids[:k]:
        if context_id in reference_context_ids:
            relevant_count += 1

    return relevant_count


def compute_recall_at_k(
    context_ids: list[str],
    reference_context_ids: list[str],
    k: int,
) -> float:
    if not reference_context_ids:
        return 0.0

    relevant_count = count_relevant_in_top_k(
        context_ids,
        reference_context_ids,
        k,
    )

    return relevant_count / len(reference_context_ids)


def compute_precision_at_k(
    context_ids: list[str],
    reference_context_ids: list[str],
    k: int,
) -> float:
    relevant_count = count_relevant_in_top_k(
        context_ids,
        reference_context_ids,
        k,
    )

    return relevant_count / k


def compute_reciprocal_rank(
    context_ids: list[str],
    reference_context_ids: list[str],
) -> float:
    for rank, context_id in enumerate(context_ids, start=1):
        if context_id in reference_context_ids:
            return 1 / rank

    return 0.0


def evaluate_single_case(
    retrieve_node_name: str,
    retrieve_docs_node: Callable[[str], list[RagDocument]],
    case: dict,
) -> dict:
    question = case["question"]
    reference_context_ids = case["reference_context_ids"]
    retrieved_docs = retrieve_docs_node(question)
    context_ids = extract_context_ids(retrieved_docs)

    top_contents = [doc.page_content[:120].replace("\n", " ") for doc in retrieved_docs[:3]]

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "note": case["note"],
        "retrieve_node_name": retrieve_node_name,
        "question": question,
        "reference_context_ids": reference_context_ids,
        "retrieved_context_ids": context_ids,
        "recall_at_1": compute_recall_at_k(context_ids, reference_context_ids, 1),
        "recall_at_3": compute_recall_at_k(context_ids, reference_context_ids, 3),
        "precision_at_1": compute_precision_at_k(context_ids, reference_context_ids, 1),
        "precision_at_3": compute_precision_at_k(context_ids, reference_context_ids, 3),
        "reciprocal_rank": compute_reciprocal_rank(context_ids, reference_context_ids),
        "top_contents": top_contents,
    }


def evaluate_retrieve_node(
    retrieve_node_name: str,
    retrieve_docs_node: Callable[[str], list[RagDocument]],
    eval_cases: list[dict],
) -> list[dict]:
    return [
        evaluate_single_case(retrieve_node_name, retrieve_docs_node, case)
        for case in eval_cases
    ]


def summarize_results(case_results: list[dict]) -> dict:
    total_cases = len(case_results)

    if total_cases == 0:
        return {
            "total_cases": 0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "precision_at_1": 0.0,
            "precision_at_3": 0.0,
            "mrr": 0.0,
        }

    return {
        "total_cases": total_cases,
        "recall_at_1": sum(item["recall_at_1"] for item in case_results) / total_cases,
        "recall_at_3": sum(item["recall_at_3"] for item in case_results) / total_cases,
        "precision_at_1": sum(item["precision_at_1"] for item in case_results) / total_cases,
        "precision_at_3": sum(item["precision_at_3"] for item in case_results) / total_cases,
        "mrr": sum(item["reciprocal_rank"] for item in case_results) / total_cases,
    }


def summarize_results_by_category(case_results: list[dict]) -> dict[str, dict]:
    grouped_results: dict[str, list[dict]] = {}

    for case_result in case_results:
        category = case_result["category"]
        grouped_results.setdefault(category, []).append(case_result)

    return {
        category: summarize_results(group_case_results)
        for category, group_case_results in grouped_results.items()
    }


def print_eval_dataset_overview() -> None:
    print("=" * 80)
    print("Eval dataset overview")
    print("=" * 80)
    print("Current dataset: project_local_eval")
    print("Knowledge base source: built-in project knowledge base documents")
    print("Question source: built-in project QA eval set")
    print(f"Default embedding model: {BEDROCK_EMBEDDING_MODEL_ID}")
    print(f"Execution mode: {EXECUTION_MODE}")
    print(f"Model provider: {MODEL_PROVIDER}")
    print(f"Vector backend: {VECTOR_BACKEND}")
    print(f"Number of eval cases: {len(EVAL_CASES)}")
    print("Note: reference_context_ids contains the explicitly labeled reference contexts.")


def print_case_results(case_results: list[dict]) -> None:
    for case_result in case_results:
        print("-" * 80)
        print(f"case_id: {case_result['case_id']}")
        print(f"retrieve node: {case_result['retrieve_node_name']}")
        print(f"category: {case_result['category']}")
        print(f"question: {case_result['question']}")
        print(f"reference context_ids: {case_result['reference_context_ids']}")
        print(f"retrieved context_ids: {case_result['retrieved_context_ids']}")
        print(f"Recall@1: {case_result['recall_at_1']:.4f}")
        print(f"Recall@3: {case_result['recall_at_3']:.4f}")
        print(f"Precision@1: {case_result['precision_at_1']:.4f}")
        print(f"Precision@3: {case_result['precision_at_3']:.4f}")
        print(f"Reciprocal Rank: {case_result['reciprocal_rank']:.4f}")


def print_summary(all_results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 80)
    print("Retrieval evaluation summary")
    print("=" * 80)

    for retrieve_node_name, case_results in all_results.items():
        summary = summarize_results(case_results)
        category_summary = summarize_results_by_category(case_results)

        print(f"retrieve node: {retrieve_node_name}")
        print(f"  total cases: {summary['total_cases']}")
        print(f"  Recall@1: {summary['recall_at_1']:.4f}")
        print(f"  Recall@3: {summary['recall_at_3']:.4f}")
        print(f"  Precision@1: {summary['precision_at_1']:.4f}")
        print(f"  Precision@3: {summary['precision_at_3']:.4f}")
        print(f"  MRR: {summary['mrr']:.4f}")
        print("  per-category results:")

        for category, category_result in category_summary.items():
            print(
                "    "
                f"- {category}: "
                f"Recall@1={category_result['recall_at_1']:.4f}, "
                f"Recall@3={category_result['recall_at_3']:.4f}, "
                f"Precision@1={category_result['precision_at_1']:.4f}, "
                f"Precision@3={category_result['precision_at_3']:.4f}, "
                f"MRR={category_result['mrr']:.4f}"
            )

        print()


def main() -> None:
    print_eval_dataset_overview()

    documents = build_documents()
    split_documents_list = split_documents(documents)
    retrieve_nodes = build_eval_retrieve_nodes(split_documents_list)

    all_results: dict[str, list[dict]] = {}

    for retrieve_node_name, retrieve_docs_node in retrieve_nodes.items():
        case_results = evaluate_retrieve_node(
            retrieve_node_name,
            retrieve_docs_node,
            EVAL_CASES,
        )
        all_results[retrieve_node_name] = case_results
        print_case_results(case_results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
