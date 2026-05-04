import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

# Add the project root to sys.path explicitly here. Same reason as the other
# eval scripts: this script is typically executed directly.
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
from backend.types import RagDocument
from evals.rag_eval import EVAL_CASES, print_eval_dataset_overview
from evals.ragas_retrieval_eval import (
    evaluate_retrieve_node_with_ragas,
    summarize_ragas_results,
    summarize_ragas_results_by_category,
)


# The focus of task 4 is still "only compare chunk strategies", so we keep a
# single retrieve node fixed and only vary the chunk configuration.
TARGET_RETRIEVE_NODE_NAME = "fuse_retrieve"


def build_retrieve_docs_node_for_profile(
    split_documents_list,
    profile_name: str,
) -> Callable[[str], list[RagDocument]]:
    """Build the retrieve_docs node function used for this chunk profile run."""
    # Each profile uses its own collection_name to prevent cross-contamination
    # between experiments in the temporary vector store.
    collection_name = f"chunk_eval_{profile_name}_{uuid4().hex}"

    vectorstore = build_evaluation_vectorstore(
        split_documents_list,
        collection_name=collection_name,
    )

    if TARGET_RETRIEVE_NODE_NAME == "vector_retrieve":
        vector_retriever = build_vector_retriever(vectorstore)
        return vector_retriever.invoke

    if TARGET_RETRIEVE_NODE_NAME == "keyword_retrieve":
        keyword_retriever = build_bm25_retriever(split_documents_list)
        return keyword_retriever.invoke

    if TARGET_RETRIEVE_NODE_NAME == "fuse_retrieve":
        hybrid_retriever = build_hybrid_retriever(split_documents_list, vectorstore)
        return hybrid_retriever.invoke

    raise ValueError(
        f"Unknown retrieve node name: {TARGET_RETRIEVE_NODE_NAME}. "
        "Allowed values: vector_retrieve, keyword_retrieve, fuse_retrieve"
    )


def print_chunk_profile_overview() -> None:
    """Print the chunk strategies that are being compared."""
    print("=" * 80)
    print("Chunk strategy comparison overview")
    print("=" * 80)
    print(f"Fixed retrieve node: {TARGET_RETRIEVE_NODE_NAME}")
    print(f"Default online profile: {DEFAULT_CHUNK_PROFILE_NAME}")
    print("Chunk configurations under comparison:")

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
    """Print RAGAS results for a single chunk strategy."""
    summary = summarize_ragas_results(case_results)
    category_summary = summarize_ragas_results_by_category(case_results)

    print("\n" + "=" * 80)
    print(f"Chunk strategy: {profile_name}")
    print("=" * 80)
    print(f"Number of chunks after split: {split_document_count}")
    print(f"Total cases: {summary['total_cases']}")
    print(f"RAGAS Context Precision: {summary['ragas_context_precision']:.4f}")
    print(f"RAGAS Context Recall: {summary['ragas_context_recall']:.4f}")
    print("Per-category results:")

    for category, category_result in category_summary.items():
        print(
            "  "
            f"- {category}: "
            f"Context Precision={category_result['ragas_context_precision']:.4f}, "
            f"Context Recall={category_result['ragas_context_recall']:.4f}"
        )


def print_final_profile_comparison(all_profile_results: dict[str, dict]) -> None:
    """Print the final cross-strategy comparison for all chunk profiles."""
    print("\n" + "=" * 80)
    print("Chunk strategy final comparison")
    print("=" * 80)

    for profile_name, result in all_profile_results.items():
        print(f"profile: {profile_name}")
        print(f"  number of chunks after split: {result['split_document_count']}")
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
    """Run the chunk strategy comparison evaluation."""
    # Step 1: describe the eval dataset.
    print_eval_dataset_overview()
    print()

    # Step 2: describe the chunk comparison rules.
    print_chunk_profile_overview()

    # Step 3: load the source documents.
    documents = build_documents()

    # Step 4: apply each chunk strategy to the same set of documents.
    all_profile_results: dict[str, dict] = {}

    for profile_name in CHUNK_PROFILES:
        split_documents_list = split_documents(documents, profile_name=profile_name)
        retrieve_docs_node = build_retrieve_docs_node_for_profile(
            split_documents_list,
            profile_name,
        )
        case_results = await evaluate_retrieve_node_with_ragas(
            TARGET_RETRIEVE_NODE_NAME,
            retrieve_docs_node,
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

    # Step 5: print the final cross-strategy comparison.
    print_final_profile_comparison(all_profile_results)


if __name__ == "__main__":
    asyncio.run(main())
