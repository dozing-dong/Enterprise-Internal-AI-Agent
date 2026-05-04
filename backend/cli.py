"""Command-line debug entry point.

Notes:
- FastAPI service entry: `python run_api.py`
- This module is for local interactive debugging only and is not the standard service entry.
"""

from backend.config import BM25_WEIGHT, QUERY_REWRITE_ENABLED, VECTOR_WEIGHT
from backend.runtime import DemoRuntime, create_demo_runtime
from backend.storage.history import build_history_path, clear_session_history
from backend.types import RagDocument


def show_document_pipeline(
    documents: list[RagDocument],
    split_documents_list: list[RagDocument],
    vector_document_count: int,
    vector_backend_name: str,
) -> None:
    """Print the demo's basic processing pipeline for learning purposes."""
    total_char_count = 0

    for doc in documents:
        total_char_count += len(doc.page_content)

    print("=" * 80)
    print("Step 1: Text loading")
    print("=" * 80)
    print(f"Raw document count: {len(documents)}")
    print(f"Total raw text size: {total_char_count} characters")

    for index, doc in enumerate(documents[:5], start=1):
        print(f"Raw document {index} source: {doc.metadata}")

    print()

    print("=" * 80)
    print("Step 2: Text splitting")
    print("=" * 80)
    print(f"Split document count: {len(split_documents_list)}")

    for index, doc in enumerate(split_documents_list[:5], start=1):
        print(f"Chunk {index}: size={len(doc.page_content)}, content={doc.page_content}")

    print()

    print("=" * 80)
    print("Step 3: Retriever construction")
    print("=" * 80)
    print(f"Vector backend: {vector_backend_name}")
    print(f"Total documents in vector store: {vector_document_count}")
    print("Retrieval mode: vector + BM25 hybrid retrieval")
    print(f"Fusion weights: vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}")
    print(f"Query rewrite enabled: {QUERY_REWRITE_ENABLED}\n")


def print_sources(sources: list[dict]) -> None:
    """Print retrieved reference snippets in the command line."""
    if not sources:
        print("No reference snippets returned.")
        return

    print("Reference snippets:")

    for source in sources:
        print(f"- Rank: {source['rank']}")
        print(f"  metadata: {source['metadata']}")
        print(f"  content: {source['content']}")


def interactive_chat(runtime: DemoRuntime) -> None:
    """Start command-line Q&A and persist chat history to the configured backend."""
    session_id = input("Enter session ID (press Enter for default): ").strip() or "default"
    history_path = build_history_path(session_id)

    print("\n" + "=" * 80)
    print("Step 4: RAG Q&A with persistent history")
    print("=" * 80)
    print(f"Current session: {session_id}")
    print(f"History location: {history_path}")
    print("Type 'quit' or 'exit' to leave; type '/clear' to wipe the current session history.\n")

    rag_graph = runtime.rag_graph

    while True:
        user_question = input("Question: ").strip()

        if user_question.lower() in {"quit", "exit"}:
            print("\nExited.")
            break

        if user_question == "/clear":
            clear_session_history(session_id)
            print("Current session history cleared.\n")
            continue

        if not user_question:
            print("Question cannot be empty, please try again.\n")
            continue

        try:
            result = rag_graph.invoke(
                {"question": user_question, "session_id": session_id}
            )

            print(f"\nOriginal question: {result.get('original_question', user_question)}")
            print(f"Retrieval question: {result.get('retrieval_question', user_question)}")
            print(f"\nAnswer:\n{result.get('answer', '')}\n")
            print_sources(result.get("sources", []))
            print("\n" + "-" * 80)
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")
            print("Check AWS credentials, Bedrock model permissions, and network connectivity.\n")


def main() -> None:
    """CLI entry point."""
    try:
        runtime = create_demo_runtime()
    except RuntimeError as exc:
        print(exc)
        return

    show_document_pipeline(
        runtime.documents,
        runtime.split_documents_list,
        runtime.vector_document_count,
        "pgvector",
    )
    interactive_chat(runtime)
