from backend.runtime import rebuild_demo_index


def main() -> None:
    """Manually rebuild the project index."""
    # This script has a single responsibility: rebuild the index.
    # Keeping it separate is easier to reason about than mixing index
    # building into the API startup script.
    result = rebuild_demo_index()

    print("=" * 80)
    print("Index rebuild complete")
    print("=" * 80)
    print(f"Raw document count: {result['raw_document_count']}")
    print(f"Split chunk count: {result['split_document_count']}")
    print(f"Vector store document count: {result['vector_document_count']}")


if __name__ == "__main__":
    main()
