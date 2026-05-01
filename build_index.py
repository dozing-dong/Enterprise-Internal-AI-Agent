from backend.runtime import rebuild_demo_index


def main() -> None:
    """手动重建项目索引。"""
    # 这个脚本的职责只有一件事：重建索引。
    # 这样做比把建索引逻辑混进 API 启动脚本更容易理解。
    result = rebuild_demo_index()

    print("=" * 80)
    print("索引重建完成")
    print("=" * 80)
    print(f"原始文档数: {result['raw_document_count']}")
    print(f"切分后片段数: {result['split_document_count']}")
    print(f"向量库文档数: {result['vector_document_count']}")


if __name__ == "__main__":
    main()
