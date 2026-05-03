"""命令行调试入口。

说明：
- FastAPI 服务入口：`python run_api.py`
- 本模块仅用于本地交互式调试，不作为标准服务入口。
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
    """打印 demo 的基础处理流程，方便学习时观察。"""
    total_char_count = 0

    for doc in documents:
        total_char_count += len(doc.page_content)

    print("=" * 80)
    print("步骤 1: 文本加载")
    print("=" * 80)
    print(f"原始文档数量：{len(documents)}")
    print(f"原始文本总大小：{total_char_count} 字符")

    for index, doc in enumerate(documents[:5], start=1):
        print(f"原始文档 {index} 来源：{doc.metadata}")

    print()

    print("=" * 80)
    print("步骤 2: 文本切分")
    print("=" * 80)
    print(f"切分后文档数量：{len(split_documents_list)}")

    for index, doc in enumerate(split_documents_list[:5], start=1):
        print(f"片段 {index}: 大小={len(doc.page_content)}，内容={doc.page_content}")

    print()

    print("=" * 80)
    print("步骤 3: 检索器构建")
    print("=" * 80)
    print(f"向量后端：{vector_backend_name}")
    print(f"向量库中文档总数：{vector_document_count}")
    print("检索模式：向量检索 + BM25 混合检索")
    print(f"融合权重：vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}")
    print(f"查询改写启用状态：{QUERY_REWRITE_ENABLED}\n")


def print_sources(sources: list[dict]) -> None:
    """在命令行里打印检索到的参考片段。"""
    if not sources:
        print("未返回可展示的参考片段。")
        return

    print("参考片段：")

    for source in sources:
        print(f"- 排名：{source['rank']}")
        print(f"  metadata：{source['metadata']}")
        print(f"  content：{source['content']}")


def interactive_chat(runtime: DemoRuntime) -> None:
    """启动命令行问答，并把聊天历史保存到当前配置的存储后端。"""
    session_id = input("请输入会话 ID（直接回车使用 default）：").strip() or "default"
    history_path = build_history_path(session_id)

    print("\n" + "=" * 80)
    print("步骤 4: 带历史存储的 RAG 问答")
    print("=" * 80)
    print(f"当前会话：{session_id}")
    print(f"历史定位：{history_path}")
    print("输入 quit 或 exit 退出，输入 /clear 清空当前会话历史。\n")

    rag_graph = runtime.rag_graph

    while True:
        user_question = input("请输入问题：").strip()

        if user_question.lower() in {"quit", "exit"}:
            print("\n已退出。")
            break

        if user_question == "/clear":
            clear_session_history(session_id)
            print("当前会话历史已清空。\n")
            continue

        if not user_question:
            print("问题不能为空，请重新输入。\n")
            continue

        try:
            result = rag_graph.invoke(
                {"question": user_question, "session_id": session_id}
            )

            print(f"\n原问题：{result.get('original_question', user_question)}")
            print(f"检索问题：{result.get('retrieval_question', user_question)}")
            print(f"\n回答：\n{result.get('answer', '')}\n")
            print_sources(result.get("sources", []))
            print("\n" + "-" * 80)
        except Exception as exc:
            print(f"\n发生错误：{exc}")
            print("请检查 AWS 凭证、Bedrock 模型权限与网络连接。\n")


def main() -> None:
    """CLI 入口函数。"""
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
