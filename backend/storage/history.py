import json

from langchain_community.chat_message_histories import FileChatMessageHistory

from backend.config import HISTORY_DIR


def build_history_path(session_id: str):
    """根据会话 ID 生成历史文件路径。"""
    # 每次使用前都确保目录存在，避免第一次运行时因为目录缺失而报错。
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / f"{session_id}.json"


def get_session_history(session_id: str) -> FileChatMessageHistory:
    """返回指定会话对应的 LangChain 历史对象。"""
    history_path = build_history_path(session_id)
    return FileChatMessageHistory(
        str(history_path),
        encoding="utf-8",
        ensure_ascii=False,
    )


def read_session_history(session_id: str) -> list[dict]:
    """直接读取历史文件内容，方便 FastAPI 接口返回 JSON。"""
    history_path = build_history_path(session_id)

    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clear_session_history(session_id: str) -> None:
    """清空指定会话的历史记录。"""
    get_session_history(session_id).clear()
