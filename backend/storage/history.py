import json
from typing import Any

from backend.config import HISTORY_DIR


def build_history_path(session_id: str):
    """根据会话 ID 生成历史文件路径。"""
    # 每次使用前都确保目录存在，避免第一次运行时因为目录缺失而报错。
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / f"{session_id}.json"


def _normalize_message(message: dict[str, Any]) -> dict[str, str] | None:
    """兼容老格式并统一成 {role, content}。"""
    role = message.get("role")
    content = message.get("content")

    if isinstance(role, str) and isinstance(content, str):
        return {"role": role, "content": content}

    # 兼容 LangChain 的历史格式。
    lc_type = message.get("type")
    lc_data = message.get("data")
    if isinstance(lc_type, str) and isinstance(lc_data, dict):
        lc_content = lc_data.get("content")
        if isinstance(lc_content, str):
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            return {"role": role_map.get(lc_type, lc_type), "content": lc_content}

    return None


def _read_raw_messages(session_id: str) -> list[dict[str, Any]]:
    history_path = build_history_path(session_id)
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


def read_session_history(session_id: str) -> list[dict]:
    """直接读取历史文件内容，方便 FastAPI 接口返回 JSON。"""
    normalized_messages: list[dict[str, str]] = []
    for item in _read_raw_messages(session_id):
        normalized = _normalize_message(item)
        if normalized is not None:
            normalized_messages.append(normalized)
    return normalized_messages


def append_session_messages(session_id: str, messages: list[dict]) -> list[dict]:
    """向历史中追加消息并返回最新历史。"""
    current_messages = read_session_history(session_id)
    normalized_new_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized = _normalize_message(message)
        if normalized is not None:
            normalized_new_messages.append(normalized)

    current_messages.extend(normalized_new_messages)
    history_path = build_history_path(session_id)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(current_messages, file, ensure_ascii=False, indent=2)
    return current_messages


def clear_session_history(session_id: str) -> None:
    """清空指定会话的历史记录。"""
    history_path = build_history_path(session_id)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump([], file, ensure_ascii=False, indent=2)
