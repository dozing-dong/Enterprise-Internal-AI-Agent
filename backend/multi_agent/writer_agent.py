"""WriterAgent：唯一的"用户可见生成节点"。

设计要点：
- 整合 Supervisor / Policy / External 三家结果，调用一次 LLM 产出最终
  Markdown 风格答复。
- 用唯一 tag ``WRITER_TAG = "agent_writer"``，orchestrator 据此把 token
  流转给前端，子 Agent 内部 LLM 输出全部过滤掉。
- 子图只有一个节点；不开 ReAct 循环，避免 writer 还要再调外部工具。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.llm import get_chat_model
from backend.multi_agent.policy import (
    AGENT_NAME_WRITER,
    WRITER_SYSTEM_PROMPT,
    WRITER_TEMPERATURE,
)
from backend.multi_agent.state import MultiAgentState


logger = logging.getLogger(__name__)


WRITER_TAG = "agent_writer"


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        lines.append(f"{role}: {content[:240]}")
    return "\n".join(lines)


def _format_employee_context(records: list[dict]) -> str:
    if not records:
        return "(none)"
    lines = []
    for rec in records[:5]:
        if not isinstance(rec, dict):
            continue
        lines.append(
            f"- {rec.get('name', '?')} "
            f"({rec.get('title', '?')}, {rec.get('department', '?')}, "
            f"{rec.get('email', '')})"
        )
    return "\n".join(lines) or "(none)"


def _format_policy_section(policy_result: dict | None) -> str:
    if not policy_result or not policy_result.get("ok"):
        return "(no policy summary available)"
    answer = (policy_result.get("answer") or "").strip()
    if not answer:
        return "(policy agent returned no useful information)"
    return answer


def _format_external_section(external_result: dict | None) -> str:
    if not external_result:
        return "(no external context available)"
    answer = (external_result.get("answer") or "").strip()
    tools_used = external_result.get("tools_used") or []
    if not answer and not tools_used:
        return "(no external context available)"
    parts: list[str] = []
    if tools_used:
        parts.append("Tools used: " + ", ".join(tools_used))
    if answer:
        parts.append(answer)
    return "\n".join(parts)


def build_writer_node():
    """工厂：返回 writer 节点函数（唯一可见生成节点）。"""

    chat_model = get_chat_model(temperature=WRITER_TEMPERATURE).with_config(
        tags=[WRITER_TAG]
    )

    def writer_node(state: MultiAgentState) -> dict[str, Any]:
        question = state.get("question") or ""
        history_text = _format_history(state.get("history") or [])
        employee_text = _format_employee_context(state.get("employee_context") or [])
        policy_text = _format_policy_section(state.get("policy_result"))
        external_text = _format_external_section(state.get("external_result"))

        plan = state.get("plan")
        rationale = ""
        if plan is not None:
            try:
                rationale = getattr(plan, "rationale", "") or ""
            except Exception:  # noqa: BLE001
                rationale = ""

        user_prompt = (
            f"User question: {question}\n\n"
            f"Recent conversation:\n{history_text or '(no prior turns)'}\n\n"
            f"Supervisor rationale: {rationale or '(none)'}\n\n"
            f"PolicyAgent summary:\n{policy_text}\n\n"
            f"ExternalContextAgent summary:\n{external_text}\n\n"
            f"Employee directory context:\n{employee_text}\n\n"
            "Now produce the final user-facing answer following the required structure."
        )

        try:
            ai_msg: AIMessage = chat_model.invoke(
                [SystemMessage(WRITER_SYSTEM_PROMPT), HumanMessage(user_prompt)]
            )
            text = _flatten_text(ai_msg.content)
        except Exception:  # noqa: BLE001
            logger.exception("WriterAgent generation failed")
            text = ""

        return {
            "final_answer": text or "未能生成可用回答。",
            "agents_invoked": [AGENT_NAME_WRITER],
        }

    return writer_node


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)
