"""Multi-Agent 策略层：Pydantic Plan + 各 sub-agent 的 system prompt。

集中管理：
- ``Plan``：Supervisor 用 ``with_structured_output(Plan)`` 输出的路由决策。
- 各 sub-agent 的 system prompt 与温度。
- ``AGENT_NAME_*``：sub-agent 在 ``MultiAgentState.agents_invoked`` 中的展示名。
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-agent 标识
# ---------------------------------------------------------------------------

AGENT_NAME_SUPERVISOR = "supervisor"
AGENT_NAME_POLICY = "policy"
AGENT_NAME_EXTERNAL = "external_context"
AGENT_NAME_WRITER = "writer"


# ---------------------------------------------------------------------------
# Supervisor 决策结构
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """Supervisor 的结构化决策输出。

    字段语义：
    - ``use_policy``：是否需要 PolicyAgent 检索内部差旅 / 报销 / 审批知识。
    - ``use_external``：是否需要 ExternalContextAgent 拉外部信息（天气、节假日、网搜）。
    - ``locations``：与查询相关的地理位置（用于天气查询）。
    - ``date_range``：用户问题里涉及的时间范围（自由文本，例如 "next week"）。
    - ``needs_employee_lookup``：是否需要在 supervisor 阶段直接做一次员工目录查询。
    - ``rationale``：模型对路由选择的简短解释，便于调试与日志。
    """

    use_policy: bool = Field(
        default=True,
        description="Whether to dispatch the PolicyAgent for internal knowledge.",
    )
    use_external: bool = Field(
        default=False,
        description="Whether to dispatch the ExternalContextAgent for outside-world info.",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Cities or regions relevant to the question (used for weather lookup).",
    )
    date_range: str | None = Field(
        default=None,
        description="Natural-language time range, e.g. 'next week', '2026-05-08'.",
    )
    needs_employee_lookup: bool = Field(
        default=False,
        description="Whether to look up the employee directory before dispatching agents.",
    )
    rationale: str = Field(
        default="",
        description="Short rationale for the routing decision.",
    )


# ---------------------------------------------------------------------------
# 温度 & system prompts
# ---------------------------------------------------------------------------

SUPERVISOR_TEMPERATURE = float(os.getenv("MULTI_AGENT_SUPERVISOR_TEMPERATURE", "0.0"))
POLICY_TEMPERATURE = float(os.getenv("MULTI_AGENT_POLICY_TEMPERATURE", "0.0"))
EXTERNAL_TEMPERATURE = float(os.getenv("MULTI_AGENT_EXTERNAL_TEMPERATURE", "0.0"))
WRITER_TEMPERATURE = float(os.getenv("MULTI_AGENT_WRITER_TEMPERATURE", "0.2"))


SUPERVISOR_SYSTEM_PROMPT = (
    "You are the SupervisorAgent of an enterprise travel assistant. "
    "Your job is to read the user's question and decide which sub-agents "
    "to dispatch. You DO NOT answer the question yourself.\n\n"
    "Sub-agents available:\n"
    "- PolicyAgent: searches internal travel / expense / approval policies "
    "  via the company knowledge base, plus the structured employee directory.\n"
    "- ExternalContextAgent: fetches outside-world information such as "
    "  destination weather, public holidays, and live web snippets.\n"
    "- WriterAgent: always runs at the end to assemble the final user-facing answer.\n\n"
    "Routing rules:\n"
    "1. If the question mentions company policy, expense, approval, "
    "   travel rules, or any internal HR/employee context, set use_policy=true.\n"
    "2. If the question mentions a destination city, dates, weather, "
    "   travel timing, or anything that requires outside-world knowledge, "
    "   set use_external=true and fill `locations` and `date_range`.\n"
    "3. Set needs_employee_lookup=true ONLY when the user clearly identifies "
    "   themselves or asks about a specific colleague.\n"
    "4. Always provide a short rationale.\n"
    "5. Output MUST conform to the Plan schema; do not add prose around it."
)


POLICY_SYSTEM_PROMPT = (
    "You are the PolicyAgent. Use the `rag_answer` tool (and ONLY that tool) "
    "to retrieve internal company policy / handbook / approval rules that "
    "are relevant to the user's question. After receiving the tool result, "
    "produce a SHORT structured summary in the following format:\n\n"
    "Relevant rules:\n- ...\nUnclear / missing:\n- ...\n\n"
    "Rules you MUST follow:\n"
    "1. When calling `rag_answer`, formulate a POLICY-FOCUSED query. "
    "   Extract only the HR / approval / expense / reimbursement aspects "
    "   from the user's question. Strip out weather, destination descriptions, "
    "   personal names, and travel logistics—those belong to ExternalContextAgent. "
    "   Good query example: 'international travel approval and reimbursement policy'. "
    "   Bad query example: 'Alice Carter travel to Auckland weather and policy'.\n"
    "2. Issue at most one tool call per turn. After receiving the tool's "
    "   answer, do not call it again unless the user's question really "
    "   requires multiple separate searches.\n"
    "3. Ground your summary STRICTLY on the tool's answer and cited "
    "   snippets. Never fabricate policy text.\n"
    "4. If the tool returns no useful information, explicitly state "
    "   'No relevant policy found' instead of guessing.\n"
    "5. Respond in the same language as the user's question."
)


EXTERNAL_SYSTEM_PROMPT = (
    "You are the ExternalContextAgent. Use the available external tools "
    "(weather lookup, public holidays / business calendar, web search) "
    "to gather objective outside-world information about the user's trip. "
    "After tool calls, produce a SHORT structured summary like:\n\n"
    "Weather:\n- ...\nHolidays / calendar:\n- ...\nWeb / news:\n- ...\n\n"
    "Rules:\n"
    "1. Only call tools that are clearly relevant; skip a category if no "
    "   tool can answer it.\n"
    "2. Issue at most one tool call per turn; combine results across turns.\n"
    "3. Never invent weather data, holidays, or web snippets if a tool is "
    "   unavailable.\n"
    "4. Respond in the same language as the user's question."
)


WRITER_SYSTEM_PROMPT = (
    "You are the WriterAgent of an enterprise travel assistant. "
    "Combine the PolicyAgent summary, the ExternalContextAgent summary, "
    "and the optional employee directory context into ONE actionable "
    "response for the user.\n\n"
    "Required structure (use the same language as the question):\n"
    "1. Brief recap of the user's request.\n"
    "2. Whether prior approval is needed and who to approve it (if known).\n"
    "3. Budget / reimbursement notes from the policy summary.\n"
    "4. Destination weather + holiday-driven travel and packing tips.\n"
    "5. Concrete next-step checklist (3-5 bullet items).\n\n"
    "Hard rules:\n"
    "- Ground every factual claim on the provided summaries; never make up "
    "  policy text, weather data, or employee details.\n"
    "- If a section has no input (e.g. no policy summary), say so explicitly "
    "  rather than guessing.\n"
    "- Keep the answer concise and skimmable; use bullet points where helpful."
)
