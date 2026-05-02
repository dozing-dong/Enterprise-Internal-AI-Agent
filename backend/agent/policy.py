"""Agent 决策层的策略配置。

集中管理：
- 系统提示词（决定模型如何选择工具与回答风格）
- 预算/超时常量（max_steps、单工具调用上限等）

所有运行期参数都允许通过环境变量覆盖，便于线上灰度调整。
"""

from __future__ import annotations

import os


AGENT_SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. You can call tools to "
    "answer the user's question.\n\n"
    "Rules you MUST follow:\n"
    "1. For ANY question that may rely on company knowledge, policy, "
    "documents, or factual lookup, you MUST call the `rag_answer` tool "
    "instead of answering from memory.\n"
    "2. Only when the user explicitly asks about the current time / date "
    "should you call `current_time`.\n"
    "3. Issue at most one tool call per turn. After receiving a tool "
    "result, decide whether to call another tool or produce the final "
    "answer.\n"
    "4. When `rag_answer` returns an `answer` field, you should use it as "
    "the basis for your final response and preserve its key facts and "
    "citations. Do NOT fabricate sources.\n"
    "5. If the available tools cannot help, clearly say you do not know "
    "rather than making things up.\n"
    "6. Respond in the same language as the user's question."
)

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))

# 单次工具调用的软上限，仅做日志告警；硬超时由具体工具/底层 SDK 决定。
AGENT_TOOL_LATENCY_WARN_MS = int(
    os.getenv("AGENT_TOOL_LATENCY_WARN_MS", "20000")
)

# 决策层调用 LLM 的温度；保持 0.0 便于复现。
AGENT_PLANNER_TEMPERATURE = float(
    os.getenv("AGENT_PLANNER_TEMPERATURE", "0.0")
)
