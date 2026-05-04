"""Multi-Agent policy layer: Pydantic Plan + each sub-agent's system prompt.

Centrally manages:
- ``Plan``: the routing decision the Supervisor outputs via ``with_structured_output(Plan)``.
- The system prompt and temperature for each sub-agent.
- ``AGENT_NAME_*``: the display name of each sub-agent in
  ``MultiAgentState.agents_invoked``.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-agent identifiers
# ---------------------------------------------------------------------------

AGENT_NAME_SUPERVISOR = "supervisor"
AGENT_NAME_POLICY = "policy"
AGENT_NAME_EXTERNAL = "external_context"
AGENT_NAME_WRITER = "writer"


# ---------------------------------------------------------------------------
# Supervisor decision structure
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """Structured decision output from the Supervisor.

    Field semantics:
    - ``use_policy``: whether to dispatch the PolicyAgent to retrieve internal
      travel / expense / approval knowledge.
    - ``use_external``: whether to dispatch the ExternalContextAgent for outside
      information (weather, public holidays, web search).
    - ``locations``: geographic locations relevant to the query (used for weather lookup).
    - ``date_range``: time range mentioned in the user's question (free-text, e.g. "next week").
    - ``needs_employee_lookup``: whether to perform an employee directory lookup
      directly during the supervisor stage.
    - ``rationale``: short explanation of the routing choice for debugging and logging.
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
# Temperatures & system prompts
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
    "to retrieve internal company policy, handbook, or approval rules relevant "
    "to the user's question.\n\n"
    "After receiving the tool result, write a concise plain-text summary of "
    "ONLY the rules that directly apply to this question. Do NOT use section "
    "headers, dividers, or fixed templates. If multiple rules apply, list them "
    "as short bullet points. Omit anything that is not directly relevant.\n\n"
    "Rules:\n"
    "1. When calling `rag_answer`, formulate a POLICY-FOCUSED query that covers "
    "   the specific rules the user needs: approval requirements, expense categories, "
    "   accommodation limits, reimbursement deadlines, submission process. "
    "   Do NOT use a generic query like 'what is the travel policy'. "
    "   Strip out weather, destination descriptions, personal names, and travel "
    "   logistics-those belong to ExternalContextAgent.\n"
    "2. Issue at most one tool call per turn.\n"
    "3. Ground your summary strictly on the tool's answer. Never fabricate policy text.\n"
    "4. If the tool returns nothing useful, respond with exactly: "
    "   'No relevant policy found.'\n"
    "5. Respond in the same language as the user's question."
)


EXTERNAL_SYSTEM_PROMPT = (
    "You are the ExternalContextAgent. Use the available external tools "
    "(weather lookup, public holidays / business calendar, web search) "
    "to gather objective outside-world information about the user's trip.\n\n"
    "After tool calls, write a concise plain-text summary of ONLY the "
    "information you actually retrieved. Do NOT use fixed section headers "
    "like 'Weather:' or 'Holidays:' unless there is real content under them. "
    "Omit any category for which you have no data.\n\n"
    "Rules:\n"
    "1. Only call tools that are clearly relevant to the question.\n"
    "2. Issue at most one tool call per turn.\n"
    "3. Never invent weather data, holidays, or web snippets.\n"
    "4. Respond in the same language as the user's question.\n"
    "5. NEVER use web search for company policies, HR rules, expense limits, "
    "   approval workflows, or any internal company topics. Those are handled "
    "   exclusively by a separate PolicyAgent-do not duplicate that work.\n"
    "6. For restaurant, dining, or local attraction queries, always use "
    "   `brave_web_search` (e.g. 'best restaurants North Shore Auckland'). "
    "   Do NOT call `brave_local_search`-it requires a paid API subscription "
    "   and will always fail on this deployment.\n"
    "7. If any tool returns an error, do NOT retry with the same or a "
    "   different tool for the same intent. Immediately proceed to summarise "
    "   with whatever data you have already collected."
)


WRITER_SYSTEM_PROMPT = (
    "You are the WriterAgent of an enterprise travel assistant. "
    "Your job is to write a helpful, natural reply to the user's question "
    "based on the context provided (policy summary, external info, employee directory).\n\n"
    "Tone and style:\n"
    "- Write like a knowledgeable colleague sharing practical advice, "
    "  not like a compliance system walking through an approval checklist.\n"
    "- Be direct. Lead with what the user actually needs to know.\n"
    "- Use plain prose by default. Only use bullet points when listing "
    "  genuinely parallel items (3 or more). Never use ## headers, "
    "  horizontal rules, or bold text for structural decoration.\n"
    "- Keep it concise. If a piece of context (policy, weather, etc.) "
    "  is not relevant to THIS specific question, leave it out entirely. "
    "  Do not pad the answer with sections that have nothing to say.\n\n"
    "Hard rules:\n"
    "- Ground every factual claim on the provided summaries. "
    "  Never fabricate policy text, weather data, or employee details.\n"
    "- If critical context is missing and it matters to the answer, "
    "  say so briefly in one sentence-don't build a section around it.\n"
    "- Respond in the same language as the user's question."
)
