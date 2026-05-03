"""Multi-Agent 顶层 LangGraph 装配。

拓扑：
- 入口 ``supervisor`` → 条件 fan-out → ``policy`` 与/或 ``external`` → ``writer`` → END。
- ``policy`` / ``external`` 都是单节点子图 wrapper（内部各自有微型 ReAct）。
- ``writer`` 是唯一带 ``WRITER_TAG`` 的可见生成节点。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from backend.config import EMPLOYEE_LOOKUP_TOP_K
from backend.multi_agent.external_context_agent import build_external_subgraph
from backend.multi_agent.policy_agent import build_policy_subgraph
from backend.multi_agent.state import MultiAgentState
from backend.multi_agent.supervisor import build_supervisor_node
from backend.multi_agent.writer_agent import WRITER_TAG, build_writer_node
from backend.rag.employee_retriever import EmployeeStore


# Re-export 给 orchestrator 用作 token 过滤 tag。
__all__ = ["WRITER_TAG", "build_multi_agent_graph"]


def build_multi_agent_graph(
    *,
    rag_graph: Any,
    mcp_tools: list[Any] | None,
    employee_store: EmployeeStore | None,
    employee_top_k: int = EMPLOYEE_LOOKUP_TOP_K,
):
    """装配并编译多 Agent 顶层 LangGraph。

    依赖项：
    - ``rag_graph``：已编译的 RAG LangGraph，PolicyAgent 通过 ``rag_answer``
      工具调用它。
    - ``mcp_tools``：``backend.mcp.load_external_mcp_tools()`` 的产物；为空
      时 ExternalContextAgent 自动降级为"无外部工具"。
    - ``employee_store``：员工目录；Supervisor 仅在 ``Plan.needs_employee_lookup``
      为真时才会查一次。
    """

    supervisor_node = build_supervisor_node(
        employee_store=employee_store,
        employee_top_k=employee_top_k,
    )
    policy_node = build_policy_subgraph(rag_graph)
    external_node = build_external_subgraph(mcp_tools)
    writer_node = build_writer_node()

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("policy", policy_node)
    graph.add_node("external", external_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("supervisor")

    def _route(state: MultiAgentState) -> list[str]:
        plan = state.get("plan")
        targets: list[str] = []
        if plan is not None:
            if getattr(plan, "use_policy", False):
                targets.append("policy")
            if getattr(plan, "use_external", False):
                targets.append("external")
        if not targets:
            # 既不需要 policy 也不需要 external → 直接交给 writer。
            return ["writer"]
        return targets

    graph.add_conditional_edges(
        "supervisor",
        _route,
        {"policy": "policy", "external": "external", "writer": "writer"},
    )

    # Policy / External 都是父图节点（不是 subgraph node），langgraph 会按
    # add_conditional_edges 的并行 fan-out 触发它们；都汇合到 writer。
    graph.add_edge("policy", "writer")
    graph.add_edge("external", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
