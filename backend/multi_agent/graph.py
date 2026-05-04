"""Multi-Agent top-level LangGraph assembly.

Topology:
- Entry ``supervisor`` -> conditional fan-out -> ``policy`` and/or ``external`` -> ``writer`` -> END.
- ``policy`` / ``external`` are single-node subgraph wrappers (each containing a tiny ReAct).
- ``writer`` is the only visible generation node tagged with ``WRITER_TAG``.
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


# Re-exported for the orchestrator to use as a token-filtering tag.
__all__ = ["WRITER_TAG", "build_multi_agent_graph"]


def build_multi_agent_graph(
    *,
    rag_graph: Any,
    mcp_tools: list[Any] | None,
    employee_store: EmployeeStore | None,
    employee_top_k: int = EMPLOYEE_LOOKUP_TOP_K,
):
    """Assemble and compile the top-level multi-agent LangGraph.

    Dependencies:
    - ``rag_graph``: the compiled RAG LangGraph; the PolicyAgent calls it via
      the ``rag_answer`` tool.
    - ``mcp_tools``: the output of ``backend.mcp.load_external_mcp_tools()``;
      when empty, the ExternalContextAgent automatically degrades to "no
      external tools".
    - ``employee_store``: the employee directory; the Supervisor only queries
      it when ``Plan.needs_employee_lookup`` is true.
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
            # Neither policy nor external is needed -> hand off directly to writer.
            return ["writer"]
        return targets

    graph.add_conditional_edges(
        "supervisor",
        _route,
        {"policy": "policy", "external": "external", "writer": "writer"},
    )

    # Policy / External are parent-graph nodes (not subgraph nodes); langgraph
    # triggers them in parallel via the fan-out from add_conditional_edges,
    # both converging into writer.
    graph.add_edge("policy", "writer")
    graph.add_edge("external", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
