"""Multi-Agent orchestration layer.

Splits a single ReAct Agent into four hierarchical subgraphs:
SupervisorAgent + PolicyAgent + ExternalContextAgent + WriterAgent. Each
is responsible for:
- Supervisor: identify the question type / break it into subtasks /
  route to downstream sub-agents / decide whether the writer needs to assemble.
- Policy: internal travel / expense / approval RAG plus employee directory
  lookup (reusing the existing rag_graph and employee_lookup).
- ExternalContext: weather, public holidays, web search, and other outside
  information (via MCP tools).
- Writer: the only "user-visible generation node"; integrates all context
  to output the final advice.

External entry points:
- ``build_multi_agent_graph(...)``: assemble the top-level LangGraph
  containing the four subgraphs.
- ``MultiAgentRunner``: thin wrapper for the orchestrator.
- ``MultiAgentState`` / ``Plan``: shared state and the supervisor's
  decision structure.
- ``WRITER_TAG``: the only tag the orchestrator uses to filter "user-visible tokens".
"""

from backend.multi_agent.graph import (
    WRITER_TAG,
    build_multi_agent_graph,
)
from backend.multi_agent.policy import (
    AGENT_NAME_EXTERNAL,
    AGENT_NAME_POLICY,
    AGENT_NAME_SUPERVISOR,
    AGENT_NAME_WRITER,
    Plan,
)
from backend.multi_agent.runner import MultiAgentRunner
from backend.multi_agent.state import MultiAgentState, build_initial_multi_agent_state


__all__ = [
    "AGENT_NAME_EXTERNAL",
    "AGENT_NAME_POLICY",
    "AGENT_NAME_SUPERVISOR",
    "AGENT_NAME_WRITER",
    "MultiAgentRunner",
    "MultiAgentState",
    "Plan",
    "WRITER_TAG",
    "build_initial_multi_agent_state",
    "build_multi_agent_graph",
]
