"""Multi-Agent 编排层。

把单个 ReAct Agent 拆成 SupervisorAgent + PolicyAgent + ExternalContextAgent +
WriterAgent 四个层级 subgraph，分别负责：
- Supervisor：识别问题类型 / 拆子任务 / 路由到下游 sub-agent / 决定最终是否需要写作整理。
- Policy：内部差旅 / 报销 / 审批等 RAG + 员工目录检索（复用现有 rag_graph 与 employee_lookup）。
- ExternalContext：天气、节假日、网络搜索等外部信息（通过 MCP 工具）。
- Writer：唯一的"用户可见生成节点"，整合所有上下文输出最终建议。

外部消费入口：
- ``build_multi_agent_graph(...)``：装配四个 subgraph 的顶层 LangGraph。
- ``MultiAgentRunner``：薄包装，给 orchestrator 调用。
- ``MultiAgentState`` / ``Plan``：共享状态与 supervisor 决策结构。
- ``WRITER_TAG``：orchestrator 用来过滤 "用户可见 token" 的唯一 tag。
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
