from typing import Any
from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END, StateGraph

from backend.data.processing import convert_docs_to_sources, format_docs
from backend.rag.models import create_chat_model
from backend.rag.rewrite import rewrite_question_for_retrieval
from backend.storage.history import get_session_history


def build_chat_chain() -> RunnableWithMessageHistory:
    """构建只负责生成答案的对话链。"""
    # 这里继续把“检索”和“生成”分开。
    # 原因是你现在要学习的重点之一，就是清楚区分：
    # 1. 检索阶段做了什么
    # 2. 生成阶段又做了什么
    llm = create_chat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个基于检索结果回答问题的助手。"
                "请优先依据提供的知识库片段和对话历史作答。"
                "如果参考内容不足以支持结论，就明确说明不知道，不能编造。",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "问题：{question}\n\n"
                "参考知识：\n{context}\n\n"
                "请基于参考知识回答。",
            ),
        ]
    )

    base_chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


class GraphState(TypedDict, total=False):
    question: str
    session_id: str
    original_question: str
    retrieval_question: str
    retrieved_docs: list
    context: str
    sources: list[dict]
    answer: str
    tool_trace: list[dict]
    iteration: int
    should_continue: bool


def build_langgraph_executor(
    retriever: BaseRetriever,
    chat_chain: RunnableWithMessageHistory,
    rewrite_chain: Any | None,
    max_iterations: int,
    min_sources: int,
) -> Runnable:
    graph = StateGraph(GraphState)

    def route_intent(state: GraphState) -> GraphState:
        return {
            "retrieval_question": state["question"],
            "tool_trace": state.get("tool_trace", []),
            "iteration": state.get("iteration", 0),
        }

    def rewrite_query(state: GraphState) -> GraphState:
        retrieval_question = rewrite_question_for_retrieval(
            state["question"],
            rewrite_chain,
        )
        trace = state.get("tool_trace", [])
        trace.append(
            {
                "tool": "rewrite_query",
                "input": state["question"],
                "output": retrieval_question,
            }
        )
        return {
            "retrieval_question": retrieval_question,
            "tool_trace": trace,
        }

    def retrieve_docs(state: GraphState) -> GraphState:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = retriever.invoke(retrieval_question)
        trace = state.get("tool_trace", [])
        trace.append(
            {
                "tool": "retrieve_docs",
                "input": retrieval_question,
                "output_count": len(docs),
            }
        )
        return {
            "retrieved_docs": docs,
            "context": format_docs(docs),
            "sources": convert_docs_to_sources(docs),
            "tool_trace": trace,
            "iteration": state.get("iteration", 0) + 1,
        }

    def quality_gate(state: GraphState) -> GraphState:
        sources = state.get("sources", [])
        iteration = state.get("iteration", 0)
        should_continue = len(sources) < min_sources and iteration < max_iterations
        return {"should_continue": should_continue}

    def generate_answer(state: GraphState) -> GraphState:
        answer = chat_chain.invoke(
            {
                "question": state["question"],
                "context": state.get("context", "未检索到可用知识片段。"),
            },
            config={"configurable": {"session_id": state["session_id"]}},
        )
        return {"answer": answer}

    def finalize(state: GraphState) -> GraphState:
        return {
            "answer": state.get("answer", "没有生成可用回答。"),
            "sources": state.get("sources", []),
            "original_question": state["question"],
            "retrieval_question": state.get("retrieval_question", state["question"]),
            "tool_trace": state.get("tool_trace", []),
        }

    graph.add_node("route_intent", route_intent)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("route_intent")
    graph.add_edge("route_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        lambda state: "retrieve_docs" if state.get("should_continue") else "generate_answer",
        {
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("generate_answer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
