from collections.abc import Callable
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from backend.data.processing import convert_docs_to_sources, format_docs
from backend.rag.models import chat_completion
from backend.rag.retrievers import SearchRetriever, fuse_retrieval_results
from backend.rag.rewrite import rewrite_question_for_retrieval
from backend.storage.history import append_session_messages, read_session_history


class GraphState(TypedDict, total=False):
    question: str
    session_id: str
    original_question: str
    retrieval_question: str
    retrieved_docs: list
    vector_docs: list
    keyword_docs: list
    context: str
    sources: list[dict]
    history: list[dict]
    messages: list[dict]
    answer: str
    tool_trace: list[dict]
    iteration: int
    should_continue: bool


def build_langgraph_executor(
    vector_retriever: SearchRetriever,
    keyword_retriever: SearchRetriever,
    rewrite_chain: Callable[[str], str] | None,
    max_iterations: int,
    min_sources: int,
    top_k: int,
):
    graph = StateGraph(GraphState)

    def route_intent(state: GraphState) -> GraphState:
        return {
            "retrieval_question": state["question"],
            "tool_trace": state.get("tool_trace", []),
            "iteration": state.get("iteration", 0),
        }

    def load_history(state: GraphState) -> GraphState:
        history = read_session_history(state["session_id"])
        return {"history": history}

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

    def vector_retrieve(state: GraphState) -> GraphState:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = vector_retriever.invoke(retrieval_question)
        trace = state.get("tool_trace", [])
        trace.append(
            {
                "tool": "vector_retrieve",
                "input": retrieval_question,
                "output_count": len(docs),
            }
        )
        return {
            "vector_docs": docs,
            "tool_trace": trace,
        }

    def keyword_retrieve(state: GraphState) -> GraphState:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = keyword_retriever.invoke(retrieval_question)
        trace = state.get("tool_trace", [])
        trace.append(
            {
                "tool": "keyword_retrieve",
                "input": retrieval_question,
                "output_count": len(docs),
            }
        )
        return {
            "keyword_docs": docs,
            "tool_trace": trace,
        }

    def fuse_docs(state: GraphState) -> GraphState:
        vector_docs = state.get("vector_docs", [])
        keyword_docs = state.get("keyword_docs", [])
        docs = fuse_retrieval_results(
            vector_docs,
            keyword_docs,
            top_k=top_k,
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

    def build_messages(state: GraphState) -> GraphState:
        context = state.get("context", "未检索到可用知识片段。")
        history = state.get("history", [])
        messages = list(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"问题：{state['question']}\n\n"
                    f"参考知识：\n{context}\n\n"
                    "请基于参考知识回答。"
                ),
            }
        )
        return {"messages": messages}

    def generate_answer(state: GraphState) -> GraphState:
        answer = chat_completion(
            state.get("messages", []),
            system_prompt=(
                "你是一个基于检索结果回答问题的助手。"
                "请优先依据提供的知识库片段和对话历史作答。"
                "如果参考内容不足以支持结论，就明确说明不知道，不能编造。"
            ),
            temperature=0.0,
        )
        return {"answer": answer}

    def save_history(state: GraphState) -> GraphState:
        updated_history = append_session_messages(
            state["session_id"],
            [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state.get("answer", "")},
            ],
        )
        return {"history": updated_history}

    def finalize(state: GraphState) -> GraphState:
        return {
            "answer": state.get("answer", "没有生成可用回答。"),
            "sources": state.get("sources", []),
            "original_question": state["question"],
            "retrieval_question": state.get("retrieval_question", state["question"]),
            "tool_trace": state.get("tool_trace", []),
        }

    graph.add_node("route_intent", route_intent)
    graph.add_node("load_history", load_history)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("vector_retrieve", vector_retrieve)
    graph.add_node("keyword_retrieve", keyword_retrieve)
    graph.add_node("fuse_docs", fuse_docs)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("build_messages", build_messages)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_history", save_history)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("route_intent")
    graph.add_edge("route_intent", "load_history")
    graph.add_edge("load_history", "rewrite_query")
    graph.add_edge("rewrite_query", "vector_retrieve")
    graph.add_edge("rewrite_query", "keyword_retrieve")
    graph.add_edge("vector_retrieve", "fuse_docs")
    graph.add_edge("keyword_retrieve", "fuse_docs")
    graph.add_edge("fuse_docs", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        lambda state: "rewrite_query" if state.get("should_continue") else "build_messages",
        {
            "rewrite_query": "rewrite_query",
            "build_messages": "build_messages",
        },
    )
    graph.add_edge("build_messages", "generate_answer")
    graph.add_edge("generate_answer", "save_history")
    graph.add_edge("save_history", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
