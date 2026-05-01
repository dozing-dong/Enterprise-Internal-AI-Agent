import operator
from collections.abc import Callable
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from backend.data.processing import convert_docs_to_sources, format_docs
from backend.rag.models import chat_completion
from backend.rag.rerank import Reranker
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
    tool_trace: Annotated[list[dict], operator.add]
    iteration: int
    should_continue: bool


def build_langgraph_executor(
    vector_retriever: SearchRetriever,
    keyword_retriever: SearchRetriever,
    rewrite_chain: Callable[[str], str] | None,
    max_iterations: int,
    min_sources: int,
    top_k: int,
    reranker: Reranker | None = None,
    rerank_top_k: int | None = None,
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
    #把原问题改写成更适合检索的问题。
    def rewrite_query(state: GraphState) -> GraphState:
        retrieval_question = rewrite_question_for_retrieval(
            state["question"],
            rewrite_chain,
        )
        return {
            "retrieval_question": retrieval_question,
            "tool_trace": [
                {
                    "tool": "rewrite_query",
                    "input": state["question"],
                    "output": retrieval_question,
                }
            ],
        }

    #用向量相似度检索器检索相关文档。
    def vector_retrieve(state: GraphState) -> GraphState:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = vector_retriever.invoke(retrieval_question)
        return {
            "vector_docs": docs,
            "tool_trace": [
                {
                    "tool": "vector_retrieve",
                    "input": retrieval_question,
                    "output_count": len(docs),
                }
            ],
        }

    #用关键词检索器检索相关文档。
    def keyword_retrieve(state: GraphState) -> GraphState:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = keyword_retriever.invoke(retrieval_question)
        return {
            "keyword_docs": docs,
            "tool_trace": [
                {
                    "tool": "keyword_retrieve",
                    "input": retrieval_question,
                    "output_count": len(docs),
                }
            ],
        }

    #用倒数排名合并前面两个检索器的检索结果。
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
            "tool_trace": [
                {
                    "tool": "fuse_docs",
                    "input_vector_count": len(vector_docs),
                    "input_keyword_count": len(keyword_docs),
                    "output_count": len(docs),
                }
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    #对融合后的候选文档做精排，截断到 rerank_top_k。
    def rerank_docs(state: GraphState) -> GraphState:
        candidates = state.get("retrieved_docs", [])
        if reranker is None or not candidates:
            return {}

        final_k = rerank_top_k if rerank_top_k is not None else top_k
        retrieval_question = state.get("retrieval_question", state["question"])
        reranked = reranker.invoke(retrieval_question, candidates, final_k)

        return {
            "retrieved_docs": reranked,
            "context": format_docs(reranked),
            "sources": convert_docs_to_sources(reranked),
            "tool_trace": [
                {
                    "tool": "rerank_docs",
                    "input_count": len(candidates),
                    "output_count": len(reranked),
                }
            ],
        }

    #判断检索质量是否达标，决定是继续循环还是进入生成阶段。
    def quality_gate(state: GraphState) -> GraphState:
        sources = state.get("sources", [])
        iteration = state.get("iteration", 0)
        should_continue = len(sources) < min_sources and iteration < max_iterations
        return {"should_continue": should_continue}

    #把历史对话、检索上下文和当前问题拼装成发给 LLM 的消息列表。这里还没有发给模型，只是拼装。
    def build_messages(state: GraphState) -> GraphState:
        context = state.get("context", "No relevant knowledge snippets were retrieved.")
        history = state.get("history", [])
        messages = list(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Reference knowledge:\n{context}\n\n"
                    "Please answer the question based on the reference knowledge above."
                ),
            }
        )
        return {"messages": messages}

    #调用模型生成回答。
    def generate_answer(state: GraphState) -> GraphState:
        answer = chat_completion(
            state.get("messages", []),
            system_prompt=(
                "You are an assistant that answers questions based on retrieval results. "
                "Prefer to rely on the provided knowledge base snippets and the conversation history. "
                "If the reference content is not sufficient to support a conclusion, "
                "clearly say that you do not know and do not fabricate an answer."
            ),
            temperature=0.0,
        )
        return {"answer": answer}

    #把本轮的用户问题和模型回答追加写入历史文件。
    def save_history(state: GraphState) -> GraphState:
        updated_history = append_session_messages(
            state["session_id"],
            [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state.get("answer", "")},
            ],
        )
        return {"history": updated_history}

    #整理并输出图执行的最终结果。
    def finalize(state: GraphState) -> GraphState:
        return {
            "answer": state.get("answer", "没有生成可用回答。"),
            "sources": state.get("sources", []),
            "original_question": state["question"],
            "retrieval_question": state.get("retrieval_question", state["question"]),
            "tool_trace": state.get("tool_trace", []),
        }

    #添加节点。
    graph.add_node("route_intent", route_intent)
    graph.add_node("load_history", load_history)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("vector_retrieve", vector_retrieve)
    graph.add_node("keyword_retrieve", keyword_retrieve)
    graph.add_node("fuse_docs", fuse_docs)
    graph.add_node("rerank_docs", rerank_docs)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("build_messages", build_messages)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_history", save_history)
    graph.add_node("finalize", finalize)
    #添加边。
    graph.set_entry_point("route_intent")
    graph.add_edge("route_intent", "load_history")
    graph.add_edge("load_history", "rewrite_query")
    graph.add_edge("rewrite_query", "vector_retrieve")
    graph.add_edge("rewrite_query", "keyword_retrieve")
    graph.add_edge("vector_retrieve", "fuse_docs")
    graph.add_edge("keyword_retrieve", "fuse_docs")
    graph.add_edge("fuse_docs", "rerank_docs")
    graph.add_edge("rerank_docs", "quality_gate")
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
    #编译图。
    return graph.compile()
