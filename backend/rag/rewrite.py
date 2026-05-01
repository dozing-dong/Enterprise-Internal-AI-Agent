from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from backend.rag.models import create_chat_model


def build_query_rewrite_chain():
    """构建一个最小查询改写链。"""
    # 这里不额外引入新模型，继续复用当前聊天模型。
    # 对学习阶段来说，这样更容易把注意力放在“改写策略”上，而不是模型接入本身。
    llm = create_chat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个检索问题改写助手。"
                "你的任务是把用户原问题改写成更适合知识库检索的短查询。"
                "请保留原问题里的时间、人物、地点、对象、朝代、因果关系和关键词。"
                "不要回答问题，不要解释原因，只输出一条改写后的检索问题。",
            ),
            (
                "human",
                "原问题：{question}\n\n"
                "请输出更适合检索的改写问题。",
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def normalize_rewritten_question(rewritten_question: str, original_question: str) -> str:
    """清理模型输出，尽量只保留检索问题本身。"""
    cleaned_question = rewritten_question.strip()

    # 有些模型会带上“改写后问题：”这种前缀。
    # 这里用最简单、可读的方式去掉常见标签。
    prefixes = [
        "改写后问题：",
        "改写后的问题：",
        "检索问题：",
        "改写问题：",
        "问题：",
    ]

    for prefix in prefixes:
        if cleaned_question.startswith(prefix):
            cleaned_question = cleaned_question[len(prefix):].strip()

    # 如果模型没有认真按要求输出，导致清理后变成空字符串，
    # 就直接退回原问题，避免后面的检索链断掉。
    if not cleaned_question:
        return original_question

    return cleaned_question


def rewrite_question_for_retrieval(question: str, rewrite_chain: Any | None) -> str:
    """把原问题改写成更适合检索的问题。"""
    # 如果当前没有启用查询改写，就直接返回原问题。
    if rewrite_chain is None:
        return question

    rewritten_question = rewrite_chain.invoke({"question": question})
    return normalize_rewritten_question(rewritten_question, question)


class QueryRewriteRetriever(BaseRetriever):
    """先改写查询，再调用下游检索器。"""

    # base_retriever 负责真正的检索。
    base_retriever: BaseRetriever

    # rewrite_chain 负责把用户问题改写成更适合检索的形式。
    rewrite_chain: Any

    def _get_relevant_documents(self, query: str, *, run_manager) -> list:
        """先改写 query，再把改写后的 query 交给下游检索器。"""
        retrieval_question = rewrite_question_for_retrieval(query, self.rewrite_chain)
        return self.base_retriever.invoke(retrieval_question)


def build_query_rewrite_retriever(base_retriever: BaseRetriever) -> BaseRetriever:
    """把已有检索器包装成“查询改写 + 检索”的新检索器。"""
    rewrite_chain = build_query_rewrite_chain()

    return QueryRewriteRetriever(
        base_retriever=base_retriever,
        rewrite_chain=rewrite_chain,
    )
