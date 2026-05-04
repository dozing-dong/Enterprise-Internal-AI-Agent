"""Generation-quality evaluation across three modes: rag / agent / multi_agent.

This script consumes the ``reference`` field that already exists in
``backend/data/data/local_eval/eval_cases.json`` (which previous eval scripts
did not use) and scores the model's final output with RAGAS LLM-as-judge
metrics:

- ``Faithfulness``: whether the answer is grounded only in retrieved context,
  with no factual hallucination.
- ``AnswerRelevancy``: whether the answer actually answers the question.
- ``AnswerCorrectness``: semantic match between the answer and ``reference``
  (combines LLM judgment and vector similarity).

LLM judge: reuses the project's own ``ChatBedrockConverse`` to avoid bringing
in an additional API key. The embedding-similarity part wraps the project's
``embed_texts`` as a LangChain Embeddings object.

Each case requires 3 modes x 3 metrics worth of LLM calls, so running the
full 69 cases is slow. By default we only evaluate the first
``DEFAULT_SAMPLE_SIZE`` cases; pass ``main(sample_size=None)`` to run the
full set.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage

from backend.agent.graph import build_initial_messages
from backend.data.knowledge_base import build_eval_cases
from backend.llm import embed_texts, get_chat_model
from backend.multi_agent.state import build_initial_multi_agent_state
from backend.runtime import DemoRuntime, create_demo_runtime


SESSION_ID = "eval_generation"

ModeInvoker = Callable[[str], tuple[str, list[dict]]]


def _invoke_rag(graph: Any, question: str) -> tuple[str, list[dict]]:
    result = graph.invoke({"question": question, "session_id": SESSION_ID})
    return result.get("answer", "") or "", result.get("sources", []) or []


def _invoke_agent(graph: Any, question: str) -> tuple[str, list[dict]]:
    initial_state = {
        "messages": build_initial_messages([], question),
        "session_id": SESSION_ID,
        "sources": [],
        "retrieval_question": None,
        "original_question": question,
    }
    result = graph.invoke(initial_state)
    sources = result.get("sources", []) or []
    answer = ""
    for msg in reversed(result.get("messages", []) or []):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            if isinstance(content, str):
                answer = content
            elif isinstance(content, list):
                answer = "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            break
    return answer, sources


def _invoke_multi_agent(graph: Any, question: str) -> tuple[str, list[dict]]:
    initial_state = build_initial_multi_agent_state(
        question=question, session_id=SESSION_ID
    )
    result = graph.invoke(initial_state)
    return result.get("final_answer", "") or "", result.get("sources", []) or []


def build_mode_invokers(runtime: DemoRuntime) -> dict[str, ModeInvoker | None]:
    invokers: dict[str, ModeInvoker | None] = {
        "rag": lambda q: _invoke_rag(runtime.rag_graph, q),
        "agent": lambda q: _invoke_agent(runtime.agent_graph, q),
    }
    if runtime.multi_agent_graph is not None:
        invokers["multi_agent"] = lambda q: _invoke_multi_agent(
            runtime.multi_agent_graph, q
        )
    else:
        invokers["multi_agent"] = None
    return invokers


DEFAULT_SAMPLE_SIZE = 10


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# RAGAS LLM / Embeddings wrappers
# ---------------------------------------------------------------------------


class _BedrockRagasEmbedding:
    """Wrap the project-built ``embed_texts`` so it satisfies both the old and
    new RAGAS Embedding interfaces simultaneously.

    In RAGAS 0.4.x, importing ``AnswerRelevancy`` via the legacy path
    ``from ragas.metrics import AnswerRelevancy`` still returns the old
    ``_answer_relevance.py`` implementation, which calls ``embed_query`` /
    ``embed_documents`` (old interface). Meanwhile the ``AnswerSimilarity``
    used inside ``AnswerCorrectness`` calls ``aembed_text`` (new interface).

    This class implements both interfaces to keep all three metrics working.
    """

    # --- New interface (AnswerCorrectness / AnswerSimilarity / new AnswerRelevancy) ---

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return embed_texts([text])[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return embed_texts([text])[0]

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return embed_texts(texts)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return embed_texts(texts)

    # --- Legacy interface (older AnswerRelevancy calls these methods directly) ---

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)


def _build_ragas_wrappers() -> tuple[Any, Any]:
    """Build the LLM / Embeddings wrapper objects used by RAGAS."""
    from ragas.llms import LangchainLLMWrapper

    chat_model = get_chat_model(temperature=0.0)
    llm_wrapper = LangchainLLMWrapper(chat_model)
    embeddings_wrapper = _BedrockRagasEmbedding()
    return llm_wrapper, embeddings_wrapper


# ---------------------------------------------------------------------------
# Single case evaluation
# ---------------------------------------------------------------------------


def _extract_contexts(sources: list[dict]) -> list[str]:
    contexts: list[str] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        content = source.get("content")
        if isinstance(content, str) and content.strip():
            contexts.append(content)
    return contexts


async def evaluate_single_case(
    mode_name: str,
    invoker: Callable[[str], tuple[str, list[dict]]],
    case: dict,
    *,
    faithfulness_metric: Any,
    relevancy_metric: Any,
    correctness_metric: Any,
) -> dict:
    """Run a single mode against one case and compute the three RAGAS metrics."""
    from ragas.dataset_schema import SingleTurnSample

    error: str | None = None
    answer = ""
    contexts: list[str] = []
    try:
        answer, sources = invoker(case["question"])
        contexts = _extract_contexts(sources)
    except Exception as exc:  # noqa: BLE001
        error = f"invoke_failed: {exc}"

    # When the answer is empty or context is missing, some metrics raise; we
    # normalize those cases by recording None as the missing value.
    sample = SingleTurnSample(
        user_input=case["question"],
        response=answer or "",
        retrieved_contexts=contexts,
        reference=case["reference"],
    )

    faithfulness_score: float | None = None
    relevancy_score: float | None = None
    correctness_score: float | None = None

    if error is None:
        # Faithfulness requires contexts; skip it when they are missing.
        if contexts and answer:
            try:
                faithfulness_score = float(
                    await faithfulness_metric.single_turn_ascore(sample)
                )
            except Exception as exc:  # noqa: BLE001
                error = f"faithfulness_failed: {exc}"

        if answer:
            try:
                relevancy_score = float(
                    await relevancy_metric.single_turn_ascore(sample)
                )
            except Exception as exc:  # noqa: BLE001
                error = (error + "; " if error else "") + f"relevancy_failed: {exc}"

        try:
            correctness_score = float(
                await correctness_metric.single_turn_ascore(sample)
            )
        except Exception as exc:  # noqa: BLE001
            error = (error + "; " if error else "") + f"correctness_failed: {exc}"

    return {
        "case_id": case["case_id"],
        "category": case.get("category", ""),
        "mode": mode_name,
        "question": case["question"],
        "answer": answer,
        "reference": case["reference"],
        "context_count": len(contexts),
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "answer_correctness": correctness_score,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Per-mode evaluation
# ---------------------------------------------------------------------------


async def evaluate_generation_for_mode(
    mode_name: str,
    invoker: Callable[[str], tuple[str, list[dict]]],
    eval_cases: list[dict],
    *,
    faithfulness_metric: Any,
    relevancy_metric: Any,
    correctness_metric: Any,
) -> list[dict]:
    case_results: list[dict] = []
    for index, case in enumerate(eval_cases, start=1):
        print(
            f"  [{mode_name}] {index}/{len(eval_cases)} case_id={case['case_id']}"
        )
        result = await evaluate_single_case(
            mode_name,
            invoker,
            case,
            faithfulness_metric=faithfulness_metric,
            relevancy_metric=relevancy_metric,
            correctness_metric=correctness_metric,
        )
        if result.get("error"):
            print(f"    !! error: {result['error']}")
        case_results.append(result)
    return case_results


def _mean_skip_none(values: list[float | None]) -> float:
    valid = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(valid) if valid else 0.0


def _count_valid(values: list[float | None]) -> int:
    return sum(1 for v in values if isinstance(v, (int, float)))


def summarize_results(case_results: list[dict]) -> dict:
    total = len(case_results)
    faithfulness_values = [r["faithfulness"] for r in case_results]
    relevancy_values = [r["answer_relevancy"] for r in case_results]
    correctness_values = [r["answer_correctness"] for r in case_results]

    return {
        "total_cases": total,
        "faithfulness_avg": _mean_skip_none(faithfulness_values),
        "faithfulness_n": _count_valid(faithfulness_values),
        "answer_relevancy_avg": _mean_skip_none(relevancy_values),
        "answer_relevancy_n": _count_valid(relevancy_values),
        "answer_correctness_avg": _mean_skip_none(correctness_values),
        "answer_correctness_n": _count_valid(correctness_values),
        "errors": sum(1 for r in case_results if r.get("error")),
    }


def summarize_by_category(case_results: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for r in case_results:
        grouped.setdefault(r["category"] or "uncategorized", []).append(r)
    return {cat: summarize_results(group) for cat, group in grouped.items()}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_mode_results(mode_name: str, summary: dict) -> None:
    print("=" * 80)
    print(f"Mode: {mode_name}")
    print("=" * 80)
    print(f"Total cases: {summary['total_cases']}")
    print(
        f"Faithfulness: {summary['faithfulness_avg']:.4f} "
        f"(valid n={summary['faithfulness_n']})"
    )
    print(
        f"AnswerRelevancy: {summary['answer_relevancy_avg']:.4f} "
        f"(valid n={summary['answer_relevancy_n']})"
    )
    print(
        f"AnswerCorrectness: {summary['answer_correctness_avg']:.4f} "
        f"(valid n={summary['answer_correctness_n']})"
    )
    print(f"Number of cases with errors: {summary['errors']}")


def print_comparison_table(all_summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 96)
    print("Generation-quality comparison across the three modes")
    print("=" * 96)
    header = (
        f"{'mode':<14}{'cases':<8}"
        f"{'faithful':<12}{'relevancy':<12}{'correctness':<14}"
        f"{'errors':<8}"
    )
    print(header)
    print("-" * 96)
    for mode, summary in all_summaries.items():
        if summary.get("status") == "skipped":
            print(f"{mode:<14}{'skipped':<8}{'-':<12}{'-':<12}{'-':<14}{'-':<8}")
            continue
        print(
            f"{mode:<14}"
            f"{summary['total_cases']:<8}"
            f"{summary['faithfulness_avg']:<12.4f}"
            f"{summary['answer_relevancy_avg']:<12.4f}"
            f"{summary['answer_correctness_avg']:<14.4f}"
            f"{summary['errors']:<8}"
        )


def print_category_breakdown(mode_name: str, case_results: list[dict]) -> None:
    by_cat = summarize_by_category(case_results)
    print(f"\n{mode_name} per-category breakdown:")
    for cat, cat_summary in by_cat.items():
        print(
            f"  - {cat}: cases={cat_summary['total_cases']}, "
            f"faithful={cat_summary['faithfulness_avg']:.3f}, "
            f"relevancy={cat_summary['answer_relevancy_avg']:.3f}, "
            f"correctness={cat_summary['answer_correctness_avg']:.3f}"
        )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


async def evaluate_all_modes(
    runtime: DemoRuntime,
    eval_cases: list[dict],
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Run the generation-quality evaluation across the three modes; unavailable modes are marked as skipped."""
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
    )
    from ragas.metrics._answer_similarity import AnswerSimilarity

    llm_wrapper, embeddings_wrapper = _build_ragas_wrappers()

    faithfulness_metric = Faithfulness(llm=llm_wrapper)
    relevancy_metric = AnswerRelevancy(
        llm=llm_wrapper, embeddings=embeddings_wrapper
    )
    # AnswerCorrectness uses weights=[0.75, 0.25] internally and the second
    # weight requires an AnswerSimilarity sub-metric. That sub-metric is
    # normally created by ragas.evaluate() when it calls metric.init(run_config),
    # but we call single_turn_ascore directly here, so we must inject it
    # manually. Otherwise _ascore raises:
    # "AssertionError: AnswerSimilarity must be set"
    correctness_metric = AnswerCorrectness(
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        answer_similarity=AnswerSimilarity(embeddings=embeddings_wrapper),
    )

    invokers = build_mode_invokers(runtime)
    summaries: dict[str, dict] = {}
    raw_results: dict[str, list[dict]] = {}

    for mode_name in ("rag", "agent", "multi_agent"):
        invoker = invokers.get(mode_name)
        if invoker is None:
            summaries[mode_name] = {
                "status": "skipped",
                "reason": "multi_agent_graph is not wired up.",
                "total_cases": 0,
            }
            raw_results[mode_name] = []
            continue

        print("\n" + "=" * 80)
        print(f"Starting mode: {mode_name}")
        print("=" * 80)
        case_results = await evaluate_generation_for_mode(
            mode_name,
            invoker,
            eval_cases,
            faithfulness_metric=faithfulness_metric,
            relevancy_metric=relevancy_metric,
            correctness_metric=correctness_metric,
        )
        raw_results[mode_name] = case_results
        summaries[mode_name] = summarize_results(case_results)

    return summaries, raw_results


async def amain(sample_size: int | None = DEFAULT_SAMPLE_SIZE) -> None:
    runtime = create_demo_runtime()
    eval_cases = build_eval_cases()

    if sample_size is not None:
        eval_cases = eval_cases[:sample_size]

    print("=" * 80)
    print("Generation-quality evaluation overview")
    print("=" * 80)
    print(f"Total cases: {len(eval_cases)}")
    print(f"multi_agent available: {runtime.multi_agent_graph is not None}")
    print("LLM judge: project's own ChatBedrockConverse")
    print("Metrics: Faithfulness / AnswerRelevancy / AnswerCorrectness")

    summaries, raw_results = await evaluate_all_modes(runtime, eval_cases)

    for mode_name, summary in summaries.items():
        if summary.get("status") == "skipped":
            print(f"\n{mode_name}: skipped ({summary.get('reason', 'unknown')})")
            continue
        print_mode_results(mode_name, summary)
        print_category_breakdown(mode_name, raw_results[mode_name])

    print_comparison_table(summaries)


def main(sample_size: int | None = DEFAULT_SAMPLE_SIZE) -> None:
    asyncio.run(amain(sample_size=sample_size))


if __name__ == "__main__":
    main()
