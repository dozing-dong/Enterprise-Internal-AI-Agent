import statistics
import time

from backend.data.knowledge_base import build_eval_cases
from backend.runtime import create_demo_runtime


def evaluate_langgraph(sample_size: int | None = None) -> dict:
    runtime = create_demo_runtime()
    rag_graph = runtime.rag_graph
    eval_cases = build_eval_cases()

    if sample_size is not None:
        eval_cases = eval_cases[:sample_size]

    latencies: list[float] = []
    source_counts: list[int] = []
    failures = 0

    for case in eval_cases:
        start = time.perf_counter()
        try:
            result = rag_graph.invoke(
                {"question": case["question"], "session_id": "eval_langgraph"}
            )
            source_counts.append(len(result.get("sources", [])))
        except Exception:
            failures += 1
        finally:
            latencies.append(time.perf_counter() - start)

    return {
        "mode": "langgraph",
        "total_cases": len(eval_cases),
        "failures": failures,
        "failure_rate": failures / len(eval_cases) if eval_cases else 0.0,
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_s": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
        ),
        "avg_source_count": statistics.mean(source_counts) if source_counts else 0.0,
    }


def print_summary(summary: dict) -> None:
    print("=" * 80)
    print(f"Mode: {summary['mode']}")
    print("=" * 80)
    print(f"Total cases: {summary['total_cases']}")
    print(f"Failures: {summary['failures']} ({summary['failure_rate']:.2%})")
    print(f"Average latency: {summary['avg_latency_s']:.2f}s")
    print(f"P95 latency: {summary['p95_latency_s']:.2f}s")
    print(f"Average source count: {summary['avg_source_count']:.2f}")


def main() -> None:
    summary = evaluate_langgraph()
    print_summary(summary)


if __name__ == "__main__":
    main()
