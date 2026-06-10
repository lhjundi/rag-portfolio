from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.routing import classify_complexity


QUERIES = [
    "Quais as disciplinas do primeiro semestre?",
    "Quais as disciplinas do primeiro semestre?",
    "Qual é o perfil do egresso?",
    "Qual é o perfil do egresso?",
    "Segundo o PPC, como funciona o estágio?",
    "Como o estágio é tratado no PPC?",
    "O que o PPC diz sobre TCC?",
    "O que o PPC diz sobre TCC?",
    "Quais são as regras do trabalho de conclusão de curso?",
    "Quais são as regras do trabalho de conclusão de curso?",
]


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(0.95 * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    print("Inicializando pipeline...")
    pipeline = build_rag_pipeline(corpus_dir=str(ROOT / "data" / "corpus"))

    exact_cache = ExactCache()
    semantic_cache = SemanticCache(threshold=0.93)

    records = []

    exact_hits = 0
    semantic_hits = 0
    llm_calls = 0
    fallback_count = 0

    for i, query in enumerate(QUERIES, start=1):
        print(f"\n[{i:02d}] {query}")
        started = time.perf_counter()

        route = classify_complexity(query)

        cached = exact_cache.get(query)

        if cached:
            layer = "exact_cache"
            answer = cached
            exact_hits += 1
            sources = []
            rate_limited = False

        else:
            try:
                cached = semantic_cache.get(query)
            except Exception as e:
                print(f"  Semantic cache get indisponivel: {type(e).__name__}")
                cached = None

            if cached:
                layer = "semantic_cache"
                answer = cached
                semantic_hits += 1
                sources = []
                rate_limited = False
                exact_cache.put(query, answer)

            else:
                result = pipeline.answer(query)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                rate_limited = bool(result.get("rate_limited"))

                llm_calls += 1
                if rate_limited:
                    fallback_count += 1
                    layer = "fallback"
                else:
                    layer = "llm"

                exact_cache.put(query, answer)

                try:
                    semantic_cache.put(query, answer)
                except Exception as e:
                    print(f"  Semantic cache put indisponivel: {type(e).__name__}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        preview = " ".join(answer.split())[:180]

        record = {
            "i": i,
            "query": query,
            "layer": layer,
            "route_complexity": route.complexity,
            "route_model": route.model,
            "latency_ms": round(elapsed_ms, 2),
            "rate_limited": rate_limited,
            "sources_count": len(sources),
            "answer_preview": preview,
        }
        records.append(record)

        print(f"  layer={layer}")
        print(f"  route={route.complexity} -> {route.model}")
        print(f"  latency_ms={record['latency_ms']}")
        print(f"  preview={preview}")

    total = len(QUERIES)
    baseline_llm_calls = total
    optimized_llm_calls = llm_calls
    reduction_pct = ((baseline_llm_calls - optimized_llm_calls) / baseline_llm_calls) * 100

    latencies = [r["latency_ms"] for r in records]

    summary = {
        "total_queries": total,
        "chunks_indexed": pipeline.collection.count(),
        "baseline_llm_calls": baseline_llm_calls,
        "optimized_llm_calls": optimized_llm_calls,
        "exact_cache_hits": exact_hits,
        "semantic_cache_hits": semantic_hits,
        "fallback_count": fallback_count,
        "reduction_pct": round(reduction_pct, 1),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(p95(latencies), 2),
        "records": records,
    }

    bench_dir = ROOT / "bench"
    bench_dir.mkdir(exist_ok=True)

    output_path = bench_dir / "results.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESUMO ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()