"""Week 3: RAGAS evaluation across three retrieval pipeline configs, on a
stratified subset of FinanceBench's public 150-question eval set (see
eval/dataset.py for why a subset rather than the full 150 — free-tier
daily quota + measured RAGAS latency make the full set impractical in one
run). Auto-ingests any filing the subset needs but MongoDB doesn't have
yet, rebuilds both chunking-variant indexes, then for each pipeline config
runs our own generation, RAGAS's faithfulness/context precision/context
recall, and a custom numerical-accuracy check — logging everything to
MLflow and checkpointing raw results to disk as it goes, since a ~40
question x 3 config run against a 15-requests/min free tier takes hours
and *will* occasionally hit a transient error.

Usage:
    uv run scripts/run_eval.py --n-questions 40
"""

import argparse
import json
import statistics
from pathlib import Path

import mlflow

from findocqa.config import DATA_DIR, settings
from findocqa.eval.dataset import select_eval_questions
from findocqa.eval.numerical_accuracy import numerical_accuracy
from findocqa.eval.ragas_harness import build_eval_rows, run_ragas_metrics
from findocqa.ingestion.financebench_docs import get_document_info
from findocqa.ingestion.pipeline import ingest_document
from findocqa.retrieval.build import build_variant_index
from findocqa.storage.mongo import get_existing_doc_names

RUNS_DIR = DATA_DIR / "processed" / "eval_runs"

CONFIGS = [
    {"name": "naive_dense", "variant": "naive", "mode": "dense"},
    {"name": "table_aware_hybrid_rerank", "variant": "table_aware", "mode": "hybrid"},
    {"name": "table_aware_hybrid_no_rerank", "variant": "table_aware", "mode": "hybrid_no_rerank"},
]


def ensure_filings_ingested(questions: list[dict]) -> None:
    needed = {q["doc_name"] for q in questions}
    existing = get_existing_doc_names()
    missing_names = needed - existing
    if not missing_names:
        print(f"All {len(needed)} required filings already ingested.")
        return

    doc_info_by_name = {d["doc_name"]: d for d in get_document_info()}
    missing_docs = [doc_info_by_name[n] for n in missing_names if n in doc_info_by_name]

    print(f"Ingesting {len(missing_docs)} missing filing(s)...")
    cik_cache: dict[str, str | None] = {}
    for i, doc in enumerate(missing_docs, start=1):
        print(f"  [{i}/{len(missing_docs)}] {doc['doc_name']}")
        try:
            ingest_document(doc, cik_cache)
        except Exception as exc:
            print(f"    FAILED: {exc}")


def mean_or_none(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def run_config(config: dict, questions: list[dict]) -> dict:
    name, variant, mode = config["name"], config["variant"], config["mode"]
    print(f"\n=== {name} (variant={variant}, mode={mode}) ===")

    rows = build_eval_rows(questions, variant=variant, mode=mode)
    print(f"Generated {len(rows)} answers ({sum(1 for r in rows if r['error'])} failed).")

    ragas_scores = run_ragas_metrics(rows)
    for i, row in enumerate(rows):
        row["faithfulness"] = ragas_scores["faithfulness"][i]
        row["context_precision"] = ragas_scores["context_precision"][i]
        row["context_recall"] = ragas_scores["context_recall"][i]
        row["numerical_accuracy"] = numerical_accuracy(row["answer"], row["ground_truth"])

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RUNS_DIR / f"{name}.json"
    checkpoint_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary = {
        "config": name,
        "n_questions": len(rows),
        "n_errors": sum(1 for r in rows if r["error"]),
        "faithfulness": mean_or_none([r["faithfulness"] for r in rows]),
        "context_precision": mean_or_none([r["context_precision"] for r in rows]),
        "context_recall": mean_or_none([r["context_recall"] for r in rows]),
        "numerical_accuracy": mean_or_none([r["numerical_accuracy"] for r in rows]),
        "mean_latency_seconds": mean_or_none([r["latency_seconds"] for r in rows]),
    }

    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "chunking_variant": variant,
                "retrieval_mode": mode,
                "top_k": 5,
                "generation_model": settings.generation_model,
                "embedding_model": settings.embedding_model,
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "n_questions": len(rows),
            }
        )
        for key in (
            "faithfulness",
            "context_precision",
            "context_recall",
            "numerical_accuracy",
            "mean_latency_seconds",
            "n_errors",
        ):
            if summary[key] is not None:
                mlflow.log_metric(key, summary[key])
        mlflow.log_artifact(str(checkpoint_path))

    return summary


def print_summary_table(summaries: list[dict]) -> None:
    cols = ["config", "faithfulness", "context_precision", "context_recall", "numerical_accuracy"]
    print("\n" + "=" * 100)
    print("Week 3 eval summary")
    print("=" * 100)
    header = f"{'config':<32}" + "".join(f"{c:>18}" for c in cols[1:])
    print(header)
    for s in summaries:
        row = f"{s['config']:<32}"
        for c in cols[1:]:
            v = s[c]
            row += f"{v:>18.3f}" if v is not None else f"{'n/a':>18}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = select_eval_questions(n=args.n_questions, seed=args.seed)
    print(f"Selected {len(questions)} questions across "
          f"{len({q['company'] for q in questions})} companies.")

    ensure_filings_ingested(questions)

    for variant in ("naive", "table_aware"):
        print(f"Building {variant} index over full corpus...")
        n_chunks = build_variant_index(variant)
        print(f"  {n_chunks} chunks indexed.")

    mlflow.set_experiment("findocqa-week3-rag-eval")

    summaries = [run_config(config, questions) for config in CONFIGS]
    print_summary_table(summaries)


if __name__ == "__main__":
    main()
