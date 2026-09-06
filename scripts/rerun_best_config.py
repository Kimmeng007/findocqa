"""One-off re-run of just the best config (table_aware_hybrid_rerank) on
the same 20-question deterministic subset, under a new checkpoint name
(table_aware_hybrid_rerank_v2) so it doesn't get skipped by run_eval.py's
resume logic against the old pre-fix checkpoint, and so both results are
preserved side by side for an honest before/after comparison.

Measures the real effect of this session's retrieval fixes (company/
fiscal-year filtering, finer table chunking, wider candidate pool when
filtered) on the metrics that were measured low before any of them:
context precision/recall.

Usage:
    uv run scripts/rerun_best_config.py
"""

import mlflow

from scripts.run_eval import print_summary_table, run_config
from findocqa.eval.dataset import select_eval_questions

CONFIG = {"name": "table_aware_hybrid_rerank_v2", "variant": "table_aware", "mode": "hybrid"}


def main() -> None:
    questions = select_eval_questions(n=20, seed=42)
    print(f"Selected {len(questions)} questions across "
          f"{len({q['company'] for q in questions})} companies.")

    mlflow.set_experiment("findocqa-week3-rag-eval")
    summary = run_config(CONFIG, questions)
    print_summary_table([summary])


if __name__ == "__main__":
    main()
