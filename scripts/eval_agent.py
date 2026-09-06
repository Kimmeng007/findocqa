"""Week 4 gap-fill: agent-level eval measuring tool-call correctness,
task completion, and failure handling on the real multi-step agent
pipeline (decompose -> answer_subquestion -> synthesize) -- distinct
from Week 3's RAG-level RAGAS eval (single-shot retrieval+generation
quality) and from tests/unit/test_agent_*.py (unit-level correctness of
individual functions/graph wiring, no real questions or API calls).

Scope: FinanceBench's own "Numerical reasoning" question category --
these need a derived figure (ratio, average, YoY change), exactly the
case the calculator tool exists for -- restricted to filings already in
this project's corpus, so retrieval-corpus-coverage isn't a confound on
what this eval is actually trying to measure. Deliberately small
(quota-bounded): each multi-step agent question can cost 10+ real
Gemini calls (decompose + per-subquestion generation + retries +
synthesize), unlike Week 3's ~1-call-per-question RAG eval.

Usage:
    uv run scripts/eval_agent.py
"""

import json
import time

from findocqa.agent.graph import run_agent
from findocqa.config import DATA_DIR
from findocqa.eval.numerical_accuracy import numerical_accuracy
from findocqa.ingestion.financebench_docs import get_questions
from findocqa.storage.mongo import get_existing_doc_names

RESULTS_PATH = DATA_DIR / "processed" / "eval_runs" / "agent_eval.json"

# Hand-picked: diverse companies, diverse derived-calculation types (YoY
# change, tax-rate delta, turnover ratio, days-payable-outstanding), all
# answerable from a single already-ingested filing.
QUESTION_IDS = [
    "financebench_id_07507",  # Adobe YoY operating income change
    "financebench_id_08135",  # Amazon YoY revenue change %
    "financebench_id_01351",  # American Express effective tax rate change
    "financebench_id_10499",  # Kraft Heinz inventory turnover ratio
    "financebench_id_00552",  # Microsoft debt increase (numeric comparison)
    "financebench_id_06655",  # Amazon days payable outstanding
]


def main() -> None:
    existing = get_existing_doc_names()
    all_questions = {q["financebench_id"]: q for q in get_questions()}
    questions = [all_questions[qid] for qid in QUESTION_IDS]
    for q in questions:
        assert q["doc_name"] in existing, f"{q['doc_name']} not ingested"

    results = []
    for q in questions:
        print(f"\n=== {q['financebench_id']}: {q['question'][:80]} ===")
        started = time.monotonic()
        try:
            result = run_agent(q["question"])
            error = None
        except Exception as exc:  # noqa: BLE001 - one bad question shouldn't kill the run
            result = None
            error = str(exc)
        latency = time.monotonic() - started

        if error:
            print(f"  FAILED: {error}")
            results.append({"financebench_id": q["financebench_id"], "error": error})
            continue

        tool_calls = [tc for sa in result["sub_answers"] for tc in sa["tool_calls"]]
        max_retry = max((sa["retry_count"] for sa in result["sub_answers"]), default=0)
        accuracy = numerical_accuracy(result["final_answer"], q["answer"])

        print(f"  final answer: {result['final_answer'][:150]}")
        print(f"  ground truth: {q['answer']}")
        print(
            f"  tool calls: {len(tool_calls)}, max retry: {max_retry}, "
            f"numerical_accuracy: {accuracy}, latency: {latency:.1f}s"
        )

        results.append(
            {
                "financebench_id": q["financebench_id"],
                "question": q["question"],
                "final_answer": result["final_answer"],
                "ground_truth": q["answer"],
                "n_sub_questions": len(result["sub_answers"]),
                "n_tool_calls": len(tool_calls),
                "tool_calls": tool_calls,
                "max_retry_count": max_retry,
                "numerical_accuracy": accuracy,
                "latency_seconds": latency,
                "error": None,
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("error") is None)
    n_tool_used = sum(1 for r in results if r.get("n_tool_calls", 0) > 0)
    n_correct = sum(1 for r in results if r.get("numerical_accuracy") is True)
    n_scored = sum(1 for r in results if r.get("numerical_accuracy") is not None)

    print("\n" + "=" * 80)
    print(f"Agent eval: {n_ok}/{len(results)} completed without error")
    print(f"Tool used on {n_tool_used}/{len(results)} numerical-reasoning questions")
    print(f"Task completion (numerical accuracy): {n_correct}/{n_scored}")


if __name__ == "__main__":
    main()
