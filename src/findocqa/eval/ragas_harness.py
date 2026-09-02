"""RAGAS evaluation wired to this project's existing Gemini client and
local embeddings model — verified empirically against the installed ragas
version (0.4.x) before building this, since its docs describe a newer
`ragas.metrics.collections` API that `evaluate()` doesn't yet accept in
this release; the classic `ragas.metrics` classes (deprecated but
functional) are used deliberately for that reason.
"""

import time

from google import genai
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from findocqa.config import settings
from findocqa.generation.answer import answer_question
from findocqa.generation.rate_limit import throttle


def _judge_llm():
    """Patches the client's actual generate_content call (not RAGAS's
    llm.generate() wrapper) so every real network attempt is throttled —
    including Instructor's own internal retries on a 429, which happen
    *inside* a single generate() call and would otherwise slip past a
    throttle wrapped only around the outer call. See
    generation/rate_limit.py for why this has to be the innermost
    interception point, not just paced externally."""
    client = genai.Client(api_key=settings.google_api_key)
    original_generate_content = client.models.generate_content

    def throttled_generate_content(*args, **kwargs):
        throttle()
        return original_generate_content(*args, **kwargs)

    client.models.generate_content = throttled_generate_content
    return llm_factory(settings.generation_model, provider="google", client=client)


def _local_embeddings():
    return HuggingFaceEmbeddings(
        model=settings.embedding_model, device="cpu", normalize_embeddings=True
    )


def build_eval_rows(questions: list[dict], variant: str, mode: str) -> list[dict]:
    """Runs our own RAG pipeline per question, producing rows with both the
    fields RAGAS needs (question/answer/contexts/ground_truth) and enough
    extra metadata (financebench_id, latency) for the numerical-accuracy
    metric and the MLflow-logged per-question artifact."""
    rows = []
    for q in questions:
        started = time.monotonic()
        try:
            result = answer_question(
                q["question"], top_k=5, variant=variant, mode=mode
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - a single bad question shouldn't kill the run
            result = {"answer": "", "retrieved_chunks": []}
            error = str(exc)
        latency = time.monotonic() - started

        rows.append(
            {
                "financebench_id": q["financebench_id"],
                "question": q["question"],
                "answer": result["answer"],
                "contexts": [c["text"] for c in result["retrieved_chunks"]],
                "ground_truth": q["answer"],
                "latency_seconds": latency,
                "error": error,
            }
        )
    return rows


def run_ragas_metrics(rows: list[dict]) -> dict[str, list[float | None]]:
    """Runs Faithfulness/ContextPrecision/ContextRecall over rows that
    generated successfully. Returns per-row scores (None for skipped rows)
    so results still line up positionally with `rows`."""
    llm = _judge_llm()
    embeddings = _local_embeddings()
    metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]

    usable_indices = [i for i, r in enumerate(rows) if r["error"] is None and r["contexts"]]
    samples = [
        SingleTurnSample(
            user_input=rows[i]["question"],
            response=rows[i]["answer"],
            retrieved_contexts=rows[i]["contexts"],
            reference=rows[i]["ground_truth"],
        )
        for i in usable_indices
    ]

    scores_by_metric: dict[str, list[float | None]] = {
        "faithfulness": [None] * len(rows),
        "context_precision": [None] * len(rows),
        "context_recall": [None] * len(rows),
    }
    if not samples:
        return scores_by_metric

    dataset = EvaluationDataset(samples=samples)
    # Concurrency no longer needs to do the rate-limiting work — every real
    # call is paced by the shared throttle in _judge_llm() regardless of
    # how many workers dispatch it. max_workers=1 keeps behavior simple and
    # deterministic rather than relying on concurrent threads all queuing
    # on the same lock.
    run_config = RunConfig(max_workers=1)
    result = evaluate(
        dataset, metrics=metrics, llm=llm, embeddings=embeddings, run_config=run_config
    )
    result_df = result.to_pandas()

    for pos, row_idx in enumerate(usable_indices):
        scores_by_metric["faithfulness"][row_idx] = result_df["faithfulness"].iloc[pos]
        scores_by_metric["context_precision"][row_idx] = result_df["context_precision"].iloc[pos]
        scores_by_metric["context_recall"][row_idx] = result_df["context_recall"].iloc[pos]

    return scores_by_metric
