"""Selects a stratified subset of FinanceBench's public 150-question eval
set. The full set spans ~70+ companies; evaluating all of it across 3
pipeline configs would cost more API calls than a single day's free-tier
quota (and take hours longer at measured RAGAS latency) — see the Week 3
plan for the full reasoning. A fixed-seed round-robin sample across
companies keeps the subset diverse and reproducible from code alone.
"""

import random
from collections import defaultdict

from findocqa.ingestion.financebench_docs import get_document_info, get_questions

_SCOPED_DOC_TYPES = {"10k", "10q"}


def select_eval_questions(n: int = 40, seed: int = 42) -> list[dict]:
    doc_type_by_name = {
        d["doc_name"]: d["doc_type"] for d in get_document_info()
    }
    scoped = [
        q
        for q in get_questions()
        if doc_type_by_name.get(q["doc_name"]) in _SCOPED_DOC_TYPES
    ]

    by_company: dict[str, list[dict]] = defaultdict(list)
    for q in scoped:
        by_company[q["company"]].append(q)

    rng = random.Random(seed)
    for questions in by_company.values():
        rng.shuffle(questions)

    companies = sorted(by_company.keys())
    rng.shuffle(companies)

    max_per_company = max((len(qs) for qs in by_company.values()), default=0)

    selected: list[dict] = []
    round_idx = 0
    while len(selected) < n and round_idx < max_per_company:
        for company in companies:
            if len(selected) >= n:
                break
            if round_idx < len(by_company[company]):
                selected.append(by_company[company][round_idx])
        round_idx += 1

    return selected
