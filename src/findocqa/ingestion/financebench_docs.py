"""Pulls FinanceBench's public document + question metadata.

FinanceBench (PatronusAI) publishes the exact filings its 150-question public
eval subset depends on, plus the questions themselves. We target ingestion at
these filings so the Week 1-2 corpus is exactly what Week 3 evaluation needs.
"""

import json
from pathlib import Path

import requests

from findocqa.config import DATA_DIR

_BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data"
DOCUMENT_INFO_URL = f"{_BASE_URL}/financebench_document_information.jsonl"
QUESTIONS_URL = f"{_BASE_URL}/financebench_open_source.jsonl"

FINANCEBENCH_DIR = DATA_DIR / "financebench"
DOCUMENT_INFO_PATH = FINANCEBENCH_DIR / "document_information.jsonl"
QUESTIONS_PATH = FINANCEBENCH_DIR / "questions.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    return dest


def get_document_info(refresh: bool = False) -> list[dict]:
    """Returns records like {doc_name, company, gics_sector, doc_type,
    doc_period, doc_link} for every filing FinanceBench's questions cite."""
    if refresh or not DOCUMENT_INFO_PATH.exists():
        _download(DOCUMENT_INFO_URL, DOCUMENT_INFO_PATH)
    return _load_jsonl(DOCUMENT_INFO_PATH)


def get_questions(refresh: bool = False) -> list[dict]:
    """Returns the 150-question public eval subset: {financebench_id,
    company, doc_name, question, answer, evidence, ...}."""
    if refresh or not QUESTIONS_PATH.exists():
        _download(QUESTIONS_URL, QUESTIONS_PATH)
    return _load_jsonl(QUESTIONS_PATH)
