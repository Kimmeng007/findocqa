"""Fetch + parse + store FinanceBench-referenced 10-K/10-Q filings.

FinanceBench's document metadata tells us which (company, doc_type, fiscal
year) we need; the actual bytes are always fetched from SEC EDGAR directly
(its own doc_link URLs are hotlink-protected and don't resolve as real SEC
accession numbers — see ingestion/edgar_client.py).

Scoped to 10-K/10-Q only, matching the project's stated domain
(FinanceBench also includes 8-K/earnings-release/annual-report entries,
which aren't standard EDGAR forms and are out of scope here).

Usage:
    uv run scripts/ingest.py --limit 5
    uv run scripts/ingest.py --company "3M"
    uv run scripts/ingest.py --doc-name AMAZON_2017_10K --doc-name ADOBE_2016_10K
"""

import argparse

from findocqa.ingestion.financebench_docs import get_document_info
from findocqa.ingestion.pipeline import FORM_TYPE_BY_DOC_TYPE, ingest_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument(
        "--doc-name", action="append", default=None, help="Exact doc_name(s) to ingest"
    )
    args = parser.parse_args()

    docs = [d for d in get_document_info() if d["doc_type"] in FORM_TYPE_BY_DOC_TYPE]
    if args.doc_name:
        wanted = set(args.doc_name)
        docs = [d for d in docs if d["doc_name"] in wanted]
    elif args.company:
        docs = [d for d in docs if args.company.lower() in d["company"].lower()]
    if args.limit:
        docs = docs[: args.limit]

    print(f"Ingesting {len(docs)} filing(s)...")
    cik_cache: dict[str, str | None] = {}

    for i, doc in enumerate(docs, start=1):
        print(f"[{i}/{len(docs)}] {doc['doc_name']}")
        try:
            n_blocks = ingest_document(doc, cik_cache)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        print(f"  stored {n_blocks} blocks")


if __name__ == "__main__":
    main()
