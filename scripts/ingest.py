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
"""

import argparse

from findocqa.config import DATA_DIR
from findocqa.ingestion.edgar_client import download_file, find_filing, lookup_cik
from findocqa.ingestion.financebench_docs import get_document_info
from findocqa.ingestion.parser import parse_html
from findocqa.storage.mongo import upsert_filing

RAW_DIR = DATA_DIR / "raw"

_FORM_TYPE_BY_DOC_TYPE = {"10k": "10-K", "10q": "10-Q"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--company", type=str, default=None)
    args = parser.parse_args()

    docs = [d for d in get_document_info() if d["doc_type"] in _FORM_TYPE_BY_DOC_TYPE]
    if args.company:
        docs = [d for d in docs if args.company.lower() in d["company"].lower()]
    if args.limit:
        docs = docs[: args.limit]

    print(f"Ingesting {len(docs)} filing(s)...")
    cik_cache: dict[str, str | None] = {}

    for i, doc in enumerate(docs, start=1):
        doc_name = doc["doc_name"]
        company = doc["company"]
        form_type = _FORM_TYPE_BY_DOC_TYPE[doc["doc_type"]]
        print(f"[{i}/{len(docs)}] {doc_name}")

        try:
            cik = cik_cache.setdefault(company, lookup_cik(company))
            if not cik:
                raise ValueError(f"could not resolve CIK for company {company!r}")

            filing = find_filing(cik, form_type, doc["doc_period"])
            if not filing:
                raise ValueError(
                    f"no {form_type} found on EDGAR for CIK {cik} / FY{doc['doc_period']}"
                )

            html_path = RAW_DIR / f"{doc_name}.htm"
            download_file(filing["url"], html_path)
            blocks = parse_html(html_path)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        upsert_filing(
            {
                "doc_name": doc_name,
                "company": company,
                "gics_sector": doc.get("gics_sector"),
                "filing_type": doc["doc_type"],
                "fiscal_year": doc["doc_period"],
                "cik": cik,
                "accession_number": filing["accession_number"],
                "source_url": filing["url"],
                "blocks": blocks,
            }
        )
        print(f"  stored {len(blocks)} blocks from {filing['url']}")


if __name__ == "__main__":
    main()
