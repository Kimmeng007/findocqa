"""Shared single-filing ingestion logic used by scripts/ingest.py and
scripts/run_eval.py's auto-ingest step, so there's one place that resolves
CIK -> finds the filing -> downloads -> parses -> stores.
"""

from findocqa.config import DATA_DIR
from findocqa.ingestion.edgar_client import download_file, find_filing, lookup_cik
from findocqa.ingestion.parser import parse_html
from findocqa.storage.mongo import upsert_filing

RAW_DIR = DATA_DIR / "raw"

FORM_TYPE_BY_DOC_TYPE = {"10k": "10-K", "10q": "10-Q"}


def ingest_document(doc: dict, cik_cache: dict[str, str | None]) -> int:
    """Ingests one FinanceBench document-info record. Returns the number of
    parsed blocks stored. Raises on failure (caller decides how to handle)."""
    doc_name = doc["doc_name"]
    company = doc["company"]
    form_type = FORM_TYPE_BY_DOC_TYPE[doc["doc_type"]]

    cik = cik_cache.setdefault(company, lookup_cik(company))
    if not cik:
        raise ValueError(f"could not resolve CIK for company {company!r}")

    filing = find_filing(cik, form_type, doc["doc_period"])
    if not filing:
        raise ValueError(f"no {form_type} found on EDGAR for CIK {cik} / FY{doc['doc_period']}")

    html_path = RAW_DIR / f"{doc_name}.htm"
    download_file(filing["url"], html_path)
    blocks = parse_html(html_path)

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
    return len(blocks)
