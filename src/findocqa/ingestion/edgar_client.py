"""SEC EDGAR client: resolve a company + form type + fiscal year to the
actual filing document, and download it politely.

We use FinanceBench's document metadata only to know *which* filings we
need (company, doc_type, fiscal year) — its `doc_link` URLs point at
IR-vendor-hosted copies that are hotlink-protected (return S3 AccessDenied
for programmatic requests) and whose embedded "accession numbers" turned out
not to be real SEC accession numbers. So the actual bytes are always fetched
from sec.gov / data.sec.gov, matching the spec's "via SEC EDGAR API" design.
"""

import time
from pathlib import Path

import requests

from findocqa.config import settings

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{filename}"

_MIN_INTERVAL_SECONDS = 0.15  # well under SEC's 10 req/s cap
_last_request_time = 0.0


def _headers() -> dict:
    return {"User-Agent": settings.sec_edgar_user_agent}


def _throttle() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_INTERVAL_SECONDS:
        time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def _get_json(url: str) -> dict:
    _throttle()
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_file(url: str, dest: Path) -> Path:
    """Downloads `url` to `dest`, skipping if already present on disk."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _throttle()
    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def lookup_cik(company_name: str, _cache: dict = {}) -> str | None:
    """Best-effort CIK lookup by company-name substring match against
    EDGAR's company_tickers.json. Returns a zero-padded 10-digit CIK."""
    if not _cache:
        for row in _get_json(_TICKERS_URL).values():
            _cache[row["ticker"].upper()] = row
            _cache[row["title"].upper()] = row

    key = company_name.upper()
    row = _cache.get(key)
    if row is None:
        for name, candidate in _cache.items():
            if key in name:
                row = candidate
                break
    return f"{row['cik_str']:010d}" if row else None


def _iter_submission_pages(cik: str):
    """Yields dicts of parallel arrays (form, accessionNumber, primaryDocument,
    reportDate, filingDate), newest first: the 'recent' page plus every
    older paginated page EDGAR splits large filers' history into."""
    data = _get_json(_SUBMISSIONS_URL.format(cik=cik))
    yield data["filings"]["recent"]
    for page in data["filings"].get("files", []):
        page_url = f"https://data.sec.gov/submissions/{page['name']}"
        yield _get_json(page_url)


def find_filing(cik: str, form_type: str, fiscal_year: int) -> dict | None:
    """Finds the filing of `form_type` (e.g. "10-K") whose reported fiscal
    period falls in `fiscal_year`. Returns
    {accession_number, primary_document, filing_date, report_date, url} or
    None if no match is found."""
    for page in _iter_submission_pages(cik):
        for i, form in enumerate(page["form"]):
            if form != form_type:
                continue
            report_date = page["reportDate"][i]
            if not report_date.startswith(str(fiscal_year)):
                continue
            accession_number = page["accessionNumber"][i]
            primary_document = page["primaryDocument"][i]
            accn_nodash = accession_number.replace("-", "")
            cik_int = int(cik)
            return {
                "accession_number": accession_number,
                "primary_document": primary_document,
                "filing_date": page["filingDate"][i],
                "report_date": report_date,
                "url": _ARCHIVE_URL.format(
                    cik_int=cik_int, accn_nodash=accn_nodash, filename=primary_document
                ),
            }
    return None
