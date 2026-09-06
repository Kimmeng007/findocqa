"""Detects which company and/or fiscal year a question is about, so
retrieval can be scoped down before similarity search ever runs.

Root cause this fixes: dense embeddings don't strongly distinguish
"FY2018" from "FY2017" -- a same-topic chunk from the wrong fiscal year
of the *same* company scores just as high as, or higher than, the right
one (e.g. 3M's FY2018 capex chunk ranked ~584th out of ~600 3M-only
chunks purely on cosine similarity, because 3M's FY2015-2023 filings all
contain a near-identical capex table). Company filtering alone (matching
"3M" in the query) is not enough for this case; the fiscal year named in
the question has to narrow the candidate pool too. See TECHNICAL_REPORT.md.
"""

import re
from functools import lru_cache

from findocqa.storage.mongo import get_filings_collection

_DOC_YEAR_RE = re.compile(r"_(\d{4})")
_QUERY_YEAR_RE = re.compile(r"FY\s?(\d{4})|fiscal\s+year\s+(\d{4})|\b(20\d{2})\b", re.IGNORECASE)


@lru_cache
def get_known_companies() -> tuple[str, ...]:
    companies = get_filings_collection().distinct("company")
    # Longest first, so "American Express" matches before any shorter,
    # coincidentally-overlapping name would.
    return tuple(sorted(companies, key=len, reverse=True))


def detect_company(query: str) -> str | None:
    """Returns the single company named in the query, or None if zero or
    *more than one* are named. A comparison question ("Compare 3M and
    Amazon's revenue...") must not be narrowed to just one of them --
    that would silently drop the other company's chunks entirely, which
    is worse than not filtering at all."""
    query_lower = query.lower()
    matches: list[str] = []
    for company in get_known_companies():
        # get_known_companies() is longest-first, so a later, shorter match
        # already contained in one we kept (e.g. "Express" inside an
        # already-matched "American Express") is the same company
        # mentioned once, not a second one -- skip it rather than let it
        # count toward the multi-company check below.
        if company.lower() in query_lower and not any(company.lower() in m.lower() for m in matches):
            matches.append(company)
    return matches[0] if len(matches) == 1 else None


def detect_fiscal_year(query: str) -> int | None:
    match = _QUERY_YEAR_RE.search(query)
    if not match:
        return None
    return int(next(g for g in match.groups() if g))


def doc_fiscal_year(doc_name: str) -> int | None:
    """Extracts the fiscal year embedded in a doc_name like "3M_2018_10K"
    or "BESTBUY_2024Q2_10Q" -- avoids requiring a corpus-wide reindex just
    to add a fiscal_year field to chunk metadata."""
    match = _DOC_YEAR_RE.search(doc_name)
    return int(match.group(1)) if match else None
