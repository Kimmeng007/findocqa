from findocqa.retrieval import company_filter


def test_detect_company_prefers_longest_match(monkeypatch):
    """"American Express" and a hypothetical shorter overlapping name must
    resolve to the longest/most specific match, not whichever sorts first."""
    monkeypatch.setattr(
        company_filter,
        "get_known_companies",
        lambda: ("American Express", "Express"),
    )
    assert company_filter.detect_company("American Express FY2022 revenue") == "American Express"


def test_detect_company_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(company_filter, "get_known_companies", lambda: ("3M", "Nike"))
    assert company_filter.detect_company("What was Tesla's revenue?") is None


def test_detect_company_returns_none_when_multiple_named(monkeypatch):
    """A comparison question naming two companies must not be narrowed to
    just one -- that would silently drop the other company's chunks."""
    monkeypatch.setattr(company_filter, "get_known_companies", lambda: ("Amazon", "3M"))
    assert company_filter.detect_company("Compare 3M and Amazon's revenue growth") is None


def test_detect_fiscal_year_handles_fy_prefix():
    assert company_filter.detect_fiscal_year("What is the FY2018 capex for 3M?") == 2018


def test_detect_fiscal_year_handles_bare_year():
    assert company_filter.detect_fiscal_year("What was 3M's 2019 revenue?") == 2019


def test_detect_fiscal_year_no_year_returns_none():
    assert company_filter.detect_fiscal_year("What is 3M's capex?") is None


def test_doc_fiscal_year_extracts_year_from_doc_name():
    assert company_filter.doc_fiscal_year("3M_2018_10K") == 2018
    assert company_filter.doc_fiscal_year("BESTBUY_2024Q2_10Q") == 2024


def test_doc_fiscal_year_returns_none_when_absent():
    assert company_filter.doc_fiscal_year("NO_YEAR_HERE") is None
