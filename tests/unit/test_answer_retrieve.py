from findocqa.generation import answer


def _patch_detection(monkeypatch, company, fiscal_year):
    monkeypatch.setattr(answer, "detect_company", lambda q: company)
    monkeypatch.setattr(answer, "detect_fiscal_year", lambda q: fiscal_year)


def test_retrieve_falls_back_to_company_only_when_year_filter_finds_nothing(monkeypatch):
    """Found via a real eval run: a question naming a year outside what's
    actually ingested for that company (e.g. "FY2018-2020 average" when
    only FY2020 is in the corpus) returned *zero* chunks under a hard
    company+year filter, even though the company's other-year filing
    could partially answer. A filter must never do worse than no filter."""
    _patch_detection(monkeypatch, company="MGM Resorts", fiscal_year=2018)

    calls = []

    def fake_retrieve_with(question, top_k, variant, mode, company, fiscal_year):
        calls.append((company, fiscal_year))
        if fiscal_year is not None:
            return []
        return [{"doc_name": "MGMRESORTS_2020_10K", "chunk_index": 1}]

    monkeypatch.setattr(answer, "_retrieve_with", fake_retrieve_with)

    result = answer._retrieve("some question", top_k=5, variant="table_aware", mode="hybrid")

    assert result == [{"doc_name": "MGMRESORTS_2020_10K", "chunk_index": 1}]
    assert calls == [("MGM Resorts", 2018), ("MGM Resorts", None)]


def test_retrieve_falls_back_to_no_filter_when_company_alone_finds_nothing(monkeypatch):
    _patch_detection(monkeypatch, company="3M", fiscal_year=None)

    calls = []

    def fake_retrieve_with(question, top_k, variant, mode, company, fiscal_year):
        calls.append((company, fiscal_year))
        if company is not None:
            return []
        return [{"doc_name": "SOME_DOC", "chunk_index": 0}]

    monkeypatch.setattr(answer, "_retrieve_with", fake_retrieve_with)

    result = answer._retrieve("some question", top_k=5, variant="table_aware", mode="hybrid")

    assert result == [{"doc_name": "SOME_DOC", "chunk_index": 0}]
    assert calls == [("3M", None), (None, None)]


def test_retrieve_does_not_fall_back_when_filtered_results_are_found(monkeypatch):
    _patch_detection(monkeypatch, company="3M", fiscal_year=2019)

    calls = []

    def fake_retrieve_with(question, top_k, variant, mode, company, fiscal_year):
        calls.append((company, fiscal_year))
        return [{"doc_name": "3M_2019_10K", "chunk_index": 5}]

    monkeypatch.setattr(answer, "_retrieve_with", fake_retrieve_with)

    result = answer._retrieve("some question", top_k=5, variant="table_aware", mode="hybrid")

    assert result == [{"doc_name": "3M_2019_10K", "chunk_index": 5}]
    assert calls == [("3M", 2019)]
