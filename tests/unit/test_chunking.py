from findocqa.retrieval.chunking import chunk_filing


def test_chunk_filing_splits_long_text_with_overlap():
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [{"type": "text", "content": "A" * 3000, "rows": None}],
    }

    chunks = chunk_filing(filing)

    assert len(chunks) > 1
    assert all(c.doc_name == "TEST_2020_10K" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_filing_flattens_table_rows_to_text():
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [
            {"type": "text", "content": "Intro paragraph.", "rows": None},
            {"type": "table", "content": None, "rows": [["Revenue", "100"], ["Costs", "40"]]},
        ],
    }

    chunks = chunk_filing(filing)

    assert len(chunks) == 1
    assert "Intro paragraph." in chunks[0].text
    assert "Revenue | 100" in chunks[0].text
