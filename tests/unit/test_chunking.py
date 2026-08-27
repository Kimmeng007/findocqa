from findocqa.retrieval.chunking import chunk_filing_naive, chunk_filing_table_aware


def test_naive_chunk_filing_splits_long_text_with_overlap():
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [{"type": "text", "content": "A" * 3000, "rows": None}],
    }

    chunks = chunk_filing_naive(filing)

    assert len(chunks) > 1
    assert all(c.doc_name == "TEST_2020_10K" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_naive_chunk_filing_flattens_table_rows_to_text():
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [
            {"type": "text", "content": "Intro paragraph.", "rows": None},
            {"type": "table", "content": None, "rows": [["Revenue", "100"], ["Costs", "40"]]},
        ],
    }

    chunks = chunk_filing_naive(filing)

    assert len(chunks) == 1
    assert "Intro paragraph." in chunks[0].text
    assert "Revenue | 100" in chunks[0].text


def test_table_aware_small_table_kept_in_one_chunk_with_header():
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [
            {"type": "text", "content": "Capital spending by segment.", "rows": None},
            {
                "type": "table",
                "content": None,
                "rows": [
                    ["Segment", "2018", "2017"],
                    ["Industrial", "500", "480"],
                    ["Total Company", "1577", "1373"],
                ],
            },
        ],
    }

    chunks = chunk_filing_table_aware(filing)

    table_chunks = [c for c in chunks if "1577" in c.text]
    assert len(table_chunks) == 1
    chunk = table_chunks[0]
    assert "Segment" in chunk.text and "2018" in chunk.text
    assert "Total Company" in chunk.text
    assert "Capital spending by segment." in chunk.text


def test_table_aware_large_table_repeats_header_when_split():
    header = ["Segment", "2018", "2017"]
    rows = [header] + [[f"Segment{i}", str(i * 10), str(i * 9)] for i in range(200)]
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [{"type": "table", "content": None, "rows": rows}],
    }

    chunks = chunk_filing_table_aware(filing)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "Segment | 2018 | 2017" in chunk.text.replace("| ", "").replace(" |", "") or (
            "Segment" in chunk.text and "2018" in chunk.text and "2017" in chunk.text
        )


def test_table_aware_text_block_still_slides_window():
    filing = {
        "doc_name": "TEST_2020_10K",
        "company": "Test Co",
        "blocks": [{"type": "text", "content": "B" * 3000, "rows": None}],
    }

    chunks = chunk_filing_table_aware(filing)

    assert len(chunks) > 1
