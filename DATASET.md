# FinDocQA — What the Data Actually Looks Like

Real records pulled live from the dataset and our database — nothing
paraphrased or reconstructed from memory.

---

## 1. The benchmark: two files

**FinanceBench** (from PatronusAI) publishes two JSONL files we pull
straight from GitHub:

- `financebench_document_information.jsonl` — 361 records, one per source
  filing (across ~70 companies total in this file)
- `financebench_open_source.jsonl` — **150 records**, the actual test
  questions with ground-truth answers. This subset only draws on **32
  unique companies**, not all 70+ in the document file.

### A real document record
```json
{
  "doc_name": "3M_2015_10K",
  "company": "3M",
  "gics_sector": "Industrials",
  "doc_type": "10k",
  "doc_period": 2015,
  "doc_link": "https://investors.3m.com/financials/sec-filings/content/0001558370-16-003162/0001558370-16-003162.pdf"
}
```
(Note: `doc_link` turned out to be hotlink-protected and unusable — see
`TECHNICAL_REPORT.md` §2.1. We fetch the real filing from SEC EDGAR
ourselves instead, using just `company` + `doc_type` + `doc_period` from
this record to find it.)

### A real question record
```json
{
  "financebench_id": "financebench_id_00807",
  "company": "3M",
  "doc_name": "3M_2023Q2_10Q",
  "question_type": "domain-relevant",
  "question_reasoning": "Logical reasoning (based on numerical reasoning)",
  "question": "Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If the quick ratio is not relevant to measure liquidity, please state that and explain why.",
  "answer": "No. The quick ratio for 3M was 0.96 by Jun'23 close, which needs a bit of an improvement to touch the 1x mark",
  "justification": "Quick Ratio= (Total current assets-Total inventories)/Total current liabilities\n(15,754-5,280)/10,936",
  "evidence": [
    {
      "evidence_text": "...Consolidated Balance Sheet...Cash and cash equivalents $4,258 $3,655...Total current assets 15,754 14,688...",
      "evidence_page_num": 3
    }
  ]
}
```

Every question also ships the **exact source passage** (`evidence`) the
ground-truth answer was derived from, and a `justification` showing the
actual formula/reasoning — useful for spot-checking whether our system's
retrieved context actually contains what it needs.

---

## 2. Three question types (50 of each, 150 total)

| Type | What it tests | Real example |
|---|---|---|
| `metrics-generated` | Pull one number straight from a statement | "What is the FY2018 capital expenditure amount (in USD millions) for 3M?" → **`$1577.00`** |
| `domain-relevant` | Apply financial-analyst judgment/ratios | "Is 3M a capital-intensive business based on FY2022 data?" → **"No, ... CAPEX/Revenue Ratio: 5.1%, Fixed assets/Total Assets: 20%, ROA: 12.4%"** |
| `novel-generated` | Open-ended reasoning across the filing | "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?" → **"The consumer segment shrunk by 0.9% organically."** |

`metrics-generated` is the type our custom numerical-accuracy metric
targets (see `TECHNICAL_REPORT.md` §4.3) — the other two types have
prose/judgment answers that a simple number-match can't score, which is
exactly why RAGAS's LLM-judged metrics (faithfulness, context precision/
recall) exist alongside it.

---

## 3. Companies in our current corpus

As of the Week 3 eval run, **32 companies** are represented, spanning very
different filing styles and sectors — not just tech: 3M, AES Corporation,
AMD, Activision Blizzard, Adobe, Amazon, Amcor, American Express, American
Water Works, Best Buy, Block, Boeing, CVS Health, Coca-Cola, Corning,
Costco, Foot Locker, General Mills, JPMorgan, Johnson & Johnson, and more
(full list keeps growing as we ingest more of the 150-question set).

---

## 4. What a filing looks like once it's in MongoDB

One document per filing. Metadata + an ordered list of parsed **blocks**
(the actual filing content is in `blocks`, omitted below for size):

```json
{
  "doc_name": "3M_2022_10K",
  "company": "3M",
  "gics_sector": "Industrials",
  "filing_type": "10k",
  "fiscal_year": 2022,
  "cik": "0000066740",
  "accession_number": "0000066740-23-000014",
  "source_url": "https://www.sec.gov/Archives/edgar/data/66740/000006674023000014/mmm-20221231.htm"
}
```
This one filing parses into **222 blocks** (mix of text paragraphs and
tables), which the table-aware chunker turns into **680 chunks** for
retrieval.

### A real TABLE block (raw, before any chunking)
```json
[
  ["Title of each class", "", "Trading Symbol(s)", "", "Name of each exchange on which registered"],
  ["Common Stock, Par Value $.01 Per Share", "", "MMM", "", "New York Stock Exchange"],
  ["", "", "MMM", "", "Chicago Stock Exchange, Inc."],
  ["0.950% Notes due 2023", "", "MMM23", "", "New York Stock Exchange"]
]
```
Notice the blank `""` cells — SEC HTML tables routinely split a single
visual cell into several `<td>`s (currency symbols, spacers). This
irregularity is exactly what caused the truncation bug in
`TECHNICAL_REPORT.md` §3.2.

### A real, readable TEXT block
```
Note: The common stock of the registrant is also traded on the SIX Swiss
Exchange.
Securities registered pursuant to section 12(g) of the Act:
None
Indicate by check mark if the registrant is a well-known seasoned issuer...
```

---

## 5. A known, minor data-quality issue (found while writing this doc)

Modern 10-Ks embed **inline XBRL** — hidden machine-readable financial
tags — directly in the HTML. Our parser occasionally sweeps this in as if
it were narrative text:

```
mmm-20221231
0000066740
FALSE
2022
FY
http://fasb.org/us-gaap/2022#Revenues
us-gaap:FairValueInputsLevel3Member
...
```

**Measured impact**: 1 of 109 text blocks in the 3M 2022 10-K looked like
this — it's confined to a single leading metadata block per filing, not
spread throughout. Not fixed yet, but now documented rather than
silently present. A future improvement: strip elements with
`display:none` / `ix:hidden`-style attributes during parsing.

---

## 6. A real chunk, as the LLM actually sees it

This is what gets embedded, retrieved, and pasted into the Gemini prompt
— chunk #269 from `chunk_filing_table_aware()` on 3M's real 2022 10-K,
pulled directly from the running system just now, verbatim:

```
ningful to investors as 3M considers net debt and its components to be
important indicators of liquidity and financial position. The following
table provides net debt as of December 31, 2022 and 2021.

|  |  | December 31, |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (Millions) |  | 2022 |  | 2021 |  | Change |  |  |  |  |
| Total debt |  | $ | 15,939 |  | $ | 17,363 |  | $ | (1,424) |  |
| Less: Cash, cash equivalents and marketable securities |  | 3,916 |  | 4,792 |  | (876) |  |  |  |  |
| Net debt (non-GAAP measure) |  | $ | 12,023 |  | $ | 12,571 |  | $ | (548) |  |
```

The leading `"ningful to investors..."` is a mid-word cut — the tail end
of the previous text block's sliding window bleeding in as context. The
table itself is intact: headers, both fiscal years, and every dollar
figure survive together in one chunk, which is exactly what the Week 1
naive chunker failed to guarantee. The extra blank `|  |` columns are the
same SEC HTML cell-splitting quirk noted in §4 — cosmetic noise, but the
actual numbers ($15,939, $17,363, $12,023, $12,571) are all there and
correctly paired with their labels and years.
