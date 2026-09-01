"""Manual spot-check: naive dense-only (Week 1 baseline) vs. table-aware
hybrid+rerank (Week 2) on ~10 real FinanceBench questions.

Not automated scoring — that's Week 3 with RAGAS. This is for eyeballing
whether Week 2's changes move retrieval in the right direction before
investing in a full eval harness.

Usage:
    uv run scripts/compare_retrieval.py
"""

import time

from findocqa.generation.answer import answer_question

# Gemini free tier: 15 requests/min for gemini-3.5-flash-lite. Two calls per
# question (naive + hybrid); pace them to stay comfortably under that.
CALL_DELAY_SECONDS = 5

QUESTIONS = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
    "Give a response to the question by relying on the details shown in the cash flow statement.",
    "Assume that you are a public equities analyst. Answer the following question by primarily "
    "using information that is shown in the balance sheet: what is the year end FY2018 net PPNE "
    "for 3M? Answer in USD billions.",
    "Is 3M a capital-intensive business based on FY2022 data?",
    "What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful "
    "metric for a company like this, then please state that and explain why.",
    "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?",
    "Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? "
    "If the quick ratio is not relevant to measure liquidity, please state that and explain why.",
    "Which debt securities are registered to trade on a national securities exchange under 3M's "
    "name as of Q2 of 2023?",
    "Does 3M maintain a stable trend of dividend distribution?",
    "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents "
    "and round to one decimal place). Calculate what was asked by utilizing the line items clearly "
    "shown in the statement of income.",
    "What is Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016 "
    "(in units of percents and round to one decimal place)? Give a solution to the question by "
    "using the income statement.",
    "According to the details clearly outlined within the P&L statement and the statement of cash "
    "flows, what is the FY2015 depreciation and amortization (D&A from cash flow statement) % "
    "margin for AMD?",
]

EXPECTED = [
    "$1577.00",
    "$8.70",
    "No (CAPEX/Revenue 5.1%, Fixed assets/Total Assets 20%, ROA 12.4%)",
    "Decreased 1.7%, driven by gross margin decline + one-off charges",
    "Consumer segment shrunk 0.9% organically",
    "No, quick ratio 0.96",
    "MMM26, MMM30, MMM31 notes",
    "Yes, 65 consecutive years of dividend increases",
    "30.8%",
    "65.4%",
    "4.2%",
]


def main() -> None:
    for question, expected in zip(QUESTIONS, EXPECTED):
        print("=" * 100)
        print(f"Q: {question}")
        print(f"Expected: {expected}")

        naive = answer_question(question, top_k=5, variant="naive", mode="dense")
        time.sleep(CALL_DELAY_SECONDS)
        hybrid = answer_question(question, top_k=5, variant="table_aware", mode="hybrid")
        time.sleep(CALL_DELAY_SECONDS)

        print(f"\n[naive dense-only]\n{naive['answer']}")
        print(f"\n[table_aware hybrid+rerank]\n{hybrid['answer']}")
        print()


if __name__ == "__main__":
    main()
