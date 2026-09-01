"""RAGAS has no built-in metric for "did the answer contain the right
number" — faithfulness/context precision/recall all judge relevance and
grounding, not numeric correctness. FinanceBench's `metrics-generated`
questions have numeric ground truths (e.g. "$1577.00", "30.8%"), so this
fills that gap with a simple, transparent, non-LLM check: no judge-model
cost, no ambiguity about what it's measuring.
"""

import re

_NUMBER_RE = re.compile(r"\(?-?\$?\d[\d,]*\.?\d*\)?%?")

_YEAR_RANGE = range(1900, 2100)


def _is_probable_year(raw: str, value: float) -> bool:
    """A bare 4-digit integer with no currency/percent/decimal/comma marker
    in a plausible calendar-year range is almost certainly a fiscal year
    reference (e.g. "FY2023", "as of 2024"), not a financial figure. Nearly
    every question and context chunk in this domain mentions a year, so
    without this filter two unrelated numbers can spuriously "match" on a
    shared year rather than the actual metric being asked about."""
    has_marker = any(c in raw for c in "$%.,")
    return not has_marker and value == int(value) and int(value) in _YEAR_RANGE


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group()
        # A trailing "." or "," with no digits after it (within this match)
        # is sentence punctuation the regex swallowed, not a decimal point
        # or thousands separator -- e.g. "in FY2024," or "FY2024 total."
        # A genuine decimal/thousands comma is always followed by digits,
        # which would already be part of the same match.
        while raw and raw[-1] in ".,":
            raw = raw[:-1]
        if not raw:
            continue
        negative = raw.startswith("(") and raw.endswith(")")
        cleaned = raw.strip("()").replace("$", "").replace(",", "").replace("%", "")
        if not cleaned or cleaned == "-":
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        value = -value if negative else value
        if _is_probable_year(raw, value):
            continue
        numbers.append(value)
    return numbers


def numerical_accuracy(answer: str, ground_truth: str, tol: float = 0.01) -> bool | None:
    """Returns True/False if the ground truth is numeric and a matching
    number (within relative tolerance) appears in the answer, or None if
    the ground truth has no number to check (a qualitative question —
    excluded from this metric's aggregate rather than scored as wrong)."""
    expected_numbers = extract_numbers(ground_truth)
    if not expected_numbers:
        return None

    answer_numbers = extract_numbers(answer)
    for expected in expected_numbers:
        for actual in answer_numbers:
            if expected == 0:
                if abs(actual) < 1e-9:
                    return True
            elif abs(actual - expected) / abs(expected) <= tol:
                return True
    return False
