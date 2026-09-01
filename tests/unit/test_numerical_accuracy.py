from findocqa.eval.numerical_accuracy import extract_numbers, numerical_accuracy


def test_extract_numbers_handles_currency_commas_and_percent():
    assert extract_numbers("Revenue was $1,577.00 million") == [1577.00]
    assert extract_numbers("grew by 30.8%") == [30.8]
    assert extract_numbers("$8.70") == [8.70]


def test_extract_numbers_handles_parenthesized_negatives():
    assert extract_numbers("a decrease of (200) million") == [-200.0]


def test_extract_numbers_handles_multiple_values():
    assert extract_numbers("from $135,987 to $177,866") == [135987.0, 177866.0]


def test_numerical_accuracy_true_when_answer_contains_matching_number():
    answer = "The FY2018 capital expenditure was $1,577.00 million."
    assert numerical_accuracy(answer, "$1577.00") is True


def test_numerical_accuracy_true_within_tolerance():
    answer = "It was approximately $1577.5 million."
    assert numerical_accuracy(answer, "$1577.00") is True


def test_numerical_accuracy_false_when_no_matching_number():
    answer = "There is no mention of the figure in the context."
    assert numerical_accuracy(answer, "$1577.00") is False


def test_numerical_accuracy_none_for_non_numeric_ground_truth():
    answer = "Yes, 3M has a stable dividend history."
    assert numerical_accuracy(answer, "Yes, consistent dividend growth") is None


def test_extract_numbers_excludes_bare_years():
    assert extract_numbers("FY2023 and FY2024 figures") == []
    assert extract_numbers("as of 2024") == []


def test_numerical_accuracy_does_not_false_match_on_shared_years():
    """Regression guard: two answers that both mention FY2023/FY2024 but
    disagree on the actual metric (42% vs no percent found) must not be
    scored as matching just because the years happen to coincide. (Small
    bare integers like quarter numbers can still coincidentally collide —
    an inherent limitation of a deliberately simple, non-LLM check — this
    test isolates the year-specific bug, not that broader class.)"""
    answer = (
        "The filing shows cash and cash equivalents balances for FY2023 "
        "and FY2024, but no percentage change is given."
    )
    ground_truth = "Yes, there was a decline of ~42% between FY2023 and FY2024."
    assert numerical_accuracy(answer, ground_truth) is False


def test_numerical_accuracy_still_matches_a_real_percent_alongside_years():
    answer = "Cash declined 42% between FY2023 (as of October 28, 2023) and FY2024."
    ground_truth = "Yes, there was a decline of ~42% between FY2023 and Q2 of FY2024."
    assert numerical_accuracy(answer, ground_truth) is True
