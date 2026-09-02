import pytest

from findocqa.agent.tools import CalculatorError, calculate


def test_calculate_matches_real_amd_da_margin_from_week2():
    # $167M is the real D&A figure our own retrieval found for AMD's 2015
    # 10-K in Week 2/3 testing; FinanceBench's documented ground truth for
    # the D&A margin is 4.2% -- this confirms the calculator reproduces
    # that exactly rather than relying on an LLM's arithmetic.
    result = calculate("167 / 3976.19 * 100")
    assert round(result, 1) == 4.2


def test_calculate_matches_real_amazon_revenue_growth_from_week2():
    result = calculate("(177866 - 135987) / 135987 * 100")
    assert round(result, 1) == 30.8


def test_calculate_handles_parentheses_and_negatives():
    assert calculate("-(5 - 10)") == 5
    assert calculate("(2 + 3) * 4") == 20


def test_calculate_rejects_division_by_zero():
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


def test_calculate_rejects_names_and_attribute_access():
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('echo hi')")


def test_calculate_rejects_function_calls():
    with pytest.raises(CalculatorError):
        calculate("abs(-5)")


def test_calculate_rejects_non_expression_syntax():
    with pytest.raises(CalculatorError):
        calculate("1; 2")


def test_calculate_rejects_garbage_input():
    with pytest.raises(CalculatorError):
        calculate("not a number")
