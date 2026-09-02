"""Agent tools: a calculator for exact arithmetic (LLMs are unreliable at
precise math on ratios/deltas) and an EDGAR live-fetch tool that lets the
agent ingest a filing on demand.

The calculator deliberately does not use `eval()`. It walks a restricted
AST — numbers and +-*/ only — and rejects anything else (names, calls,
attribute access), so there's no code-injection surface even though the
"user input" here is model-generated, not directly user-supplied.
"""

import ast
import operator

from findocqa.ingestion.financebench_docs import get_document_info
from findocqa.ingestion.pipeline import ingest_document
from findocqa.retrieval.build import build_variant_index
from findocqa.storage.mongo import get_existing_doc_names

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Expression contains a disallowed element: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Evaluates a restricted arithmetic expression exactly, e.g.
    "1577 / 135987 * 100" for a percentage. Only numbers, +, -, *, /, and
    parentheses are allowed -- anything else (names, function calls,
    attribute access) raises CalculatorError rather than silently
    executing."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Not a valid arithmetic expression: {expression!r}") from exc
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero") from exc


def ensure_filing_ingested(company: str, fiscal_year: int) -> bool:
    """EDGAR live-fetch tool: if the agent needs a filing that isn't in
    MongoDB yet, resolve it from FinanceBench's document list and ingest
    it on the fly, reusing the exact same ingestion path Week 3's
    auto-ingest uses. Returns True if a new filing was ingested (the
    caller should rebuild the retrieval indexes), False if nothing was
    needed or the filing couldn't be resolved.

    Known limitation: rebuilding the index after a live fetch means a
    full FAISS/BM25 rebuild (retrieval.build.build_variant_index), not an
    incremental add -- fine at this project's scale, not how a production
    system would do it.
    """
    existing = get_existing_doc_names()
    candidates = [
        d
        for d in get_document_info()
        if d["doc_type"] in ("10k", "10q")
        and company.lower() in d["company"].lower()
        and d["doc_period"] == fiscal_year
        and d["doc_name"] not in existing
    ]
    if not candidates:
        return False

    ingest_document(candidates[0], cik_cache={})
    return True


def reindex_after_live_fetch(variant: str = "table_aware") -> int:
    return build_variant_index(variant)
