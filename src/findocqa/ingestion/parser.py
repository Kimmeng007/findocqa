"""Parses a filing's primary HTML document into an ordered list of typed
blocks, preserving document order between narrative text and tables.

Tables are kept as structured row data (not flattened prose) so Week 2's
table-aware chunker has real structure to work with. Week 1's baseline
chunker (retrieval/chunking.py) still flattens everything to plain text —
that's intentional naive-baseline scope, not a limitation of the parser.
"""

import re
from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup

_PLACEHOLDER_RE = re.compile(r"\x00TABLE_(\d+)\x00")


class Block(TypedDict):
    type: str  # "text" | "table"
    content: str | None  # for type == "text"
    rows: list[list[str]] | None  # for type == "table"


def _normalize_text(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _extract_table_rows(table) -> list[list[str]]:
    rows = [
        [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        for tr in table.find_all("tr")
    ]
    return [row for row in rows if any(cell for cell in row)]


def parse_html(path: Path) -> list[Block]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    tables_by_id: dict[str, list[list[str]]] = {}
    for i, table in enumerate(soup.find_all("table")):
        rows = _extract_table_rows(table)
        if rows:
            tables_by_id[str(i)] = rows
            table.replace_with(f"\x00TABLE_{i}\x00")
        else:
            table.decompose()

    full_text = soup.get_text(separator="\n")

    blocks: list[Block] = []
    pos = 0
    for match in _PLACEHOLDER_RE.finditer(full_text):
        text_piece = _normalize_text(full_text[pos : match.start()])
        if text_piece:
            blocks.append({"type": "text", "content": text_piece, "rows": None})
        blocks.append(
            {"type": "table", "content": None, "rows": tables_by_id[match.group(1)]}
        )
        pos = match.end()

    tail = _normalize_text(full_text[pos:])
    if tail:
        blocks.append({"type": "text", "content": tail, "rows": None})

    return blocks
