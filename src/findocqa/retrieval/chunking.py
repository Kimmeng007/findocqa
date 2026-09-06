"""Two chunking strategies, kept side by side so Week 3's eval can report a
measured baseline -> improvement comparison instead of overwriting Week 1's
numbers:

- `chunk_filing_naive`: Week 1 baseline. Flattens every block (text and
  tables alike) into one string and slices it at a fixed character window,
  with no regard for block boundaries.
- `chunk_filing_table_aware`: Week 2. Fixes the concrete failure the Week 1
  baseline produced on a real FinanceBench question (3M FY2018 capex): the
  correct figure was retrieved, but its chunk's fixed-size window had cut
  the table's header row into the *previous* chunk, leaving the model a
  wall of unlabeled numbers it correctly declined to guess from. Table
  blocks are now never split across a chunk boundary — each table becomes
  one chunk (or, if too large, several with the header row repeated in
  each) — and text-block windows no longer bleed into neighboring tables.
"""

from dataclasses import dataclass

CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200

# Markdown-table rows this large need splitting so a single chunk doesn't
# blow past the embedding model's effective input window. Kept small
# (rather than the embedding model's actual ~2000-char budget) on purpose:
# a large multi-line-item statement (e.g. a 20-row cash flow statement)
# fitting in one chunk buries any single figure among many others, which
# measurably hurt both dense and BM25 ranking for a real, specific-figure
# question (3M FY2018 capex -- see TECHNICAL_REPORT.md section 7.4/7.5).
# Splitting into small header-repeating groups keeps each retrievable unit
# focused on a handful of related line items instead of a whole statement.
MAX_TABLE_CHUNK_CHARS = 500
CONTEXT_TAIL_CHARS = 200


@dataclass
class Chunk:
    doc_name: str
    company: str
    text: str
    chunk_index: int


def _block_to_text(block: dict) -> str:
    if block["type"] == "text":
        return block["content"] or ""
    rows = block.get("rows") or []
    return "\n".join(" | ".join(cell or "" for cell in row) for row in rows)


def chunk_filing_naive(filing: dict) -> list[Chunk]:
    """Week 1 baseline: flatten everything, slide a fixed-size window over
    the whole document with no awareness of table boundaries."""
    full_text = "\n\n".join(_block_to_text(b) for b in filing.get("blocks", []))

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE_CHARS
        piece = full_text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    doc_name=filing["doc_name"],
                    company=filing["company"],
                    text=piece,
                    chunk_index=index,
                )
            )
            index += 1
        start = end - CHUNK_OVERLAP_CHARS
    return chunks


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    # SEC filing tables routinely have a short caption/title row (e.g. a
    # single merged cell) followed by wider data rows, since blank spacer
    # cells and currency-symbol cells each get their own <td>. Sizing off
    # the first row alone would truncate every wider row and silently
    # drop data — width must cover the widest row in the table.
    width = max(len(r) for r in rows)
    header, *body = rows

    def fmt_row(row: list[str]) -> str:
        cells = list(row) + [""] * (width - len(row))
        return "| " + " | ".join(c or "" for c in cells) + " |"

    lines = [fmt_row(header), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(fmt_row(r) for r in body)
    return "\n".join(lines)


def _table_block_to_chunks(rows: list[list[str]], context: str) -> list[str]:
    """Serializes a table to Markdown, repeating the header row if the table
    needs splitting to stay under MAX_TABLE_CHUNK_CHARS."""
    if not rows:
        return []

    prefix = f"{context}\n\n" if context else ""
    full_md = _rows_to_markdown(rows)
    if len(prefix) + len(full_md) <= MAX_TABLE_CHUNK_CHARS:
        return [prefix + full_md]

    header, *body = rows
    budget = MAX_TABLE_CHUNK_CHARS - len(prefix) - len(_rows_to_markdown([header]))
    pieces: list[str] = []
    batch: list[list[str]] = []
    batch_chars = 0
    for row in body:
        row_chars = len(" | ".join(row)) + 4
        if batch and batch_chars + row_chars > budget:
            pieces.append(prefix + _rows_to_markdown([header, *batch]))
            batch, batch_chars = [], 0
        batch.append(row)
        batch_chars += row_chars
    if batch:
        pieces.append(prefix + _rows_to_markdown([header, *batch]))
    return pieces


def chunk_filing_table_aware(filing: dict) -> list[Chunk]:
    """Week 2: chunk per block. Tables are never split across a chunk
    boundary (or, if too large for one chunk, split with the header row
    repeated); text blocks keep Week 1's sliding window but scoped to a
    single block so it can't bleed into an adjacent table."""
    doc_name = filing["doc_name"]
    company = filing["company"]
    blocks = filing.get("blocks", [])

    pieces: list[str] = []
    preceding_text_tail = ""

    for block in blocks:
        if block["type"] == "text":
            text = block["content"] or ""
            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE_CHARS
                piece = text[start:end].strip()
                if piece:
                    pieces.append(piece)
                start = end - CHUNK_OVERLAP_CHARS
            preceding_text_tail = text[-CONTEXT_TAIL_CHARS:].strip()
        else:
            table_pieces = _table_block_to_chunks(
                block.get("rows") or [], preceding_text_tail
            )
            pieces.extend(table_pieces)

    return [
        Chunk(doc_name=doc_name, company=company, text=piece, chunk_index=i)
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]
