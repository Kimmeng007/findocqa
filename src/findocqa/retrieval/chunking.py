"""Naive fixed-size chunking — the Week 1 baseline to be beaten by Week 2's
table-aware chunker. Flattens every block (including tables) to plain text."""

from dataclasses import dataclass

CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200


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
    return "\n".join(
        " | ".join(cell or "" for cell in row) for row in rows
    )


def chunk_filing(filing: dict) -> list[Chunk]:
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
