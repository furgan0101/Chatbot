from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, List

from src.ingest import PDFDocument, PDFPage, normalize_extracted_text


@dataclass
class TextChunk:
    """A chunk of document text stored with source and page metadata."""

    chunk_id: str
    source_name: str
    source_path: str
    text: str
    start_char: int
    end_char: int
    page_number: int | None = None
    section_heading: str | None = None
    source_variants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageBlock:
    """A semantically grouped piece of a PDF page."""

    text: str
    heading: str | None
    start_char: int
    end_char: int


def normalize_bullet(line: str) -> str:
    """Normalize bullet-like prefixes into a stable textual form."""

    return re.sub(r"^[\u2022\-*]+\s*", "- ", line.strip())


def looks_like_heading(line: str) -> bool:
    """Heuristic heading detector for lecture slide text."""

    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    word_count = len(stripped.split())
    if word_count == 0 or word_count > 14:
        return False
    if stripped.endswith("."):
        return False
    if stripped.endswith(":"):
        return True
    uppercase_letters = sum(1 for character in stripped if character.isupper())
    alpha_letters = sum(1 for character in stripped if character.isalpha())
    if alpha_letters and uppercase_letters / alpha_letters > 0.55:
        return True
    return word_count <= 8


def split_large_block(text: str, max_length: int) -> list[str]:
    """Split an oversized block along sentence boundaries when possible."""

    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_length:
        window = remaining[:max_length]
        boundary = max(
            window.rfind("\n\n"),
            window.rfind(". "),
            window.rfind("; "),
            window.rfind(", "),
            window.rfind(" "),
        )
        if boundary <= max_length // 2:
            boundary = max_length
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return [part for part in parts if part]


def extract_page_blocks(page: PDFPage) -> list[PageBlock]:
    """Split a page into coherent paragraph or bullet-group blocks."""

    raw_lines = [line.rstrip() for line in page.raw_text.splitlines()]
    blocks: list[PageBlock] = []
    current_lines: list[str] = []
    active_heading = page.heading
    char_cursor = 0

    def flush_current() -> None:
        nonlocal current_lines, char_cursor
        if not current_lines:
            return
        text = normalize_extracted_text("\n".join(current_lines))
        if not text:
            current_lines = []
            return
        start_char = char_cursor
        end_char = start_char + len(text)
        blocks.append(
            PageBlock(
                text=text,
                heading=active_heading,
                start_char=start_char,
                end_char=end_char,
            )
        )
        char_cursor = end_char + 2
        current_lines = []

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            flush_current()
            continue
        if looks_like_heading(line) and not current_lines:
            active_heading = line.rstrip(":")
            continue
        if looks_like_heading(line) and current_lines:
            flush_current()
            active_heading = line.rstrip(":")
            continue
        if re.match(r"^[\u2022\-*]\s+", line):
            current_lines.append(normalize_bullet(line))
            continue
        current_lines.append(line)

    flush_current()

    if not blocks and page.text:
        blocks.append(
            PageBlock(
                text=page.text,
                heading=page.heading,
                start_char=0,
                end_char=len(page.text),
            )
        )

    expanded_blocks: list[PageBlock] = []
    for block in blocks:
        offset = block.start_char
        for segment in split_large_block(block.text, max_length=450):
            expanded_blocks.append(
                PageBlock(
                    text=segment,
                    heading=block.heading,
                    start_char=offset,
                    end_char=offset + len(segment),
                )
            )
            offset += len(segment) + 1

    return expanded_blocks


def build_page_chunks(
    document: PDFDocument,
    page: PDFPage,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Build chunks for one page while preserving page and heading metadata."""

    blocks = extract_page_blocks(page)
    if not blocks:
        return []

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_heading = page.heading
    current_start = blocks[0].start_char
    chunk_index = 0

    def emit_chunk() -> str:
        nonlocal chunk_index
        chunk_text = "\n\n".join(part for part in current_parts if part).strip()
        if not chunk_text:
            return ""
        chunks.append(
            TextChunk(
                chunk_id=f"{document.source_name}-p{page.page_number}-chunk-{chunk_index}",
                source_name=document.source_name,
                source_path=document.source_path,
                text=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                page_number=page.page_number,
                section_heading=current_heading,
                source_variants=list(document.source_variants),
            )
        )
        chunk_index += 1
        return chunk_text

    for block in blocks:
        candidate_parts = [*current_parts, block.text]
        candidate_text = "\n\n".join(candidate_parts).strip()
        if current_parts and len(candidate_text) > chunk_size:
            previous_chunk_text = emit_chunk()
            overlap_text = previous_chunk_text[-chunk_overlap:].strip() if chunk_overlap else ""
            current_parts = [overlap_text] if overlap_text else []
            current_heading = block.heading or page.heading
            current_start = max(0, block.start_char - len(overlap_text))

        if not current_parts:
            current_heading = block.heading or page.heading
            current_start = block.start_char

        current_parts.append(block.text)

    emit_chunk()
    return chunks


def chunk_documents(
    documents: Iterable[PDFDocument],
    chunk_size: int = 850,
    chunk_overlap: int = 120,
) -> List[TextChunk]:
    """Split parsed documents into page-aware chunks while preserving metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for document in documents:
        for page in document.pages:
            chunks.extend(
                build_page_chunks(
                    document=document,
                    page=page,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

    return chunks
