from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Sequence

from pypdf import PdfReader

from src.utils import DOCS_DIR, ensure_directories, sanitize_filename, unique_path


NEAR_DUPLICATE_THRESHOLD = 0.9


@dataclass
class PDFPage:
    """Extracted PDF page content with lightweight structural metadata."""

    page_number: int
    raw_text: str
    text: str
    heading: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PDFDocument:
    """A parsed PDF document ready for chunking and indexing."""

    source_name: str
    source_path: str
    text: str
    page_count: int
    pages: list[PDFPage]
    file_hash: str
    text_hash: str
    fingerprint: str
    source_variants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DuplicateRecord:
    """Information about a skipped duplicate file."""

    skipped_source_name: str
    skipped_source_path: str
    canonical_source_name: str
    canonical_source_path: str
    duplicate_type: str
    similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentLoadResult:
    """Documents plus duplicate-tracking metadata."""

    documents: list[PDFDocument]
    duplicates: list[DuplicateRecord]


def normalize_extracted_text(text: str) -> str:
    """Normalize PDF text while preserving paragraph breaks."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for raw_paragraph in text.split("\n\n"):
        lines = [" ".join(line.split()) for line in raw_paragraph.splitlines() if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))
    return "\n\n".join(paragraphs).strip()


def normalize_similarity_text(text: str) -> str:
    """Normalize text for cross-document similarity checks."""

    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def hash_text(value: str) -> str:
    """Return a stable hash for a text value."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenize_for_similarity(text: str) -> list[str]:
    """Return normalized tokens for duplicate detection."""

    return [token for token in normalize_similarity_text(text).split() if token]


def shingle_text(text: str, size: int = 5, max_shingles: int = 4000) -> set[str]:
    """Build a bounded set of token shingles for rough near-duplicate checks."""

    tokens = tokenize_for_similarity(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}

    shingles: set[str] = set()
    upper_bound = min(len(tokens) - size + 1, max_shingles)
    for index in range(upper_bound):
        shingles.add(" ".join(tokens[index : index + size]))
    return shingles


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity for two sets."""

    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def document_similarity(left: PDFDocument, right: PDFDocument) -> float:
    """Estimate whether two extracted PDFs are effectively the same content."""

    if left.text_hash == right.text_hash:
        return 1.0

    left_shingles = shingle_text(left.fingerprint)
    right_shingles = shingle_text(right.fingerprint)
    shingle_score = jaccard_similarity(left_shingles, right_shingles)

    preview_length = 20000
    sequence_score = SequenceMatcher(
        None,
        left.fingerprint[:preview_length],
        right.fingerprint[:preview_length],
    ).ratio()

    return (0.7 * shingle_score) + (0.3 * sequence_score)


def infer_heading(lines: list[str]) -> str | None:
    """Guess a page heading from the leading lines when one is obvious."""

    for raw_line in lines[:6]:
        line = " ".join(raw_line.split())
        if not line:
            continue
        word_count = len(line.split())
        if word_count > 14 or len(line) > 120:
            continue
        if line.endswith("."):
            continue
        if re.match(r"^(page|slide)\s+\d+$", line.lower()):
            continue
        return line
    return None


def extract_text_from_pdf(pdf_path: Path) -> PDFDocument | None:
    """Extract text from a single PDF file and retain page-level metadata."""

    reader = PdfReader(str(pdf_path))
    pages: list[PDFPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if not raw_text.strip():
            continue
        cleaned_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        normalized_text = normalize_extracted_text(raw_text)
        if not normalized_text:
            continue
        pages.append(
            PDFPage(
                page_number=page_number,
                raw_text=raw_text,
                text=normalized_text,
                heading=infer_heading(cleaned_lines),
            )
        )

    if not pages:
        return None

    file_bytes = pdf_path.read_bytes()
    combined_text = "\n\n".join(page.text for page in pages)
    normalized_similarity = normalize_similarity_text(combined_text)

    return PDFDocument(
        source_name=pdf_path.name,
        source_path=str(pdf_path),
        text=combined_text,
        page_count=len(reader.pages),
        pages=pages,
        file_hash=hashlib.sha256(file_bytes).hexdigest(),
        text_hash=hash_text(combined_text),
        fingerprint=normalized_similarity,
        source_variants=[pdf_path.name],
    )


def deduplicate_documents(documents: Sequence[PDFDocument]) -> DocumentLoadResult:
    """Remove exact and near-duplicate PDFs while preserving traceability."""

    canonical_documents: list[PDFDocument] = []
    duplicates: list[DuplicateRecord] = []

    for document in documents:
        matched_canonical: PDFDocument | None = None
        duplicate_type = ""
        similarity = 0.0

        for canonical in canonical_documents:
            if document.file_hash == canonical.file_hash:
                matched_canonical = canonical
                duplicate_type = "exact-file"
                similarity = 1.0
                break
            if document.text_hash == canonical.text_hash:
                matched_canonical = canonical
                duplicate_type = "exact-text"
                similarity = 1.0
                break

            similarity_score = document_similarity(document, canonical)
            if similarity_score >= NEAR_DUPLICATE_THRESHOLD:
                matched_canonical = canonical
                duplicate_type = "near-duplicate"
                similarity = similarity_score
                break

        if matched_canonical is None:
            canonical_documents.append(document)
            continue

        matched_canonical.source_variants.append(document.source_name)
        duplicates.append(
            DuplicateRecord(
                skipped_source_name=document.source_name,
                skipped_source_path=document.source_path,
                canonical_source_name=matched_canonical.source_name,
                canonical_source_path=matched_canonical.source_path,
                duplicate_type=duplicate_type,
                similarity=round(similarity, 4),
            )
        )

    return DocumentLoadResult(documents=canonical_documents, duplicates=duplicates)


def load_documents(
    docs_dir: Path = DOCS_DIR,
    deduplicate: bool = False,
) -> List[PDFDocument]:
    """Load all readable PDFs from the configured documents directory."""

    return load_documents_with_deduplication(docs_dir, deduplicate=deduplicate).documents


def load_documents_with_deduplication(
    docs_dir: Path = DOCS_DIR,
    deduplicate: bool = False,
) -> DocumentLoadResult:
    """Load PDFs and optionally collapse exact and near-duplicate files."""

    ensure_directories()
    documents: list[PDFDocument] = []

    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        try:
            document = extract_text_from_pdf(pdf_path)
        except Exception:
            continue
        if document is not None:
            documents.append(document)

    if not deduplicate:
        return DocumentLoadResult(documents=documents, duplicates=[])

    return deduplicate_documents(documents)


def save_uploaded_pdfs(uploaded_files: Sequence[Any], docs_dir: Path = DOCS_DIR) -> list[str]:
    """Save uploaded PDF files into the documents directory with safe filenames."""

    ensure_directories()
    docs_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for uploaded_file in uploaded_files:
        safe_name = sanitize_filename(uploaded_file.name)
        target_path = unique_path(docs_dir / safe_name)
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(target_path.name)

    return saved_files
