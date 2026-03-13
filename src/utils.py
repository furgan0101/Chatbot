from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"
BUILD_REPORT_FILE = INDEX_DIR / "build_report.json"


def ensure_directories() -> None:
    """Create project directories used by the application if they do not exist."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Save JSON data to disk using UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Load JSON data from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-like environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable with fallback."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable with fallback."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value.strip())
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    """Read a string environment variable with fallback."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip()
    return value if value else default


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets first, then environment variables."""

    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:
        pass
    return env_str(name, default)


def sanitize_filename(filename: str) -> str:
    """Return a Windows-friendly filename."""

    invalid_characters = '<>:"/\\|?*'
    sanitized = "".join("_" if character in invalid_characters else character for character in filename).strip()
    return sanitized or "uploaded_document.pdf"


def unique_path(path: Path) -> Path:
    """Return a non-conflicting file path by appending a counter if needed."""

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def truncate_text(text: str, max_length: int = 300) -> str:
    """Return a shortened preview string for UI display."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 3]}..."
