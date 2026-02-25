"""
Structure-aware and semantic chunking for ingestion.
Preserves logical paragraphs and Markdown hierarchy for better graph extraction
and conceptual continuity (replaces naive fixed-token splitting).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app import config


def _word_count(text: str) -> int:
    return len((text or "").split())


def semantic_chunk_markdown(
    text: str,
    max_words: int | None = None,
    min_words: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split Markdown into chunks that respect structure: headers (##, ###) and
    paragraph boundaries (\\n\\n). Merges small segments up to max_words for
    conceptual continuity. Returns same shape as chunk_text: id, text, index,
    start_char, end_char.
    """
    max_words = max_words or config.SEMANTIC_CHUNK_MAX_WORDS
    min_words = min_words or config.SEMANTIC_CHUNK_MIN_WORDS
    if not (text or "").strip():
        return []

    # Split into segments: by ## or ### (keep header with following content) or by double newline
    # First split by double newline to get paragraphs/blocks
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[str] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        # If block is large, split by ### or ## while keeping header with next segment
        if _word_count(block) > max_words:
            parts = re.split(r"(?=^#{2,3}\s+)", block, flags=re.MULTILINE)
            for p in parts:
                p = p.strip()
                if p:
                    segments.append(p)
        else:
            segments.append(block)

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_words = 0
    char_offset = 0

    for i, seg in enumerate(segments):
        seg_words = _word_count(seg)
        if current_words + seg_words > max_words and current:
            chunk_text = "\n\n".join(current)
            start_char = char_offset
            char_offset += len(chunk_text) + 2
            chunks.append({
                "id": str(uuid.uuid4())[:12],
                "text": chunk_text,
                "index": len(chunks),
                "start_char": start_char,
                "end_char": start_char + len(chunk_text),
            })
            current = [seg]
            current_words = seg_words
        else:
            current.append(seg)
            current_words += seg_words

        # If a single segment is already over max, emit it alone
        if seg_words >= max_words and len(current) == 1:
            chunk_text = current[0]
            start_char = char_offset
            char_offset += len(chunk_text) + 2
            chunks.append({
                "id": str(uuid.uuid4())[:12],
                "text": chunk_text,
                "index": len(chunks),
                "start_char": start_char,
                "end_char": start_char + len(chunk_text),
            })
            current = []
            current_words = 0

    if current:
        chunk_text = "\n\n".join(current)
        start_char = char_offset
        chunks.append({
            "id": str(uuid.uuid4())[:12],
            "text": chunk_text,
            "index": len(chunks),
            "start_char": start_char,
            "end_char": start_char + len(chunk_text),
        })

    return chunks


def semantic_chunk_plain(
    text: str,
    max_words: int | None = None,
    min_words: int | None = None,
) -> list[dict[str, Any]]:
    """
    Plain-text semantic chunking: split on paragraph (\\n\\n) and sentence
    boundaries, merge up to max_words. Use when input is not Markdown.
    """
    max_words = max_words or config.SEMANTIC_CHUNK_MAX_WORDS
    min_words = min_words or config.SEMANTIC_CHUNK_MIN_WORDS
    if not (text or "").strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    chunks = []
    current: list[str] = []
    current_words = 0
    char_offset = 0

    for p in paragraphs:
        w = _word_count(p)
        if current_words + w > max_words and current:
            chunk_text = "\n\n".join(current)
            start_char = char_offset
            char_offset += len(chunk_text) + 2
            chunks.append({
                "id": str(uuid.uuid4())[:12],
                "text": chunk_text,
                "index": len(chunks),
                "start_char": start_char,
                "end_char": start_char + len(chunk_text),
            })
            current = []
            current_words = 0
        current.append(p)
        current_words += w

    if current:
        chunk_text = "\n\n".join(current)
        start_char = char_offset
        chunks.append({
            "id": str(uuid.uuid4())[:12],
            "text": chunk_text,
            "index": len(chunks),
            "start_char": start_char,
            "end_char": start_char + len(chunk_text),
        })

    return chunks


def is_likely_markdown(text: str) -> bool:
    """Heuristic: text contains # headers or markdown list/code."""
    if not text or len(text) < 50:
        return False
    return bool(re.search(r"^#{1,6}\s+\S", text, re.MULTILINE)) or "```" in text[:2000]


def chunk_text_semantic(
    text: str,
    max_words: int | None = None,
    min_words: int | None = None,
    force_fixed: bool = False,
) -> list[dict[str, Any]]:
    """
    Entry point: use structure-aware (Markdown or paragraph) chunking when
    USE_SEMANTIC_CHUNKING is True and text is Markdown or paragraph-friendly;
    otherwise fall back to fixed-word chunking (chunk_text from contextual).
    """
    if force_fixed or not config.USE_SEMANTIC_CHUNKING:
        return []  # Caller should use chunk_text from contextual for fixed
    if is_likely_markdown(text):
        return semantic_chunk_markdown(text, max_words=max_words, min_words=min_words)
    return semantic_chunk_plain(text, max_words=max_words, min_words=min_words)
