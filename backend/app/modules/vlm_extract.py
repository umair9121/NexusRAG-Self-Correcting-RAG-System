"""
VLM-based text extraction for documents where Docling / traditional OCR
produces insufficient output (handwritten notes, complex layouts, etc.).

Uses a Vision Language Model (qwen3-vl:2b by default) via the Ollama chat
API: each PDF page is rendered to an image, sent to the VLM, and the
response is collected as Markdown with <!-- PAGE N --> markers so that
downstream PageIndex tree-building can preserve page numbers.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF – used to render PDF pages to images
import requests

from app import config

logger = logging.getLogger(__name__)


# ── Heuristic: should we fall back to the VLM? ─────────────────────────────

def should_use_vlm(file_path: str | Path, docling_markdown: str | None) -> bool:
    """
    Decide whether VLM extraction is needed by comparing the amount of text
    Docling produced against the raw PDF file size.

    A scanned / handwritten PDF that Docling barely reads will have a very low
    chars-per-KB ratio.  For example a 2 MB scanned PDF with only 80 chars of
    Markdown → 0.04 chars/KB, well below the default 2.0 threshold.

    Returns ``True`` when the VLM should be tried.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    file_size_kb = file_path.stat().st_size / 1024

    # Tiny files: not worth the overhead of the VLM
    if file_size_kb < config.VLM_MIN_FILE_SIZE_KB:
        return False

    md_chars = len(docling_markdown.strip()) if docling_markdown else 0
    chars_per_kb = md_chars / max(file_size_kb, 1)

    logger.info(
        "VLM heuristic: file=%.1f KB, markdown=%d chars, ratio=%.2f chars/KB "
        "(threshold=%.2f)",
        file_size_kb,
        md_chars,
        chars_per_kb,
        config.VLM_CHARS_PER_KB_THRESHOLD,
    )

    return chars_per_kb < config.VLM_CHARS_PER_KB_THRESHOLD


# ── Page rendering ──────────────────────────────────────────────────────────

def _pdf_page_to_base64(pdf_path: str | Path, page_idx: int, dpi: int = 200) -> str:
    """Render *one* PDF page to a base64-encoded PNG string."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")
    finally:
        doc.close()


def _image_file_to_base64(image_path: str | Path) -> str:
    """Read an image file (.png, .jpg, .jpeg) and return base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_with_vlm_image(
    image_path: str | Path,
    callback: Any | None = None,
) -> str:
    """
    Extract text from a single image using the VLM (e.g. for .png, .jpg uploads).
    Returns transcribed text or empty string on failure.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return ""
    suf = image_path.suffix.lower()
    if suf not in (".png", ".jpg", ".jpeg"):
        return ""
    try:
        img_b64 = _image_file_to_base64(image_path)
        if not img_b64:
            return ""
    except Exception as exc:
        logger.warning("Cannot read image for VLM: %s", exc)
        return ""

    if callback:
        callback(f"Extracting text from image with {config.VLM_MODEL}…")

    payload: dict[str, Any] = {
        "model": config.VLM_MODEL,
        "messages": [
            {"role": "system", "content": _VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Extract all text visible in this image.",
                "images": [img_b64],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        if callback and text:
            callback(f"VLM extracted {len(text)} chars from image")
        return text
    except Exception as exc:
        logger.warning("VLM image extraction failed: %s", exc)
        if callback:
            callback(f"VLM failed: {exc}")
        return ""


# ── Core extraction ─────────────────────────────────────────────────────────

_VLM_SYSTEM_PROMPT = (
    "You are a precise document transcription assistant. "
    "Your ONLY job is to extract every piece of text visible on the page "
    "and reproduce it faithfully in Markdown. "
    "Use # for main headings, ## for sub-headings, ### for smaller headings, "
    "- for bullet points, and blank lines between paragraphs. "
    "If the text is handwritten, transcribe it as accurately as you can. "
    "Output ONLY the transcribed text — no commentary, no preamble."
)


def extract_text_with_vlm(
    pdf_path: str | Path,
    callback: Any | None = None,
) -> str:
    """
    Extract text from every page of a PDF using a Vision Language Model.

    Each page is rendered to a PNG, sent to the Ollama ``/api/chat`` endpoint
    with the configured VLM model, and the response is collected.  Pages are
    separated by ``<!-- PAGE N -->`` markers so the downstream tree builder
    can reconstruct page associations.

    Returns the combined Markdown text, or an empty string on failure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    except Exception as exc:
        logger.warning("Cannot open PDF for VLM extraction: %s", exc)
        return ""

    if callback:
        callback(
            f"Starting VLM text extraction ({config.VLM_MODEL}) "
            f"for {total_pages} page(s)…"
        )

    all_page_texts: list[tuple[int, str]] = []  # (1-based page number, text)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if callback:
            callback(f"VLM processing page {page_num}/{total_pages}…")

        try:
            img_b64 = _pdf_page_to_base64(pdf_path, page_idx)

            payload: dict[str, Any] = {
                "model": config.VLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": _VLM_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "Extract all text from this document page.",
                        "images": [img_b64],
                    },
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 4096,
                },
            }

            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()

            text = (
                resp.json()
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if text:
                all_page_texts.append((page_num, text))
                if callback:
                    callback(
                        f"VLM extracted {len(text)} chars from page {page_num}"
                    )
            else:
                if callback:
                    callback(f"VLM found no text on page {page_num}")

        except Exception as exc:
            logger.warning("VLM extraction failed for page %d: %s", page_num, exc)
            if callback:
                callback(f"VLM failed on page {page_num}: {exc}")

    if not all_page_texts:
        if callback:
            callback("VLM produced no text from any page")
        return ""

    # Combine pages with markers that build_tree_from_markdown can parse
    combined_parts: list[str] = []
    for page_num, text in all_page_texts:
        combined_parts.append(f"<!-- PAGE {page_num} -->\n{text}")

    combined = "\n\n".join(combined_parts)
    if callback:
        callback(
            f"VLM extraction complete: {len(combined)} total chars "
            f"from {len(all_page_texts)} page(s)"
        )

    return combined
