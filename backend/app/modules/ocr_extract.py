"""
Fast OCR for scanned/unselectable text PDFs (no heavy VLM).

- Tesseract: fastest, good for plain text scans.
- OCR_MODEL (Ollama, e.g. PaddleOCR-VL): better for complex columns/tables.
Use triage to route scanned-text docs here instead of full-page VLM.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import fitz
import requests

from app import config

logger = logging.getLogger(__name__)

_TESSERACT_AVAILABLE: bool | None = None


def is_tesseract_available() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract  # noqa: F401
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        return True
    except Exception:
        _TESSERACT_AVAILABLE = False
        return False


# Lower DPI for OCR model requests to avoid 400 (huge payload). Tesseract can use higher DPI.
OCR_MODEL_IMAGE_DPI = 120


def _pdf_page_to_png_bytes(
    pdf_path: str | Path, page_idx: int, dpi: int = 200, max_long_side: int | None = None
) -> bytes:
    """Render one PDF page to PNG bytes. If max_long_side is set, scale so longest side <= that (avoids 400 from oversized payload)."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        if max_long_side and max(pix.width, pix.height) > max_long_side:
            scale = max_long_side / max(pix.width, pix.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale * dpi / 72, scale * dpi / 72))
        return pix.tobytes("png")
    finally:
        doc.close()


def extract_text_with_tesseract(
    pdf_path: str | Path,
    callback: Any | None = None,
) -> str:
    """
    OCR every page with Tesseract. Fast, no GPU. Good for plain scanned text.
    Returns Markdown with <!-- PAGE N --> markers for downstream PageIndex.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return ""
    if not is_tesseract_available():
        logger.warning("Tesseract not available; pip install pytesseract and install tesseract-ocr")
        return ""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError as e:
        logger.warning("Tesseract dependencies missing: %s", e)
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.warning("Cannot open PDF for Tesseract: %s", e)
        return ""

    parts: list[str] = []
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if callback:
            callback(f"Tesseract OCR page {page_num}/{total_pages}…")
        try:
            png_bytes = _pdf_page_to_png_bytes(pdf_path, page_idx)
            img = Image.open(io.BytesIO(png_bytes))
            text = (pytesseract.image_to_string(img) or "").strip()
            if text:
                parts.append(f"<!-- PAGE {page_num} -->\n{text}")
        except Exception as e:
            logger.warning("Tesseract failed page %d: %s", page_num, e)
    if not parts:
        return ""
    out = "\n\n".join(parts)
    if callback:
        callback(f"Tesseract OCR complete: {len(out)} chars from {len(parts)} page(s)")
    return out


def extract_text_with_ocr_model(
    pdf_path: str | Path,
    callback: Any | None = None,
) -> str:
    """
    OCR every page using Ollama OCR_MODEL (e.g. PaddleOCR-VL). Better for columns/tables.
    Returns Markdown with <!-- PAGE N --> markers.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.warning("Cannot open PDF for OCR model: %s", e)
        return ""

    model = (getattr(config, "OCR_MODEL", None) or "").strip()
    if not model:
        logger.warning("OCR_MODEL not set; cannot run OCR model extraction")
        return ""

    parts: list[str] = []
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if callback:
            callback(f"OCR model page {page_num}/{total_pages}…")
        try:
            png_bytes = _pdf_page_to_png_bytes(
                pdf_path, page_idx, dpi=OCR_MODEL_IMAGE_DPI, max_long_side=1024
            )
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Extract all text from this document page. Output only the extracted text in Markdown.",
                        "images": [b64],
                    },
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            }
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            text = (resp.json().get("message", {}).get("content", "") or "").strip()
            if text:
                parts.append(f"<!-- PAGE {page_num} -->\n{text}")
        except requests.HTTPError as e:
            body = ""
            if e.response is not None:
                try:
                    body = (e.response.text or "").strip()
                except Exception:
                    pass
            logger.warning("OCR model failed page %d: %s", page_num, e)
            if body:
                logger.warning(
                    "Ollama response body (true error): %s", body[:2000]
                )
            else:
                logger.warning(
                    "Ollama returned %d with no body; try smaller images or check Ollama logs",
                    e.response.status_code if e.response is not None else 0,
                )
        except Exception as e:
            logger.warning("OCR model failed page %d: %s", page_num, e)
    if not parts:
        return ""
    out = "\n\n".join(parts)
    if callback:
        callback(f"OCR model complete: {len(out)} chars from {len(parts)} page(s)")
    return out


def extract_text_with_fast_ocr(
    pdf_path: str | Path,
    prefer_tesseract: bool = True,
    callback: Any | None = None,
) -> str:
    """
    Run fast OCR for scanned text: try Tesseract first (fastest), else OCR_MODEL.
    Use for triage route ocr_scanned. Do not use the heavy VLM here.
    """
    if prefer_tesseract and is_tesseract_available():
        out = extract_text_with_tesseract(pdf_path, callback=callback)
        if out and len(out.strip()) > 100:
            return out
        logger.info("Tesseract produced little text; trying OCR_MODEL")
    out = extract_text_with_ocr_model(pdf_path, callback=callback)
    if out:
        return out
    if not out and prefer_tesseract and is_tesseract_available():
        return extract_text_with_tesseract(pdf_path, callback=callback)
    return ""
