"""
Fallback for PDFs that yield no text from PyMuPDF (e.g. image-based/scanned).
Uses deepdoctection dataflow API: layout detection + OCR, optional figure extraction.
Install with: pip install deepdoctection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app import config

logger = logging.getLogger(__name__)

_DD_AVAILABLE: bool | None = None


def _deepdoctection_available() -> bool:
    global _DD_AVAILABLE
    if _DD_AVAILABLE is not None:
        return _DD_AVAILABLE
    try:
        import deepdoctection as dd  # noqa: F401
        _DD_AVAILABLE = True
        return True
    except ImportError:
        _DD_AVAILABLE = False
        return False


def _text_from_dp(dp: Any) -> str:
    """Extract text from a single document/page (dp) from the analyzer."""
    if hasattr(dp, "export_to_text"):
        try:
            out = dp.export_to_text()
            if out and isinstance(out, str):
                return out.strip()
        except Exception:
            pass
    if hasattr(dp, "text"):
        t = getattr(dp, "text", None)
        if t is not None:
            return str(t).strip()
    # If dp has pages (full document), collect text from each page
    pages = getattr(dp, "pages", None)
    if pages:
        page_texts = [_text_from_dp(p) for p in (pages if isinstance(pages, (list, tuple)) else [pages])]
        return "\n\n".join(p for p in page_texts if p).strip()
    # Build text from layout/words if available
    text_parts: list[str] = []
    for attr in ("words", "text_container", "get_text"):
        if not hasattr(dp, attr):
            continue
        obj = getattr(dp, attr)
        if callable(obj):
            try:
                obj = obj()
            except Exception:
                continue
        if obj is None:
            continue
        if isinstance(obj, str) and obj.strip():
            text_parts.append(obj.strip())
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if hasattr(item, "get_text"):
                    try:
                        text_parts.append(str(item.get_text()).strip())
                    except Exception:
                        pass
                elif isinstance(item, str) and item.strip():
                    text_parts.append(item.strip())
    if text_parts:
        return "\n".join(text_parts)
    return ""


def _extract_figures_from_dp(dp: Any, doc_stem: str, page_index: int, callback: Any | None) -> list[str]:
    """Extract figure regions from dp and save as PNGs. Returns list of saved file paths."""
    import deepdoctection as dd

    saved: list[str] = []
    try:
        layout_type = getattr(dd, "LayoutType", None)
        if layout_type is None:
            layout_type = getattr(dd, "layout_type", None)
        figure_cat = getattr(layout_type, "figure", None) if layout_type else None
        if figure_cat is None:
            return saved

        get_ann = getattr(dp, "get_annotation", None)
        if not get_ann:
            return saved

        figures = get_ann(category_names=figure_cat)
        if not figures:
            return saved
        figures = figures if isinstance(figures, (list, tuple)) else [figures]

        out_dir = config.FIGURES_DIR / doc_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        viz_handler = getattr(dd, "viz_handler", None)
        write_image = getattr(viz_handler, "write_image", None) if viz_handler else None

        for fig in figures:
            try:
                ann_id = getattr(fig, "annotation_id", None) or f"page{page_index}_{len(saved)}"
                viz_fn = getattr(fig, "viz", None)
                if not callable(viz_fn):
                    continue
                # Prefer non-interactive so we don't open windows during batch ingestion
                try:
                    np_array = viz_fn(interactive=False)
                except TypeError:
                    np_array = viz_fn()
                if np_array is None or write_image is None:
                    continue
                ext = ".png"
                path = out_dir / f"{ann_id}{ext}"
                write_image(str(path), np_array)
                saved.append(str(path))
            except Exception as e:
                logger.debug("Skipping figure save: %s", e)
        if callback and saved:
            callback(f"Saved {len(saved)} figure(s) for page {page_index + 1}")
    except Exception as e:
        logger.warning("Figure extraction failed: %s", e)
    return saved


def extract_text_with_deepdoctection(
    pdf_path: str | Path,
    callback: Any | None = None,
    doc_stem: str | None = None,
    extract_figures: bool = True,
) -> str:
    """
    Extract text from a PDF using deepdoctection (dataflow API: layout + OCR).
    Use when PyMuPDF returns no text (e.g. scanned/image PDFs).
    If extract_figures is True, figure regions are saved under storage/figures/<doc_stem>/.
    Returns empty string if deepdoctection is not installed or extraction fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return ""

    if not _deepdoctection_available():
        logger.warning(
            "deepdoctection not installed; cannot extract text for empty PDF. pip install deepdoctection"
        )
        return ""

    # Safe dir name (no path separators)
    doc_stem = (doc_stem or pdf_path.stem).replace("\\", "_").replace("/", "_").strip() or "document"

    try:
        import deepdoctection as dd

        if callback:
            callback("Running deepdoctection (layout + OCR) for empty document...")

        analyzer = dd.get_dd_analyzer()
        df = analyzer.analyze(path=str(pdf_path))

        if hasattr(df, "reset_state"):
            df.reset_state()

        parts: list[str] = []
        page_index = 0
        for dp in df:
            if dp is None:
                page_index += 1
                continue
            text = _text_from_dp(dp)
            if text:
                parts.append(text)
            if extract_figures:
                _extract_figures_from_dp(dp, doc_stem, page_index, callback)
            page_index += 1

        text = "\n\n".join(p for p in parts if p).strip()
        if callback and text:
            callback(f"deepdoctection extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.warning("deepdoctection extraction failed for %s: %s", pdf_path.name, e)
        if callback:
            callback(f"deepdoctection failed: {e}")
        return ""


def is_pageindex_empty(pageindex_json_path: str | Path) -> bool:
    """Return True if the PageIndex JSON has root title 'Empty Document'."""
    import json

    path = Path(pageindex_json_path)
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("title") or "").strip() == "Empty Document"
    except (json.JSONDecodeError, TypeError):
        return False
