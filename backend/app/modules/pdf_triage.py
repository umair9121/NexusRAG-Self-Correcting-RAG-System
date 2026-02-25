"""
PDF triage: page-by-page classification and document-level routing.

Uses PyMuPDF to measure text density and image count per page (no AI).
Routes documents to the cheapest/fastest extraction path:
- fast_pageindex: selectable text only, no OCR/VLM
- ocr_scanned: unselectable text → fast OCR (Tesseract / OCR_MODEL)
- mixed_layout: textbooks (text + figures) → layout parser + VLM only for figures
- complex_visual: drawings/blueprints → full-page VLM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from app import config

logger = logging.getLogger(__name__)

# Per-page classification
PAGE_TEXT_HEAVY = "text_heavy"  # High char count, few/no images → fast PageIndex
PAGE_IMAGE_HEAVY = "image_heavy"  # Low/zero chars, images → OCR or full-page VLM
PAGE_MIXED = "mixed"  # High chars AND images → layout + VLM for figures only

# Document-level routes (for ingestion)
ROUTE_FAST_PAGEINDEX = "fast_pageindex"
ROUTE_OCR_SCANNED = "ocr_scanned"
ROUTE_MIXED_LAYOUT = "mixed_layout"
ROUTE_COMPLEX_VISUAL = "complex_visual"


@dataclass
class PageStats:
    page_idx: int  # 0-based
    char_count: int
    image_count: int
    classification: str = ""


def get_page_stats(pdf_path: str | Path) -> list[PageStats]:
    """
    For each page: extract text-layer char count and embedded image count.
    No OCR, no AI — pure PyMuPDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return []
    result: list[PageStats] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = (page.get_text() or "").strip()
            char_count = len(text)
            image_list = page.get_images()
            image_count = len(image_list) if image_list else 0
            result.append(PageStats(page_idx=page_idx, char_count=char_count, image_count=image_count))
        doc.close()
    except Exception as e:
        logger.warning("pdf_triage get_page_stats failed: %s", e)
    return result


def _classify_page(
    s: PageStats,
    text_heavy_threshold: int,
    image_heavy_char_max: int,
) -> str:
    """
    Classify a single page:
    - text_heavy: enough selectable text, few images (fast path).
    - mixed: selectable text AND images (textbook) → layout + VLM for figures.
    - image_heavy: little/no text, has images or is blank (OCR or full-page VLM).
    """
    if s.char_count >= text_heavy_threshold and s.image_count <= 1:
        return PAGE_TEXT_HEAVY
    if s.char_count >= text_heavy_threshold and s.image_count > 1:
        return PAGE_MIXED
    if s.char_count >= text_heavy_threshold and s.image_count == 1:
        return PAGE_MIXED  # one figure on an otherwise text page
    if s.char_count < image_heavy_char_max:
        if s.image_count >= 1:
            return PAGE_IMAGE_HEAVY  # scan or diagram
        return PAGE_IMAGE_HEAVY  # likely scanned text (no text layer)
    return PAGE_TEXT_HEAVY


def classify_pages(
    stats: list[PageStats],
    text_heavy_threshold: int | None = None,
    image_heavy_char_max: int | None = None,
) -> list[PageStats]:
    """Fill classification for each PageStats."""
    th = text_heavy_threshold if text_heavy_threshold is not None else config.PDF_TRIAGE_TEXT_HEAVY_CHARS
    im = image_heavy_char_max if image_heavy_char_max is not None else config.PDF_TRIAGE_IMAGE_HEAVY_CHAR_MAX
    for s in stats:
        s.classification = _classify_page(s, th, im)
    return stats


@dataclass
class TriageResult:
    route: str
    page_stats: list[PageStats]
    summary: dict[str, int] = field(default_factory=dict)
    reason: str = ""


def get_document_route(
    pdf_path: str | Path,
    text_heavy_threshold: int | None = None,
    image_heavy_char_max: int | None = None,
) -> TriageResult:
    """
    Determine document-level route from per-page stats.

    - fast_pageindex: (almost) all pages text_heavy → selectable text only.
    - ocr_scanned: most pages image_heavy with 0 images → unselectable text, use fast OCR.
    - complex_visual: most pages image_heavy with images → full-page VLM.
    - mixed_layout: mix of text_heavy and mixed, or any mixed → layout + VLM for figures only.
    """
    stats = get_page_stats(pdf_path)
    if not stats:
        return TriageResult(route=ROUTE_FAST_PAGEINDEX, page_stats=[], reason="no pages or not PDF")

    classify_pages(stats, text_heavy_threshold, image_heavy_char_max)
    summary: dict[str, int] = {}
    for s in stats:
        summary[s.classification] = summary.get(s.classification, 0) + 1

    n = len(stats)
    n_text = summary.get(PAGE_TEXT_HEAVY, 0)
    n_image = summary.get(PAGE_IMAGE_HEAVY, 0)
    n_mixed = summary.get(PAGE_MIXED, 0)

    # All or almost all text_heavy → fast PageIndex (no OCR/VLM)
    if n_text >= n * 0.9 and n_mixed == 0:
        return TriageResult(
            route=ROUTE_FAST_PAGEINDEX,
            page_stats=stats,
            summary=summary,
            reason=f"{n_text}/{n} pages text_heavy (selectable); fast PageIndex only",
        )

    # Many mixed pages (text + figures) → mixed layout pipeline
    if n_mixed >= 1 or (n_text >= 1 and n_image >= 1):
        return TriageResult(
            route=ROUTE_MIXED_LAYOUT,
            page_stats=stats,
            summary=summary,
            reason=f"textbook-style: text_heavy={n_text} mixed={n_mixed} image_heavy={n_image}; layout + VLM for figures only",
        )

    # Most pages image_heavy: distinguish scanned text (0–1 image per page) vs complex visual (drawings/blueprints)
    # Single-page image_heavy (e.g. 1-page engineering drawing) → complex_visual so it gets VLM per-page summaries.
    im_max = image_heavy_char_max if image_heavy_char_max is not None else config.PDF_TRIAGE_IMAGE_HEAVY_CHAR_MAX
    pages_no_text = sum(1 for s in stats if s.char_count < im_max)
    pages_with_many_images = sum(1 for s in stats if s.image_count > 1)
    if n_image >= n * 0.5:
        if n == 1 and n_image == 1:
            return TriageResult(
                route=ROUTE_COMPLEX_VISUAL,
                page_stats=stats,
                summary=summary,
                reason="single-page image_heavy (e.g. engineering drawing); complex_visual for VLM summaries",
            )
        if pages_with_many_images >= max(1, n * 0.3):
            return TriageResult(
                route=ROUTE_COMPLEX_VISUAL,
                page_stats=stats,
                summary=summary,
                reason=f"complex visual: {n_image} image_heavy pages, many with multiple figures; full-page VLM",
            )
        return TriageResult(
            route=ROUTE_OCR_SCANNED,
            page_stats=stats,
            summary=summary,
            reason=f"scanned text: {n_image} image_heavy pages (0–1 image/page); use fast OCR (Tesseract/OCR_MODEL)",
        )

    # Default: mixed or uncertain → mixed_layout to preserve figure context
    return TriageResult(
        route=ROUTE_MIXED_LAYOUT,
        page_stats=stats,
        summary=summary,
        reason=f"mixed/uncertain summary={summary}; use layout + VLM for figures",
    )
