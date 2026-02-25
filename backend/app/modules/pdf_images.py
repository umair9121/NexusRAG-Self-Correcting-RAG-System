"""
PDF page image extraction for reasoning-based retrieval and VLM visual answer generation.

Extracts one image per PDF page (e.g. for PageIndex node -> page range -> images)
and provides helpers to resolve stored images by doc stem and page number.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app import config

logger = logging.getLogger(__name__)


def extract_pdf_page_images(
    pdf_path: str | Path,
    output_dir: str | Path = "pdf_images",
) -> tuple[dict[int, str], int]:
    """
    Render each PDF page to a JPEG and save under output_dir.
    Returns (page_images, total_pages) where page_images maps 1-based page number to image path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        logger.info("pdf_images: extract skipped — path missing or not PDF: %s", pdf_path)
        return {}, 0
    logger.info("pdf_images: extracting page images from %s -> %s", pdf_path.name, output_dir.resolve())
    page_images: dict[int, str] = {}
    try:
        pdf_document = fitz.open(str(pdf_path))
        total_pages = len(pdf_document)
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("jpeg")
            image_path = output_dir / f"page_{page_number + 1}.jpg"
            image_path.write_bytes(img_data)
            page_images[page_number + 1] = str(image_path)
            logger.debug("pdf_images: saved page %s -> %s", page_number + 1, image_path)
        pdf_document.close()
        logger.info("pdf_images: extracted %d/%d pages to %s", len(page_images), total_pages, output_dir.resolve())
        return page_images, total_pages
    except Exception as e:
        logger.warning("pdf_images: extract_pdf_page_images failed: %s", e)
        return {}, 0


def get_stored_page_images(doc_stem: str) -> dict[int, str] | None:
    """
    Load stored page images for a document (from PDF_IMAGES_DIR / doc_stem).
    Returns dict mapping 1-based page number to image path, or None if dir missing/empty.
    """
    base = config.PDF_IMAGES_DIR / doc_stem
    if not base.is_dir():
        logger.debug("pdf_images: no stored images for doc_stem=%s (dir missing: %s)", doc_stem, base)
        return None
    page_images: dict[int, str] = {}
    for p in sorted(base.glob("page_*.jpg")):
        try:
            num = int(p.stem.split("_")[-1])
            page_images[num] = str(p)
        except (ValueError, IndexError):
            continue
    if page_images:
        logger.info("pdf_images: loaded %d stored images for doc_stem=%s from %s", len(page_images), doc_stem, base.resolve())
    else:
        logger.debug("pdf_images: no page_*.jpg in %s for doc_stem=%s", base, doc_stem)
    return page_images if page_images else None


def get_page_images_for_nodes(
    node_list: list[str],
    node_map: dict[str, dict[str, Any]],
    page_images: dict[int, str],
) -> list[str]:
    """
    Return list of image paths for the page range of the given nodes.
    node_map entries must have 'start_index' and 'end_index' (1-based).
    Deduplicates by page and preserves order.
    """
    image_paths: list[str] = []
    seen_pages: set[int] = set()
    for node_id in node_list:
        node_info = node_map.get(node_id)
        if not node_info:
            logger.debug("pdf_images: node_id=%s not in node_map, skipping", node_id)
            continue
        start = node_info.get("start_index")
        end = node_info.get("end_index")
        if start is None or end is None:
            continue
        for page_num in range(start, end + 1):
            if page_num not in seen_pages and page_num in page_images:
                image_paths.append(page_images[page_num])
                seen_pages.add(page_num)
    logger.info("pdf_images: get_page_images_for_nodes node_list=%s -> %d image path(s)", node_list, len(image_paths))
    return image_paths
