"""
Structured OCR pipeline for scanned/image-based PDFs.
Uses Docling with OCR enabled (RapidOCR), layout and table structure, so that
scanned PDFs are converted to Markdown (H1, H2, ...) before PageIndex.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DOCLING_AVAILABLE: bool | None = None
_OCR_CONVERTER_CACHE: Any = None


def is_docling_available() -> bool:
    global _DOCLING_AVAILABLE
    if _DOCLING_AVAILABLE is not None:
        return _DOCLING_AVAILABLE
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
        _DOCLING_AVAILABLE = True
        return True
    except ImportError:
        _DOCLING_AVAILABLE = False
        return False


def _get_ocr_converter():
    """
    Build DocumentConverter with OCR and table structure enabled so scanned PDFs
    are actually processed (layout + RapidOCR). Uses local CPU/Intel Arc when available.
    """
    global _OCR_CONVERTER_CACHE
    if _OCR_CONVERTER_CACHE is False:
        return None
    if _OCR_CONVERTER_CACHE is not None:
        return _OCR_CONVERTER_CACHE

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
            AcceleratorOptions,
        )
    except ImportError as e:
        logger.debug("Docling pipeline options not available: %s", e)
        _OCR_CONVERTER_CACHE = False
        return None

    try:
        # OCR and table structure for scanned/image PDFs
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
        )
        pipeline_options.ocr_options = RapidOcrOptions()

        # Optional: table cell matching, accelerator (Intel Arc / CPU)
        try:
            from docling.datamodel.pipeline_options import (
                TableStructureOptions,
                AcceleratorDevice,
            )
            if hasattr(TableStructureOptions, "do_cell_matching"):
                pipeline_options.table_structure_options = TableStructureOptions(
                    do_cell_matching=True
                )
            accelerator_options = AcceleratorOptions(
                num_threads=8,
                device=AcceleratorDevice.AUTO,
            )
        except (ImportError, AttributeError):
            accelerator_options = AcceleratorOptions(num_threads=8, device="auto")

        pipeline_options.accelerator_options = accelerator_options

        # Format option for PDF
        try:
            from docling.datamodel.base_models import InputFormat
            format_key = InputFormat.PDF
        except ImportError:
            format_key = "pdf"

        converter = DocumentConverter(
            format_options={
                format_key: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        _OCR_CONVERTER_CACHE = converter
        return converter
    except Exception as e:
        logger.warning("Could not build OCR-enabled Docling converter: %s", e)
        _OCR_CONVERTER_CACHE = False
        return None


def convert_pdf_to_markdown(pdf_path: str | Path, callback: Any | None = None) -> str:
    """
    Convert a PDF to structured Markdown using Docling with OCR enabled.
    Scanned/image PDFs are run through layout detection + RapidOCR so text is
    actually extracted. Returns empty string if Docling is not installed or conversion fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return ""

    if not is_docling_available():
        logger.warning("Docling not installed; pip install docling")
        return ""

    try:
        from docling.document_converter import DocumentConverter

        if callback:
            callback("Converting PDF to Markdown with Docling (OCR + layout + tables)...")

        converter = _get_ocr_converter()
        if converter is None:
            # Fallback to default converter (may not run OCR on scanned PDFs)
            logger.warning("Using default Docling converter (OCR options not applied)")
            converter = DocumentConverter()

        result = converter.convert(str(pdf_path))
        md = result.document.export_to_markdown()
        if md and isinstance(md, str):
            md = md.strip()
        else:
            md = ""
        if callback and md:
            callback(f"Docling produced {len(md)} characters of Markdown")
        return md
    except Exception as e:
        logger.warning("Docling conversion failed for %s: %s", pdf_path.name, e)
        if callback:
            callback(f"Docling failed: {e}")
        return ""


def validate_and_fix_headers(md_content: str) -> str:
    """
    Ensure proper H1/H2 hierarchy for PageIndex.
    If no H1 is found, promote the first H2 (or ###) to H1 so the tree has an anchor.
    """
    if not md_content or not md_content.strip():
        return md_content

    lines = md_content.splitlines()
    has_h1 = any(re.match(r"^#\s+", line) for line in lines)

    if has_h1:
        return md_content

    logger.warning("No H1 header found in Markdown. Promoting first heading to H1.")
    new_lines: list[str] = []
    promoted = False

    for line in lines:
        if not promoted:
            if re.match(r"^##\s+", line):
                new_lines.append(re.sub(r"^##\s+", "# ", line, count=1))
                promoted = True
                continue
            if re.match(r"^###\s+", line):
                new_lines.append(re.sub(r"^###\s+", "# ", line, count=1))
                promoted = True
                continue
        new_lines.append(line)

    result = "\n".join(new_lines)
    if promoted:
        logger.info("Header hierarchy repaired (first H2/H3 promoted to H1).")
    return result
