"""
Extract plain text from supported document types for ingestion and CAG.
Dispatches by file extension; used for non-PDF and for a single entry point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app import config
from app.modules.contextual import extract_text_from_pdf

logger = logging.getLogger(__name__)


def _is_allowed(path: str | Path) -> bool:
    return Path(path).suffix.lower() in config.ALLOWED_EXTENSIONS


def extract_text_from_docx(path: str | Path) -> str:
    """Extract text from .docx using python-docx."""
    try:
        import docx  # python-docx
    except ImportError:
        logger.warning("python-docx not installed; pip install python-docx")
        return ""
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".docx":
        return ""
    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning("docx extraction failed for %s: %s", path.name, e)
        return ""


def extract_text_from_doc(path: str | Path) -> str:
    """Extract text from legacy .doc using mammoth."""
    try:
        import mammoth
    except ImportError:
        logger.warning("mammoth not installed; pip install mammoth")
        return ""
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".doc":
        return ""
    try:
        with open(path, "rb") as f:
            result = mammoth.extract_raw_text(f)
        return result.value or ""
    except Exception as e:
        logger.warning("mammoth (.doc) extraction failed for %s: %s", path.name, e)
        return ""


def extract_text_from_excel(path: str | Path) -> str:
    """Extract text from .xlsx, .xlsm (and .xls if xlrd available)."""
    path = Path(path)
    if not path.exists():
        return ""
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            logger.warning("openpyxl not installed; pip install openpyxl")
            return ""
        try:
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    line = "\t".join(str(c) if c is not None else "" for c in row)
                    if line.strip():
                        parts.append(line)
            wb.close()
            return "\n".join(parts)
        except Exception as e:
            logger.warning("openpyxl extraction failed for %s: %s", path.name, e)
            return ""
    if suf == ".xls":
        try:
            import xlrd
        except ImportError:
            logger.warning("xlrd not installed for .xls; pip install xlrd")
            return ""
        try:
            wb = xlrd.open_workbook(str(path))
            parts = []
            for sheet in wb.sheets():
                parts.append(f"Sheet: {sheet.name}")
                for row_idx in range(sheet.nrows):
                    row = sheet.row_values(row_idx)
                    line = "\t".join(str(c) if c else "" for c in row)
                    if line.strip():
                        parts.append(line)
            return "\n".join(parts)
        except Exception as e:
            logger.warning("xlrd (.xls) extraction failed for %s: %s", path.name, e)
            return ""
    return ""


def extract_text_from_json(path: str | Path) -> str:
    """Read JSON and produce a string (indented dump or extract text-like values)."""
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".json":
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("JSON read failed for %s: %s", path.name, e)
        return ""


def extract_text_from_html(path: str | Path) -> str:
    """Extract text from HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed; pip install beautifulsoup4")
        return ""
    path = Path(path)
    if not path.exists():
        return ""
    suf = path.suffix.lower()
    if suf not in (".html", ".htm"):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.warning("HTML extraction failed for %s: %s", path.name, e)
        return ""


def extract_text_from_image(path: str | Path, callback: Any | None = None) -> str:
    """Extract text from an image using VLM (same as PDF page extraction)."""
    from app.modules.vlm_extract import extract_text_with_vlm_image
    return extract_text_with_vlm_image(path, callback=callback)


def extract_text_from_file(
    file_path: str | Path,
    callback: Any | None = None,
) -> str:
    """
    Extract plain text from a file based on its extension.
    Used for ingestion (contextual, graph, minimal PageIndex) and CAG loading.
    Returns empty string if type is unsupported or extraction fails.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return ""
    suf = path.suffix.lower()
    if suf not in config.ALLOWED_EXTENSIONS:
        return ""

    if suf == ".pdf":
        if callback:
            callback(f"Extracting text from PDF {path.name}...")
        return extract_text_from_pdf(path)
    if suf == ".docx":
        if callback:
            callback(f"Extracting text from Word document {path.name}...")
        return extract_text_from_docx(path)
    if suf == ".doc":
        if callback:
            callback(f"Extracting text from Word document {path.name}...")
        return extract_text_from_doc(path)
    if suf in (".xlsx", ".xlsm", ".xls"):
        if callback:
            callback(f"Extracting text from spreadsheet {path.name}...")
        return extract_text_from_excel(path)
    if suf == ".json":
        if callback:
            callback(f"Reading JSON {path.name}...")
        return extract_text_from_json(path)
    if suf in (".html", ".htm"):
        if callback:
            callback(f"Extracting text from HTML {path.name}...")
        return extract_text_from_html(path)
    if suf in (".png", ".jpg", ".jpeg"):
        if callback:
            callback(f"Extracting text from image {path.name} (VLM)...")
        return extract_text_from_image(path, callback=callback)

    return ""
