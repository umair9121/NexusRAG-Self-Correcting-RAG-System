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
import json
import logging
import re
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


def get_pdf_text_layer_stats(pdf_path: str | Path) -> tuple[int, int]:
    """
    Get (total_pages, total_chars) from the PDF's text layer only (no OCR).
    Used to detect image-heavy PDFs where most content is unselectable.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return 0, 0
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        total_chars = 0
        for i in range(total_pages):
            total_chars += len((doc[i].get_text() or "").strip())
        doc.close()
        return total_pages, total_chars
    except Exception as exc:
        logger.warning("get_pdf_text_layer_stats failed: %s", exc)
        return 0, 0


def should_use_vlm_for_sparse_pages(pdf_path: str | Path) -> bool:
    """
    For PDFs that have a text layer but very little text per page (e.g. image-heavy,
    unselectable text, forms), run full-document VLM extraction so we persist
    a single transcript for PageIndex, vector, and graph instead of losing context.

    Returns True when total_pages >= 2 and (total_chars / total_pages) < VLM_MIN_CHARS_PER_PAGE.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return False
    total_pages, total_chars = get_pdf_text_layer_stats(pdf_path)
    if total_pages < 2:
        return False
    chars_per_page = total_chars / total_pages
    use = chars_per_page < config.VLM_MIN_CHARS_PER_PAGE
    logger.info(
        "VLM sparse-pages check: pages=%d total_chars=%d chars/page=%.0f threshold=%d -> %s",
        total_pages,
        total_chars,
        chars_per_page,
        config.VLM_MIN_CHARS_PER_PAGE,
        "use VLM extraction" if use else "skip",
    )
    return use


# ── Page rendering ──────────────────────────────────────────────────────────

def _pdf_page_to_base64(
    pdf_path: str | Path, page_idx: int, dpi: int = 200, max_long_side: int | None = None
) -> str:
    """Render *one* PDF page to a base64-encoded PNG string. If max_long_side set, scale down to avoid huge payloads."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        if max_long_side and max(pix.width, pix.height) > max_long_side:
            scale = max_long_side / max(pix.width, pix.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale * dpi / 72, scale * dpi / 72))
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


def generate_answer_with_images(
    prompt: str,
    image_paths: list[str] | list[Path],
    callback: Any | None = None,
) -> str:
    """
    Generate an answer using the VLM with the given prompt and document page images.
    Used for visual reasoning after PageIndex reasoning-based retrieval.
    """
    if not image_paths:
        logger.info("vlm.generate_answer_with_images: no image_paths provided, skipping")
        return ""
    path_preview = [str(Path(p).resolve()) for p in image_paths[:3]]
    if len(image_paths) > 3:
        path_preview.append(f"...+{len(image_paths) - 3} more")
    logger.info("vlm.generate_answer_with_images: prompt_len=%d image_count=%d paths=%s", len(prompt), len(image_paths), path_preview)
    images_b64: list[str] = []
    for p in image_paths:
        p = Path(p)
        if not p.exists():
            logger.warning("vlm.generate_answer_with_images: image missing: %s", p)
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        try:
            b64 = _image_file_to_base64(p)
            if b64:
                images_b64.append(b64)
        except Exception as exc:
            logger.warning("vlm.generate_answer_with_images: cannot read image %s: %s", p, exc)
    if not images_b64:
        logger.warning("vlm.generate_answer_with_images: no valid images loaded from %d path(s)", len(image_paths))
        return ""
    logger.info("vlm.generate_answer_with_images: calling VLM model=%s with %d image(s)", config.VLM_MODEL, len(images_b64))
    if callback:
        callback(f"Generating answer with VLM and {len(images_b64)} page image(s)…")
    payload: dict[str, Any] = {
        "model": config.VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            },
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": config.DEFAULT_NUM_CTX},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "").strip()
        logger.info("vlm.generate_answer_with_images: success answer_len=%d", len(answer))
        if callback and answer:
            callback("VLM answer generated")
        return answer
    except Exception as exc:
        logger.warning("vlm.generate_answer_with_images: VLM call failed: %s", exc)
        if callback:
            callback(f"VLM failed: {exc}")
        return ""


_FIGURE_CAPTION_PROMPT = (
    "Describe this figure, diagram, or chart in 1–3 concise sentences. "
    "Include what it shows, key labels or data if visible, and how it relates to a document (e.g. schematic, flowchart, table). "
    "Output only the description, no preamble."
)


def caption_image_with_vlm(
    image_path: str | Path,
    callback: Any | None = None,
) -> str:
    """
    Generate a short caption for a single image (e.g. cropped figure from a textbook)
    using the VLM. Used in the mixed-layout pipeline so figures get context without
    running full-page VLM on text-heavy pages.
    """
    path = Path(image_path)
    if not path.exists() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return ""
    try:
        img_b64 = _image_file_to_base64(path)
    except Exception as exc:
        logger.warning("caption_image_with_vlm: cannot read %s: %s", path, exc)
        return ""
    if not img_b64:
        return ""
    payload: dict[str, Any] = {
        "model": config.VLM_MODEL,
        "messages": [
            {"role": "user", "content": _FIGURE_CAPTION_PROMPT, "images": [img_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        caption = resp.json().get("message", {}).get("content", "").strip()
        if callback and caption:
            callback("VLM figure caption generated")
        return caption
    except Exception as exc:
        logger.warning("caption_image_with_vlm failed for %s: %s", path.name, exc)
        if callback:
            callback(f"VLM caption failed: {exc}")
        return ""


_VLM_PAGE_SUMMARY_PROMPT = (
    "Describe this document page in detail. Focus on: diagrams, figures, labels, "
    "spatial layout, and any key visual information that would help a reader understand the content. "
    "Output a concise paragraph (2–5 sentences). No preamble."
)


def get_per_page_visual_summaries(
    pdf_path: str | Path,
    callback: Any | None = None,
    dpi: int = 150,
    max_long_side: int = 1024,
) -> list[tuple[int, str]]:
    """
    For each page of the PDF, render to image and ask the main VLM (VLM_MODEL) for a
    short visual summary. Used for complex_visual documents so embeddings get extra
    context that OCR may have missed (e.g. diagrams, layout).
    Returns [(page_num, summary), ...] (1-based page numbers).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return []
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    except Exception as exc:
        logger.warning("get_per_page_visual_summaries: cannot open PDF %s: %s", pdf_path.name, exc)
        return []

    results: list[tuple[int, str]] = []
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if callback:
            callback(f"VLM visual summary page {page_num}/{total_pages}…")
        try:
            img_b64 = _pdf_page_to_base64(
                pdf_path, page_idx, dpi=dpi, max_long_side=max_long_side
            )
            if not img_b64:
                continue
            payload: dict[str, Any] = {
                "model": config.VLM_MODEL,
                "messages": [
                    {"role": "user", "content": _VLM_PAGE_SUMMARY_PROMPT, "images": [img_b64]},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 4096},
            }
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            summary = resp.json().get("message", {}).get("content", "").strip()
            if summary:
                results.append((page_num, summary))
        except Exception as exc:
            logger.warning("get_per_page_visual_summaries page %d failed: %s", page_num, exc)
            if callback:
                callback(f"VLM summary page {page_num} failed: {exc}")

    if callback and results:
        callback(f"VLM visual summaries: {len(results)} page(s)")
    return results


# ── Screen Object Model (SOM) for UI screenshots (Enrico, MUD, MobileViews) ───

_SOM_SYSTEM_PROMPT = (
    "You are a UI analyst. Your task is to produce a Screen Object Model (SOM): "
    "a hierarchical, machine-readable description of the interface. "
    "For each visible component, output: type (button, text, image, input, container, etc.), "
    "text content if any, approximate position (e.g. top/center/bottom, left/right), "
    "parent-child relationships, and any accessibility-related attributes (e.g. label presence, contrast). "
    "Output ONLY valid JSON. No commentary, no markdown fences."
)

_SOM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string", "description": "e.g. login, settings, list"},
        "root": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {"$ref": "#/def/component"},
                },
            },
        },
        "components_flat": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "text": {"type": "string"},
                    "bounds": {"type": "object", "properties": {"left": {"type": "number"}, "top": {"type": "number"}, "width": {"type": "number"}, "height": {"type": "number"}}},
                    "accessibility_label": {"type": "string"},
                    "parent_id": {"type": "string"},
                },
            },
        },
    },
    "required": ["screen_type", "components_flat"],
}


def extract_screen_object_model(
    image_path: str | Path,
    view_hierarchy_json: dict | None = None,
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Generate a Screen Object Model (SOM) from a UI screenshot using the VLM.
    Hierarchical description of UI components, spatial relationships, and text contents.
    If view_hierarchy_json is provided (e.g. from Enrico or MobileViews metadata), it can be
    included in the prompt to improve structure. Returns a dict with screen_type and components_flat at minimum.
    """
    path = Path(image_path)
    if not path.exists() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return {}
    try:
        img_b64 = _image_file_to_base64(path)
    except Exception as exc:
        logger.warning("extract_screen_object_model: cannot read image %s: %s", path, exc)
        return {}
    if not img_b64:
        return {}

    if callback:
        callback("Extracting Screen Object Model (SOM) from screenshot…")

    user_content = (
        "Analyze this UI screenshot and produce a Screen Object Model (SOM) as JSON. "
        "Include: (1) screen_type (e.g. login, list, settings), (2) components_flat: array of every visible component with "
        "id, type (button, text, image, input, etc.), text (visible label/caption), bounds (left, top, width, height in normalized 0–1 or pixels), "
        "accessibility_label (if inferable), parent_id. Describe spatial relationships (above, below, inside). "
        "Output ONLY valid JSON with keys screen_type and components_flat."
    )
    if view_hierarchy_json:
        user_content += "\n\nOptional view hierarchy metadata (use to refine structure):\n" + json.dumps(view_hierarchy_json)[:2000]

    payload: dict[str, Any] = {
        "model": config.VLM_MODEL,
        "messages": [
            {"role": "system", "content": _SOM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content, "images": [img_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        # Strip markdown code fence if present
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        out = json.loads(raw) if raw else {}
        if callback and out:
            callback(f"SOM extracted: {len(out.get('components_flat', []))} components")
        return out
    except Exception as exc:
        logger.warning("extract_screen_object_model failed: %s", exc)
        if callback:
            callback(f"SOM extraction failed: {exc}")
        return {}


# ── UI compliance audit (WCAG / HIPAA); output matches UIAuditResult schema ──

_UI_AUDIT_SYSTEM_PROMPT = (
    "You are an accessibility and regulatory compliance auditor for digital interfaces. "
    "Evaluate the screenshot against WCAG 2.2 Level AA and, when relevant, HIPAA technical safeguards. "
    "For each violation: provide guideline_reference (e.g. 'WCAG 2.2 Success Criterion 1.4.3'), "
    "issue_description, severity (low|medium|high|critical), bounding_box (left, top, width, height; normalized 0–1), "
    "element_type, and suggested_fix. Output ONLY valid JSON matching the required schema. No markdown, no commentary."
)


def run_ui_compliance_audit(
    image_path: str | Path,
    som_json: dict[str, Any] | None = None,
    regulatory_context: str | None = None,
    evaluation_standard: str = "WCAG 2.2 Level AA",
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Run a compliance audit on a UI screenshot. Uses VLM to detect violations and produce
    audit_metadata, detected_violations (with bounding boxes and guideline_reference), and generative_synthesis.
    If regulatory_context is provided (e.g. from Graph RAG retrieval), it is included in the prompt
    to ground guideline_reference and issue_description.
    Returns a dict suitable for UIAuditResult / audit_result JSON.
    """
    path = Path(image_path)
    if not path.exists() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return {}
    try:
        img_b64 = _image_file_to_base64(path)
    except Exception as exc:
        logger.warning("run_ui_compliance_audit: cannot read image %s: %s", path, exc)
        return {}
    if not img_b64:
        return {}

    if callback:
        callback("Running UI compliance audit (VLM)…")

    user_content = (
        f"Evaluate this UI screenshot for accessibility and compliance against {evaluation_standard}. "
        "Output a JSON object with exactly these keys:\n"
        "1. audit_metadata: { screen_id (string), timestamp (ISO8601), evaluation_standard (string), overall_compliance_status (Pass|Failed|Partial) }\n"
        "2. detected_violations: array of { guideline_reference, issue_description, severity, bounding_box (left, top, width, height normalized 0–1), element_type, suggested_fix }\n"
        "3. generative_synthesis: { global_edit_prompt (instructions to fix the UI for an image model), negative_prompt (what not to change) }\n"
        "Identify violations with exact coordinates (bounding_box) for each violating element. "
        "For contrast issues cite WCAG 1.4.3; for non-text content cite 1.1.1; for focus/keyboard cite 2.4.7; for touch targets cite 2.5.5."
    )
    if som_json:
        user_content += "\n\nScreen Object Model (use to locate elements):\n" + json.dumps(som_json)[:3000]
    if regulatory_context:
        user_content += "\n\nRelevant regulatory text (use to align guideline_reference and issue_description):\n" + regulatory_context[:4000]

    payload = {
        "model": config.VLM_MODEL,
        "messages": [
            {"role": "system", "content": _UI_AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content, "images": [img_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        out = json.loads(raw) if raw else {}
        if callback and out:
            n = len(out.get("detected_violations", []))
            callback(f"Audit complete: {out.get('audit_metadata', {}).get('overall_compliance_status', 'Unknown')}, {n} violation(s)")
        return out
    except Exception as exc:
        logger.warning("run_ui_compliance_audit failed: %s", exc)
        if callback:
            callback(f"Audit failed: {exc}")
        return {}


def inject_vlm_summaries_into_ocr_markdown(
    ocr_md: str,
    summaries: list[tuple[int, str]],
) -> str:
    """
    Inject each page's VLM summary into the OCR markdown right after that page's content.
    So when PageIndex builds the tree (with flush on page change), each node gets both
    OCR text and [VLM visual context] for that page, and summarization sees both.
    summaries: [(page_num, summary), ...] from get_per_page_visual_summaries.
    """
    if not ocr_md or not summaries:
        return ocr_md
    summary_by_page = dict(summaries)
    # Split on <!-- PAGE N --> (capture marker and page number)
    parts = re.split(r'(<!--\s*PAGE\s+(\d+)\s*-->)', ocr_md)
    if len(parts) < 3:
        return ocr_md
    out: list[str] = [parts[0]]
    for i in range(1, len(parts), 3):
        marker = parts[i]
        try:
            page_num = int(parts[i + 1])
        except (ValueError, IndexError):
            out.append(parts[i])
            if i + 1 < len(parts):
                out.append(parts[i + 1])
            if i + 2 < len(parts):
                out.append(parts[i + 2])
            continue
        content = (parts[i + 2] if i + 2 < len(parts) else "").strip()
        summary = summary_by_page.get(page_num, "").strip()
        out.append(marker)
        out.append("\n\n")
        out.append(content)
        if summary:
            out.append("\n\n[VLM visual context]: ")
            out.append(summary)
        out.append("\n\n")
    return "".join(out)
