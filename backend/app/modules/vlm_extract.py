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
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF – used to render PDF pages to images
import requests
from PIL import Image

from app import config

logger = logging.getLogger(__name__)


# ── Ollama VRAM management ──────────────────────────────────────────────────

def unload_vlm_from_vram() -> None:
    """Ask Ollama to immediately unload the VLM from VRAM (keep_alive=0).

    Call this after the last VLM inference before switching to the
    embedding model so VRAM is freed and the embedding model can load
    without contention.
    """
    try:
        requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": config.VLM_MODEL, "prompt": "", "keep_alive": 0},
            timeout=15,
        )
        logger.info("Unloaded %s from VRAM (keep_alive=0)", config.VLM_MODEL)
    except Exception as exc:
        logger.debug("VLM unload request failed (non-critical): %s", exc)


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
    """Read/resize an image file and return base64-encoded string.

    Large screenshots can trigger Ollama 500 errors (OOM/context pressure). We
    downscale to a safer long side before encoding while preserving readability.
    """
    path = Path(image_path)
    if not path.exists():
        return ""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            max_long_side = 1280
            w, h = img.size
            long_side = max(w, h)
            if long_side > max_long_side:
                scale = max_long_side / float(long_side)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            # JPEG keeps payload small for VLM calls and avoids 500s on large images.
            img.save(buf, format="JPEG", quality=88, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def _extract_json_from_text(text: str) -> str | None:
    """Find and extract a JSON object from text that may have preamble/commentary.

    qwen3-vl often emits chain-of-thought reasoning before the actual
    JSON output.  This scans for the first top-level ``{`` and last ``}``
    and attempts to parse the substring as JSON.  Returns the cleaned
    JSON string on success, or None if no valid JSON was found.
    """
    if not text:
        return None
    # Strip markdown code fences first
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Fast path: already valid JSON
    stripped = cleaned.strip()
    if stripped.startswith("{"):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # Find the first '{' and last '}' — the JSON object may be buried in text
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def _extract_chat_text(body: dict[str, Any]) -> str:
    """Extract generated text from an Ollama /api/chat response.

    qwen3-vl may place its main output in ``message.content``, in
    ``message.thinking`` / ``message.reasoning_content``, or in the
    top-level ``response`` key depending on Ollama version and model
    variant.  We check all known locations.
    """
    msg = body.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                return part["text"].strip()
    response = body.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    reasoning = msg.get("reasoning_content") or msg.get("thinking")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    if isinstance(body.get("thinking"), str) and body["thinking"].strip():
        return body["thinking"].strip()
    return ""


def _post_chat_with_fallback(
    payload: dict[str, Any],
    callback: Any | None = None,
    purpose: str = "VLM",
) -> str:
    """Call Ollama /api/chat with fallback num_ctx values for robustness."""
    option_template = dict(payload.get("options") or {})
    fallback_ctx = [option_template.get("num_ctx", 4096), 2048, 1024]
    seen: set[int] = set()
    ctx_values = [c for c in fallback_ctx if isinstance(c, int) and c > 0 and not (c in seen or seen.add(c))]
    last_error = ""

    for i, num_ctx in enumerate(ctx_values, start=1):
        attempt_payload = dict(payload)
        attempt_opts = dict(option_template)
        attempt_opts["num_ctx"] = num_ctx
        if i > 1 and isinstance(attempt_opts.get("num_predict"), int):
            attempt_opts["num_predict"] = max(128, int(attempt_opts["num_predict"]) // 2)
        attempt_payload["options"] = attempt_opts
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=attempt_payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            if resp.status_code >= 400:
                body = (resp.text or "").strip()
                last_error = f"{resp.status_code} {body[:300]}"
                logger.warning("%s call failed (num_ctx=%s): %s", purpose, num_ctx, last_error)
                if callback:
                    callback(f"{purpose} retrying with lower context (num_ctx={num_ctx})...")
                continue
            parsed = resp.json()
            text = _extract_chat_text(parsed)
            if text:
                return text
            last_error = "empty response content"
            logger.warning(
                "%s empty content diagnostic (status=200): keys=%s message.keys=%s",
                purpose, list(parsed.keys()), list((parsed.get("message") or {}).keys()),
            )
        except Exception as exc:
            last_error = str(exc)
            logger.warning("%s call exception (num_ctx=%s): %s", purpose, num_ctx, exc)

    # Fallback: /api/generate can sometimes succeed where /api/chat does not
    try:
        user_msg = {}
        for m in payload.get("messages", []):
            if m.get("role") == "user":
                user_msg = m
        gen_payload = {
            "model": payload.get("model"),
            "prompt": (user_msg.get("content") or "").strip(),
            "images": user_msg.get("images") or [],
            "stream": False,
            "options": option_template,
        }
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=gen_payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        if resp.status_code < 400:
            parsed = resp.json()
            text = _extract_chat_text(parsed)
            if text:
                logger.info("%s succeeded via /api/generate fallback", purpose)
                return text
    except Exception as exc:
        last_error = f"/api/generate fallback: {exc}"

    if callback and last_error:
        callback(f"{purpose} failed: {last_error}")
    return ""


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
    text = _post_chat_with_fallback(payload, callback=callback, purpose="VLM image extraction")
    # If the general VLM yields nothing, fall back to OCR_MODEL for text-centric extraction.
    ocr_model = (getattr(config, "OCR_MODEL", "") or "").strip()
    if (not text) and ocr_model and ocr_model != config.VLM_MODEL:
        if callback:
            callback(f"No text from {config.VLM_MODEL}; retrying with OCR model {ocr_model}…")
        ocr_payload: dict[str, Any] = {
            "model": ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract all readable text visible in this image. "
                        "Output only extracted text in plain Markdown."
                    ),
                    "images": [img_b64],
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 2048},
        }
        text = _post_chat_with_fallback(
            ocr_payload, callback=callback, purpose="OCR-model image extraction"
        )
    if callback and text:
        callback(f"VLM extracted {len(text)} chars from image")
    return text


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

            text = _extract_chat_text(resp.json())

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
        answer = _extract_chat_text(resp.json())
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
        caption = _extract_chat_text(resp.json())
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
            summary = _extract_chat_text(resp.json())
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
    "You are a UI analyst. Produce a Screen Object Model (SOM) as a JSON object. "
    "For each visible component include: type, text content, approximate position, "
    "and accessibility-related attributes."
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

    prompt = (
        "Produce a JSON SOM for this screenshot with keys: "
        "screen_type (string), components_flat (array of "
        "{id, type, text, bounds:{left,top,width,height}, accessibility_label, parent_id})."
    )
    if view_hierarchy_json:
        prompt += "\nView hierarchy hint:\n" + json.dumps(view_hierarchy_json)[:1500]

    payload: dict[str, Any] = {
        "model": config.VLM_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1024},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        if callback:
            callback(f"SOM VLM response: {len(raw)} chars")
        json_str = _extract_json_from_text(raw)
        if not json_str:
            logger.warning("SOM response has no valid JSON (len=%d): %.200s", len(raw), raw)
            if callback:
                callback("SOM response contained no valid JSON — skipping")
            return {}
        out = json.loads(json_str)
        if callback and out:
            callback(f"SOM extracted: {len(out.get('components_flat', []))} components")
        return out
    except Exception as exc:
        logger.warning("extract_screen_object_model failed: %s", exc)
        if callback:
            callback(f"SOM extraction failed: {exc}")
        return {}


# ── WCAG-grounded UI compliance audit ────────────────────────────────────────

_UI_AUDIT_SYSTEM_PROMPT = (
    "You are a WCAG 2.2 compliance auditor. Analyze the UI against the provided WCAG criteria. "
    "For each violation cite the WCAG criterion, describe the issue, rate severity, and suggest a fix. "
    "Respond as a JSON object."
)


def retrieve_wcag_context_for_ui(
    ui_description: str,
    callback: Any | None = None,
) -> str:
    """Query the ingested WCAG data (vector + graph) using the UI description
    to retrieve the most relevant success criteria, techniques, and test procedures.

    Returns a combined string of WCAG reference text suitable for grounding the audit.
    """
    import time as _time
    from app.modules.ollama_client import OllamaClient
    from app.modules.contextual import _get_chroma_collection, DocStore
    from app.modules.graph import GraphSearcher

    wcag_stem = config.WCAG_DOC_STEM
    llm = OllamaClient()
    context_parts: list[str] = []

    # Keep the query short — long queries slow down embedding models drastically
    wcag_query = (
        "WCAG 2.2 accessibility: contrast, text alternatives, focus indicators, "
        "touch targets, keyboard, color, form labels, error identification, navigation"
    )

    # Vector search against WCAG chunks (filtered by doc_name)
    try:
        if callback:
            callback("Embedding WCAG query for vector search…")
        t0 = _time.monotonic()
        query_embedding = llm.embed(wcag_query)
        embed_ms = int((_time.monotonic() - t0) * 1000)
        if callback:
            callback(f"WCAG query embedded in {embed_ms}ms")
        if query_embedding:
            collection = _get_chroma_collection()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=10,
                where={"doc_name": wcag_stem},
                include=["documents", "metadatas", "distances"],
            )
            docstore = DocStore()
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    parent = docstore.get(doc_id)
                    text = parent["text"] if parent else (results["documents"][0][i] or "")
                    text = text.strip()
                    if text:
                        context_parts.append(text)
                if callback:
                    callback(f"Retrieved {len(results['ids'][0])} WCAG vector chunks")
            else:
                if callback:
                    callback("No WCAG vector chunks found (collection may be empty for this doc)")
    except Exception as exc:
        logger.warning("WCAG vector search failed: %s", exc)
        if callback:
            callback(f"WCAG vector search failed: {exc}")

    # Graph search for entity/relationship context
    try:
        graph_file = f"{wcag_stem}_graph.json"
        if (config.GRAPH_DIR / graph_file).exists():
            if callback:
                callback("Searching WCAG knowledge graph…")
            gs = GraphSearcher(llm)
            gs.load_graph(wcag_stem)
            graph_result = gs.search(
                wcag_query,
                mode="local",
                doc_name=wcag_stem,
            )
            graph_ctx = (graph_result.get("context") or "").strip()
            if graph_ctx:
                context_parts.append(graph_ctx)
                if callback:
                    callback(f"Retrieved WCAG graph context ({len(graph_ctx)} chars)")
        else:
            if callback:
                callback("WCAG graph not found — skipping graph search")
    except Exception as exc:
        logger.warning("WCAG graph search failed: %s", exc)
        if callback:
            callback(f"WCAG graph search failed: {exc}")

    combined = "\n\n---\n\n".join(context_parts)
    if callback:
        callback(f"Total WCAG context: {len(combined)} chars from {len(context_parts)} sources")
    return combined


def _build_audit_from_freetext(raw: str, evaluation_standard: str) -> dict[str, Any]:
    """Parse a free-text VLM audit response into the structured audit dict.

    When format:json fails and the VLM returns prose, this extracts whatever
    violation information it can from the text and wraps it in the expected
    audit schema so downstream code works identically.
    """
    violations: list[dict[str, str]] = []
    lines = raw.split("\n")
    current: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        # Detect numbered violation lines or lines referencing WCAG SC
        sc_match = re.search(r'(\d+\.\d+\.\d+)', stripped)
        if sc_match or any(kw in low for kw in ("violation", "issue", "fail", "non-compliant", "does not meet")):
            if current.get("issue_description"):
                violations.append(current)
            current = {
                "guideline_reference": sc_match.group(1) if sc_match else "",
                "issue_description": stripped,
                "severity": "needs review",
                "element_type": "",
                "suggested_fix": "",
            }
        elif current and any(kw in low for kw in ("fix", "remediat", "suggest", "recommend", "solution")):
            current["suggested_fix"] = stripped
        elif current and any(kw in low for kw in ("contrast", "color", "alt", "label", "focus", "touch", "target", "keyboard", "heading", "navigation")):
            if not current["element_type"]:
                current["element_type"] = stripped
            else:
                current["issue_description"] += " " + stripped
        elif current:
            current["issue_description"] += " " + stripped
    if current.get("issue_description"):
        violations.append(current)

    status = "Needs Review"
    if not violations:
        status = "Could Not Determine"
    elif len(violations) >= 3:
        status = "Non-Compliant"

    return {
        "audit_metadata": {
            "overall_compliance_status": status,
            "evaluation_standard": evaluation_standard,
            "parse_mode": "freetext_fallback",
        },
        "detected_violations": violations,
        "generative_synthesis": {
            "compliance_summary": raw[:1500],
        },
    }


def run_text_only_compliance_audit(
    ui_description: str,
    som_json: dict[str, Any] | None = None,
    *,
    regulatory_context: str | None = None,
    evaluation_standard: str = "WCAG 2.2 Level AA",
    callback: Any | None = None,
) -> dict[str, Any]:
    """WCAG compliance audit using the main LLM (text-only, no VLM).

    The UI description was already generated by the VLM in step 1 — this
    function receives that full description, retrieves relevant WCAG criteria
    from the RAG pipeline, and asks the main LLM to identify violations.

    This is the primary audit path (not a fallback).  Small VLMs cannot
    reliably do multi-step accessibility reasoning with structured output;
    the main LLM handles it much better as a pure text task.
    """
    from app.modules.ollama_client import OllamaClient

    ui_description = (ui_description or "").strip()
    if not ui_description:
        return {}

    import time as _time

    if callback:
        callback("Running WCAG compliance audit with main LLM…")

    if not regulatory_context:
        t_retrieval = _time.monotonic()
        try:
            regulatory_context = retrieve_wcag_context_for_ui(ui_description, callback)
        except Exception as exc:
            logger.warning("WCAG context retrieval failed: %s", exc)
            if callback:
                callback(f"WCAG retrieval failed: {exc}")
            regulatory_context = ""
        retrieval_s = _time.monotonic() - t_retrieval
        if callback:
            callback(
                f"WCAG retrieval step done in {retrieval_s:.1f}s "
                f"({len(regulatory_context or '')} chars)"
            )

    # Hard caps — keep prompt small enough that even qwen3:1.7b can respond.
    UI_DESC_CAP = 1500
    WCAG_CTX_CAP = 1500
    SOM_CAP = 500

    prompt = (
        f"Audit this UI for {evaluation_standard} compliance.\n\n"
        f"UI DESCRIPTION:\n{ui_description[:UI_DESC_CAP]}\n\n"
    )
    if regulatory_context:
        prompt += f"WCAG CRITERIA:\n{regulatory_context[:WCAG_CTX_CAP]}\n\n"
    if som_json:
        prompt += f"COMPONENTS:\n{json.dumps(som_json)[:SOM_CAP]}\n\n"
    prompt += (
        "List every WCAG violation. Return JSON:\n"
        '{"audit_metadata":{"overall_compliance_status":"Non-Compliant",'
        f'"evaluation_standard":"{evaluation_standard}"}}'
        ','
        '"detected_violations":[{"guideline_reference":"1.4.3",'
        '"issue_description":"...","severity":"major",'
        '"element_type":"...","suggested_fix":"..."}],'
        '"generative_synthesis":{"compliance_summary":"..."}}\n'
    )

    prompt_len = len(prompt)
    logger.info("WCAG audit prompt: %d chars (ui=%d wcag=%d) model=%s",
                prompt_len, min(len(ui_description), UI_DESC_CAP),
                min(len(regulatory_context or ""), WCAG_CTX_CAP), config.LLM_MODEL)
    if callback:
        callback(f"Sending WCAG audit prompt to {config.LLM_MODEL} ({prompt_len} chars)…")

    llm = OllamaClient()
    result: dict[str, Any] = {}

    # Do NOT use format:"json" — small models (qwen3:1.7b) return empty {}
    # with that constraint.  Just generate free text and parse it ourselves.
    try:
        t_gen = _time.monotonic()
        raw_text = llm.generate(
            prompt,
            options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 2048},
        )
        gen_s = _time.monotonic() - t_gen
        logger.info("WCAG audit LLM responded in %.1fs (%d chars): %.300s",
                     gen_s, len(raw_text), raw_text[:300])
        if callback:
            callback(f"LLM responded in {gen_s:.1f}s ({len(raw_text)} chars)")
    except Exception as exc:
        logger.warning("WCAG audit generate failed: %s", exc)
        raw_text = ""
        if callback:
            callback(f"WCAG audit LLM call failed: {exc}")

    if raw_text.strip():
        # Try to parse as JSON first
        json_str = _extract_json_from_text(raw_text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and parsed:
                    result = parsed
            except json.JSONDecodeError:
                pass
        # If JSON parsing failed or returned empty, extract from freetext
        if not result:
            result = _build_audit_from_freetext(raw_text, evaluation_standard)

    if not isinstance(result, dict) or not result:
        if callback:
            callback("WCAG audit could not produce results — check LLM model availability")
        return {}

    # Normalise shape
    audit_metadata = result.get("audit_metadata") or {}
    detected_violations = result.get("detected_violations") or []
    generative_synthesis = result.get("generative_synthesis") or {}

    if isinstance(audit_metadata, dict):
        audit_metadata.setdefault("evaluation_standard", evaluation_standard)
        audit_metadata.setdefault("overall_compliance_status", "Needs Review")
        audit_metadata.setdefault("parse_mode", "text_only_llm")

    out = {
        "audit_metadata": audit_metadata,
        "detected_violations": detected_violations,
        "generative_synthesis": generative_synthesis,
    }

    if callback:
        n = len(detected_violations) if isinstance(detected_violations, list) else 0
        status = audit_metadata.get("overall_compliance_status", "Unknown")
        callback(f"WCAG audit complete: {status}, {n} violation(s) detected")

    return out


def run_ui_compliance_audit(
    image_path: str | Path,
    ui_description: str | None = None,
    som_json: dict[str, Any] | None = None,
    regulatory_context: str | None = None,
    evaluation_standard: str = "WCAG 2.2 Level AA",
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Run a WCAG-grounded compliance audit on a UI screenshot.

    The VLM's job was already done in step 1 (extract_image_description) which
    produced a rich natural-language UI description.  The audit itself is a
    *text reasoning* task: cross-reference the UI description against WCAG
    criteria retrieved from the RAG pipeline.  This is handled by the main LLM
    (config.LLM_MODEL), not the small VLM — small vision models choke on long
    structured prompts and return empty responses.

    Returns a dict with audit_metadata, detected_violations, and generative_synthesis.
    """
    ui_description = (ui_description or "").strip()
    if not ui_description:
        logger.warning("run_ui_compliance_audit: no UI description provided — cannot audit")
        return {}

    return run_text_only_compliance_audit(
        ui_description,
        som_json=som_json,
        regulatory_context=regulatory_context,
        evaluation_standard=evaluation_standard,
        callback=callback,
    )


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
