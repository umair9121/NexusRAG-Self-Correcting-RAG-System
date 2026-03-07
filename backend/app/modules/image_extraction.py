"""
Image Extraction Module — VLM-based description extraction for images.

Uses a Vision Language Model (qwen2.5vl:3b by default) via the Ollama chat API
to extract detailed UI/visual descriptions from images, similar to the notebook
workflow in Extracting_Description_From_Image.ipynb.

Extractions are saved as JSON metadata per file in storage/extractions/.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app import config

logger = logging.getLogger(__name__)

# ── Image handling ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def _image_file_to_base64(image_path: str | Path) -> str:
    """Read/resize image and return base64-encoded string for VLM."""
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
            img.save(buf, format="JPEG", quality=88, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def _call_vlm_with_fallback(
    payload: dict[str, Any],
    purpose: str,
) -> str:
    """Call Ollama /api/chat with conservative fallback contexts."""
    def _extract_text(body: dict[str, Any]) -> str:
        msg = body.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # Content can be a list of parts (e.g. [{"type": "text", "text": "..."}]).
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"].strip()
        # Some runtimes/models surface text on `response` (generate-style shape).
        response = body.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        # Reasoning / thinking: some models (e.g. qwen3) put main output here.
        reasoning = msg.get("reasoning_content") or msg.get("thinking")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        if isinstance(body.get("thinking"), str) and body["thinking"].strip():
            return body["thinking"].strip()
        return ""

    options = dict(payload.get("options") or {})
    ctx_candidates = [options.get("num_ctx", 4096), 2048, 1024]
    seen: set[int] = set()
    ctx_values = [c for c in ctx_candidates if isinstance(c, int) and c > 0 and not (c in seen or seen.add(c))]
    last_error = ""
    for i, num_ctx in enumerate(ctx_values, start=1):
        attempt = dict(payload)
        attempt_opts = dict(options)
        attempt_opts["num_ctx"] = num_ctx
        if i > 1 and isinstance(attempt_opts.get("num_predict"), int):
            attempt_opts["num_predict"] = max(128, int(attempt_opts["num_predict"]) // 2)
        attempt["options"] = attempt_opts
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=attempt,
                timeout=config.OLLAMA_TIMEOUT,
            )
            if resp.status_code >= 400:
                body = (resp.text or "").strip()
                last_error = f"{resp.status_code} {body[:300]}"
                logger.warning("%s failed (num_ctx=%s): %s", purpose, num_ctx, last_error)
                continue
            parsed = resp.json()
            text = _extract_text(parsed)
            if text:
                return text
            # Diagnose why we got empty content (qwen3-vl is capable; often 500 or wrong key).
            msg = parsed.get("message") or {}
            content_val = msg.get("content")
            err_piece = f" keys={list(parsed.keys())}"
            if parsed.get("error"):
                err_piece += f" error={str(parsed.get('error'))[:200]}"
            err_piece += f" message.keys={list(msg.keys())}"
            err_piece += f" message.content type={type(content_val).__name__}"
            if content_val is not None:
                err_piece += f" len={getattr(content_val, '__len__', lambda: 0)()}"
                err_piece += f" repr={repr(content_val)[:300]}"
            logger.warning(
                "VLM empty content diagnostic (status=200):%s",
                err_piece,
            )
            last_error = f"empty response body{err_piece}"
        except Exception as exc:
            last_error = str(exc)
            logger.warning("%s exception (num_ctx=%s): %s", purpose, num_ctx, exc)

    # Fallback: some Ollama model builds behave better with /api/generate for image prompts.
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
            "options": options,
        }
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=gen_payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        if resp.status_code < 400:
            parsed = resp.json()
            text = _extract_text(parsed)
            if text:
                logger.info("%s succeeded via /api/generate fallback", purpose)
                return text
        else:
            last_error = f"/api/generate {resp.status_code} {(resp.text or '')[:240]}"
    except Exception as exc:
        last_error = f"/api/generate exception: {exc}"

    if last_error:
        logger.warning("%s final failure: %s", purpose, last_error)
    return ""


def is_image_file(file_path: str | Path) -> bool:
    """Check if a file is a supported image type."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


# ── Prompts (adapted from notebook) ─────────────────────────────────────────

SCREENSHOT_PROMPT = """You are a senior mobile UI and UX analyst.

You are looking at ONE FINAL IMPLEMENTED MOBILE APP SCREENSHOT (a fully designed UI).
Describe the screen in detail for a developer who cannot see the image.

ABSOLUTE RULES:
- Always talk in terms of UI components (status bar, header/app bar, title, subtitle, image, list row, toggle, chevron, button, footer, bottom nav).
- If any text is readable (titles, labels, button text), include it verbatim.
- Do NOT skip the bottom area: capture CTA buttons and any footer/bottom nav if present.

OUTPUT STRUCTURE (use these headings exactly):
A) Screen purpose and type:
B) Layout top → bottom:
C) Component inventory (grouped):
D) Primary actions + navigation:
Write 14–22 sentences total (compact, but complete).
End with a bullet recap:
- Screen type/purpose
- Key components
- Primary CTA(s)
- Navigation/footer
"""

STATUSBAR_PROMPT = """You are a senior mobile UI analyst.

Focus ONLY on the STATUS BAR (the thin strip at the very top of the screen).
Ignore everything below it.

Your task is to report what is visually present, with confidence awareness.

Rules:
- If a symbol is clearly recognizable by its visual shape or logo, name it.
- If a symbol looks like a known system icon but is not fully certain, write:
  "likely <icon name> (low confidence)".
- If a symbol cannot be reliably identified, write:
  "unknown icon" with a brief shape description.
- Do NOT assume icons just because they are common.
- Do NOT infer icons from context.
- Time should be reported if readable.

Scan strictly from LEFT to RIGHT.

Return EXACTLY two sections:

1) Left-side notification/app icons:
   - List icons in order.
   - Use app names only when logos are unmistakable.

2) Right-side system/status icons + time:
   - List icons in order.
   - Use confidence labels where needed.
   - End with time if readable.

Be literal, visual, and conservative.
"""


# ── VLM generation via Ollama ────────────────────────────────────────────────

def _vlm_describe_image(
    img_b64: str,
    prompt: str,
    max_new_tokens: int = config.VLM_MAX_NEW_TOKENS,
    model_override: str | None = None,
) -> str:
    """Send an image + prompt to the VLM via Ollama chat API and return the response."""
    payload: dict[str, Any] = {
        "model": model_override or config.VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": max_new_tokens,
        },
    }
    return _call_vlm_with_fallback(payload, purpose="VLM image description")


# ── Main extraction function ─────────────────────────────────────────────────

def extract_image_description(
    image_path: str | Path,
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Extract a detailed description from an image file using the VLM.

    Produces a full screenshot description and (optionally) a status bar
    analysis, then merges them into a single metadata dict.

    Returns a dict with keys:
        - screenshot_description: full UI description
        - statusbar_description: status bar details (if applicable)
        - merged_description: combined text
        - extracted_at: ISO timestamp
        - vlm_model: model used
        - file_path: original image path
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning("Image file not found: %s", image_path)
        return {}

    if not is_image_file(image_path):
        logger.warning("Not a supported image file: %s", image_path)
        return {}

    img_b64 = _image_file_to_base64(image_path)
    if not img_b64:
        logger.warning("Could not read image: %s", image_path)
        return {}

    if callback:
        callback(f"Extracting image description with {config.VLM_MODEL}...")

    # Full screenshot description (UI analysis model)
    full_desc = _vlm_describe_image(img_b64, SCREENSHOT_PROMPT)
    if not full_desc.strip():
        # A shorter retry prompt can help when long structured prompts are dropped
        # by constrained runtimes.
        full_desc = _vlm_describe_image(
            img_b64,
            (
                "Describe this UI screenshot for a developer who cannot see it. "
                "Explain layout top-to-bottom, components, and visible text verbatim."
            ),
            max_new_tokens=min(700, config.VLM_MAX_NEW_TOKENS),
        )
    if callback:
        callback(f"Screenshot description: {len(full_desc)} chars")

    # Status bar analysis (UI analysis model)
    if callback:
        callback("Analyzing status bar region...")
    sb_desc = _vlm_describe_image(img_b64, STATUSBAR_PROMPT, max_new_tokens=300)
    if not sb_desc.strip():
        sb_desc = _vlm_describe_image(
            img_b64,
            "Describe only the top status bar icons and visible time, left to right.",
            max_new_tokens=180,
        )
    if callback:
        callback(f"Status bar description: {len(sb_desc)} chars")

    # Merge descriptions
    merged = full_desc.strip()
    if sb_desc.strip():
        merged += "\n\nSTATUS BAR (detail pass):\n" + sb_desc.strip()
    if not merged:
        logger.warning("Image extraction produced empty output for %s", image_path.name)
        if callback:
            callback(
                "UI description model returned empty output for screenshot. "
                "This pipeline requires UI description generation (not OCR fallback)."
            )

    result = {
        "screenshot_description": full_desc,
        "statusbar_description": sb_desc,
        "merged_description": merged,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "vlm_model": config.VLM_MODEL,
        "file_path": str(image_path),
    }

    if callback:
        callback(f"Image extraction complete: {len(merged)} total chars")

    return result


def save_extraction(doc_stem: str, extraction_data: dict[str, Any]) -> Path:
    """Save extraction metadata to storage/extractions/{doc_stem}.json."""
    config.EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EXTRACTION_DIR / f"{doc_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extraction_data, f, indent=2, ensure_ascii=False)
    return out_path


def load_extraction(doc_stem: str) -> dict[str, Any] | None:
    """Load extraction metadata from storage. Returns None if not found."""
    path = config.EXTRACTION_DIR / f"{doc_stem}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
