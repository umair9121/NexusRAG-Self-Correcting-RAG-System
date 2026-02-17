"""
Image Extraction Module — VLM-based description extraction for images.

Uses a Vision Language Model (qwen2.5vl:3b by default) via the Ollama chat API
to extract detailed UI/visual descriptions from images, similar to the notebook
workflow in Extracting_Description_From_Image.ipynb.

Extractions are saved as JSON metadata per file in storage/extractions/.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from app import config

logger = logging.getLogger(__name__)

# ── Image handling ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def _image_file_to_base64(image_path: str | Path) -> str:
    """Read an image file and return base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
) -> str:
    """Send an image + prompt to the VLM via Ollama chat API and return the response."""
    payload: dict[str, Any] = {
        "model": config.VLM_MODEL,
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
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        logger.warning("VLM image description failed: %s", exc)
        return ""


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

    # Full screenshot description
    full_desc = _vlm_describe_image(img_b64, SCREENSHOT_PROMPT)
    if callback:
        callback(f"Screenshot description: {len(full_desc)} chars")

    # Status bar analysis
    if callback:
        callback("Analyzing status bar region...")
    sb_desc = _vlm_describe_image(img_b64, STATUSBAR_PROMPT, max_new_tokens=300)
    if callback:
        callback(f"Status bar description: {len(sb_desc)} chars")

    # Merge descriptions
    merged = full_desc.strip()
    if sb_desc.strip():
        merged += "\n\nSTATUS BAR (detail pass):\n" + sb_desc.strip()

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
