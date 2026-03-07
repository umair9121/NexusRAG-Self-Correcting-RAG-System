#!/usr/bin/env python3
"""
Test Ollama vision API with a single image. Use to verify qwen3-vl (or another
vision model) returns non-empty content outside the main app.

  From repo root:
    python -m backend.scripts.test_ollama_vision path/to/screenshot.png
    python -m backend.scripts.test_ollama_vision path/to/screenshot.png --model llava

Prints: HTTP status, response keys, and any text found in message.content,
response, reasoning_content, thinking.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

# Add backend to path so app.config is importable (run from repo root: python -m backend.scripts.test_ollama_vision ...)
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

import requests
from PIL import Image

from app import config


def image_to_base64(path: Path, max_long_side: int = 1280) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_long_side:
            scale = max_long_side / float(long_side)
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Test Ollama vision with one image")
    p.add_argument("image", type=Path, help="Path to PNG/JPEG image")
    p.add_argument("--model", default=None, help="Ollama model (default: VLM_MODEL)")
    p.add_argument("--url", default=None, help="Ollama base URL (default: OLLAMA_BASE_URL)")
    args = p.parse_args()

    if not args.image.exists():
        print(f"File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    model = args.model or config.VLM_MODEL
    base_url = (args.url or config.OLLAMA_BASE_URL).rstrip("/")
    img_b64 = image_to_base64(args.image)
    print(f"Model: {model}, URL: {base_url}, image base64 length: {len(img_b64)}")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Describe this image in one short paragraph.",
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 256},
    }

    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT or 120,
        )
    except Exception as e:
        print(f"Request error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"HTTP status: {resp.status_code}")
    if resp.status_code >= 400:
        print(f"Body: {resp.text[:500]}")
        sys.exit(3)

    try:
        data = resp.json()
    except Exception as e:
        print(f"JSON decode error: {e}\nRaw: {resp.text[:400]}")
        sys.exit(4)

    print(f"Top-level keys: {list(data.keys())}")
    msg = data.get("message") or {}
    print(f"message keys: {list(msg.keys())}")

    for key in ("content", "reasoning_content", "thinking"):
        val = msg.get(key) or data.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            print(f"\n--- {key} (str, len={len(val)}) ---\n{val[:800]}")
        else:
            print(f"\n--- {key} (type={type(val).__name__}) ---\n{repr(val)[:500]}")

    response_text = data.get("response")
    if response_text and isinstance(response_text, str):
        print(f"\n--- response (top-level, len={len(response_text)}) ---\n{response_text[:800]}")

    if not (msg.get("content") or data.get("response") or msg.get("reasoning_content") or msg.get("thinking")):
        print("\n(No text found in content/response/reasoning_content/thinking; raw message below)")
        print(json.dumps(msg, indent=2, default=repr)[:1200])


if __name__ == "__main__":
    main()
