"""
EDI Evaluation Module — LLM-based evaluation of images against EDI Guidelines.

Uses the main LLM model (gemma3:4b-it-qat by default) to evaluate whether
an image's extracted description conforms to EDI (Equality, Diversity, and
Inclusion) guidelines.

The evaluation is run on the *text description* produced by the image
extraction pipeline (not the raw image), so it operates as a post-extraction
analysis step.

Results are saved alongside extraction data in storage/extractions/{doc_stem}.json
under the "edi_evaluation" key.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config
from app.modules.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ── EDI Evaluation Prompt ────────────────────────────────────────────────────

EDI_EVALUATION_PROMPT = """\
You are an expert reviewer evaluating mobile app UI screenshots against \
Equality, Diversity, and Inclusion (EDI) guidelines.

You have been given a detailed textual description of a mobile app screenshot. \
Your job is to assess whether this UI design follows EDI best practices.

Evaluate the following areas and provide a structured assessment:

1) ACCESSIBILITY:
   - Are there sufficient contrast indicators mentioned?
   - Are interactive elements (buttons, toggles) clearly described with labels?
   - Is text readable (appropriate sizing mentioned)?
   - Are touch targets likely large enough?

2) INCLUSIVE LANGUAGE:
   - Is the text content (labels, titles, button text) inclusive and non-discriminatory?
   - Are there any gendered, ableist, or exclusionary terms?
   - Is the language clear and simple enough for non-native speakers?

3) VISUAL REPRESENTATION:
   - Does the UI appear to use universally understood icons/symbols?
   - Are there any culturally biased visual elements?
   - Is the color usage considerate of color-blind users (based on described colors)?

4) NAVIGATION & USABILITY:
   - Is the navigation structure clear and intuitive?
   - Would the layout work for users with motor impairments?
   - Are there clear visual hierarchies described?

5) CONTENT INCLUSIVITY:
   - Does the content structure accommodate diverse user needs?
   - Are there any assumptions about user demographics?

For each area, provide:
- A rating: PASS, PARTIAL, FAIL, or N/A (if not enough info)
- A brief explanation (1-2 sentences)

Then provide an OVERALL assessment:
- Overall EDI Rating: PASS / PARTIAL / FAIL
- Summary: 2-3 sentences summarizing the key findings
- Recommendations: List up to 5 specific, actionable improvement suggestions

Return your response as valid JSON with this structure:
{
  "accessibility": {"rating": "...", "explanation": "..."},
  "inclusive_language": {"rating": "...", "explanation": "..."},
  "visual_representation": {"rating": "...", "explanation": "..."},
  "navigation_usability": {"rating": "...", "explanation": "..."},
  "content_inclusivity": {"rating": "...", "explanation": "..."},
  "overall_rating": "PASS|PARTIAL|FAIL",
  "summary": "...",
  "recommendations": ["...", "..."]
}
"""


# ── Evaluation function ──────────────────────────────────────────────────────

def evaluate_edi_compliance(
    description: str,
    llm: OllamaClient | None = None,
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Evaluate an image description against EDI guidelines using the LLM.

    Args:
        description: The merged text description from image extraction.
        llm: Optional OllamaClient instance (creates one if not provided).
        callback: Optional callback for progress logging.

    Returns:
        A dict containing the EDI evaluation results, or an error dict.
    """
    if not description or not description.strip():
        logger.warning("No description provided for EDI evaluation")
        return {"error": "No description available for evaluation"}

    if llm is None:
        llm = OllamaClient()

    if callback:
        callback(f"Running EDI evaluation with {config.LLM_MODEL}...")

    prompt = (
        f"{EDI_EVALUATION_PROMPT}\n\n"
        f"--- SCREENSHOT DESCRIPTION ---\n"
        f"{description[:4000]}\n"
        f"--- END DESCRIPTION ---\n\n"
        f"Provide your EDI evaluation as JSON."
    )

    try:
        result = llm.generate_json(prompt, options=config.REASONING_OPTIONS)

        # Check if we got a valid structured response
        if "raw" in result and len(result) == 1:
            logger.warning("EDI evaluation returned unparseable response")
            if callback:
                callback("EDI evaluation: LLM returned non-JSON response, attempting text extraction")
            # Fall back to plain text generation
            text_result = llm.generate(prompt, options=config.REASONING_OPTIONS)
            result = {
                "raw_evaluation": text_result,
                "overall_rating": "UNKNOWN",
                "summary": "Evaluation completed but response was not structured JSON.",
            }

        result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        result["llm_model"] = config.LLM_MODEL

        if callback:
            overall = result.get("overall_rating", "UNKNOWN")
            callback(f"EDI evaluation complete: {overall}")

        return result

    except Exception as exc:
        logger.error("EDI evaluation failed: %s", exc)
        if callback:
            callback(f"EDI evaluation failed: {exc}")
        return {
            "error": str(exc),
            "overall_rating": "ERROR",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "llm_model": config.LLM_MODEL,
        }


def save_edi_evaluation(doc_stem: str, evaluation_data: dict[str, Any]) -> Path:
    """
    Save EDI evaluation results into the extraction JSON file.

    The evaluation is stored under the "edi_evaluation" key in the existing
    extraction metadata file.
    """
    config.EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    extraction_path = config.EXTRACTION_DIR / f"{doc_stem}.json"

    # Load existing extraction data if it exists
    existing: dict[str, Any] = {}
    if extraction_path.exists():
        try:
            with open(extraction_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    existing["edi_evaluation"] = evaluation_data

    with open(extraction_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return extraction_path


def load_edi_evaluation(doc_stem: str) -> dict[str, Any] | None:
    """Load EDI evaluation from the extraction file. Returns None if not found."""
    extraction_path = config.EXTRACTION_DIR / f"{doc_stem}.json"
    if not extraction_path.exists():
        return None
    try:
        with open(extraction_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("edi_evaluation")
    except (OSError, json.JSONDecodeError):
        return None
