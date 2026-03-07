"""
WCAG reference data ingestion — converts wcag_data.json into structured
Markdown for the RAG pipeline (PageIndex, contextual vectors, Graph RAG).

The resulting text is organised so each success criterion becomes a
PageIndex-friendly section with its techniques, examples, and test
procedures embedded.  This lets the vector and graph pipelines index
WCAG knowledge at technique-level granularity, which the screenshot
compliance audit can then retrieve at query time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app import config

logger = logging.getLogger(__name__)


def wcag_json_to_markdown(data: dict[str, Any]) -> str:
    """Convert the nested wcag_data.json structure into rich Markdown text.

    Structure: principles > guidelines > success_criteria > techniques
    Each success criterion gets its own ## heading so PageIndex can build
    a meaningful tree, and each technique becomes a ### block with its
    description, examples, and test procedure.
    """
    parts: list[str] = []
    parts.append(f"# WCAG {data.get('version', '2.2')} Compliance Reference\n")
    parts.append(
        f"Source: {data.get('source', 'https://www.w3.org/WAI/WCAG22/Understanding/')}\n"
    )

    for principle in data.get("principles", []):
        p_name = principle.get("name", "Unknown Principle")
        parts.append(f"\n# Principle: {p_name}\n")

        for guideline in principle.get("guidelines", []):
            g_name = guideline.get("name", "Unknown Guideline")
            parts.append(f"\n## Guideline {g_name}\n")

            for sc in guideline.get("success_criteria", []):
                sc_num = sc.get("num", "")
                sc_name = sc.get("name", "")
                parts.append(f"\n### Success Criterion {sc_num}: {sc_name}\n")

                url = sc.get("understanding_url", "")
                if url:
                    parts.append(f"Understanding: {url}\n")

                techniques = sc.get("techniques", {})
                for category in ("sufficient", "advisory", "failures"):
                    techs = techniques.get(category, [])
                    if not techs:
                        continue
                    cat_label = category.capitalize()
                    parts.append(f"\n#### {cat_label} Techniques for {sc_num}\n")

                    for tech in techs:
                        label = tech.get("label", "")
                        tech_url = tech.get("url", "")
                        parts.append(f"\n##### {label}\n")
                        if tech_url:
                            parts.append(f"URL: {tech_url}\n")

                        content = tech.get("content", {})
                        if not content:
                            continue

                        desc = content.get("description", "").strip()
                        if desc:
                            parts.append(f"\n**Description:** {desc}\n")

                        examples = content.get("examples", "").strip()
                        if examples:
                            parts.append(f"\n**Examples:** {examples}\n")

                        test_proc = content.get("test_procedure", "").strip()
                        if test_proc:
                            parts.append(f"\n**Test Procedure:** {test_proc}\n")

    return "\n".join(parts)


def extract_wcag_text(wcag_path: Path | None = None) -> str:
    """Load wcag_data.json and return structured Markdown for ingestion."""
    wcag_path = wcag_path or config.WCAG_DATA_PATH
    if not wcag_path.exists():
        logger.warning("WCAG data file not found: %s", wcag_path)
        return ""
    try:
        with open(wcag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        md = wcag_json_to_markdown(data)
        logger.info(
            "WCAG text extraction: %d chars from %s", len(md), wcag_path.name
        )
        return md
    except Exception as exc:
        logger.error("Failed to extract WCAG text: %s", exc)
        return ""


def is_wcag_ingested() -> bool:
    """Check whether wcag_data has already been ingested (PageIndex + graph exist)."""
    pi = config.PAGEINDEX_DIR / f"{config.WCAG_DOC_STEM}.json"
    graph = config.GRAPH_DIR / f"{config.WCAG_DOC_STEM}_graph.json"
    return pi.exists() and graph.exists()


async def ensure_wcag_ingested(callback=None) -> bool:
    """Run the full RAG pipeline on wcag_data.json if not already ingested.

    Called at startup so the WCAG knowledge base is always available for
    screenshot compliance analysis.  All heavy work is dispatched to a thread
    via ``run_in_executor`` to avoid blocking the event loop.

    Returns True if ingestion ran (or was already done), False on failure.
    """
    import asyncio
    from app.modules.ollama_client import OllamaClient
    from app.modules.pageindex import TreeBuilder
    from app.modules.contextual import ContextualRetriever
    from app.modules.graph import GraphBuilder
    from app.modules.knowledge_base import get_knowledge_base

    if is_wcag_ingested():
        logger.info("WCAG data already ingested — skipping")
        return True

    if not config.WCAG_DATA_PATH.exists():
        logger.warning(
            "wcag_data.json not found at %s — WCAG compliance analysis will be limited",
            config.WCAG_DATA_PATH,
        )
        return False

    def _log(msg: str):
        logger.info("[WCAG ingest] %s", msg)
        if callback:
            callback(msg)

    _log("Extracting WCAG reference text…")
    md_text = extract_wcag_text()
    if not md_text or not md_text.strip():
        _log("No text extracted from wcag_data.json")
        return False

    doc_stem = config.WCAG_DOC_STEM
    loop = asyncio.get_event_loop()

    def _run_full_pipeline():
        llm = OllamaClient()

        # 1. PageIndex
        _log("Building PageIndex tree for WCAG data…")
        try:
            config.PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
            builder = TreeBuilder(llm)
            builder.ingest_markdown(
                md_text, doc_name=doc_stem, out_name=doc_stem, callback=_log,
            )
            _log(f"PageIndex tree saved")
        except Exception as exc:
            _log(f"PageIndex failed: {exc}")
            logger.error("WCAG PageIndex failed: %s", exc)

        # 2. Contextual (vector embeddings)
        _log("Creating vector embeddings for WCAG data…")
        try:
            retriever = ContextualRetriever(llm)
            count = retriever.ingest_text(doc_stem, md_text, callback=_log)
            _log(f"Vector embeddings: {count} chunks indexed")
        except Exception as exc:
            _log(f"Vector embeddings failed: {exc}")
            logger.error("WCAG contextual failed: %s", exc)

        # 3. Graph RAG
        _log("Building knowledge graph for WCAG data…")
        try:
            graph_builder = GraphBuilder(llm)
            stats = graph_builder.ingest(md_text, doc_name=doc_stem, callback=_log)
            _log(f"Graph built: {stats}")

            try:
                kb = get_knowledge_base()
                kb_stats = kb.update_from_graph(
                    doc_stem, graph_builder.graph, callback=_log
                )
                _log(
                    f"Knowledge base updated: +{kb_stats['new_entities']} entities, "
                    f"+{kb_stats['new_relations']} relations"
                )
            except Exception as kb_exc:
                logger.warning("WCAG KB update failed: %s", kb_exc)
        except Exception as exc:
            _log(f"Graph RAG failed: {exc}")
            logger.error("WCAG graph failed: %s", exc)

        _log("WCAG data ingestion complete")

    await loop.run_in_executor(None, _run_full_pipeline)
    return True
