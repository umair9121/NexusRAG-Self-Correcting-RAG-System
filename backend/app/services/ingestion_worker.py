"""
Background ingestion worker using asyncio tasks.
Processes documents sequentially to respect hardware constraints.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

import json
import uuid

from app import config
from app.database import async_session
from app.logger import broadcaster
from app.models import Document
from app.modules.ollama_client import OllamaClient
from app.modules.pageindex import TreeBuilder, minimal_official_tree
from app.modules.contextual import ContextualRetriever, extract_text_from_pdf, sanitize_extracted_text
from app.modules.graph import GraphBuilder, GraphSearcher
from app.modules.knowledge_base import get_knowledge_base
from app.modules.deepdoctection_extract import (
    extract_text_with_deepdoctection,
    is_pageindex_empty,
)
from app.modules.docling_pipeline import validate_and_fix_headers
from app.modules.vlm_extract import (
    extract_text_with_vlm,
    extract_text_with_vlm_image,
    extract_screen_object_model,
    run_ui_compliance_audit,
    run_text_only_compliance_audit,
    get_per_page_visual_summaries,
    inject_vlm_summaries_into_ocr_markdown,
    unload_vlm_from_vram,
    should_use_vlm_for_sparse_pages,
)
from app.modules.image_extraction import (
    extract_image_description,
    save_extraction,
)
from app.modules.pdf_images import extract_pdf_page_images
from app.modules.pdf_triage import (
    get_document_route,
    ROUTE_FAST_PAGEINDEX,
    ROUTE_OCR_SCANNED,
    ROUTE_MIXED_LAYOUT,
    ROUTE_COMPLEX_VISUAL,
)
from app.modules.ocr_extract import extract_text_with_fast_ocr
from app.modules.text_extract import extract_text_from_file
from app.services.ingestion_state import ingestion_state

logger = logging.getLogger(__name__)


def _is_regulatory_document(doc) -> bool:
    """True if document should be forced into Graph RAG (regulatory/EDI/HIPAA/WCAG)."""
    tags_raw = (doc.tags or "[]").strip()
    try:
        tags = json.loads(tags_raw) if tags_raw else []
    except (json.JSONDecodeError, TypeError):
        tags = []
    tags_lower = [str(t).lower() for t in tags]
    filename_lower = (getattr(doc, "filename", "") or "").lower()
    for tag in getattr(config, "REGULATORY_TAGS", []):
        if tag.lower() in tags_lower or tag.lower() in filename_lower:
            return True
    for pattern in getattr(config, "REGULATORY_FILENAME_PATTERNS", []):
        if pattern.lower() in filename_lower:
            return True
    return False


def _is_image_only(path: str) -> bool:
    """True if file is an image (screenshot) for UI/SOM pipeline."""
    return Path(path).suffix.lower() in getattr(config, "IMAGE_EXTENSIONS", {".png", ".jpg", ".jpeg"})


def _load_view_hierarchy_if_present(file_path: str) -> dict | None:
    """Load optional sidecar JSON (same stem + .view_hierarchy.json or .metadata.json)."""
    base = Path(file_path)
    stem = base.stem
    parent = base.parent
    for name in (f"{stem}.view_hierarchy.json", f"{stem}.metadata.json", f"{stem}.json"):
        p = parent / name
        if p.exists() and p != base:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
    return None


def _text_from_pageindex_tree(doc_stem: str) -> str | None:
    """
    Build contextually flattened text from PageIndex JSON for graph and vector ingestion.
    Each leaf is emitted as one block: hierarchy (Document / Section / Subsection) + summaries
    + content, so embeddings and search get full context (no empty-parent or orphaned-child loss).
    Uses summary in addition to (or instead of) raw text so natural language is indexed.
    Returns None if missing/empty.
    """
    path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    doc_title = (root.get("title") or doc_stem or "Document").strip()
    blocks: list[str] = []

    def walk(node: dict, ancestors: list[dict]) -> None:
        # Official PageIndex uses "nodes"; legacy uses "children"
        children = node.get("nodes") or node.get("children") or []
        if not children:
            # Leaf: build one flattened block with full hierarchy and summaries
            hierarchy_parts = [f"Document: {doc_title}."]
            for a in ancestors:
                if (a.get("title") or "").strip():
                    hierarchy_parts.append(f"Section ({a.get('level', 0)}): {(a.get('title') or '').strip()}.")
                if (a.get("summary") or "").strip():
                    hierarchy_parts.append(f"Summary: {(a.get('summary') or '').strip()}")
            subsection = (node.get("title") or "").strip()
            if subsection:
                hierarchy_parts.append(f"Subsection: {subsection}.")
            summary = (node.get("summary") or "").strip()
            # Use summary only for flattened blocks (official PageIndex style); keep text in tree but don't index it (formatting issues).
            if summary:
                block = " ".join(hierarchy_parts) + " Content: " + summary
                blocks.append(block)
            return
        for ch in children:
            walk(ch, ancestors + [node])

    walk(root, [])
    if not blocks:
        return None
    return config.PAGEINDEX_FLATTENED_BLOCK_DELIMITER.join(blocks)


@dataclass
class IngestionJob:
    document_id: int
    filename: str
    file_path: str
    pipelines: list[str]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "queued"  # queued | running | done | cancelled | error
    current_pipeline: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "pipelines": self.pipelines,
            "status": self.status,
            "current_pipeline": self.current_pipeline,
            "error_message": self.error_message,
        }


class IngestionWorker:
    """Manages a sequential queue of ingestion jobs."""

    def __init__(self):
        self._queue: asyncio.Queue[IngestionJob] = asyncio.Queue()
        self._active_job: IngestionJob | None = None
        self._jobs: dict[int, IngestionJob] = {}  # doc_id -> job
        self._worker_task: asyncio.Task | None = None
        self._llm: OllamaClient | None = None
        self._pause_requested = False
        self._resume_event: asyncio.Event = asyncio.Event()

    def _get_llm(self) -> OllamaClient:
        if self._llm is None:
            self._llm = OllamaClient()
        return self._llm

    def start(self):
        """Start the background worker loop."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Ingestion worker started")

    async def enqueue(self, document_id: int, filename: str, file_path: str, pipelines: list[str]) -> IngestionJob:
        """Add a document to the ingestion queue. Skips if already queued or running."""
        existing = self._jobs.get(document_id)
        if existing and existing.status in ("queued", "running"):
            broadcaster.log(
                f"Already queued or running: {filename} (skipping duplicate)",
                module="ingestion", document_id=document_id, level="WARNING",
            )
            return existing
        # Clear stale finished/errored jobs so re-ingestion works
        if existing and existing.status in ("done", "error", "cancelled"):
            del self._jobs[document_id]
        job = IngestionJob(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            pipelines=pipelines,
        )
        self._jobs[document_id] = job
        await self._queue.put(job)
        ingestion_state.track_job(document_id, filename, file_path, pipelines)
        broadcaster.log(
            f"Queued for ingestion: {filename} (pipelines: {', '.join(pipelines)})",
            module="ingestion", document_id=document_id,
        )
        return job

    def cancel(self, document_id: int) -> bool:
        """Cancel a queued or running job."""
        job = self._jobs.get(document_id)
        if job and job.status in ("queued", "running"):
            job.cancel_event.set()
            job.status = "cancelled"
            ingestion_state.mark_cancelled(document_id)
            broadcaster.log(
                f"Cancellation requested: {job.filename}",
                module="ingestion", document_id=document_id, level="WARNING",
            )
            return True
        return False

    def cancel_all(self) -> list[int]:
        """Cancel all queued and running jobs. Returns list of cancelled document IDs."""
        cancelled_ids: list[int] = []
        for doc_id, job in list(self._jobs.items()):
            if job.status in ("queued", "running"):
                job.cancel_event.set()
                job.status = "cancelled"
                ingestion_state.mark_cancelled(doc_id)
                cancelled_ids.append(doc_id)
                broadcaster.log(
                    f"Cancellation requested: {job.filename}",
                    module="ingestion", document_id=doc_id, level="WARNING",
                )
        return cancelled_ids

    def pause(self) -> None:
        """Request pause: current pipeline runs to completion, then worker and queue stall until resume."""
        self._pause_requested = True
        broadcaster.log("Ingestion pause requested; will pause after current pipeline.", module="ingestion", level="WARNING")

    def unpause(self) -> None:
        """Resume processing after a pause."""
        self._pause_requested = False
        self._resume_event.set()
        broadcaster.log("Ingestion resumed.", module="ingestion")

    async def _wait_if_paused(self, job: IngestionJob) -> None:
        """If pause was requested, wait until unpause (or job cancelled). Called at pipeline boundaries."""
        while self._pause_requested and not job.cancel_event.is_set():
            self._resume_event.clear()
            await self._resume_event.wait()
        self._resume_event.clear()

    def get_status(self) -> dict[str, Any]:
        return {
            "active_job": self._active_job.to_dict() if self._active_job else None,
            "queue_length": self._queue.qsize(),
            "queue": [j.to_dict() for j in self._jobs.values() if j.status == "queued"],
            "paused": self._pause_requested,
        }

    def get_active_or_queued_document_ids(self) -> set[int]:
        """Document IDs that are currently being processed or queued. Used to exclude them from recovery (they are not interrupted)."""
        ids: set[int] = set()
        if self._active_job:
            ids.add(self._active_job.document_id)
        for job in self._jobs.values():
            if job.status == "queued":
                ids.add(job.document_id)
        return ids

    async def _worker_loop(self):
        """Main worker loop: process jobs sequentially."""
        while True:
            try:
                while self._pause_requested:
                    self._resume_event.clear()
                    await self._resume_event.wait()
                self._resume_event.clear()

                job = await self._queue.get()
                if job.cancel_event.is_set():
                    self._queue.task_done()
                    continue

                self._active_job = job
                job.status = "running"
                await self._process_job(job)
                self._active_job = None
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Worker loop error: %s", exc)
                self._active_job = None

    async def _process_job(self, job: IngestionJob):
        """Process a single ingestion job across all selected pipelines."""
        llm = self._get_llm()
        doc_id = job.document_id
        file_path = job.file_path
        cb = broadcaster.make_callback("ingestion", document_id=doc_id)

        async with async_session() as db:
            doc = await db.get(Document, doc_id)
            if not doc:
                job.status = "error"
                job.error_message = "Document not found in database"
                return
            doc.status = "ingesting"
            doc.ingestion_duration_seconds = None  # clear so re-ingestion gets a fresh duration
            await db.commit()

            # Force regulatory documents into Graph RAG pipeline (EDI, WCAG, HIPAA).
            if _is_regulatory_document(doc):
                if getattr(config, "REGULATORY_GRAPH_ONLY", False):
                    job.pipelines = ["graph"]
                    broadcaster.log(
                        f"Regulatory doc: forcing pipelines to graph-only for {job.filename}",
                        module="ingestion", document_id=doc_id,
                    )
                else:
                    if "graph" not in job.pipelines:
                        job.pipelines = list(job.pipelines) + ["graph"]
                    broadcaster.log(
                        f"Regulatory doc: ensuring Graph RAG for {job.filename}",
                        module="ingestion", document_id=doc_id,
                    )

            ingestion_start_time = datetime.now(timezone.utc)
            total_paused_seconds = 0.0  # exclude pause time from ingestion duration

            pipelines_list = [p for p in ["pageindex", "contextual", "graph"] if p in job.pipelines]
            total_pipelines = len(pipelines_list)
            broadcaster.log(
                f"Starting ingestion for {job.filename} ({total_pipelines} pipelines: {', '.join(pipelines_list)})",
                module="ingestion", document_id=doc_id,
            )

            try:
                doc_stem = Path(file_path).stem
                fallback_text: str | None = None
                pageindex_handled_by_ocr = False
                is_pdf = Path(file_path).suffix.lower() == ".pdf"
                is_image = _is_image_only(file_path)

                # ── Image-only (UI screenshot): VLM UI analysis → WCAG RAG compliance audit ──
                if is_image:
                    view_hierarchy = _load_view_hierarchy_if_present(file_path)

                    # Step 1: Rich VLM UI description (notebook-style: full screenshot + status bar)
                    broadcaster.log(
                        "Step 1/4: Extracting detailed UI description from screenshot…",
                        module="ingestion", document_id=doc_id,
                    )
                    extraction_data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: extract_image_description(file_path, callback=cb)
                    )
                    ui_description = (extraction_data.get("merged_description") or "").strip()
                    fallback_text = ui_description
                    broadcaster.log(
                        f"UI description extracted: {len(ui_description)} chars",
                        module="ingestion", document_id=doc_id,
                    )

                    # For screenshot compliance, UI-description generation is mandatory.
                    # If VLM returns empty output, fail fast instead of silently switching
                    # to OCR-like extraction behavior.
                    if not ui_description:
                        msg = (
                            "Screenshot UI analysis returned empty output. "
                            "Cannot run WCAG compliance reasoning without a UI description."
                        )
                        broadcaster.log(msg, module="ingestion", document_id=doc_id, level="ERROR")
                        await db.refresh(doc)
                        doc.status = "error"
                        doc.ingestion_type = "screenshot"
                        doc.error_message = msg
                        if "pageindex" in job.pipelines:
                            doc.pageindex_status = "error"
                        if "contextual" in job.pipelines:
                            doc.contextual_status = "error"
                        if "graph" in job.pipelines:
                            doc.graph_status = "error"
                        await db.commit()
                        job.status = "error"
                        job.error_message = msg
                        ingestion_state.mark_error(doc_id)
                        return

                    # Save extraction metadata for later reference
                    if extraction_data:
                        try:
                            await asyncio.get_event_loop().run_in_executor(
                                None, lambda: save_extraction(doc_stem, extraction_data)
                            )
                        except Exception:
                            pass

                    # Step 2: SOM extraction (optional structural model for audit)
                    som_json: dict = {}
                    try:
                        broadcaster.log(
                            "Step 2/4: Extracting Screen Object Model (SOM)…",
                            module="ingestion", document_id=doc_id,
                        )
                        som_json = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: extract_screen_object_model(file_path, view_hierarchy, callback=cb)
                        )
                        n_components = len(som_json.get("components_flat", [])) if som_json else 0
                        broadcaster.log(
                            f"SOM extraction complete: {n_components} component(s) detected",
                            module="ingestion", document_id=doc_id,
                        )
                    except Exception as som_exc:
                        broadcaster.log(
                            f"SOM extraction failed (non-critical): {som_exc}",
                            module="ingestion", document_id=doc_id, level="WARNING",
                        )

                    # Free VRAM: unload VLM before switching to embedding model for WCAG retrieval
                    await asyncio.get_event_loop().run_in_executor(None, unload_vlm_from_vram)

                    # Step 3: WCAG-grounded compliance audit using the UI description
                    # and retrieved WCAG criteria from the RAG pipeline.
                    # Prefer the VLM-based audit when it works; if it returns an empty
                    # result (common with very small VLMs), fall back to a text-only
                    # audit that uses the UI description + WCAG knowledge base with
                    # the main LLM model.
                    try:
                        broadcaster.log(
                            "Step 3/4: Running WCAG compliance audit (retrieving WCAG criteria + VLM audit)…",
                            module="ingestion", document_id=doc_id,
                        )
                        audit_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: run_ui_compliance_audit(
                                file_path,
                                ui_description=ui_description or None,
                                som_json=som_json if som_json else None,
                                callback=cb,
                            ),
                        )
                        if audit_result and (
                            audit_result.get("detected_violations") is not None
                            or audit_result.get("audit_metadata")
                        ):
                            await db.refresh(doc)
                            doc.audit_result = json.dumps(audit_result, ensure_ascii=False)
                            doc.ingestion_type = "screenshot"
                            await db.commit()
                            n_violations = len(audit_result.get("detected_violations", []))
                            status = audit_result.get("audit_metadata", {}).get("overall_compliance_status", "Unknown")
                            broadcaster.log(
                                f"WCAG compliance audit saved: {status}, {n_violations} violation(s)",
                                module="ingestion", document_id=doc_id,
                            )
                        else:
                            # VLM audit returned empty / unparseable result; fall back to
                            # a text-only audit that uses the UI description and WCAG KB.
                            broadcaster.log(
                                "WCAG audit returned empty result from VLM — running text-only WCAG audit with main LLM…",
                                module="ingestion", document_id=doc_id, level="WARNING",
                            )
                            text_audit = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: run_text_only_compliance_audit(
                                    ui_description or "",
                                    som_json if som_json else None,
                                    callback=cb,
                                ),
                            )
                            if text_audit and (
                                text_audit.get("detected_violations") is not None
                                or text_audit.get("audit_metadata")
                            ):
                                await db.refresh(doc)
                                doc.audit_result = json.dumps(text_audit, ensure_ascii=False)
                                doc.ingestion_type = "screenshot"
                                await db.commit()
                                n_violations = len(text_audit.get("detected_violations", []))
                                status = text_audit.get("audit_metadata", {}).get("overall_compliance_status", "Unknown")
                                broadcaster.log(
                                    f"Text-only WCAG audit saved: {status}, {n_violations} violation(s)",
                                    module="ingestion", document_id=doc_id,
                                )
                            else:
                                broadcaster.log(
                                    "WCAG audit returned empty result even in text-only mode (see logs for details).",
                                    module="ingestion", document_id=doc_id, level="WARNING",
                                )
                    except Exception as audit_exc:
                        broadcaster.log(
                            f"WCAG compliance audit failed: {audit_exc}",
                            module="ingestion", document_id=doc_id, level="WARNING",
                        )

                    # Step 4: PageIndex for screenshot: minimal tree from VLM description
                    if "pageindex" in job.pipelines and not job.cancel_event.is_set() and (fallback_text or "").strip():
                        job.current_pipeline = "pageindex"
                        ingestion_state.mark_running(doc_id, "pageindex")
                        broadcaster.log(
                            "Step 4/4: PageIndex — building minimal tree from VLM UI description",
                            module="ingestion", document_id=doc_id,
                        )
                        await db.refresh(doc)
                        doc.pageindex_status = "running"
                        await db.commit()
                        try:
                            config.PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
                            pageindex_json_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
                            minimal_root = minimal_official_tree(doc_stem, fallback_text or "")
                            with open(pageindex_json_path, "w", encoding="utf-8") as f:
                                json.dump(minimal_root, f, indent=2, ensure_ascii=False)
                            await db.refresh(doc)
                            doc.pageindex_status = "done"
                            await db.commit()
                            ingestion_state.mark_pipeline_done(doc_id, "pageindex")
                            broadcaster.log(f"PageIndex (VLM UI description) complete for {job.filename}", module="pageindex", document_id=doc_id)
                        except Exception as exc:
                            await db.refresh(doc)
                            doc.pageindex_status = "error"
                            doc.error_message = str(exc)
                            await db.commit()
                            broadcaster.log(f"PageIndex FAILED: {exc}", module="pageindex", document_id=doc_id, level="ERROR")
                # End of image-only block; contextual and graph below use fallback_text when is_image

                # Non-PDF (and not already handled as image): extract text once for all pipelines
                if not is_pdf and not is_image:
                    fallback_text = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: extract_text_from_file(file_path, callback=cb)
                    )
                    if not fallback_text or not fallback_text.strip():
                        broadcaster.log(
                            f"No text extracted from {job.filename}; skipping ingestion",
                            module="ingestion", document_id=doc_id, level="WARNING",
                        )

                # PageIndex for non-PDF (not image: image path already did PageIndex above): minimal tree from extracted text
                if not is_pdf and not is_image and "pageindex" in job.pipelines and not job.cancel_event.is_set() and (fallback_text or "").strip():
                    job.current_pipeline = "pageindex"
                    ingestion_state.mark_running(doc_id, "pageindex")
                    step = pipelines_list.index("pageindex") + 1
                    broadcaster.log(
                        f"Pipeline {step}/{total_pipelines}: PageIndex — building minimal tree from extracted text",
                        module="ingestion", document_id=doc_id,
                    )
                    await db.refresh(doc)
                    doc.pageindex_status = "running"
                    await db.commit()
                    try:
                        config.PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
                        pageindex_json_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
                        minimal_root = minimal_official_tree(doc_stem, fallback_text or "")
                        with open(pageindex_json_path, "w", encoding="utf-8") as f:
                            json.dump(minimal_root, f, indent=2, ensure_ascii=False)
                        await db.refresh(doc)
                        doc.pageindex_status = "done"
                        await db.commit()
                        ingestion_state.mark_pipeline_done(doc_id, "pageindex")
                        broadcaster.log(f"PageIndex (minimal tree) complete for {job.filename}", module="pageindex", document_id=doc_id)
                    except Exception as exc:
                        await db.refresh(doc)
                        doc.pageindex_status = "error"
                        doc.error_message = str(exc)
                        await db.commit()
                        broadcaster.log(f"PageIndex FAILED: {exc}", module="pageindex", document_id=doc_id, level="ERROR")

                # PageIndex (digital PDF path, or when Docling-first was skipped/failed)
                if is_pdf and "pageindex" in job.pipelines and not job.cancel_event.is_set() and not pageindex_handled_by_ocr:
                    job.current_pipeline = "pageindex"
                    ingestion_state.mark_running(doc_id, "pageindex")
                    step = pipelines_list.index("pageindex") + 1
                    await db.refresh(doc)
                    doc.pageindex_status = "running"
                    await db.commit()

                    # Triage-driven routing (when enabled): fast_pageindex | ocr_scanned | mixed_layout | complex_visual
                    triage_handled = False
                    if getattr(config, "PDF_TRIAGE_ENABLED", False):
                        triage_result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: get_document_route(file_path)
                        )
                        broadcaster.log(
                            f"PDF triage: {triage_result.route} — {triage_result.reason}",
                            module="ingestion", document_id=doc_id,
                        )
                        logger.info(
                            "ingestion: triage route=%s summary=%s", triage_result.route, triage_result.summary,
                        )

                        if triage_result.route in (
                            ROUTE_OCR_SCANNED,
                            ROUTE_MIXED_LAYOUT,
                            ROUTE_COMPLEX_VISUAL,
                        ):
                            def _run_ocr_with_vlm_context():
                                ocr_md = extract_text_with_fast_ocr(file_path, callback=cb)
                                if not ocr_md or not ocr_md.strip():
                                    return None
                                summaries = get_per_page_visual_summaries(file_path, callback=cb)
                                if summaries:
                                    ocr_md = inject_vlm_summaries_into_ocr_markdown(ocr_md, summaries)
                                ocr_md = validate_and_fix_headers(ocr_md)
                                b = TreeBuilder(llm)
                                b.ingest_markdown(ocr_md, doc_name=doc_stem, out_name=doc_stem, callback=cb, pdf_path=file_path)
                                return ocr_md
                            try:
                                combined_text = await asyncio.get_event_loop().run_in_executor(
                                    None, _run_ocr_with_vlm_context
                                )
                                if combined_text:
                                    fallback_text = combined_text
                                    await db.refresh(doc)
                                    doc.pageindex_status = "done"
                                    await db.commit()
                                    ingestion_state.mark_pipeline_done(doc_id, "pageindex")
                                    broadcaster.log(
                                        f"OCR + VLM visual context complete ({len(combined_text)} chars); PageIndex summary context includes VLM",
                                        module="ingestion", document_id=doc_id,
                                    )
                                    triage_handled = True
                            except Exception as exc:
                                broadcaster.log(
                                    f"OCR+VLM context failed: {exc}; falling back",
                                    module="ingestion", document_id=doc_id, level="WARNING",
                                )

                    # fast_pageindex path or fallback: selectable text only (no OCR/VLM), or legacy sparse check
                    if not triage_handled and not (fallback_text and fallback_text.strip() and doc.pageindex_status == "done"):
                        if not getattr(config, "PDF_TRIAGE_ENABLED", False) and should_use_vlm_for_sparse_pages(file_path):
                            broadcaster.log(
                                f"PDF has sparse text per page; running full VLM extraction for {job.filename}",
                                module="ingestion", document_id=doc_id,
                            )
                            try:
                                def _run_vlm_for_sparse():
                                    vlm_md = extract_text_with_vlm(file_path, callback=cb)
                                    if not vlm_md or not vlm_md.strip():
                                        return None
                                    vlm_md = validate_and_fix_headers(vlm_md)
                                    b = TreeBuilder(llm)
                                    b.ingest_markdown(vlm_md, doc_name=doc_stem, out_name=doc_stem, callback=cb, pdf_path=file_path)
                                    return vlm_md
                                vlm_text = await asyncio.get_event_loop().run_in_executor(None, _run_vlm_for_sparse)
                                if vlm_text:
                                    fallback_text = vlm_text
                                    await db.refresh(doc)
                                    doc.pageindex_status = "done"
                                    await db.commit()
                                    ingestion_state.mark_pipeline_done(doc_id, "pageindex")
                                    broadcaster.log(
                                        f"VLM extraction complete for image-heavy PDF ({len(vlm_text)} chars)",
                                        module="ingestion", document_id=doc_id,
                                    )
                            except Exception as exc:
                                broadcaster.log(f"VLM for sparse failed: {exc}; falling back to PageIndex", module="ingestion", document_id=doc_id, level="WARNING")

                        if not (fallback_text and fallback_text.strip() and doc.pageindex_status == "done"):
                            broadcaster.log(
                                f"Pipeline {step}/{total_pipelines}: PageIndex — building section tree and summaries",
                                module="ingestion", document_id=doc_id,
                            )
                            broadcaster.log(f"Starting PageIndex for {job.filename}", module="pageindex", document_id=doc_id)
                            try:
                                builder = TreeBuilder(llm)
                                tree = await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: builder.ingest(file_path, callback=cb)
                                )
                                await broadcaster.flush_thread_queue()
                                await db.refresh(doc)
                                doc.pageindex_status = "done"
                                await db.commit()
                                ingestion_state.mark_pipeline_done(doc_id, "pageindex")
                                broadcaster.log(f"PageIndex complete for {job.filename}", module="pageindex", document_id=doc_id)
                            except Exception as exc:
                                await db.refresh(doc)
                                doc.pageindex_status = "error"
                                doc.error_message = str(exc)
                                await db.commit()
                                broadcaster.log(f"PageIndex FAILED: {exc}", module="pageindex", document_id=doc_id, level="ERROR")

                # If PageIndex produced "Empty Document", run fast OCR then deepdoctection fallback (PDF only)
                pageindex_json_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
                if (
                    is_pdf
                    and is_pageindex_empty(pageindex_json_path)
                    and (not fallback_text or not fallback_text.strip())
                    and not job.cancel_event.is_set()
                ):
                    broadcaster.log(
                        f"PageIndex empty for {job.filename}; running fast OCR pipeline",
                        module="ingestion", document_id=doc_id,
                    )
                    try:
                        ocr_md = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: extract_text_with_fast_ocr(file_path, callback=cb)
                        )
                        if ocr_md and ocr_md.strip():
                            ocr_md = validate_and_fix_headers(ocr_md)
                            builder = TreeBuilder(llm)
                            builder.ingest_markdown(
                                ocr_md, doc_name=doc_stem, out_name=doc_stem, callback=cb, pdf_path=file_path
                            )
                            fallback_text = ocr_md
                            broadcaster.log(
                                f"Fast OCR produced Markdown; PageIndex tree updated ({len(fallback_text)} chars)",
                                module="ingestion", document_id=doc_id,
                            )
                    except Exception as exc:
                        broadcaster.log(f"Fast OCR fallback failed: {exc}", module="ingestion", document_id=doc_id, level="WARNING")

                    # deepdoctection (last resort — minimal tree + text for embeddings)
                    if (not fallback_text or not fallback_text.strip()) and not job.cancel_event.is_set():
                        broadcaster.log(
                            f"Fast OCR produced no output; trying deepdoctection fallback",
                            module="ingestion", document_id=doc_id,
                        )
                        fallback_text = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: extract_text_with_deepdoctection(
                                file_path,
                                callback=cb,
                                doc_stem=doc_stem,
                                extract_figures=True,
                            ),
                        )
                        if fallback_text and fallback_text.strip():
                            config.PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
                            minimal_root = minimal_official_tree(doc_stem, fallback_text)
                            with open(pageindex_json_path, "w", encoding="utf-8") as f:
                                json.dump(minimal_root, f, indent=2, ensure_ascii=False)
                            broadcaster.log(
                                f"deepdoctection extracted text; updated PageIndex with {len(fallback_text)} chars",
                                module="ingestion", document_id=doc_id,
                            )
                        elif not fallback_text or not fallback_text.strip():
                            broadcaster.log(
                                "All extraction methods exhausted; Vector Embeddings Contextual Process/graph may have no content",
                                module="ingestion", document_id=doc_id, level="WARNING",
                            )

                # Extract PDF page images for reasoning-based retrieval + VLM visual answer (when PageIndex exists)
                if is_pdf and not job.cancel_event.is_set() and (config.PAGEINDEX_DIR / f"{doc_stem}.json").exists():
                    out_dir = config.PDF_IMAGES_DIR / doc_stem
                    logger.info(
                        "ingestion: extracting PDF page images doc_stem=%s pdf=%s -> %s",
                        doc_stem, file_path, out_dir.resolve(),
                    )
                    broadcaster.log(
                        f"Extracting PDF page images to {out_dir.resolve()} for VLM visual retrieval",
                        module="ingestion", document_id=doc_id,
                    )

                    def _extract_pdf_images():
                        config.PDF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                        page_images, total = extract_pdf_page_images(file_path, output_dir=out_dir)
                        return len(page_images), total

                    try:
                        n_images, total_pages = await asyncio.get_event_loop().run_in_executor(None, _extract_pdf_images)
                        if n_images:
                            broadcaster.log(
                                f"Extracted {n_images}/{total_pages} PDF page images to {out_dir.resolve()}",
                                module="ingestion", document_id=doc_id,
                            )
                            logger.info(
                                "ingestion: PDF page images saved doc_stem=%s n_images=%d total_pages=%d path=%s",
                                doc_stem, n_images, total_pages, out_dir.resolve(),
                            )
                        else:
                            logger.warning("ingestion: PDF page image extraction produced 0 images for doc_stem=%s", doc_stem)
                    except Exception as exc:
                        logger.warning("ingestion: PDF page image extraction failed doc_stem=%s: %s", doc_stem, exc)
                        broadcaster.log(
                            f"PDF page image extraction failed: {exc}",
                            module="ingestion", document_id=doc_id, level="WARNING",
                        )

                pause_start = datetime.now(timezone.utc)
                await self._wait_if_paused(job)
                total_paused_seconds += (datetime.now(timezone.utc) - pause_start).total_seconds()

                # Vector Embeddings Contextual Process
                if "contextual" in job.pipelines and not job.cancel_event.is_set():
                    job.current_pipeline = "contextual"
                    ingestion_state.mark_running(doc_id, "contextual")
                    step = pipelines_list.index("contextual") + 1
                    broadcaster.log(
                        f"Pipeline {step}/{total_pipelines}: Vector Embeddings Contextual Process — chunking, contextualizing, and embedding",
                        module="ingestion", document_id=doc_id,
                    )
                    await db.refresh(doc)
                    doc.contextual_status = "running"
                    await db.commit()
                    broadcaster.log(f"Starting Vector Embeddings Contextual Process for {job.filename}", module="contextual", document_id=doc_id)
                    try:
                        retriever = ContextualRetriever(llm)
                        # Prefer contextually flattened PageIndex text when available so vector
                        # chunks get hierarchy + summaries (no empty-parent or orphaned-child loss).
                        text_to_ingest = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: _text_from_pageindex_tree(doc_stem)
                        )
                        text_to_ingest = (text_to_ingest or "").strip()
                        if not text_to_ingest:
                            text_to_ingest = (fallback_text or "").strip()
                        if not text_to_ingest and not is_pdf:
                            text_to_ingest = (await asyncio.get_event_loop().run_in_executor(
                                None, lambda: extract_text_from_file(file_path, callback=cb)
                            ) or "").strip()
                        if text_to_ingest:
                            count = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: retriever.ingest_text(doc_stem, text_to_ingest, callback=cb)
                            )
                        else:
                            count = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: retriever.ingest(file_path, callback=cb)
                            )
                        await broadcaster.flush_thread_queue()
                        await db.refresh(doc)
                        doc.contextual_status = "done"
                        doc.chunk_count = count
                        await db.commit()
                        ingestion_state.mark_pipeline_done(doc_id, "contextual")
                        broadcaster.log(f"Vector Embeddings Contextual Process complete: {count} chunks indexed", module="contextual", document_id=doc_id)
                    except Exception as exc:
                        await db.refresh(doc)
                        doc.contextual_status = "error"
                        doc.error_message = str(exc)
                        await db.commit()
                        broadcaster.log(f"Vector Embeddings Contextual Process FAILED: {exc}", module="contextual", document_id=doc_id, level="ERROR")

                pause_start = datetime.now(timezone.utc)
                await self._wait_if_paused(job)
                total_paused_seconds += (datetime.now(timezone.utc) - pause_start).total_seconds()

                # Graph RAG
                if "graph" in job.pipelines and not job.cancel_event.is_set():
                    job.current_pipeline = "graph"
                    ingestion_state.mark_running(doc_id, "graph")
                    step = pipelines_list.index("graph") + 1
                    broadcaster.log(
                        f"Pipeline {step}/{total_pipelines}: Graph RAG — extracting entities, building graph, summarizing communities",
                        module="ingestion", document_id=doc_id,
                    )
                    await db.refresh(doc)
                    doc.graph_status = "running"
                    await db.commit()
                    broadcaster.log(f"Starting Graph RAG for {job.filename}", module="graph", document_id=doc_id)
                    try:
                        full_text = (fallback_text if fallback_text and fallback_text.strip() else None)
                        if not full_text and is_pdf:
                            pageindex_text = _text_from_pageindex_tree(doc_stem)
                            if pageindex_text and pageindex_text.strip():
                                full_text = pageindex_text
                                broadcaster.log(
                                    f"Using PageIndex tree text for graph ({len(full_text)} chars)",
                                    module="graph", document_id=doc_id,
                                )
                        if not full_text:
                            full_text = (
                                extract_text_from_file(file_path, callback=cb)
                                if not is_pdf
                                else extract_text_from_pdf(file_path)
                            )
                        if full_text:
                            full_text = sanitize_extracted_text(full_text)
                        if not full_text or not full_text.strip():
                            broadcaster.log(
                                "No text available for graph; skipping entity extraction",
                                module="graph", document_id=doc_id, level="WARNING",
                            )
                        graph_builder = GraphBuilder(llm)
                        stats = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: graph_builder.ingest(full_text, doc_name=Path(file_path).stem, callback=cb)
                        )
                        await broadcaster.flush_thread_queue()
                        await db.refresh(doc)
                        doc.graph_status = "done"
                        doc.node_count = stats.get("nodes", 0)
                        doc.edge_count = stats.get("edges", 0)
                        await db.commit()
                        ingestion_state.mark_pipeline_done(doc_id, "graph")
                        broadcaster.log(f"Graph complete: {stats}", module="graph", document_id=doc_id)

                        # Update global knowledge base taxonomy from the new graph
                        doc_stem_kb = Path(file_path).stem
                        logger.info("ingestion: KB update START doc_stem=%s doc_id=%s graph_nodes=%d", doc_stem_kb, doc_id, graph_builder.graph.number_of_nodes())
                        try:
                            kb = get_knowledge_base()
                            kb_stats = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: kb.update_from_graph(doc_stem_kb, graph_builder.graph, callback=cb)
                            )
                            logger.info(
                                "ingestion: KB update DONE doc_stem=%s doc_id=%s new_entities=%d updated_entities=%d new_relations=%d",
                                doc_stem_kb, doc_id, kb_stats["new_entities"], kb_stats["updated_entities"], kb_stats["new_relations"],
                            )
                            broadcaster.log(
                                f"Knowledge base updated: +{kb_stats['new_entities']} entities, +{kb_stats['new_relations']} relations",
                                module="knowledge_base", document_id=doc_id,
                            )
                        except Exception as kb_exc:
                            logger.warning("ingestion: KB update FAILED doc_stem=%s doc_id=%s: %s", doc_stem_kb, doc_id, kb_exc)
                            broadcaster.log(f"Knowledge base update warning: {kb_exc}", module="knowledge_base", document_id=doc_id, level="WARNING")
                    except Exception as exc:
                        await db.refresh(doc)
                        doc.graph_status = "error"
                        doc.error_message = str(exc)
                        await db.commit()
                        broadcaster.log(f"Graph FAILED: {exc}", module="graph", document_id=doc_id, level="ERROR")

                # Mark pipelines that weren't run as skipped only if still pending (don't overwrite done/error)
                await db.refresh(doc)
                if "pageindex" not in job.pipelines and doc.pageindex_status == "pending":
                    doc.pageindex_status = "skipped"
                if "contextual" not in job.pipelines and doc.contextual_status == "pending":
                    doc.contextual_status = "skipped"
                if "graph" not in job.pipelines and doc.graph_status == "pending":
                    doc.graph_status = "skipped"

                if job.cancel_event.is_set():
                    doc.status = "uploaded"
                    doc.error_message = ""
                    # Reset all pipeline statuses — the cancel-all route
                    # cleans up artefacts; we reset DB statuses here too
                    # to avoid races where a just-finished pipeline overwrites
                    # the route's cleanup.
                    doc.pageindex_status = "pending"
                    doc.contextual_status = "pending"
                    doc.graph_status = "pending"
                    doc.chunk_count = 0
                    doc.node_count = 0
                    doc.edge_count = 0
                    job.status = "cancelled"
                    ingestion_state.mark_cancelled(doc_id)
                else:
                    # Ready when every pipeline that was requested for this job is done
                    def _status_for(p: str) -> str:
                        if p == "pageindex":
                            return doc.pageindex_status
                        if p == "contextual":
                            return doc.contextual_status
                        if p == "graph":
                            return doc.graph_status
                        return "pending"
                    all_requested_done = all(
                        _status_for(p) == "done" for p in job.pipelines
                    )
                    if all_requested_done:
                        doc.status = "ready"
                        elapsed = (datetime.now(timezone.utc) - ingestion_start_time).total_seconds()
                        doc.ingestion_duration_seconds = int(max(0, elapsed - total_paused_seconds))
                    else:
                        doc.status = "uploaded"
                    job.status = "done"
                    ingestion_state.mark_done(doc_id)

                await db.commit()
                broadcaster.log(f"Ingestion complete for {job.filename} (status: {job.status})", module="ingestion", document_id=doc_id)

            except Exception as exc:
                logger.error("Ingestion job failed for doc %d: %s", doc_id, exc)
                job.status = "error"
                job.error_message = str(exc)
                ingestion_state.mark_error(doc_id)
                try:
                    await db.refresh(doc)
                    doc.status = "error"
                    doc.error_message = str(exc)
                    await db.commit()
                except Exception:
                    pass
                broadcaster.log(f"Ingestion FAILED: {exc}", module="ingestion", document_id=doc_id, level="ERROR")


# Global worker singleton
ingestion_worker = IngestionWorker()
