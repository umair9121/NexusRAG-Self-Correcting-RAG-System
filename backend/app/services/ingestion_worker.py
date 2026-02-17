"""
Background ingestion worker using asyncio tasks.
Processes documents sequentially to respect hardware constraints.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
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
from app.modules.pageindex import TreeBuilder
from app.modules.contextual import ContextualRetriever, extract_text_from_pdf, sanitize_extracted_text
from app.modules.graph import GraphBuilder
from app.modules.deepdoctection_extract import (
    extract_text_with_deepdoctection,
    is_pageindex_empty,
)
from app.modules.docling_pipeline import (
    convert_pdf_to_markdown,
    is_docling_available,
    validate_and_fix_headers,
)
from app.modules.vlm_extract import extract_text_with_vlm, should_use_vlm
from app.modules.text_extract import extract_text_from_file
from app.services.ingestion_state import ingestion_state

logger = logging.getLogger(__name__)


def _text_from_pageindex_tree(doc_stem: str) -> str | None:
    """Build full text from PageIndex JSON (leaf text/summary) for graph ingestion. Returns None if missing/empty."""
    path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    parts = []

    def walk(node: dict) -> None:
        text = (node.get("text") or "").strip()
        summary = (node.get("summary") or "").strip()
        if text:
            parts.append(text)
        elif summary:
            parts.append(summary)
        for ch in node.get("children", []):
            walk(ch)

    walk(root)
    if not parts:
        return None
    return "\n\n".join(parts)


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
            await db.commit()

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

                # Non-PDF: extract text once for all pipelines (PageIndex minimal tree, contextual, graph)
                if not is_pdf:
                    fallback_text = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: extract_text_from_file(file_path, callback=cb)
                    )
                    if not fallback_text or not fallback_text.strip():
                        broadcaster.log(
                            f"No text extracted from {job.filename}; skipping ingestion",
                            module="ingestion", document_id=doc_id, level="WARNING",
                        )

                # Detect scanned PDF before PageIndex: no text layer → run Docling (OCR) first
                if is_pdf and "pageindex" in job.pipelines and not job.cancel_event.is_set():
                    blocks = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: TreeBuilder.extract_blocks(file_path)
                    )
                    if len(blocks) == 0 and is_docling_available():
                        broadcaster.log(
                            f"No text layer in {job.filename}; treating as scanned, running Docling (OCR) before PageIndex",
                            module="ingestion", document_id=doc_id,
                        )
                        await db.refresh(doc)
                        doc.pageindex_status = "running"
                        await db.commit()
                        job.current_pipeline = "pageindex"

                        def _run_docling_first():
                            md = convert_pdf_to_markdown(file_path, callback=cb)
                            if not md or not md.strip():
                                return None
                            md = validate_and_fix_headers(md)
                            b = TreeBuilder(llm)
                            b.ingest_markdown(md, doc_name=doc_stem, out_name=doc_stem, callback=cb)
                            return md

                        fallback_text = await asyncio.get_event_loop().run_in_executor(None, _run_docling_first)

                        # ── VLM check: did Docling extract enough text? ──
                        if fallback_text and should_use_vlm(file_path, fallback_text):
                            broadcaster.log(
                                f"Docling output too thin ({len(fallback_text)} chars) "
                                f"relative to file size; escalating to VLM ({config.VLM_MODEL})",
                                module="ingestion", document_id=doc_id,
                            )
                            def _run_vlm_first():
                                vlm_md = extract_text_with_vlm(file_path, callback=cb)
                                if not vlm_md or not vlm_md.strip():
                                    return None
                                vlm_md = validate_and_fix_headers(vlm_md)
                                b = TreeBuilder(llm)
                                b.ingest_markdown(vlm_md, doc_name=doc_stem, out_name=doc_stem, callback=cb)
                                return vlm_md

                            vlm_text = await asyncio.get_event_loop().run_in_executor(None, _run_vlm_first)
                            if vlm_text:
                                fallback_text = vlm_text
                                broadcaster.log(
                                    f"VLM extraction replaced Docling output ({len(fallback_text)} chars)",
                                    module="ingestion", document_id=doc_id,
                                )

                        if fallback_text:
                            pageindex_handled_by_ocr = True
                            await db.refresh(doc)
                            doc.pageindex_status = "done"
                            await db.commit()
                            ingestion_state.mark_pipeline_done(doc_id, "pageindex")
                            broadcaster.log(
                                f"OCR/VLM complete; PageIndex tree built from Markdown ({len(fallback_text)} chars)",
                                module="ingestion", document_id=doc_id,
                            )
                        else:
                            broadcaster.log(
                                "Docling produced no Markdown; will run normal PageIndex and fallbacks",
                                module="ingestion", document_id=doc_id, level="WARNING",
                            )

                # PageIndex for non-PDF: minimal tree from extracted text
                if not is_pdf and "pageindex" in job.pipelines and not job.cancel_event.is_set() and (fallback_text or "").strip():
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
                        minimal_root = {
                            "id": str(uuid.uuid4())[:8],
                            "title": doc_stem,
                            "level": 0,
                            "text": "",
                            "summary": "",
                            "page_numbers": [],
                            "children": [
                                {
                                    "id": str(uuid.uuid4())[:8],
                                    "title": "Content",
                                    "level": 1,
                                    "text": (fallback_text or "")[:500000],
                                    "summary": "",
                                    "page_numbers": [],
                                    "children": [],
                                }
                            ],
                        }
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
                    broadcaster.log(
                        f"Pipeline {step}/{total_pipelines}: PageIndex — building section tree and summaries",
                        module="ingestion", document_id=doc_id,
                    )
                    await db.refresh(doc)
                    doc.pageindex_status = "running"
                    await db.commit()
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

                # If PageIndex produced "Empty Document" (and we didn't already run Docling), run OCR fallback (PDF only)
                pageindex_json_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
                if (
                    is_pdf
                    and
                    is_pageindex_empty(pageindex_json_path)
                    and (not fallback_text or not fallback_text.strip())
                    and not job.cancel_event.is_set()
                ):
                    broadcaster.log(
                        f"PageIndex empty for {job.filename}; running structured OCR pipeline",
                        module="ingestion", document_id=doc_id,
                    )
                    # Step 1: Docling (Markdown hierarchy → PageIndex tree)
                    if is_docling_available():
                        def _run_docling():
                            md = convert_pdf_to_markdown(file_path, callback=cb)
                            if not md or not md.strip():
                                return None
                            md = validate_and_fix_headers(md)
                            builder = TreeBuilder(llm)
                            builder.ingest_markdown(md, doc_name=doc_stem, out_name=doc_stem, callback=cb)
                            return md
                        fallback_text = await asyncio.get_event_loop().run_in_executor(None, _run_docling)
                        if fallback_text:
                            broadcaster.log(
                                f"Docling produced structured Markdown; PageIndex tree updated ({len(fallback_text)} chars)",
                                module="ingestion", document_id=doc_id,
                            )

                    # Step 2: VLM — if Docling produced output that is too thin for
                    # the file size (handwritten, complex layout, etc.)
                    if (
                        fallback_text
                        and should_use_vlm(file_path, fallback_text)
                        and not job.cancel_event.is_set()
                    ):
                        broadcaster.log(
                            f"Docling output too thin ({len(fallback_text)} chars) "
                            f"relative to file size; escalating to VLM ({config.VLM_MODEL})",
                            module="ingestion", document_id=doc_id,
                        )
                        def _run_vlm_postpi():
                            vlm_md = extract_text_with_vlm(file_path, callback=cb)
                            if not vlm_md or not vlm_md.strip():
                                return None
                            vlm_md = validate_and_fix_headers(vlm_md)
                            builder = TreeBuilder(llm)
                            builder.ingest_markdown(vlm_md, doc_name=doc_stem, out_name=doc_stem, callback=cb)
                            return vlm_md

                        vlm_text = await asyncio.get_event_loop().run_in_executor(None, _run_vlm_postpi)
                        if vlm_text:
                            fallback_text = vlm_text
                            broadcaster.log(
                                f"VLM replaced thin Docling output; PageIndex rebuilt ({len(fallback_text)} chars)",
                                module="ingestion", document_id=doc_id,
                            )

                    # Step 3: VLM — if Docling produced nothing at all, try VLM
                    # before falling back to deepdoctection
                    if (not fallback_text or not fallback_text.strip()) and not job.cancel_event.is_set():
                        broadcaster.log(
                            f"Docling unavailable or empty; trying VLM ({config.VLM_MODEL}) before deepdoctection",
                            module="ingestion", document_id=doc_id,
                        )
                        def _run_vlm_standalone():
                            vlm_md = extract_text_with_vlm(file_path, callback=cb)
                            if not vlm_md or not vlm_md.strip():
                                return None
                            vlm_md = validate_and_fix_headers(vlm_md)
                            builder = TreeBuilder(llm)
                            builder.ingest_markdown(vlm_md, doc_name=doc_stem, out_name=doc_stem, callback=cb)
                            return vlm_md

                        fallback_text = await asyncio.get_event_loop().run_in_executor(None, _run_vlm_standalone)
                        if fallback_text:
                            broadcaster.log(
                                f"VLM extracted text; PageIndex tree built ({len(fallback_text)} chars)",
                                module="ingestion", document_id=doc_id,
                            )

                    # Step 4: deepdoctection (last resort — minimal tree + text for embeddings)
                    if (not fallback_text or not fallback_text.strip()) and not job.cancel_event.is_set():
                        broadcaster.log(
                            f"VLM produced no output; trying deepdoctection fallback",
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
                            minimal_root = {
                                "id": str(uuid.uuid4())[:8],
                                "title": doc_stem,
                                "level": 0,
                                "text": "",
                                "summary": "",
                                "page_numbers": [],
                                "children": [
                                    {
                                        "id": str(uuid.uuid4())[:8],
                                        "title": "Document Content",
                                        "level": 1,
                                        "text": fallback_text[:500000],
                                        "summary": "",
                                        "page_numbers": [],
                                        "children": [],
                                    }
                                ],
                            }
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

                await self._wait_if_paused(job)

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

                await self._wait_if_paused(job)

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
                    # Ready only when all three pipelines (PI, VE, GR) are done; otherwise uploaded
                    all_pipelines_done = (
                        doc.pageindex_status == "done"
                        and doc.contextual_status == "done"
                        and doc.graph_status == "done"
                    )
                    doc.status = "ready" if all_pipelines_done else "uploaded"
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
