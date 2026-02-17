"""
Background ingestion worker using asyncio tasks.
Processes documents sequentially to respect hardware constraints.

Pipeline order:
  1. Extraction — VLM-based image description extraction (for image files)
  2. EDI Evaluation — LLM-based EDI compliance check on extracted description
  3. Graph RAG — entity extraction, graph building, community summarization
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

import json

from app import config
from app.database import async_session
from app.logger import broadcaster
from app.models import Document
from app.modules.ollama_client import OllamaClient
from app.modules.graph import GraphBuilder
from app.modules.image_extraction import (
    extract_image_description,
    is_image_file,
    save_extraction,
    load_extraction,
)
from app.modules.edi_evaluation import (
    evaluate_edi_compliance,
    save_edi_evaluation,
)
from app.modules.text_extract import extract_text_from_file
from app.services.ingestion_state import ingestion_state

logger = logging.getLogger(__name__)


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

            pipelines_list = [p for p in ["extraction", "edi_evaluation", "graph"] if p in job.pipelines]
            total_pipelines = len(pipelines_list)
            broadcaster.log(
                f"Starting ingestion for {job.filename} ({total_pipelines} pipelines: {', '.join(pipelines_list)})",
                module="ingestion", document_id=doc_id,
            )

            try:
                doc_stem = Path(file_path).stem
                is_image = is_image_file(file_path)
                extraction_text: str = ""

                # ── Pipeline 1: Image Extraction ─────────────────────────
                if "extraction" in job.pipelines and not job.cancel_event.is_set():
                    job.current_pipeline = "extraction"
                    ingestion_state.mark_running(doc_id, "extraction")
                    step = pipelines_list.index("extraction") + 1
                    broadcaster.log(
                        f"Pipeline {step}/{total_pipelines}: Extraction — extracting image description with VLM",
                        module="ingestion", document_id=doc_id,
                    )
                    await db.refresh(doc)
                    doc.extraction_status = "running"
                    await db.commit()

                    try:
                        if is_image:
                            extraction_data = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: extract_image_description(file_path, callback=cb)
                            )
                            await broadcaster.flush_thread_queue()

                            if extraction_data and extraction_data.get("merged_description"):
                                extraction_text = extraction_data["merged_description"]
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: save_extraction(doc_stem, extraction_data)
                                )
                                await db.refresh(doc)
                                doc.extraction_status = "done"
                                await db.commit()
                                ingestion_state.mark_pipeline_done(doc_id, "extraction")
                                broadcaster.log(
                                    f"Extraction complete: {len(extraction_text)} chars extracted",
                                    module="extraction", document_id=doc_id,
                                )
                            else:
                                broadcaster.log(
                                    "Extraction produced no description",
                                    module="extraction", document_id=doc_id, level="WARNING",
                                )
                                await db.refresh(doc)
                                doc.extraction_status = "done"
                                await db.commit()
                                ingestion_state.mark_pipeline_done(doc_id, "extraction")
                        else:
                            # Non-image files: extract text for downstream pipelines
                            extraction_text = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: extract_text_from_file(file_path, callback=cb) or ""
                            )
                            await broadcaster.flush_thread_queue()
                            if extraction_text:
                                save_data = {
                                    "merged_description": extraction_text,
                                    "file_type": Path(file_path).suffix.lower(),
                                    "file_path": str(file_path),
                                }
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: save_extraction(doc_stem, save_data)
                                )
                            await db.refresh(doc)
                            doc.extraction_status = "done"
                            await db.commit()
                            ingestion_state.mark_pipeline_done(doc_id, "extraction")
                            broadcaster.log(
                                f"Text extraction complete: {len(extraction_text)} chars",
                                module="extraction", document_id=doc_id,
                            )
                    except Exception as exc:
                        await db.refresh(doc)
                        doc.extraction_status = "error"
                        doc.error_message = str(exc)
                        await db.commit()
                        broadcaster.log(f"Extraction FAILED: {exc}", module="extraction", document_id=doc_id, level="ERROR")

                await self._wait_if_paused(job)

                # ── Pipeline 2: EDI Evaluation ───────────────────────────
                if "edi_evaluation" in job.pipelines and not job.cancel_event.is_set():
                    job.current_pipeline = "edi_evaluation"
                    ingestion_state.mark_running(doc_id, "edi_evaluation")
                    step = pipelines_list.index("edi_evaluation") + 1
                    broadcaster.log(
                        f"Pipeline {step}/{total_pipelines}: EDI Evaluation — assessing EDI compliance",
                        module="ingestion", document_id=doc_id,
                    )
                    await db.refresh(doc)
                    doc.edi_status = "running"
                    await db.commit()

                    try:
                        # Load extraction text if we don't have it yet
                        if not extraction_text:
                            existing_extraction = load_extraction(doc_stem)
                            if existing_extraction:
                                extraction_text = existing_extraction.get("merged_description", "")

                        if extraction_text:
                            edi_result = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: evaluate_edi_compliance(extraction_text, llm, callback=cb)
                            )
                            await broadcaster.flush_thread_queue()

                            await asyncio.get_event_loop().run_in_executor(
                                None, lambda: save_edi_evaluation(doc_stem, edi_result)
                            )

                            await db.refresh(doc)
                            doc.edi_status = "done"
                            doc.edi_rating = edi_result.get("overall_rating", "")
                            await db.commit()
                            ingestion_state.mark_pipeline_done(doc_id, "edi_evaluation")
                            broadcaster.log(
                                f"EDI evaluation complete: {edi_result.get('overall_rating', 'UNKNOWN')}",
                                module="edi", document_id=doc_id,
                            )
                        else:
                            broadcaster.log(
                                "No extraction text available for EDI evaluation; skipping",
                                module="edi", document_id=doc_id, level="WARNING",
                            )
                            await db.refresh(doc)
                            doc.edi_status = "skipped"
                            await db.commit()
                            ingestion_state.mark_pipeline_done(doc_id, "edi_evaluation")
                    except Exception as exc:
                        await db.refresh(doc)
                        doc.edi_status = "error"
                        doc.error_message = str(exc)
                        await db.commit()
                        broadcaster.log(f"EDI Evaluation FAILED: {exc}", module="edi", document_id=doc_id, level="ERROR")

                await self._wait_if_paused(job)

                # ── Pipeline 3: Graph RAG ────────────────────────────────
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
                        # Use extraction text if available, otherwise extract fresh
                        full_text = extraction_text
                        if not full_text:
                            existing_extraction = load_extraction(doc_stem)
                            if existing_extraction:
                                full_text = existing_extraction.get("merged_description", "")
                        if not full_text:
                            full_text = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: extract_text_from_file(file_path, callback=cb) or ""
                            )

                        if not full_text or not full_text.strip():
                            broadcaster.log(
                                "No text available for graph; skipping entity extraction",
                                module="graph", document_id=doc_id, level="WARNING",
                            )
                        graph_builder = GraphBuilder(llm)
                        stats = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: graph_builder.ingest(full_text, doc_name=doc_stem, callback=cb)
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

                # Mark pipelines that weren't run as skipped only if still pending
                await db.refresh(doc)
                if "extraction" not in job.pipelines and doc.extraction_status == "pending":
                    doc.extraction_status = "skipped"
                if "edi_evaluation" not in job.pipelines and doc.edi_status == "pending":
                    doc.edi_status = "skipped"
                if "graph" not in job.pipelines and doc.graph_status == "pending":
                    doc.graph_status = "skipped"

                if job.cancel_event.is_set():
                    doc.status = "uploaded"
                    doc.error_message = ""
                    doc.extraction_status = "pending"
                    doc.edi_status = "pending"
                    doc.graph_status = "pending"
                    doc.node_count = 0
                    doc.edge_count = 0
                    doc.edi_rating = ""
                    job.status = "cancelled"
                    ingestion_state.mark_cancelled(doc_id)
                else:
                    # Ready only when all three pipelines are done; otherwise uploaded
                    all_pipelines_done = (
                        doc.extraction_status in ("done", "skipped")
                        and doc.edi_status in ("done", "skipped")
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
