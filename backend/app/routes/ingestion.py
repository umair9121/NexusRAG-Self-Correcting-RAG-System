"""
Ingestion API routes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.schemas import IngestionRequest, IngestionStatusOut
from app.services.document_service import get_document
from app.logger import broadcaster
from app.services.ingestion_worker import ingestion_worker
from app.services.ingestion_state import ingestion_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pipelines_to_run_for_doc(doc, requested: list[str]) -> list[str]:
    """Return which pipelines to run when only_failed=True: those in requested that are error or pending."""
    run = []
    if "pageindex" in requested and doc.pageindex_status in ("error", "pending"):
        run.append("pageindex")
    if "contextual" in requested and doc.contextual_status in ("error", "pending"):
        run.append("contextual")
    if "graph" in requested and doc.graph_status in ("error", "pending"):
        run.append("graph")
    return run


def _cleanup_pipeline_data(doc_stem: str, pipeline: str) -> None:
    """Remove stored artefacts for a single pipeline so it can be re-run cleanly."""
    if pipeline == "pageindex":
        pi_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
        if pi_path.exists():
            pi_path.unlink()
            logger.info("Cleaned up PageIndex data for %s", doc_stem)
        # Also clean up extracted figures
        try:
            import shutil
            figures_dir = config.FIGURES_DIR / doc_stem
            if figures_dir.exists() and figures_dir.is_dir():
                shutil.rmtree(figures_dir)
                logger.info("Cleaned up figures for %s", doc_stem)
        except Exception as exc:
            logger.warning("Figures cleanup for %s failed: %s", doc_stem, exc)

    elif pipeline == "graph":
        for suffix in ["_graph.json", "_communities.json"]:
            gp = config.GRAPH_DIR / f"{doc_stem}{suffix}"
            if gp.exists():
                gp.unlink()
        logger.info("Cleaned up Graph data for %s", doc_stem)

    elif pipeline == "contextual":
        # Contextual retriever already cleans up on re-ingest (ingest_text deletes
        # existing chunks), but we do it proactively here for clarity.
        try:
            from app.modules.contextual import DocStore, _delete_chroma_by_doc_name
            _delete_chroma_by_doc_name(doc_stem)
            DocStore().delete_by_doc_name(doc_stem)
            logger.info("Cleaned up Contextual data for %s", doc_stem)
        except Exception as exc:
            logger.warning("Contextual cleanup for %s failed: %s", doc_stem, exc)


# ── Standard routes ──────────────────────────────────────────────────────────


@router.post("/start")
async def start_ingestion(
    req: IngestionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start ingestion for one or more documents. If only_failed=True, run only pipelines that are error or pending for each doc."""
    queued = []
    requested = req.pipelines or ["pageindex", "contextual", "graph"]
    for doc_id in req.document_ids:
        doc = await get_document(db, doc_id)
        if not doc:
            continue
        pipelines = _pipelines_to_run_for_doc(doc, requested) if req.only_failed else requested
        if not pipelines:
            continue
        # Skip Graph RAG for documents at or above the page threshold (e.g. very long textbooks)
        if (
            config.GRAPH_SKIP_PAGE_THRESHOLD > 0
            and doc.page_count is not None
            and doc.page_count >= config.GRAPH_SKIP_PAGE_THRESHOLD
            and "graph" in pipelines
        ):
            pipelines = [p for p in pipelines if p != "graph"]
        if not pipelines:
            continue
        job = await ingestion_worker.enqueue(
            document_id=doc.id,
            filename=doc.filename,
            file_path=doc.file_path,
            pipelines=pipelines,
        )
        queued.append(job.to_dict())
    return {"queued": queued, "count": len(queued)}


@router.get("/status", response_model=IngestionStatusOut)
async def ingestion_status():
    """Get current ingestion queue status."""
    return ingestion_worker.get_status()


@router.post("/cancel/{doc_id}")
async def cancel_ingestion(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Cancel ingestion for a specific document. Idempotent: returns 200 even if already cancelled."""
    success = ingestion_worker.cancel(doc_id)
    if success:
        doc = await get_document(db, doc_id)
        if doc:
            doc_stem = Path(doc.file_path).stem
            # Clean up any pipeline data that was created during partial ingestion
            for pipeline in ["pageindex", "contextual", "graph"]:
                _cleanup_pipeline_data(doc_stem, pipeline)
            # Reset all pipeline statuses to pending
            doc.pageindex_status = "pending"
            doc.contextual_status = "pending"
            doc.graph_status = "pending"
            doc.chunk_count = 0
            doc.node_count = 0
            doc.edge_count = 0
            doc.status = "uploaded"
            doc.error_message = ""
            await db.commit()
    return {"detail": "Cancellation requested" if success else "No active/queued job for this document"}


@router.post("/cancel-all")
async def cancel_all_ingestion(db: AsyncSession = Depends(get_db)):
    """Cancel all active and queued ingestion jobs. Cleans up all pipeline data for cancelled documents."""
    cancelled_ids = ingestion_worker.cancel_all()
    cleaned_docs: list[dict] = []
    for doc_id in cancelled_ids:
        doc = await get_document(db, doc_id)
        if doc:
            doc_stem = Path(doc.file_path).stem
            # Clean up any pipeline data that was created during partial ingestion
            for pipeline in ["pageindex", "contextual", "graph"]:
                _cleanup_pipeline_data(doc_stem, pipeline)
            # Reset all pipeline statuses to pending
            doc.pageindex_status = "pending"
            doc.contextual_status = "pending"
            doc.graph_status = "pending"
            doc.chunk_count = 0
            doc.node_count = 0
            doc.edge_count = 0
            doc.status = "uploaded"
            doc.error_message = ""
            cleaned_docs.append({"document_id": doc.id, "filename": doc.filename})
    await db.commit()
    broadcaster.log(
        f"Ingestion cancelled for {len(cleaned_docs)} document(s). All pipeline data cleaned up.",
        module="ingestion", level="WARNING",
    )
    return {
        "detail": f"Cancelled {len(cleaned_docs)} document(s)",
        "cancelled_count": len(cleaned_docs),
        "cancelled_documents": cleaned_docs,
    }


@router.post("/pause")
async def pause_ingestion():
    """Pause ingestion: current pipeline runs to completion, then no further pipelines or queued jobs run until unpause."""
    ingestion_worker.pause()
    return {"detail": "Pause requested"}


@router.post("/unpause")
async def unpause_ingestion():
    """Resume ingestion after a pause."""
    ingestion_worker.unpause()
    return {"detail": "Resumed"}


# ── Recovery routes (resume after crash) ─────────────────────────────────────


@router.get("/recovery")
async def get_recovery_jobs(db: AsyncSession = Depends(get_db)):
    """Return interrupted ingestion jobs that can be resumed.

    Only includes jobs that are in the persistent state but NOT currently
    in the worker queue (i.e. interrupted by backend restart). Navigating
    to other pages or refreshing the frontend must not show this modal.
    """
    interrupted = ingestion_state.get_interrupted_jobs()
    # Exclude documents that are currently being processed or queued — they are not "interrupted"
    currently_processing = ingestion_worker.get_active_or_queued_document_ids()
    interrupted = [e for e in interrupted if e["document_id"] not in currently_processing]

    # Validate that the documents still exist in the DB
    valid: list[dict[str, Any]] = []
    for entry in interrupted:
        doc = await get_document(db, entry["document_id"])
        if doc and entry.get("remaining_pipelines"):
            valid.append(entry)
        elif not doc:
            # Document was deleted while we were down — clean up state
            ingestion_state.remove(entry["document_id"])
    return {"jobs": valid, "count": len(valid)}


class ResumeRequest(BaseModel):
    document_ids: list[int] | None = None  # None = resume all


@router.post("/resume")
async def resume_ingestion(
    req: ResumeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resume interrupted ingestion jobs.

    For each document:
    1. Determine remaining pipelines (completed ones are skipped).
    2. Clean up artefacts for the first remaining pipeline (it may have
       been partially written when the process died).
    3. Reset the DB per-pipeline statuses for the remaining pipelines.
    4. Enqueue the job with only the remaining pipelines.
    """
    interrupted = ingestion_state.get_interrupted_jobs()
    if req.document_ids is not None:
        id_set = set(req.document_ids)
        interrupted = [j for j in interrupted if j["document_id"] in id_set]

    queued: list[dict[str, Any]] = []
    for entry in interrupted:
        doc_id = entry["document_id"]
        remaining: list[str] = entry.get("remaining_pipelines", [])
        if not remaining:
            ingestion_state.remove(doc_id)
            continue

        doc = await get_document(db, doc_id)
        if not doc:
            ingestion_state.remove(doc_id)
            continue

        doc_stem = Path(doc.file_path).stem

        # Clean up the first remaining pipeline (it may be partially done)
        first_pipeline = remaining[0]
        _cleanup_pipeline_data(doc_stem, first_pipeline)

        # Reset DB per-pipeline statuses for the pipelines we're about to re-run
        for p in remaining:
            if p == "pageindex":
                doc.pageindex_status = "pending"
            elif p == "contextual":
                doc.contextual_status = "pending"
            elif p == "graph":
                doc.graph_status = "pending"
        doc.status = "uploaded"
        doc.error_message = ""
        await db.commit()

        # Remove old state entry so it gets freshly tracked by the worker
        ingestion_state.remove(doc_id)

        # Enqueue
        job = await ingestion_worker.enqueue(
            document_id=doc.id,
            filename=doc.filename,
            file_path=doc.file_path,
            pipelines=remaining,
        )
        queued.append(job.to_dict())

    return {"queued": queued, "count": len(queued)}


class DismissRequest(BaseModel):
    document_ids: list[int] | None = None  # None = dismiss all


@router.post("/dismiss-recovery")
async def dismiss_recovery(req: DismissRequest | None = None):
    """Dismiss recovery entries without resuming.

    Resets their status in the persistent state so the popup won't reappear.
    If document_ids is provided, only those are dismissed; otherwise all are.
    """
    if req and req.document_ids is not None:
        for doc_id in req.document_ids:
            ingestion_state.remove(doc_id)
    else:
        ingestion_state.remove_all()
    return {"detail": "Recovery dismissed"}
