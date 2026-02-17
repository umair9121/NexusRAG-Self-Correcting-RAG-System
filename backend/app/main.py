"""
FastAPI application – entry point for the Hybrid RAG backend.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import select

from app.database import async_session, init_db
from app.logger import broadcaster, logger
from app.models import Document
from app.services.document_service import deduplicate_documents
from app.services.ingestion_worker import ingestion_worker
from app.services.ingestion_state import ingestion_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    logger.info("Initializing database...")
    await init_db()

    # Clean up any duplicate document records left over from prior imports
    async with async_session() as db:
        removed = await deduplicate_documents(db)
        if removed:
            logger.info("Removed %d duplicate document record(s).", removed)

    # ── Detect interrupted ingestion from previous run ────────────────────
    # Documents stuck in "ingesting" need to be flagged so the frontend can
    # offer a resume prompt.  We also ensure they have entries in the
    # persistent ingestion_state so /api/ingestion/recovery returns them.
    async with async_session() as db:
        result = await db.execute(
            select(Document).where(Document.status == "ingesting")
        )
        stuck_docs = list(result.scalars().all())
        if stuck_docs:
            logger.warning(
                "%d document(s) were left in 'ingesting' status from a previous run.",
                len(stuck_docs),
            )
            for doc in stuck_docs:
                # If not already tracked in the state file (e.g. older version had no
                # state tracking), synthesize a recovery entry from DB column statuses.
                existing = ingestion_state.get_all()
                if str(doc.id) not in existing:
                    # Determine which pipelines were originally requested (assume all 3)
                    all_pipelines = ["pageindex", "contextual", "graph"]
                    completed = []
                    for p, col in [
                        ("pageindex", doc.pageindex_status),
                        ("contextual", doc.contextual_status),
                        ("graph", doc.graph_status),
                    ]:
                        if col == "done":
                            completed.append(p)
                    ingestion_state.track_job(
                        doc.id, doc.filename, doc.file_path, all_pipelines,
                    )
                    # Mark completed pipelines so remaining_pipelines is correct
                    for p in completed:
                        ingestion_state.mark_pipeline_done(doc.id, p)
                    # Mark the entry as "running" so get_interrupted_jobs picks it up
                    ingestion_state.mark_running(doc.id, "")
                    logger.info(
                        "  → doc %d (%s): completed=%s, will offer resume for remaining",
                        doc.id, doc.filename, completed,
                    )

    interrupted = ingestion_state.get_interrupted_jobs()
    if interrupted:
        logger.info(
            "%d interrupted ingestion job(s) available for recovery.",
            len(interrupted),
        )

    logger.info("Starting ingestion worker...")
    ingestion_worker.start()
    broadcaster.start_background_drain()  # so ingestion logs from worker threads reach UI
    logger.info("Backend ready.")
    broadcaster.log("Backend started", module="system")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Hybrid RAGSystem",
    description="Adaptive RAG backend with PageIndex, Vector, Graph, and CAG retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow Next.js frontend from any origin (so Network URL e.g. http://172.23.144.1:3000 works)
import os
_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    "http://[::1]:3000",
]
_extra = os.getenv("CORS_ORIGINS", "")
if _extra:
    _origins.extend(o.strip() for o in _extra.split(",") if o.strip())
# Allow any origin: use "*" so opening via Next.js "Network" URL works (credentials disabled for "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
from app.routes.documents import router as documents_router
from app.routes.ingestion import router as ingestion_router
from app.routes.query import router as query_router
from app.routes.chat import router as chat_router
from app.routes.system import router as system_router
from app.routes.websocket import router as ws_router

from app.schemas import (
    HardwareProfileResponse,
    PerformanceOptionUpdate,
    PerformanceOptionsResponse,
)
from app.routes.hardware import (
    get_hardware_profile as _get_hardware_profile,
    get_performance_options as _get_performance_options,
    update_performance_option as _update_performance_option,
)

app.include_router(documents_router)
app.include_router(ingestion_router)
app.include_router(query_router)
app.include_router(chat_router)
app.include_router(system_router)
app.include_router(ws_router)

# Hardware/options routes registered on app so they are always available
@app.get("/api/hardware/profile", response_model=HardwareProfileResponse)
async def hardware_profile():
    return await _get_hardware_profile()


@app.get("/api/hardware/options", response_model=PerformanceOptionsResponse)
async def performance_options():
    return await _get_performance_options()


@app.patch("/api/hardware/options", response_model=PerformanceOptionsResponse)
async def performance_options_update(body: PerformanceOptionUpdate):
    return await _update_performance_option(body)


@app.get("/")
async def root():
    return {"message": "Hybrid RAG System API", "docs": "/docs"}


@app.get("/api")
async def api_info():
    """Confirm API is this app and list main endpoints (avoids 404 confusion)."""
    return {
        "api": "local-rag",
        "version": "1.0",
        "endpoints": [
            "GET  /api/documents",
            "POST /api/documents/upload",
            "GET  /api/chat/sessions",
            "GET  /api/system/health",
            "GET  /api/hardware/profile",
            "GET  /api/hardware/options",
            "PATCH /api/hardware/options",
            "POST /api/query",
        ],
    }
