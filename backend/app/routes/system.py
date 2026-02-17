"""
System health and status endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.models import Document
from app.modules.ollama_client import OllamaClient
from app.modules.graph import GraphSearcher
from app.modules.cag import get_available_ram_gb, CAGManager
from app.schemas import SystemHealthResponse
from app.routes.query import get_cag

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=SystemHealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    llm = OllamaClient()
    connected = llm.is_alive()
    models = llm.list_models() if connected else []

    # Document counts
    total_q = await db.execute(select(func.count(Document.id)))
    total = total_q.scalar() or 0

    ready_q = await db.execute(select(func.count(Document.id)).where(Document.status == "ready"))
    ready = ready_q.scalar() or 0

    ingesting_q = await db.execute(select(func.count(Document.id)).where(Document.status == "ingesting"))
    ingesting = ingesting_q.scalar() or 0

    graph_searcher = GraphSearcher(llm)
    cag = get_cag()

    return SystemHealthResponse(
        ollama_connected=connected,
        ollama_models=models,
        available_ram_gb=round(get_available_ram_gb(), 1),
        total_documents=total,
        documents_ready=ready,
        documents_ingesting=ingesting,
        knowledge_graphs=graph_searcher.list_graphs(),
        cag_status=cag.get_status(),
    )
