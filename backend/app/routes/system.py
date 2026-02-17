"""
System health and status endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.models import Document
from app.modules.ollama_client import OllamaClient
from app.modules.pageindex import TreeSearcher
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

    # Index info
    tree_searcher = TreeSearcher(llm)
    graph_searcher = GraphSearcher(llm)

    cag = get_cag()

    return SystemHealthResponse(
        ollama_connected=connected,
        ollama_models=models,
        available_ram_gb=round(get_available_ram_gb(), 1),
        total_documents=total,
        documents_ready=ready,
        documents_ingesting=ingesting,
        pageindex_trees=tree_searcher.list_trees(),
        knowledge_graphs=graph_searcher.list_graphs(),
        cag_status=cag.get_status(),
    )


@router.post("/reset-vector-store")
async def reset_vector_store(db: AsyncSession = Depends(get_db)):
    """
    Reset the entire vector store (ChromaDB collection + DocStore).
    Deletes all embeddings so a new embedding model can be used on re-ingestion.
    Documents and their files are NOT deleted — only the vector/contextual data.
    All documents' contextual_status is reset to 'pending'.
    """
    from app.modules.contextual import _reset_chroma_collection, DocStore

    # 1. Reset ChromaDB collection (delete & recreate)
    try:
        _reset_chroma_collection()
        chroma_reset = True
    except Exception as exc:
        logger.error("Failed to reset ChromaDB collection: %s", exc)
        chroma_reset = False

    # 2. Clear the DocStore
    try:
        docstore = DocStore()
        old_count = docstore.count()
        docstore.clear()
        docstore_cleared = old_count
    except Exception as exc:
        logger.error("Failed to clear DocStore: %s", exc)
        docstore_cleared = 0

    # 3. Reset all documents' contextual status and chunk count in the DB
    result = await db.execute(
        update(Document)
        .where(Document.contextual_status != "pending")
        .values(contextual_status="pending", chunk_count=0)
    )
    docs_reset = result.rowcount  # type: ignore[union-attr]
    await db.commit()

    logger.info(
        "Vector store reset: chroma=%s, docstore_entries_cleared=%d, documents_reset=%d",
        chroma_reset, docstore_cleared, docs_reset,
    )

    return {
        "success": chroma_reset,
        "chroma_reset": chroma_reset,
        "docstore_entries_cleared": docstore_cleared,
        "documents_reset": docs_reset,
        "message": (
            "Vector store reset successfully. Re-ingest documents to rebuild embeddings with the current model."
            if chroma_reset
            else "Partial reset — ChromaDB collection could not be deleted. Check server logs."
        ),
    }
