"""
System health and status endpoint.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.models import Document
from app.modules.ollama_client import OllamaClient
from app.modules.pageindex import TreeSearcher
from app.modules.graph import GraphSearcher
from app.modules.cag import get_available_ram_gb, CAGManager
from app.modules.knowledge_base import get_knowledge_base
from app.schemas import SystemHealthResponse, RefreshKnowledgeBaseRequest
from app.routes.query import get_cag
from app.logger import broadcaster

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
        graph_skip_page_threshold=config.GRAPH_SKIP_PAGE_THRESHOLD,
    )


@router.get("/pageindex/documents")
async def pageindex_list_documents():
    """
    List PageIndex documents with summaries and page ranges (official PageIndex-style).
    Use for list-docs → get_structure → get_page_content flow.
    """
    tree_searcher = TreeSearcher(OllamaClient())
    docs = tree_searcher.list_documents_with_summaries()
    return {
        "success": True,
        "docs": docs,
        "ready_count": len(docs),
        "total_shown": len(docs),
        "next_steps": {
            "summary": f"Showing {len(docs)} most recent documents",
            "options": [
                "Use get_document_structure(doc_name) to explore content, then get_page_content() for specific pages."
            ]
            if docs
            else ["Upload and index documents to enable PageIndex search."],
        },
    }


@router.get("/pageindex/structure")
async def pageindex_get_structure(doc_name: str):
    """
    Return document structure (title, node_id, start_index, end_index, summary) for a PageIndex doc.
    Official PageIndex-style; use after listing documents to explore before pulling page content.
    """
    tree_searcher = TreeSearcher(OllamaClient())
    result = tree_searcher.get_document_structure(doc_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No PageIndex tree found for doc_name={doc_name!r}")
    return result


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


# ── Knowledge Base Endpoints ─────────────────────────────────────────────


@router.get("/knowledge-base/stats")
async def knowledge_base_stats():
    """Return statistics about the global knowledge base (entity taxonomy + relations)."""
    logger.info("api: GET /knowledge-base/stats")
    kb = get_knowledge_base()
    stats = kb.get_stats()
    stats["default_classification_model"] = config.INTERNAL_REASONING_AGENT
    logger.debug("api: knowledge-base/stats -> entity_count=%d relation_count=%d", stats.get("entity_count"), stats.get("relation_count"))
    return stats


@router.post("/knowledge-base/refresh")
async def refresh_knowledge_base(body: RefreshKnowledgeBaseRequest | None = Body(None)):
    """
    Rebuild the entire knowledge base from all per-document graphs.
    Classifies untyped entities using INTERNAL_REASONING_AGENT (e.g. phi3:3.8b) by default.
    Optional body.model overrides the classification model for this run.
    """
    logger.info("api: POST /knowledge-base/refresh START (model=%s)", getattr(body, "model", None) or "default")
    kb = get_knowledge_base()
    cb = broadcaster.make_callback("knowledge_base")
    model_override = body.model if body and body.model and str(body.model).strip() else None

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: kb.refresh_all(callback=cb, model=model_override)
    )
    model_used = model_override or config.INTERNAL_REASONING_AGENT
    logger.info(
        "api: POST /knowledge-base/refresh DONE model_used=%s documents_processed=%d total_entities=%d total_relations=%d",
        model_used, result.get("documents_processed", 0), result.get("total_entities", 0), result.get("total_relations", 0),
    )
    return {
        "success": True,
        **result,
        "model_used": model_used,
        "message": (
            f"Knowledge base refreshed with model {model_used}: {result['total_entities']} entities, "
            f"{result['total_relations']} relations from {result['documents_processed']} documents."
        ),
    }


@router.get("/knowledge-base/entities")
async def list_entities(entity_type: str | None = None, search: str | None = None, limit: int = 50):
    """List entities from the knowledge base, optionally filtered by type or search term."""
    kb = get_knowledge_base()
    if search:
        return {"entities": kb.search_entities(search, limit=limit)}
    if entity_type:
        names = kb.get_entities_by_type(entity_type)
        entities = []
        for name in names[:limit]:
            entry = kb.lookup_entity(name)
            if entry:
                entities.append(entry)
        return {"entities": entities}
    # Return all entities, sorted by mentions
    all_entities = sorted(
        [{"name": k, **v} for k, v in kb._taxonomy.items()],
        key=lambda x: x.get("mentions", 0), reverse=True,
    )
    return {"entities": all_entities[:limit]}


@router.get("/knowledge-base/entity/{entity_name}")
async def get_entity(entity_name: str):
    """Get details about a specific entity including its relationships."""
    kb = get_knowledge_base()
    entry = kb.lookup_entity(entity_name)
    if not entry:
        return {"found": False, "entity": None, "relations": []}
    relations = kb.get_related_entities(entity_name)
    facts = kb.get_facts_for_entity(entity_name)
    return {"found": True, "entity": entry, "relations": relations, "facts": facts}
