"""
Adaptive RAG query endpoint.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app import config
from app.schemas import QueryRequest, QueryResponse
from app.modules.ollama_client import OllamaClient
from app.modules.router import AdaptiveRAG
from app.modules.cag import CAGManager
from app.logger import broadcaster

router = APIRouter(prefix="/api", tags=["query"])

# Singletons – shared across requests
_llm: OllamaClient | None = None
_rag: AdaptiveRAG | None = None
_cag: CAGManager | None = None


def get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        _llm = OllamaClient()
    return _llm


def get_cag() -> CAGManager:
    global _cag
    if _cag is None:
        _cag = CAGManager(get_llm())
    return _cag


def get_rag() -> AdaptiveRAG:
    global _rag
    if _rag is None:
        _rag = AdaptiveRAG(llm=get_llm(), cag_manager=get_cag())
    return _rag


@router.post("/query", response_model=QueryResponse)
async def adaptive_query(req: QueryRequest):
    """Execute the Adaptive RAG pipeline."""
    broadcaster.log(
        f"Query received: {req.query[:100]}",
        module="router",
    )

    # Apply temperature override
    config.GENERATION_OPTIONS["temperature"] = req.temperature

    rag = get_rag()
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: rag.query(req.query, force_route=req.force_route),
    )

    broadcaster.log(
        f"Query complete: route={result.get('route', '?')}, retries={result.get('retries', 0)}",
        module="router",
    )

    return QueryResponse(**result)
