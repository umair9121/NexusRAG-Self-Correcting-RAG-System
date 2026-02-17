"""
Chat session and CAG management routes.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.models import ChatSession, Document
from app.schemas import (
    ChatMessageOut,
    ChatMessageRequest,
    ChatSessionCreate,
    ChatSessionOut,
)
from app.routes.query import get_cag, get_rag
from app.logger import broadcaster
from app.modules.text_extract import extract_text_from_file

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _load_doc_into_cag(file_path: str, doc_name: str):
    """Load a document into CAG: PDF by path, other types by extracted text."""
    from pathlib import Path
    cag = get_cag()
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return cag.load_document(pdf_path=file_path, doc_name=doc_name)
    text = extract_text_from_file(file_path)
    if not text or not text.strip():
        return {"success": False, "error": "No text could be extracted from this file"}
    return cag.load_document(text=text, doc_name=doc_name)

# In-memory session state (resets on server restart)
_chat_session_state: dict[str, str | None] = {"last_selected_model": None}


def _session_to_out(s: ChatSession) -> ChatSessionOut:
    msgs = []
    try:
        msgs = json.loads(s.messages) if s.messages else []
    except (json.JSONDecodeError, TypeError):
        pass
    return ChatSessionOut(
        id=s.id,
        title=s.title,
        document_id=s.document_id,
        created_at=s.created_at,
        message_count=len(msgs),
    )


@router.get("/session-state")
async def get_session_state():
    """Return current chat session state (last selected model). Resets on server restart."""
    return {"last_selected_model": _chat_session_state.get("last_selected_model")}


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).order_by(ChatSession.created_at.desc()))
    return [_session_to_out(s) for s in result.scalars().all()]


@router.post("/sessions", response_model=ChatSessionOut)
async def create_session(
    req: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(title=req.title, document_id=req.document_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # If a document is specified, load it into CAG
    if req.document_id:
        doc = await db.get(Document, req.document_id)
        if doc:
            broadcaster.log(f"Loading document into CAG: {doc.filename}", module="cag", document_id=doc.id)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: _load_doc_into_cag(doc.file_path, doc.filename)
            )

    return _session_to_out(session)


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
async def get_messages(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass
    return [ChatMessageOut(**m) for m in msgs]


@router.post("/sessions/{session_id}/message", response_model=ChatMessageOut)
async def send_message(
    session_id: int,
    req: ChatMessageRequest,
    db: AsyncSession = Depends(get_db),
):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Parse existing messages
    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass

    # Add user message
    user_msg = {"role": "user", "content": req.content, "trace": [], "metadata": {}, "route": ""}
    msgs.append(user_msg)

    # Resolve model: request body > last selected > config default; then remember for next time
    effective_model = req.model or _chat_session_state.get("last_selected_model") or config.LLM_MODEL
    _chat_session_state["last_selected_model"] = effective_model

    # Generate response
    config.GENERATION_OPTIONS["temperature"] = req.temperature
    rag = get_rag()
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: rag.query(req.content, force_route=req.force_route, model=effective_model),
    )

    # Add assistant message
    assistant_msg = {
        "role": "assistant",
        "content": result.get("answer", ""),
        "trace": result.get("trace", []),
        "metadata": result.get("metadata", {}),
        "route": result.get("route", ""),
    }
    msgs.append(assistant_msg)

    # Update session title if first message
    if len(msgs) <= 2:
        session.title = req.content[:80]

    # Save
    session.messages = json.dumps(msgs, default=str)
    await db.commit()

    return ChatMessageOut(**assistant_msg)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()

    # Unload CAG if this was the CAG session
    if session.document_id:
        cag = get_cag()
        if cag.is_active:
            cag.unload()

    return {"detail": "Session deleted"}


@router.post("/cag/load/{doc_id}")
async def load_cag(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Manually load a document into CAG context."""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _load_doc_into_cag(doc.file_path, doc.filename)
    )
    return result


@router.post("/cag/unload")
async def unload_cag():
    """Unload the current CAG document."""
    cag = get_cag()
    cag.unload()
    return {"detail": "CAG unloaded"}


@router.get("/cag/status")
async def cag_status():
    """Get CAG session status."""
    cag = get_cag()
    return cag.get_status()
