"""
Workspace-aware chat routes with remembrance cache.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import async_session, get_db
from app.models import ChatSession, Workspace
from app.routes.chat import (
    _append_screenshot_audit_context,
    _build_citations_and_sources,
    _build_effective_query,
    _chat_session_state,
    _extract_accepted_preferences,
    _normalize_markdown_headings,
    _session_to_out,
)
from app.routes.query import get_llm, get_rag
from app.schemas import (
    ChatMessageOut,
    ChatMessageRequest,
    ChatPreferenceRequest,
    ChatSessionCreate,
    ChatSessionOut,
    ChatSessionUpdate,
    WorkspaceCreate,
    WorkspaceOut,
    WorkspaceUpdate,
)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

WORKSPACE_MEMORY_MAX_CHARS = 4000
SESSION_MEMORY_MAX_CHARS = 1800


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim_cache(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _build_deep_memory_block(
    workspace_cache: str,
    session_cache: str,
    accepted_preferences: list[str],
) -> str:
    parts: list[str] = []
    if workspace_cache.strip():
        parts.append(
            "Workspace remembrance (long-term):\n"
            f"{workspace_cache.strip()[:WORKSPACE_MEMORY_MAX_CHARS]}"
        )
    if session_cache.strip():
        parts.append(
            "Session remembrance (short-term):\n"
            f"{session_cache.strip()[:SESSION_MEMORY_MAX_CHARS]}"
        )
    if accepted_preferences:
        prefs = "\n".join(f"- {p}" for p in accepted_preferences[-4:])
        parts.append(f"Confirmed user preferences:\n{prefs}")
    return "\n\n".join(parts).strip()


def _build_deep_effective_query(
    user_query: str,
    msgs: list[dict],
    workspace_cache: str,
    session_cache: str,
) -> str:
    prefs = _extract_accepted_preferences(msgs)
    memory_block = _build_deep_memory_block(workspace_cache, session_cache, prefs)
    if not memory_block:
        return _build_effective_query(user_query, msgs)
    return (
        "Use the remembrance below only as stable contextual guidance.\n"
        "If it conflicts with the user's latest request, prioritize the latest request.\n\n"
        f"{memory_block}\n\n"
        f"Current user question: {user_query}"
    )


def _build_memory_update_prompt(existing_cache: str, user_text: str, assistant_text: str, scope: str) -> str:
    return (
        f"You update a {scope} remembrance cache for an assistant.\n\n"
        "Rules:\n"
        "- Keep only durable facts, user preferences, decisions, constraints, and ongoing goals.\n"
        "- Remove redundant or stale information.\n"
        "- Keep it concise and structured with bullet points.\n"
        "- Do not include temporary small-talk.\n"
        "- If there are no durable updates, keep the cache as-is.\n\n"
        f"Existing cache:\n{existing_cache or '(empty)'}\n\n"
        f"Latest user message:\n{user_text}\n\n"
        f"Latest assistant response:\n{assistant_text}\n\n"
        "Return ONLY the updated cache text."
    )


def _llm_update_cache(existing_cache: str, user_text: str, assistant_text: str, *, scope: str, max_chars: int) -> str:
    prompt = _build_memory_update_prompt(existing_cache, user_text, assistant_text, scope)
    llm = get_llm()
    updated = llm.generate(
        prompt,
        model=config.INTERNAL_REASONING_AGENT,
        options=config.REASONING_OPTIONS,
    ).strip()
    if not updated:
        return _trim_cache(existing_cache or "", max_chars)
    return _trim_cache(updated, max_chars)


async def _update_remembrance_async(
    workspace_id: int,
    session_id: int,
    user_text: str,
    assistant_text: str,
) -> None:
    try:
        async with async_session() as db:
            workspace = await db.get(Workspace, workspace_id)
            session = await db.get(ChatSession, session_id)
            if not workspace or not session or session.workspace_id != workspace_id:
                return

            loop = asyncio.get_event_loop()
            workspace_cache = workspace.workspace_cache or ""
            session_cache = session.session_cache or ""

            updated_workspace_cache = await loop.run_in_executor(
                None,
                lambda: _llm_update_cache(
                    workspace_cache,
                    user_text,
                    assistant_text,
                    scope="workspace",
                    max_chars=WORKSPACE_MEMORY_MAX_CHARS,
                ),
            )
            updated_session_cache = await loop.run_in_executor(
                None,
                lambda: _llm_update_cache(
                    session_cache,
                    user_text,
                    assistant_text,
                    scope="chat session",
                    max_chars=SESSION_MEMORY_MAX_CHARS,
                ),
            )

            workspace.workspace_cache = updated_workspace_cache
            workspace.last_used_at = _utc_now()
            session.session_cache = updated_session_cache
            await db.commit()
    except Exception:
        # Memory update failures should not break request flow.
        return


@router.get("", response_model=list[WorkspaceOut])
async def list_workspaces(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Workspace, func.count(ChatSession.id))
        .outerjoin(ChatSession, ChatSession.workspace_id == Workspace.id)
        .group_by(Workspace.id)
        .order_by(Workspace.last_used_at.desc(), Workspace.created_at.desc())
    )
    out: list[WorkspaceOut] = []
    for workspace, count in result.all():
        out.append(
            WorkspaceOut(
                id=workspace.id,
                name=workspace.name,
                workspace_cache=workspace.workspace_cache or "",
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
                last_used_at=workspace.last_used_at or workspace.created_at,
                chat_count=int(count or 0),
            )
        )
    return out


@router.post("", response_model=WorkspaceOut)
async def create_workspace(req: WorkspaceCreate, db: AsyncSession = Depends(get_db)):
    name = (req.name or "").strip() or "New Workspace"
    now = _utc_now()
    workspace = Workspace(name=name, workspace_cache="", last_used_at=now, created_at=now, updated_at=now)
    db.add(workspace)
    await db.commit()
    await db.refresh(workspace)
    return WorkspaceOut(
        id=workspace.id,
        name=workspace.name,
        workspace_cache=workspace.workspace_cache or "",
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        last_used_at=workspace.last_used_at or workspace.created_at,
        chat_count=0,
    )


@router.patch("/{workspace_id}", response_model=WorkspaceOut)
async def update_workspace(workspace_id: int, req: WorkspaceUpdate, db: AsyncSession = Depends(get_db)):
    workspace = await db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workspace name cannot be empty")
    workspace.name = name
    workspace.updated_at = _utc_now()
    await db.commit()
    count_result = await db.execute(
        select(func.count(ChatSession.id)).where(ChatSession.workspace_id == workspace_id)
    )
    chat_count = int(count_result.scalar() or 0)
    return WorkspaceOut(
        id=workspace.id,
        name=workspace.name,
        workspace_cache=workspace.workspace_cache or "",
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        last_used_at=workspace.last_used_at or workspace.created_at,
        chat_count=chat_count,
    )


@router.delete("/{workspace_id}")
async def delete_workspace(workspace_id: int, db: AsyncSession = Depends(get_db)):
    workspace = await db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    sessions_result = await db.execute(
        select(ChatSession).where(ChatSession.workspace_id == workspace_id)
    )
    for session in sessions_result.scalars().all():
        await db.delete(session)
    await db.delete(workspace)
    await db.commit()
    return {"detail": "Workspace deleted"}


@router.get("/{workspace_id}/sessions", response_model=list[ChatSessionOut])
async def list_workspace_sessions(workspace_id: int, db: AsyncSession = Depends(get_db)):
    workspace = await db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.workspace_id == workspace_id)
        .order_by(ChatSession.created_at.desc())
    )
    return [_session_to_out(s) for s in result.scalars().all()]


@router.post("/{workspace_id}/sessions", response_model=ChatSessionOut)
async def create_workspace_session(
    workspace_id: int,
    req: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
):
    workspace = await db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    session = ChatSession(
        title=req.title,
        document_id=req.document_id,
        workspace_id=workspace_id,
        session_cache="",
    )
    db.add(session)
    workspace.last_used_at = _utc_now()
    await db.commit()
    await db.refresh(session)
    return _session_to_out(session)


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
async def get_workspace_messages(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_not(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Workspace session not found")
    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass
    visible = [m for m in msgs if m.get("role") != "preference"]
    return [ChatMessageOut(**m) for m in visible]


@router.post("/sessions/{session_id}/preference")
async def set_workspace_preference(
    session_id: int,
    req: ChatPreferenceRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_not(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Workspace session not found")

    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass

    pref_msg = {
        "role": "preference",
        "content": req.option_text.strip(),
        "trace": [],
        "metadata": {"accepted": bool(req.accepted)},
        "route": "",
    }
    msgs.append(pref_msg)
    session.messages = json.dumps(msgs, default=str)
    await db.commit()
    return {"detail": "Preference saved"}


@router.post("/sessions/{session_id}/message", response_model=ChatMessageOut)
async def send_workspace_message(
    session_id: int,
    req: ChatMessageRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_not(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session or session.workspace_id is None:
        raise HTTPException(status_code=404, detail="Workspace session not found")
    workspace = await db.get(Workspace, session.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    sent_at = _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass

    user_msg = {"role": "user", "content": req.content, "trace": [], "metadata": {}, "route": "", "sent_at": sent_at}
    msgs.append(user_msg)

    effective_model = req.model or _chat_session_state.get("last_selected_model") or config.LLM_MODEL
    _chat_session_state["last_selected_model"] = effective_model

    config.GENERATION_OPTIONS["temperature"] = req.temperature
    rag = get_rag()

    if req.fast_response:
        effective_query = _build_effective_query(req.content, msgs)
    else:
        effective_query = _build_deep_effective_query(
            req.content,
            msgs,
            workspace.workspace_cache or "",
            session.session_cache or "",
        )
    effective_query, screenshot_audit_meta = await _append_screenshot_audit_context(
        db, req.content, effective_query
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rag.query(
            effective_query,
            force_route=req.force_route,
            model=effective_model,
            fast_response=req.fast_response,
            enable_vector=req.enable_vector,
            enable_taxonomy=req.enable_taxonomy,
            enable_pageindex=req.enable_pageindex,
            enable_graph=req.enable_graph,
        ),
    )

    answer = _normalize_markdown_headings(result.get("answer", ""))
    meta = dict(result.get("metadata", {}))
    if screenshot_audit_meta:
        meta["screenshot_audit_context"] = screenshot_audit_meta
    cited_answer, meta["sources"] = await _build_citations_and_sources(
        answer, rag, db, loop, max_sources=10
    )

    response_received_at = _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    assistant_msg = {
        "role": "assistant",
        "content": cited_answer,
        "trace": result.get("trace", []),
        "metadata": meta,
        "route": result.get("route", ""),
        "sent_at": sent_at,
        "response_received_at": response_received_at,
    }
    msgs.append(assistant_msg)

    if len(msgs) <= 2:
        session.title = req.content[:80]

    workspace.last_used_at = _utc_now()
    session.messages = json.dumps(msgs, default=str)
    await db.commit()

    if not req.fast_response:
        asyncio.create_task(
            _update_remembrance_async(
                workspace_id=workspace.id,
                session_id=session.id,
                user_text=req.content,
                assistant_text=cited_answer,
            )
        )

    return ChatMessageOut(**assistant_msg)


@router.patch("/sessions/{session_id}", response_model=ChatSessionOut)
async def update_workspace_session(
    session_id: int,
    req: ChatSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_not(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Workspace session not found")
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    session.title = title[:512]
    await db.commit()
    await db.refresh(session)
    return _session_to_out(session)


@router.delete("/sessions/{session_id}")
async def delete_workspace_session(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_not(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Workspace session not found")
    workspace_id = session.workspace_id
    await db.delete(session)
    if workspace_id is not None:
        workspace = await db.get(Workspace, workspace_id)
        if workspace:
            workspace.last_used_at = _utc_now()
    await db.commit()

    return {"detail": "Workspace session deleted"}
