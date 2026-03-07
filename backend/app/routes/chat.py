"""
Chat session and CAG management routes.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.models import ChatSession, Document
from app.schemas import (
    ChatMessageOut,
    ChatPreferenceRequest,
    ChatMessageRequest,
    ChatSessionCreate,
    ChatSessionOut,
    ChatSessionUpdate,
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


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _infer_pdf_page_from_snippet(
    file_path: str,
    snippet: str,
    page_cache: dict[str, list[str]],
) -> int | None:
    """Infer the most likely 1-based PDF page containing snippet text."""
    if not snippet or not snippet.strip():
        return None
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return None

    normalized_snippet = _normalize_text(snippet)
    if len(normalized_snippet) < 20:
        return None

    if file_path not in page_cache:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page_cache[file_path] = [_normalize_text(page.get_text()) for page in doc]
            doc.close()
        except Exception:
            page_cache[file_path] = []

    pages = page_cache.get(file_path, [])
    if not pages:
        return None

    # Prefer page where the *end* of the snippet appears (reduces off-by-one when content spans two pages).
    snippet_len = len(normalized_snippet)
    best_page: int | None = None
    for i, page_text in enumerate(pages, start=1):
        if normalized_snippet in page_text:
            return i
        if normalized_snippet[:220] in page_text:
            best_page = i
    if best_page is not None:
        return best_page
    # Fallback: check if end of snippet appears on next page (content often continues there).
    if snippet_len > 120:
        tail = normalized_snippet[-200:].strip()
        for i, page_text in enumerate(pages, start=1):
            if tail in page_text:
                return i
    return None


def _search_chunks_by_answer(rag, answer: str, top_k: int = 10) -> list[dict]:
    """Run a quick vector/semantic search using the LLM answer to get top chunks. Returns list of {doc_name, page_number, snippet}."""
    if not answer or not answer.strip():
        return []
    try:
        result = rag.hybrid_searcher.search(answer.strip(), top_k=top_k)
        fused = result.get("results") or []
    except Exception:
        return []
    out = []
    for r in fused:
        doc_name = r.get("doc_name")
        if not doc_name:
            continue
        page_raw = r.get("page_number")
        page_number = page_raw if isinstance(page_raw, int) and page_raw > 0 else None
        snippet = (r.get("text") or "")[:500]
        out.append({"doc_name": doc_name, "page_number": page_number, "snippet": snippet})
    return out


async def _resolve_stem_to_doc(db: AsyncSession):
    """Return dict stem -> {document_id, filename, file_path}."""
    result = await db.execute(select(Document.id, Document.filename, Document.file_path))
    rows = result.all()
    stem_to_doc = {}
    for doc_id, filename, file_path in rows:
        stem = Path(filename).stem
        if stem not in stem_to_doc:
            stem_to_doc[stem] = {"document_id": doc_id, "filename": filename, "file_path": file_path}
    return stem_to_doc


def _ensure_page_number(
    primary_page: int | None,
    file_path: str,
    snippet: str,
    page_cache: dict[str, list[str]],
) -> int:
    """Ensure we always have a page number: infer from snippet or default to 1."""
    if primary_page is not None and primary_page > 0:
        return primary_page
    inferred = _infer_pdf_page_from_snippet(file_path, snippet, page_cache)
    return inferred if inferred is not None else 1


async def _build_sources_from_metadata(
    meta: dict, db: AsyncSession, *, always_page_numbered: bool = True
) -> list[dict]:
    """Build source refs { document_id, filename, pages, primary_page, snippet? } from RAG metadata. Ensures every source has at least one page."""
    raw: list[dict] = []
    for leaf in meta.get("pageindex_leaves") or []:
        doc_name = leaf.get("doc_name")
        if not doc_name:
            continue
        page_list = leaf.get("pages") or []
        pages = []
        for p in page_list:
            try:
                n = int(p)
                if n > 0:
                    pages.append(n)
            except (TypeError, ValueError):
                pass
        pages = sorted(set(pages))
        raw.append({
            "doc_name": doc_name,
            "pages": pages,
            "primary_page": pages[0] if pages else None,
            "snippet": leaf.get("title") or "",
        })
    for vs in meta.get("vector_sources") or []:
        doc_name = vs.get("doc_name")
        if not doc_name:
            continue
        page_raw = vs.get("page_number")
        page_number = page_raw if isinstance(page_raw, int) and page_raw > 0 else None
        raw.append({
            "doc_name": doc_name,
            "pages": [page_number] if page_number else [],
            "primary_page": page_number,
            "snippet": (vs.get("snippet") or "")[:500],
        })
    if not raw:
        return []
    stem_to_doc = await _resolve_stem_to_doc(db)
    sources = []
    page_cache: dict[str, list[str]] = {}
    for r in raw:
        stem = r["doc_name"]
        if stem not in stem_to_doc:
            continue
        info = stem_to_doc[stem]
        primary_page = r["primary_page"]
        if primary_page is None and always_page_numbered:
            primary_page = _ensure_page_number(
                None, info["file_path"], r.get("snippet") or "", page_cache
            )
        pages = r["pages"] if r["pages"] else ([primary_page] if primary_page else [])
        if always_page_numbered and not pages and primary_page is not None:
            pages = [primary_page]
        if always_page_numbered and not pages:
            pages = [1]
            primary_page = 1
        snippet = (r.get("snippet") or "")[:500] or None
        sources.append({
            "document_id": info["document_id"],
            "filename": info["filename"],
            "pages": pages,
            "primary_page": primary_page,
            "snippet": snippet,
        })
    return sources


async def _build_and_rank_sources(
    answer: str,
    meta: dict,
    db: AsyncSession,
    rag,
    loop,
    top_k: int = 8,
    max_refs: int = 4,
) -> list[dict]:
    """
    Double-check references: run a quick vector semantic search using the LLM answer,
    return only top references that are actually reflected in the response. Every source
    is always page-numbered; clicking opens the same PDF view.
    """
    # Metadata-seeded candidates (trusted retrieval context) used as constraints.
    seeded_sources = await _build_sources_from_metadata(meta, db, always_page_numbered=True)
    allowed_docs = {
        int(s["document_id"])
        for s in seeded_sources
        if isinstance(s.get("document_id"), int)
    }
    allowed_pages_by_doc: dict[int, set[int]] = {}
    for s in seeded_sources:
        doc_id = s.get("document_id")
        if not isinstance(doc_id, int):
            continue
        pages = s.get("pages") or []
        page_set = {int(p) for p in pages if isinstance(p, int) and p > 0}
        if not page_set:
            pp = s.get("primary_page")
            if isinstance(pp, int) and pp > 0:
                page_set = {pp}
        allowed_pages_by_doc.setdefault(doc_id, set()).update(page_set)

    # 1) Get top chunks semantically aligned with the answer (sync call in executor)
    def do_search():
        return _search_chunks_by_answer(rag, answer, top_k=top_k)

    ranked_chunks = await loop.run_in_executor(None, do_search)

    stem_to_doc = await _resolve_stem_to_doc(db)
    page_cache: dict[str, list[str]] = {}
    seen: set[tuple[int, int]] = set()  # (document_id, primary_page) for dedupe
    sources = []

    for hit in ranked_chunks:
        stem = hit["doc_name"]
        if stem not in stem_to_doc:
            continue
        info = stem_to_doc[stem]
        page = _ensure_page_number(
            hit.get("page_number"),
            info["file_path"],
            hit.get("snippet") or "",
            page_cache,
        )
        if info["document_id"] not in allowed_docs:
            continue
        allowed_pages = allowed_pages_by_doc.get(info["document_id"], set())
        if allowed_pages and page not in allowed_pages:
            continue
        key = (info["document_id"], page)
        if key in seen:
            continue
        seen.add(key)
        snippet = (hit.get("snippet") or "")[:500] or None
        sources.append({
            "document_id": info["document_id"],
            "filename": info["filename"],
            "pages": [page],
            "primary_page": page,
            "snippet": snippet,
        })
        if len(sources) >= max_refs:
            break

    if not sources:
        # Fallback: metadata-seeded references (already page-numbered) and capped.
        sources = seeded_sources[:max_refs]
    return sources


def _retrieve_hit_for_sentence(rag, sentence: str) -> dict | None:
    """Run hybrid search for one sentence; return top hit with doc_name, page_number, text, or None."""
    normalized = _normalize_text(sentence)
    if len(normalized) < 20 or "pick one of the options below" in normalized:
        return None
    try:
        result = rag.hybrid_searcher.search(sentence.strip(), top_k=1)
        hits = result.get("results") or []
    except Exception:
        return None
    if not hits:
        return None
    hit = hits[0]
    doc_name = hit.get("doc_name")
    if not doc_name:
        return None
    page_raw = hit.get("page_number")
    page = page_raw if isinstance(page_raw, int) and page_raw > 0 else None
    return {
        "doc_name": str(doc_name),
        "page_number": page,
        "snippet": (hit.get("text") or "")[:500],
    }


async def _build_citations_and_sources(
    answer: str,
    rag,
    db: AsyncSession,
    loop,
    max_sources: int = 10,
) -> tuple[str, list[dict]]:
    """
    Citation-first flow: derive references only from what the LLM output actually
    cites. For each sentence we retrieve the best-matching chunk; we build the
    source list from those (stem, page) pairs only, then inject [^1]..[^n] so
    superscripts and the reference list are in harmony. Only cited doc+page appear
    as references.

    Page number: used from chunk metadata when the ingestion pipeline stores it
    (contextual pipeline currently does not). Otherwise we infer from snippet via
    _infer_pdf_page_from_snippet (with end-of-snippet preference to reduce off-by-one).
    """
    if not answer or not answer.strip():
        return answer, []

    lines = answer.splitlines()
    line_sentences: list[list[str]] = []
    flat_sentences: list[str] = []
    for line in lines:
        if not line.strip():
            line_sentences.append([])
            continue
        if re.match(r"^\s*#{1,6}\s+", line):
            line_sentences.append([])
            continue
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", line.strip()) if s]
        line_sentences.append(sentences)
        flat_sentences.extend(sentences)

    def do_retrieve_all():
        out: list[dict | None] = []
        for s in flat_sentences:
            out.append(_retrieve_hit_for_sentence(rag, s))
        return out

    hits_per_sentence = await loop.run_in_executor(None, do_retrieve_all)

    # Unique (stem, page) in order of first appearance; page from chunk when present else to be inferred
    order_keys: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    hit_by_key: dict[tuple[str, int], dict] = {}

    for hit in hits_per_sentence:
        if not hit:
            continue
        stem = hit["doc_name"]
        page_from_chunk = hit.get("page_number")
        page = page_from_chunk if isinstance(page_from_chunk, int) and page_from_chunk > 0 else 0
        key = (stem, page) if page else (stem, 0)
        if key not in seen_keys:
            seen_keys.add(key)
            order_keys.append(key)
            hit_by_key[key] = hit
        if len(order_keys) >= max_sources:
            break

    stem_to_doc = await _resolve_stem_to_doc(db)
    page_cache: dict[str, list[str]] = {}
    sources: list[dict] = []
    key_to_index: dict[tuple[str, int], int] = {}

    for idx, key in enumerate(order_keys, start=1):
        stem, page_from_chunk = key
        if stem not in stem_to_doc:
            continue
        info = stem_to_doc[stem]
        h = hit_by_key.get(key, {})
        snippet = h.get("snippet") or ""
        if page_from_chunk and page_from_chunk > 0:
            page = page_from_chunk
        else:
            page = _ensure_page_number(None, info["file_path"], snippet, page_cache)
        key_to_index[(stem, page)] = idx
        if (stem, 0) not in key_to_index:
            key_to_index[(stem, 0)] = idx
        sources.append({
            "document_id": info["document_id"],
            "filename": info["filename"],
            "pages": [page],
            "primary_page": page,
            "snippet": snippet[:500] if snippet else None,
        })

    # Map each sentence to 1-based index (only if we have a matching source)
    def sentence_to_index(hit: dict | None) -> int | None:
        if not hit:
            return None
        stem = hit["doc_name"]
        page_from_chunk = hit.get("page_number")
        page = page_from_chunk if isinstance(page_from_chunk, int) and page_from_chunk > 0 else 0
        idx = key_to_index.get((stem, page)) or key_to_index.get((stem, 0))
        return idx

    mapped_indices = [sentence_to_index(h) for h in hits_per_sentence]
    mapped_iter = iter(mapped_indices)

    rebuilt_lines: list[str] = []
    for line, sentences in zip(lines, line_sentences, strict=False):
        if re.match(r"^\s*#{1,6}\s+", line):
            rebuilt_lines.append(line)
            continue
        if not sentences:
            rebuilt_lines.append(line)
            continue
        annotated: list[str] = []
        for sentence in sentences:
            idx = next(mapped_iter, None)
            if idx is not None and f"[^{idx}]" not in sentence:
                sentence = f"{sentence} [^{idx}]"
            annotated.append(sentence)
        rebuilt_lines.append(" ".join(annotated))

    cited_answer = "\n".join(rebuilt_lines)
    return cited_answer, sources


def _normalize_markdown_headings(answer: str) -> str:
    """
    Normalize malformed markdown headings so only heading text remains on heading line.
    Example:
      ## Critical Safety Instructions  . Never reverse ...
    becomes:
      ## Critical Safety Instructions
      Never reverse ...
    """
    if not answer:
        return answer
    lines = answer.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            out.append(line)
            continue

        hashes, rest = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        heading_text = rest
        body = ""

        # Split obvious malformed separators that should start body text.
        sep = re.search(r"\s+\.\s+|\s+-\s+", rest)
        if sep:
            heading_text = rest[:sep.start()].strip().rstrip(".")
            body = rest[sep.end():].strip()
        else:
            # Fallback heuristic: short heading sentence followed by another sentence.
            period_split = re.search(r"\.\s+[A-Z]", rest)
            if period_split and len(rest[: period_split.start() + 1].split()) <= 10:
                heading_text = rest[: period_split.start() + 1].strip().rstrip(".")
                body = rest[period_split.start() + 2 :].strip()

        out.append(f"{hashes} {heading_text}".rstrip())
        if body:
            out.append(body)
    return "\n".join(out)


def _session_to_out(s: ChatSession) -> ChatSessionOut:
    msgs = []
    try:
        msgs = json.loads(s.messages) if s.messages else []
    except (json.JSONDecodeError, TypeError):
        pass
    visible_count = sum(1 for m in msgs if m.get("role") != "preference")
    return ChatSessionOut(
        id=s.id,
        title=s.title,
        document_id=s.document_id,
        workspace_id=s.workspace_id,
        created_at=s.created_at,
        message_count=visible_count,
    )


def _extract_accepted_preferences(msgs: list[dict]) -> list[str]:
    prefs: list[str] = []
    for m in msgs:
        if m.get("role") != "preference":
            continue
        metadata = m.get("metadata") or {}
        if not isinstance(metadata, dict) or not metadata.get("accepted"):
            continue
        text = str(m.get("content") or "").strip()
        if text:
            prefs.append(text)
    # Keep the most recent unique preferences only.
    deduped: list[str] = []
    seen: set[str] = set()
    for p in reversed(prefs):
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    deduped.reverse()
    return deduped[-4:]


def _build_effective_query(user_query: str, msgs: list[dict]) -> str:
    prefs = _extract_accepted_preferences(msgs)
    if not prefs:
        return user_query
    pref_block = "\n".join(f"- {p}" for p in prefs)
    return (
        "Use these user-confirmed disambiguation preferences when answering.\n"
        f"{pref_block}\n\n"
        f"Current user question: {user_query}"
    )


_COMPLIANCE_QUERY_KEYWORDS = (
    "compliance",
    "wcag",
    "accessibility",
    "violation",
    "violates",
    "non-compliant",
    "non compliant",
    "audit",
    "screen",
    "screenshot",
    "ui",
    "contrast",
    "focus",
    "alt text",
)


def _is_compliance_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _COMPLIANCE_QUERY_KEYWORDS)


async def _append_screenshot_audit_context(
    db: AsyncSession,
    user_query: str,
    effective_query: str,
    *,
    max_documents: int = 3,
    max_violations_per_doc: int = 8,
) -> tuple[str, dict]:
    """
    For compliance-style queries, append stored screenshot audit results to the
    prompt so the answer reliably surfaces detected WCAG failures.
    """
    if not _is_compliance_query(user_query):
        return effective_query, {}

    result = await db.execute(
        select(Document)
        .where(
            Document.ingestion_type == "screenshot",
            Document.audit_result.is_not(None),
        )
        .order_by(Document.updated_at.desc())
        .limit(max_documents)
    )
    screenshot_docs = list(result.scalars().all())
    if not screenshot_docs:
        return effective_query, {"screenshot_audits_found": 0}

    blocks: list[str] = []
    total_violations = 0
    for doc in screenshot_docs:
        try:
            audit = json.loads(doc.audit_result or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        violations = audit.get("detected_violations") or []
        if not isinstance(violations, list):
            violations = []
        status = (
            (audit.get("audit_metadata") or {}).get("overall_compliance_status")
            if isinstance(audit.get("audit_metadata"), dict)
            else None
        ) or "Unknown"
        lines = [f"Screenshot: {doc.filename}", f"Overall status: {status}"]
        for i, v in enumerate(violations[:max_violations_per_doc], start=1):
            if not isinstance(v, dict):
                continue
            ref = (v.get("guideline_reference") or "").strip() or "WCAG reference not provided"
            issue = (v.get("issue_description") or "").strip()
            fix = (v.get("suggested_fix") or "").strip()
            lines.append(f"{i}. {ref}")
            if issue:
                lines.append(f"   Issue: {issue}")
            if fix:
                lines.append(f"   Suggested fix: {fix}")
        total_violations += len(violations)
        blocks.append("\n".join(lines))

    if not blocks:
        return effective_query, {"screenshot_audits_found": 0}

    injected = (
        f"{effective_query}\n\n"
        "Use the stored screenshot UI audit findings below as high-priority evidence. "
        "When asked what failed, explicitly enumerate the violations and explain why each is non-compliant.\n\n"
        "Stored Screenshot WCAG Audit Findings:\n"
        f"{'\n\n---\n\n'.join(blocks)}"
    )
    meta = {
        "screenshot_audits_found": len(blocks),
        "screenshot_violations_indexed": total_violations,
    }
    return injected, meta


@router.get("/session-state")
async def get_session_state():
    """Return current chat session state (last selected model). Resets on server restart."""
    return {"last_selected_model": _chat_session_state.get("last_selected_model")}


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.workspace_id.is_(None))
        .order_by(ChatSession.created_at.desc())
    )
    return [_session_to_out(s) for s in result.scalars().all()]


@router.post("/sessions", response_model=ChatSessionOut)
async def create_session(
    req: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(title=req.title, document_id=req.document_id, workspace_id=None)
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
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass
    visible = [m for m in msgs if m.get("role") != "preference"]
    return [ChatMessageOut(**m) for m in visible]


@router.post("/sessions/{session_id}/preference")
async def set_preference(
    session_id: int,
    req: ChatPreferenceRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

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
async def send_message(
    session_id: int,
    req: ChatMessageRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Time tracking (UTC, second precision)
    sent_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Parse existing messages
    msgs = []
    try:
        msgs = json.loads(session.messages) if session.messages else []
    except (json.JSONDecodeError, TypeError):
        pass

    # Add user message
    user_msg = {"role": "user", "content": req.content, "trace": [], "metadata": {}, "route": "", "sent_at": sent_at}
    msgs.append(user_msg)

    # Resolve model: request body > last selected > config default; then remember for next time
    effective_model = req.model or _chat_session_state.get("last_selected_model") or config.LLM_MODEL
    _chat_session_state["last_selected_model"] = effective_model

    # Generate response
    config.GENERATION_OPTIONS["temperature"] = req.temperature
    rag = get_rag()
    effective_query = _build_effective_query(req.content, msgs)
    effective_query, screenshot_audit_meta = await _append_screenshot_audit_context(
        db, req.content, effective_query
    )
    result = await asyncio.get_event_loop().run_in_executor(
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

    # Citation-first: references and superscripts derived only from what the answer actually cites
    answer = _normalize_markdown_headings(result.get("answer", ""))
    meta = dict(result.get("metadata", {}))
    if screenshot_audit_meta:
        meta["screenshot_audit_context"] = screenshot_audit_meta
    loop = asyncio.get_event_loop()
    cited_answer, meta["sources"] = await _build_citations_and_sources(
        answer, rag, db, loop, max_sources=10
    )

    # Add assistant message (with response-received time)
    response_received_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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

    # Update session title if first message
    if len(msgs) <= 2:
        session.title = req.content[:80]

    # Save
    session.messages = json.dumps(msgs, default=str)
    await db.commit()

    return ChatMessageOut(**assistant_msg)


@router.patch("/sessions/{session_id}", response_model=ChatSessionOut)
async def update_session(
    session_id: int,
    req: ChatSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    session.title = title[:512]
    await db.commit()
    await db.refresh(session)
    return _session_to_out(session)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.workspace_id.is_(None),
        )
    )
    session = result.scalar_one_or_none()
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
