"""
Document CRUD service logic.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from sqlalchemy import func, or_, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.models import Document
from app.modules.contextual import DocStore, _delete_chroma_by_doc_name

logger = logging.getLogger(__name__)


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _path_to_tags(relative_path: str) -> list[str]:
    """Build tags from a file path: full path plus path segments (for filtering by folder)."""
    normalized = relative_path.replace("\\", "/").strip("/")
    if not normalized:
        return []
    tags = [normalized]
    parts = normalized.split("/")
    for i in range(1, len(parts)):
        tags.append("/".join(parts[:i]))
    return tags


def get_target_segments_from_full_path(
    full_path: Path,
    relative_path_under_root: str = "",
) -> list[str]:
    """
    Get tags from a full path: start at the first segment in IMPORT_TAG_ANCHOR_SEGMENTS
    (e.g. Trader), then include every slash-separated folder name after that. No filename.
    relative_path_under_root: append folder segments under the root (e.g. subfolder names).
    """
    resolved = full_path.resolve()
    parts = list(resolved.parts)
    if not parts:
        return []

    # Normalize: on Windows, parts[0] may be "C:\\" or "C:"; treat as one segment for matching
    anchors = config.IMPORT_TAG_ANCHOR_SEGMENTS
    anchor_set = {a.lower() for a in anchors}

    start_idx = None
    for i, seg in enumerate(parts):
        # Strip trailing slash/backslash for comparison (e.g. "C:\" -> "C:")
        seg_clean = seg.rstrip("\\/") if seg else seg
        if seg_clean and seg_clean.lower() in anchor_set:
            start_idx = i
            break

    if start_idx is None:
        return []

    root_segments = list(parts[start_idx:])

    if not relative_path_under_root:
        return root_segments

    # Append folder segments from relative path (exclude filename so hierarchy is folder-only)
    rel_normalized = relative_path_under_root.replace("\\", "/").strip("/")
    if not rel_normalized:
        return root_segments
    rel_parts = rel_normalized.split("/")
    # If last part looks like a file (has extension), drop it for hierarchy
    if rel_parts and "." in rel_parts[-1]:
        rel_parts = rel_parts[:-1]
    return root_segments + rel_parts


def get_target_segments_from_relative_path(relative_path: str) -> list[str]:
    """
    Get tags from a relative path (e.g. folder upload): start at the first segment that
    matches IMPORT_TAG_ANCHOR_SEGMENTS, then include every slash-separated folder name
    after that. If none of those names appear in the path, use all folder segments we have.
    Never include the filename or the full path string—only the individual folder names.
    """
    normalized = relative_path.replace("\\", "/").strip("/")
    if not normalized:
        return []
    parts = normalized.split("/")
    # Folder-only: drop last part if it looks like a filename (has extension)
    if parts and "." in parts[-1]:
        parts = parts[:-1]
    if not parts:
        return []

    anchors = config.IMPORT_TAG_ANCHOR_SEGMENTS
    anchor_set = {a.lower() for a in anchors}
    start_idx = None
    for i, seg in enumerate(parts):
        if seg and seg.lower() in anchor_set:
            start_idx = i
            break
    if start_idx is not None:
        return list(parts[start_idx:])
    # No anchor (e.g. user selected Schedule folder so path is Schedule/Handover/file.docx)
    return list(parts)


def _normalize_filename(name: str) -> str:
    """Canonical form for duplicate checks: forward slashes."""
    return name.replace("\\", "/").strip("/")


async def create_document(
    db: AsyncSession,
    filename: str,
    file_path: str,
    initial_tags: list[str] | None = None,
    initial_path_hierarchy: list[str] | None = None,
) -> Document:
    """Create a new document record and extract metadata. Optionally set tags (e.g. from folder path).
    If a document with the same filename already exists (case-insensitive, normalized), returns the existing record."""
    norm = _normalize_filename(filename)
    # Duplicate guard: match existing by normalized, case-insensitive filename
    existing_result = await db.execute(
        select(Document).where(
            func.lower(func.replace(Document.filename, "\\", "/")) == norm.lower()
        )
    )
    existing = existing_result.scalar_one_or_none()
    if existing is not None:
        return existing

    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    page_count = 0
    if path.exists() and _is_pdf(path):
        try:
            doc = fitz.open(str(path))
            page_count = len(doc)
            doc.close()
        except Exception:
            pass

    # Use backend time for upload: file mtime when file exists (when we wrote it), else now
    now_utc = datetime.now(timezone.utc)
    if path.exists():
        try:
            created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except (OSError, ValueError):
            created_at = now_utc
    else:
        created_at = now_utc

    tags_json = "[]"
    if initial_tags:
        tags_json = json.dumps(initial_tags)
    path_hierarchy_json = "[]"
    if initial_path_hierarchy is not None:
        path_hierarchy_json = json.dumps(initial_path_hierarchy)

    record = Document(
        filename=filename,
        file_path=str(path),
        file_size=file_size,
        page_count=page_count,
        status="uploaded",
        tags=tags_json,
        path_hierarchy=path_hierarchy_json,
        created_at=created_at,
        updated_at=created_at,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


# Allowed sort columns for list_documents (maps API sort_by to model column)
_LIST_SORT_COLUMNS = {
    "name": Document.filename,
    "created_at": Document.created_at,
    "page_count": Document.page_count,
    "status": Document.status,
    "chunk_count": Document.chunk_count,
    "ingestion_duration": Document.ingestion_duration_seconds,
}


def _needs_ingestion_condition():
    """Documents that need ingestion: uploaded or any pipeline pending/error."""
    return or_(
        Document.status == "uploaded",
        Document.pageindex_status.in_(["pending", "error"]),
        Document.contextual_status.in_(["pending", "error"]),
        Document.graph_status.in_(["pending", "error"]),
    )


def _apply_list_filters(
    query,
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    needs_ingestion: bool = False,
):
    """Apply search, status, tag, and needs_ingestion filters to a document query."""
    if search:
        query = query.where(
            or_(
                Document.filename.ilike(f"%{search}%"),
                Document.tags.ilike(f"%{search}%"),
                Document.notes.ilike(f"%{search}%"),
            )
        )
    if status:
        query = query.where(Document.status == status)
    if tag:
        query = query.where(Document.tags.ilike(f'%"{tag}"%'))
    if needs_ingestion:
        query = query.where(_needs_ingestion_condition())
    return query


async def list_documents(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 50,
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    needs_ingestion: bool = False,
    sort_by: str | None = None,
    sort_order: str | None = None,
) -> tuple[list[Document], int]:
    """List documents with pagination, search, filters, and optional sort."""
    query = select(Document)
    query = _apply_list_filters(query, search=search, status=status, tag=tag, needs_ingestion=needs_ingestion)

    # Count total (direct count with same filters, no subquery)
    count_q = select(func.count()).select_from(Document)
    count_q = _apply_list_filters(count_q, search=search, status=status, tag=tag, needs_ingestion=needs_ingestion)
    total_result = await db.execute(count_q)
    total = total_result.scalar() or 0

    # Order by
    order_col = _LIST_SORT_COLUMNS.get(sort_by) if sort_by else None
    if order_col is not None and sort_order in ("asc", "desc"):
        query = query.order_by(order_col.desc() if sort_order == "desc" else order_col.asc())
    else:
        query = query.order_by(Document.created_at.desc())

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    docs = list(result.scalars().all())

    return docs, total


async def list_document_ids(
    db: AsyncSession,
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    needs_ingestion: bool = False,
    limit: int = 10_000,
) -> list[int]:
    """Return document IDs matching the same filters as list_documents, up to limit."""
    query = select(Document.id)
    query = _apply_list_filters(query, search=search, status=status, tag=tag, needs_ingestion=needs_ingestion)

    query = query.order_by(Document.created_at.desc()).limit(limit)
    result = await db.execute(query)
    return [row[0] for row in result.all()]


async def get_document_facets(
    db: AsyncSession,
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    tag_facet_limit: int = 50_000,
) -> tuple[dict[str, int], dict[str, int], int]:
    """Return (by_status, by_tag, needs_ingestion_count) for the given filters."""
    # Base filter (no needs_ingestion) for status and tag facets
    base = select(Document)
    base = _apply_list_filters(base, search=search, status=status, tag=tag, needs_ingestion=False)

    # Count by status: same filters, GROUP BY status on the table directly (no subquery)
    status_q = select(Document.status, func.count()).select_from(Document)
    status_q = _apply_list_filters(status_q, search=search, status=status, tag=tag, needs_ingestion=False)
    status_q = status_q.group_by(Document.status)
    status_result = await db.execute(status_q)
    by_status = {str(row[0]): row[1] for row in status_result.all() if row[0] is not None}

    # needs_ingestion_count: same filters + needs_ingestion condition
    needs_q = select(func.count()).select_from(Document)
    needs_q = _apply_list_filters(needs_q, search=search, status=status, tag=tag, needs_ingestion=True)
    needs_result = await db.execute(needs_q)
    needs_ingestion_count = needs_result.scalar() or 0

    # by_tag: fetch filtered docs (limit for performance), aggregate tag counts in Python
    filtered_with_limit = base.limit(tag_facet_limit)
    result = await db.execute(filtered_with_limit)
    docs_slice = result.scalars().all()
    tag_counter: Counter = Counter()
    for doc in docs_slice:
        try:
            tags_list = json.loads(doc.tags) if doc.tags else []
        except (json.JSONDecodeError, TypeError):
            tags_list = []
        for t in tags_list:
            if isinstance(t, str) and t.strip():
                tag_counter[t.strip()] += 1
    by_tag = dict(tag_counter)

    return by_status, by_tag, needs_ingestion_count


async def get_document(db: AsyncSession, doc_id: int) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def update_document(
    db: AsyncSession,
    doc_id: int,
    tags: list[str] | None = None,
    path_hierarchy: list[str] | None = None,
    notes: str | None = None,
) -> Document | None:
    doc = await get_document(db, doc_id)
    if not doc:
        return None
    if tags is not None:
        doc.tags = json.dumps(tags)
    if path_hierarchy is not None:
        doc.path_hierarchy = json.dumps(path_hierarchy)
    if notes is not None:
        doc.notes = notes
    await db.commit()
    await db.refresh(doc)
    return doc


async def delete_document(db: AsyncSession, doc_id: int) -> bool:
    doc = await get_document(db, doc_id)
    if not doc:
        return False

    # Delete file from disk
    try:
        file_path = Path(doc.file_path)
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    # Delete associated index data
    stem = Path(doc.filename).stem
    _cleanup_index_data(stem)

    await db.delete(doc)
    await db.commit()
    return True


async def bulk_delete_documents(db: AsyncSession, doc_ids: list[int]) -> int:
    deleted = 0
    for doc_id in doc_ids:
        if await delete_document(db, doc_id):
            deleted += 1
    return deleted


async def deduplicate_documents(db: AsyncSession) -> int:
    """Remove duplicate document records (by normalized, case-insensitive filename),
    keeping the one with the lowest id. Cleans up orphaned files on disk for deleted duplicates.
    Returns the number of duplicates removed."""
    result = await db.execute(select(Document).order_by(Document.id))
    all_docs = list(result.scalars().all())

    # Group by normalized, lowercased filename
    by_key: dict[str, list[Document]] = {}
    for doc in all_docs:
        key = _normalize_filename(doc.filename).lower()
        by_key.setdefault(key, []).append(doc)

    removed = 0
    for key, group in by_key.items():
        if len(group) <= 1:
            continue
        # Keep the first (lowest id), delete the rest
        for dup in group[1:]:
            try:
                dup_path = Path(dup.file_path)
                if dup_path.exists():
                    dup_path.unlink()
            except Exception:
                pass
            await db.delete(dup)
            removed += 1

    if removed:
        await db.commit()
    return removed


# Metadata key names for selective delete (must match API/UI)
METADATA_KEYS = frozenset({"pageindex", "graph", "communities", "chroma", "docstore", "figures"})


def _delete_metadata_pageindex(doc_stem: str) -> None:
    """Remove PageIndex tree and stored PDF page images for this document."""
    pi_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
    if pi_path.exists():
        pi_path.unlink()
    pdf_images_dir = config.PDF_IMAGES_DIR / doc_stem
    if pdf_images_dir.exists() and pdf_images_dir.is_dir():
        try:
            shutil.rmtree(pdf_images_dir)
            logger.info("document_service: removed PDF page images dir doc_stem=%s path=%s", doc_stem, pdf_images_dir.resolve())
        except Exception as e:
            logger.warning("document_service: could not remove PDF images dir %s: %s", pdf_images_dir, e)


def _delete_metadata_graph(doc_stem: str) -> None:
    """Remove graph and communities JSON files."""
    for suffix in ["_graph.json", "_communities.json"]:
        gp = config.GRAPH_DIR / f"{doc_stem}{suffix}"
        if gp.exists():
            gp.unlink()


def _delete_metadata_chroma(doc_stem: str) -> None:
    """Remove ChromaDB vectors for this document."""
    try:
        _delete_chroma_by_doc_name(doc_stem)
    except Exception:
        pass


def _delete_metadata_docstore(doc_stem: str) -> None:
    """Remove DocStore chunks for this document."""
    try:
        DocStore().delete_by_doc_name(doc_stem)
    except Exception:
        pass


def _delete_metadata_figures(doc_stem: str) -> None:
    """Remove extracted figures directory for this document."""
    try:
        figures_dir = config.FIGURES_DIR / doc_stem
        if figures_dir.exists() and figures_dir.is_dir():
            shutil.rmtree(figures_dir)
    except Exception:
        pass


def _cleanup_index_data(doc_stem: str):
    """Remove PageIndex tree, Graph data, ChromaDB vectors, DocStore chunks, figures, and KB entries for a document."""
    _delete_metadata_pageindex(doc_stem)
    _delete_metadata_graph(doc_stem)
    _delete_metadata_chroma(doc_stem)
    _delete_metadata_docstore(doc_stem)
    _delete_metadata_figures(doc_stem)
    try:
        from app.modules.knowledge_base import get_knowledge_base
        logger.info("document_service: KB remove_document START doc_stem=%s", doc_stem)
        kb = get_knowledge_base()
        kb.remove_document(doc_stem)
        logger.info("document_service: KB remove_document DONE doc_stem=%s", doc_stem)
    except Exception as e:
        logger.warning("document_service: KB remove_document FAILED doc_stem=%s: %s", doc_stem, e)


async def delete_document_metadata(
    db: AsyncSession,
    doc_id: int,
    keys: list[str] | None = None,
) -> Document | None:
    """
    Delete selective metadata (or all) for a document. Does NOT delete the document record or file.
    keys: optional list of metadata types to delete (pageindex, graph, communities, chroma, docstore, figures).
          If None or empty, deletes all metadata.
    Returns the updated document or None if not found.
    """
    doc = await get_document(db, doc_id)
    if not doc:
        return None

    stem = Path(doc.filename).stem
    to_delete = set(keys) if keys else METADATA_KEYS
    to_delete = to_delete & METADATA_KEYS  # only allow known keys

    if not to_delete:
        to_delete = METADATA_KEYS  # delete all if no valid keys specified

    if "pageindex" in to_delete:
        _delete_metadata_pageindex(stem)
        doc.pageindex_status = "pending"
    if "graph" in to_delete or "communities" in to_delete:
        _delete_metadata_graph(stem)
        doc.graph_status = "pending"
        doc.node_count = 0
        doc.edge_count = 0
        try:
            from app.modules.knowledge_base import get_knowledge_base
            logger.info("document_service: KB remove_document (metadata) stem=%s", stem)
            get_knowledge_base().remove_document(stem)
        except Exception as e:
            logger.warning("document_service: KB remove_document (metadata) FAILED stem=%s: %s", stem, e)
    if "chroma" in to_delete:
        _delete_metadata_chroma(stem)
        doc.contextual_status = "pending"
        doc.chunk_count = 0
    if "docstore" in to_delete:
        _delete_metadata_docstore(stem)
        doc.contextual_status = "pending"
        doc.chunk_count = 0
    if "figures" in to_delete:
        _delete_metadata_figures(stem)

    # If we cleared chroma or docstore, ensure counts/status reset
    if "chroma" in to_delete or "docstore" in to_delete:
        doc.chunk_count = 0
        doc.contextual_status = "pending"

    # After clearing any pipeline metadata, status is no longer "ready"
    if to_delete:
        doc.status = "uploaded"

    await db.commit()
    await db.refresh(doc)
    return doc
