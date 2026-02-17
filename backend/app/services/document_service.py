"""
Document CRUD service logic.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from sqlalchemy import func, or_, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.models import Document
from app.modules.contextual import DocStore, _delete_chroma_by_doc_name


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


def _normalize_filename(name: str) -> str:
    """Canonical form for duplicate checks: forward slashes."""
    return name.replace("\\", "/").strip("/")


async def create_document(
    db: AsyncSession,
    filename: str,
    file_path: str,
    initial_tags: list[str] | None = None,
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

    tags_json = "[]"
    if initial_tags:
        tags_json = json.dumps(initial_tags)

    record = Document(
        filename=filename,
        file_path=str(path),
        file_size=file_size,
        page_count=page_count,
        status="uploaded",
        tags=tags_json,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def list_documents(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 50,
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
) -> tuple[list[Document], int]:
    """List documents with pagination, search, and filters."""
    query = select(Document)

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

    # Count total
    count_q = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_q)
    total = total_result.scalar() or 0

    # Paginate
    query = query.order_by(Document.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    docs = list(result.scalars().all())

    return docs, total


async def list_document_ids(
    db: AsyncSession,
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    limit: int = 10_000,
) -> list[int]:
    """Return document IDs matching the same filters as list_documents, up to limit."""
    query = select(Document.id)

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

    query = query.order_by(Document.created_at.desc()).limit(limit)
    result = await db.execute(query)
    return [row[0] for row in result.all()]


async def get_document(db: AsyncSession, doc_id: int) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def update_document(
    db: AsyncSession, doc_id: int, tags: list[str] | None = None, notes: str | None = None,
) -> Document | None:
    doc = await get_document(db, doc_id)
    if not doc:
        return None
    if tags is not None:
        doc.tags = json.dumps(tags)
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
    """Remove PageIndex tree only."""
    pi_path = config.PAGEINDEX_DIR / f"{doc_stem}.json"
    if pi_path.exists():
        pi_path.unlink()


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
    """Remove PageIndex tree, Graph data, ChromaDB vectors, DocStore chunks, and figures for a document."""
    _delete_metadata_pageindex(doc_stem)
    _delete_metadata_graph(doc_stem)
    _delete_metadata_chroma(doc_stem)
    _delete_metadata_docstore(doc_stem)
    _delete_metadata_figures(doc_stem)


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
