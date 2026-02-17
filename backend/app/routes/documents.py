"""
Document CRUD API routes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.database import get_db
from app.models import Document
from app.schemas import (
    BulkDeleteRequest,
    DeleteMetadataRequest,
    DocumentIdsResponse,
    DocumentListResponse,
    DocumentOut,
    DocumentUpdate,
    ImportFolderRequest,
)
from app.modules.contextual import get_chroma_entries_for_document
from app.services.document_service import (
    _path_to_tags,
    bulk_delete_documents,
    create_document,
    delete_document,
    delete_document_metadata,
    get_document,
    list_document_ids,
    list_documents,
    update_document,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Only one folder import at a time so duplicate checks see committed rows
_import_folder_lock = asyncio.Lock()


def _doc_to_out(doc) -> DocumentOut:
    tags = []
    try:
        tags = json.loads(doc.tags) if doc.tags else []
    except (json.JSONDecodeError, TypeError):
        tags = []
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        tags=tags,
        notes=doc.notes or "",
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        pageindex_status=doc.pageindex_status or "pending",
        contextual_status=doc.contextual_status or "pending",
        graph_status=doc.graph_status or "pending",
        chunk_count=doc.chunk_count or 0,
        node_count=doc.node_count or 0,
        edge_count=doc.edge_count or 0,
        error_message=doc.error_message or "",
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse)
async def list_docs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    docs, total = await list_documents(db, page=page, page_size=page_size, search=search, status=status, tag=tag)
    return DocumentListResponse(
        documents=[_doc_to_out(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/ids", response_model=DocumentIdsResponse)
async def list_doc_ids(
    search: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    limit: int = Query(10_000, ge=1, le=50_000),
    db: AsyncSession = Depends(get_db),
):
    """Return document IDs matching the same filters as list_docs, for bulk select-all."""
    ids = await list_document_ids(db, search=search, status=status, tag=tag, limit=limit)
    return DocumentIdsResponse(ids=ids)


def _sanitize_relative_path(rel: str) -> str:
    """Normalize and strip path traversal for use as subpath under imports/."""
    parts = rel.replace("\\", "/").strip("/").split("/")
    out = []
    for p in parts:
        if p in ("", ".", ".."):
            continue
        out.append(p)
    return "/".join(out) if out else "file"


def _allowed_file(filename: str) -> bool:
    """True if filename has an allowed extension."""
    return filename and Path(filename).suffix.lower() in config.ALLOWED_EXTENSIONS


@router.post("/upload", response_model=list[DocumentOut])
async def upload_docs(
    files: list[UploadFile] = File(...),
    path_tags: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload documents (PDF, Word, Excel, JSON, HTML, images). Optionally pass path_tags
    as a JSON array of relative paths (one per file) to use folder structure as metadata/tags.
    """
    results = []
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path_list: list[str] | None = None
    if path_tags:
        try:
            path_list = json.loads(path_tags)
            if not isinstance(path_list, list):
                path_list = None
        except (json.JSONDecodeError, TypeError):
            path_list = None

    is_folder_upload = path_list and len(path_list) == len(files)
    import_dir: Path | None = None
    if is_folder_upload:
        import_dir = config.DATA_DIR / "imports" / uuid.uuid4().hex
        import_dir.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files):
        if not file.filename or not _allowed_file(file.filename):
            continue
        ext = Path(file.filename).suffix.lower()
        if is_folder_upload and import_dir and i < len(path_list):
            rel = _sanitize_relative_path(path_list[i])
            if Path(rel).suffix.lower() not in config.ALLOWED_EXTENSIONS:
                rel = rel.rstrip("/") + "/" + file.filename
            dest = import_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            display_name = rel
            tags = _path_to_tags(rel)
        else:
            dest = config.DATA_DIR / file.filename
            counter = 1
            while dest.exists():
                stem = Path(file.filename).stem
                dest = config.DATA_DIR / f"{stem}_{counter}{ext}"
                counter += 1
            display_name = file.filename
            tags = None
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        doc = await create_document(db, display_name, str(dest), initial_tags=tags)
        results.append(_doc_to_out(doc))
    return results


def _normalize_filename(name: str) -> str:
    """Canonical form for duplicate checks: forward slashes, no leading/trailing slashes."""
    return name.replace("\\", "/").strip("/")


@router.post("/import-folder", response_model=list[DocumentOut])
async def import_folder(
    req: ImportFolderRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Import all supported documents from a folder on the server. Recursively scans the given path,
    copies files into data/imports/<id>/ preserving relative structure, and creates
    document records. Use when the backend runs on the same machine as the folder (e.g. local dev).
    """
    root = Path(req.path).resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    async with _import_folder_lock:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        import_dir = config.DATA_DIR / "imports" / uuid.uuid4().hex
        import_dir.mkdir(parents=True, exist_ok=True)
        results = []
        try:
            seen: set[str] = set()
            for abs_path in root.rglob("*"):
                if not abs_path.is_file():
                    continue
                if abs_path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
                    continue
                resolved = str(abs_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    rel = abs_path.relative_to(root)
                except ValueError:
                    continue
                rel_str = rel.as_posix()
                norm = _normalize_filename(rel_str)

                # Skip if a document with this filename already exists (case-insensitive, normalized)
                existing_check = await db.execute(
                    select(Document).where(
                        func.lower(func.replace(Document.filename, "\\", "/")) == norm.lower()
                    )
                )
                if existing_check.scalar_one_or_none() is not None:
                    continue

                dest = import_dir / rel_str
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(abs_path, dest)
                tags = _path_to_tags(rel_str)
                doc = await create_document(db, rel_str, str(dest), initial_tags=tags)
                results.append(_doc_to_out(doc))
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=f"Permission denied: {e}")
    return results


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_doc(doc_id: int, db: AsyncSession = Depends(get_db)):
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_out(doc)


@router.get("/{doc_id}/metadata")
async def get_doc_metadata(doc_id: int, db: AsyncSession = Depends(get_db)):
    """
    Return all storage-layer JSON metadata for a document:
    pageindex tree, graph data, community data, and docstore chunks.
    """
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    stem = Path(doc.file_path).stem
    result: dict[str, Any] = {}

    # PageIndex tree
    pi_path = config.PAGEINDEX_DIR / f"{stem}.json"
    if pi_path.exists():
        with open(pi_path, "r", encoding="utf-8") as f:
            result["pageindex"] = json.load(f)

    # Knowledge graph
    graph_path = config.GRAPH_DIR / f"{stem}_graph.json"
    if graph_path.exists():
        with open(graph_path, "r", encoding="utf-8") as f:
            result["graph"] = json.load(f)

    # Community data
    comm_path = config.GRAPH_DIR / f"{stem}_communities.json"
    if comm_path.exists():
        with open(comm_path, "r", encoding="utf-8") as f:
            result["communities"] = json.load(f)

    # Docstore chunks belonging to this document
    docstore_path = config.STORAGE_DIR / "docstore.json"
    if docstore_path.exists():
        with open(docstore_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        doc_chunks = {
            k: v for k, v in all_chunks.items()
            if v.get("doc_name") == stem
        }
        if doc_chunks:
            result["docstore"] = doc_chunks

    # ChromaDB vector embeddings for this document
    chroma_entries = get_chroma_entries_for_document(stem)
    if chroma_entries:
        result["chroma"] = chroma_entries

    if not result:
        raise HTTPException(status_code=404, detail="No metadata files found for this document")

    return result


@router.delete("/{doc_id}/metadata", response_model=DocumentOut)
async def delete_doc_metadata(
    doc_id: int,
    body: DeleteMetadataRequest | None = Body(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete selective metadata or all metadata for a document. The document record and file are NOT deleted.
    Pass body with keys: [] to delete all metadata, or keys: ["pageindex", "chroma", ...] to delete only those.
    Allowed keys: pageindex, graph, communities, chroma, docstore, figures.
    """
    doc = await delete_document_metadata(db, doc_id, keys=body.keys if body else None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_out(doc)


# Media types for document download/view
_FILE_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xls": "application/vnd.ms-excel",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@router.get("/{doc_id}/pdf")
@router.get("/{doc_id}/file")
async def get_doc_file(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Serve the original document file for viewing or download."""
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    ext = file_path.suffix.lower()
    media_type = _FILE_MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=doc.filename,
        headers={"Content-Disposition": "inline"},
    )


@router.patch("/{doc_id}", response_model=DocumentOut)
async def update_doc(
    doc_id: int,
    update: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
):
    doc = await update_document(db, doc_id, tags=update.tags, notes=update.notes)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_out(doc)


@router.delete("/{doc_id}")
async def delete_doc(doc_id: int, db: AsyncSession = Depends(get_db)):
    success = await delete_document(db, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": "Document deleted"}


@router.post("/bulk-delete")
async def bulk_delete(req: BulkDeleteRequest, db: AsyncSession = Depends(get_db)):
    count = await bulk_delete_documents(db, req.document_ids)
    return {"deleted": count}
