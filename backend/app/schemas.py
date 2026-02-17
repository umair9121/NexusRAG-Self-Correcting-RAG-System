"""
Pydantic request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Documents ───────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: int
    filename: str
    file_size: int
    page_count: int
    status: str
    tags: list[str] = []
    notes: str = ""
    created_at: datetime
    updated_at: datetime
    pageindex_status: str
    contextual_status: str
    graph_status: str
    chunk_count: int
    node_count: int
    edge_count: int
    error_message: str = ""

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total: int
    page: int
    page_size: int


class DocumentIdsResponse(BaseModel):
    ids: list[int]


class DocumentUpdate(BaseModel):
    tags: list[str] | None = None
    notes: str | None = None


class BulkDeleteRequest(BaseModel):
    document_ids: list[int]


class DeleteMetadataRequest(BaseModel):
    """Optional list of metadata keys to delete. If empty or omitted, all metadata is deleted."""
    keys: list[str] = []


class ImportFolderRequest(BaseModel):
    path: str


# ── Ingestion ───────────────────────────────────────────────────────────────

class IngestionRequest(BaseModel):
    document_ids: list[int]
    pipelines: list[str] = Field(default=["pageindex", "contextual", "graph"])
    only_failed: bool = Field(
        default=False,
        description="If True, run only pipelines that are 'error' or 'pending' for each document (re-run failed/pending stages only).",
    )


class IngestionStatusOut(BaseModel):
    active_job: dict[str, Any] | None = None
    queue_length: int = 0
    queue: list[dict[str, Any]] = []
    paused: bool = False


# ── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    force_route: str | None = None
    temperature: float = 0.2


class QueryResponse(BaseModel):
    answer: str
    route: str
    trace: list[str]
    context: str = ""
    metadata: dict[str, Any] = {}
    retries: int = 0
    error: str = ""


# ── Chat ────────────────────────────────────────────────────────────────────

class ChatSessionOut(BaseModel):
    id: int
    title: str
    document_id: int | None
    created_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    title: str = "New Chat"
    document_id: int | None = None


class ChatMessageRequest(BaseModel):
    content: str
    force_route: str | None = None
    temperature: float = 0.2
    model: str | None = None


class ChatMessageOut(BaseModel):
    role: str
    content: str
    trace: list[str] = []
    metadata: dict[str, Any] = {}
    route: str = ""


# ── System ──────────────────────────────────────────────────────────────────

class SystemHealthResponse(BaseModel):
    ollama_connected: bool
    ollama_models: list[str] = []
    available_ram_gb: float
    total_documents: int
    documents_ready: int
    documents_ingesting: int
    pageindex_trees: list[str] = []
    knowledge_graphs: list[str] = []
    cag_status: dict[str, Any] = {}


# ── Hardware & Performance Options ──────────────────────────────────────

class CPUInfoOut(BaseModel):
    vendor: str = "Unknown"
    model_name: str = ""
    core_count: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    has_amx: bool = False
    has_npu: bool = False


class GPUInfoOut(BaseModel):
    vendor: str = "Unknown"
    model_name: str = ""
    vram_mb: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    rocm_available: bool = False


class HardwareProfileResponse(BaseModel):
    cpu: CPUInfoOut
    gpus: list[GPUInfoOut] = []
    os: str = ""
    platform_arch: str = ""
    available_vendors: list[str] = []


class PerformanceOption(BaseModel):
    """Single toggleable performance option."""
    key: str
    label: str
    description: str = ""
    enabled: bool = False
    available: bool = True
    category: str = "General"


class PerformanceOptionsResponse(BaseModel):
    categories: dict[str, list[PerformanceOption]] = {}
    hardware: HardwareProfileResponse


class PerformanceOptionUpdate(BaseModel):
    key: str
    enabled: bool
