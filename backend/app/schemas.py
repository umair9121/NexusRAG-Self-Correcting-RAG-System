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
    path_hierarchy: list[str] = []
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
    ingestion_duration_seconds: int | None = None
    ingestion_type: str = "document"   # document | screenshot | regulatory
    audit_result: dict[str, Any] | None = None  # UI audit JSON when ingestion_type is screenshot

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total: int
    page: int
    page_size: int


class DocumentIdsResponse(BaseModel):
    ids: list[int]


class DocumentFacetsResponse(BaseModel):
    """Counts by status, by tag, and total needing ingestion (for filter UI)."""
    by_status: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    needs_ingestion_count: int = 0


class DocumentUpdate(BaseModel):
    tags: list[str] | None = None
    path_hierarchy: list[str] | None = None
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

class WorkspaceOut(BaseModel):
    id: int
    name: str
    workspace_cache: str = ""
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime
    chat_count: int = 0

    class Config:
        from_attributes = True


class WorkspaceCreate(BaseModel):
    name: str = "New Workspace"


class WorkspaceUpdate(BaseModel):
    name: str


class ChatSessionOut(BaseModel):
    id: int
    title: str
    document_id: int | None
    workspace_id: int | None = None
    created_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    title: str = "New Chat"
    document_id: int | None = None
    workspace_id: int | None = None


class ChatSessionUpdate(BaseModel):
    title: str


class ChatMessageRequest(BaseModel):
    content: str
    force_route: str | None = None
    temperature: float = 0.2
    model: str | None = None
    # Fast Response = vector search + LLM only; Deeper Thinking = full adaptive RAG
    fast_response: bool = False
    # For Deeper Thinking: which retrievers to use (all True = current behaviour)
    enable_vector: bool = True
    enable_taxonomy: bool = True
    enable_pageindex: bool = True
    enable_graph: bool = True


class ChatPreferenceRequest(BaseModel):
    option_text: str
    accepted: bool


class ChatMessageOut(BaseModel):
    role: str
    content: str
    trace: list[str] = []
    metadata: dict[str, Any] = {}
    route: str = ""
    # Time tracking (ISO 8601 with second precision, UTC)
    sent_at: str | None = None
    response_received_at: str | None = None


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
    # When > 0, UI can suggest skipping Graph for docs with page_count >= this.
    graph_skip_page_threshold: int = 0


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


class RefreshKnowledgeBaseRequest(BaseModel):
    """Optional model override for entity classification. Default is INTERNAL_REASONING_AGENT (e.g. phi3:3.8b)."""
    model: str | None = None


# ── UI / Accessibility audit (SOM + compliance) ─────────────────────────────

class BoundingBox(BaseModel):
    """Normalized or pixel coordinates for a violating element (e.g. [x_min, y_min, x_max, y_max] or left, top, width, height)."""
    left: float = 0.0
    top: float = 0.0
    width: float = 0.0
    height: float = 0.0
    # Optional: normalized 0–1 for image-agnostic use
    normalized: bool = False


class DetectedViolation(BaseModel):
    """A single EDI/accessibility or regulatory violation with location and guideline reference."""
    guideline_reference: str = ""   # e.g. "WCAG 2.2 Success Criterion 1.4.3"
    issue_description: str = ""
    severity: str = "medium"       # low | medium | high | critical
    bounding_box: BoundingBox | None = None
    element_type: str = ""         # e.g. "button", "text", "image"
    suggested_fix: str = ""


class AuditMetadata(BaseModel):
    screen_id: str = ""
    timestamp: str = ""
    evaluation_standard: str = ""   # e.g. "WCAG 2.2 Level AA"
    overall_compliance_status: str = ""  # Pass | Failed | Partial


class GenerativeSynthesis(BaseModel):
    global_edit_prompt: str = ""
    negative_prompt: str = ""


class UIAuditResult(BaseModel):
    """Structured output from VLM + Graph RAG compliance audit for a UI screenshot."""
    audit_metadata: AuditMetadata = Field(default_factory=AuditMetadata)
    detected_violations: list[DetectedViolation] = Field(default_factory=list)
    generative_synthesis: GenerativeSynthesis = Field(default_factory=GenerativeSynthesis)
