export interface Document {
  id: number;
  filename: string;
  file_size: number;
  page_count: number;
  status: "uploaded" | "ingesting" | "ready" | "error";
  tags: string[];
  /** Ordered path segments (e.g. vessel → category → folder) for hierarchy/breadcrumbs. */
  path_hierarchy?: string[];
  notes: string;
  created_at: string;
  updated_at: string;
  pageindex_status: string;
  contextual_status: string;
  graph_status: string;
  chunk_count: number;
  node_count: number;
  edge_count: number;
  error_message: string;
  /** Seconds to complete all 3 pipelines; null if ingestion never completed. */
  ingestion_duration_seconds?: number | null;
  ingestion_type?: string;
  audit_result?: {
    audit_metadata?: {
      screen_id?: string;
      evaluation_standard?: string;
      overall_compliance_status?: string;
    };
    detected_violations?: {
      guideline_reference?: string;
      issue_description?: string;
      severity?: string;
      element_type?: string;
      suggested_fix?: string;
    }[];
    generative_synthesis?: {
      compliance_summary?: string;
    };
  } | null;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  page: number;
  page_size: number;
}

/** Counts by status, by tag, and needs_ingestion for filter UI. */
export interface DocumentFacets {
  by_status: Record<string, number>;
  by_tag: Record<string, number>;
  needs_ingestion_count: number;
}

export interface QueryRequest {
  query: string;
  force_route?: string | null;
  temperature?: number;
}

export interface QueryResponse {
  answer: string;
  route: string;
  trace: string[];
  context: string;
  metadata: Record<string, unknown>;
  retries: number;
  error: string;
}

export interface ChatSession {
  id: number;
  title: string;
  document_id: number | null;
  workspace_id?: number | null;
  created_at: string;
  message_count: number;
}

export interface Workspace {
  id: number;
  name: string;
  workspace_cache: string;
  created_at: string;
  updated_at: string;
  last_used_at: string;
  chat_count: number;
}

/** A document reference cited in a chat response (from RAG metadata). */
export interface SourceReference {
  document_id: number;
  filename: string;
  pages?: number[];
  primary_page?: number;
  snippet?: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  trace: string[];
  metadata: Record<string, unknown>;
  route: string;
  /** When the message was sent (ISO 8601, second precision). */
  sent_at?: string | null;
  /** When the assistant response was received (assistant messages only). */
  response_received_at?: string | null;
}

export interface SystemHealth {
  ollama_connected: boolean;
  ollama_models: string[];
  available_ram_gb: number;
  total_documents: number;
  documents_ready: number;
  documents_ingesting: number;
  pageindex_trees: string[];
  knowledge_graphs: string[];
  cag_status: Record<string, unknown>;
  /** When > 0, suggest skipping Graph for docs with page_count >= this. */
  graph_skip_page_threshold?: number;
}

export interface IngestionJobInfo {
  document_id: number;
  filename: string;
  pipelines: string[];
  status: string;
  current_pipeline?: string;
  error_message?: string;
}

export interface IngestionStatus {
  active_job: IngestionJobInfo | null;
  queue_length: number;
  queue: IngestionJobInfo[];
  paused?: boolean;
}

// ── Ingestion Recovery ──────────────────────────────────────────────────

export interface RecoveryJob {
  document_id: number;
  filename: string;
  file_path: string;
  pipelines: string[];
  completed_pipelines: string[];
  remaining_pipelines: string[];
  current_pipeline: string;
  status: string;
}

export interface RecoveryResponse {
  jobs: RecoveryJob[];
  count: number;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
  document_id: number | null;
  extra?: Record<string, unknown>;
}

// ── Hardware & Performance Options ─────────────────────────────────────

export interface CPUInfo {
  vendor: string;
  model_name: string;
  core_count: number;
  has_avx2: boolean;
  has_avx512: boolean;
  has_amx: boolean;
  has_npu: boolean;
}

export interface GPUInfo {
  vendor: string;
  model_name: string;
  vram_mb: number;
  driver_version: string;
  cuda_version: string;
  rocm_available: boolean;
}

export interface HardwareProfile {
  cpu: CPUInfo;
  gpus: GPUInfo[];
  os: string;
  platform_arch: string;
  available_vendors: string[];
}

export interface PerformanceOption {
  key: string;
  label: string;
  description: string;
  enabled: boolean;
  available: boolean;
  category: string;
}

export interface PerformanceOptionsResponse {
  categories: Record<string, PerformanceOption[]>;
  hardware: HardwareProfile;
}

// ── Knowledge Base ────────────────────────────────────────────────────

export interface KnowledgeBaseEntity {
  name: string;
  types: string[];
  description: string;
  mentions: number;
  documents: string[];
}

export interface KnowledgeBaseStats {
  entity_count: number;
  relation_count: number;
  document_count: number;
  last_updated: string | null;
  types: Record<string, number>;
  top_entities: { name: string; mentions: number; types: string[] }[];
  /** Default model for entity classification (e.g. phi3:3.8b). */
  default_classification_model?: string;
}

export interface KnowledgeBaseRefreshResult {
  success: boolean;
  documents_processed: number;
  total_entities: number;
  total_relations: number;
  newly_classified: number;
  message: string;
  /** Model used for entity classification this run. */
  model_used?: string;
}
