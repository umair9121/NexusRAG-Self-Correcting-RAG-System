export interface Document {
  id: number;
  filename: string;
  file_size: number;
  page_count: number;
  status: "uploaded" | "ingesting" | "ready" | "error";
  tags: string[];
  notes: string;
  created_at: string;
  updated_at: string;
  extraction_status: string;
  edi_status: string;
  graph_status: string;
  node_count: number;
  edge_count: number;
  edi_rating: string;
  error_message: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  page: number;
  page_size: number;
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
  created_at: string;
  message_count: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  trace: string[];
  metadata: Record<string, unknown>;
  route: string;
}

export interface SystemHealth {
  ollama_connected: boolean;
  ollama_models: string[];
  available_ram_gb: number;
  total_documents: number;
  documents_ready: number;
  documents_ingesting: number;
  knowledge_graphs: string[];
  cag_status: Record<string, unknown>;
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
