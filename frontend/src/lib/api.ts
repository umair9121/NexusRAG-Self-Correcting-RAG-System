export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

/** Check if backend is reachable (GET /api returns 200). */
export async function isBackendAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const err = await res.text();
    const msg = `API Error ${res.status}: ${err}`;
    const e = new Error(msg) as Error & { status?: number };
    e.status = res.status;
    throw e;
  }
  return res.json();
}

// ── Documents ──────────────────────────────────────────────────────────────

import type {
  Document,
  DocumentListResponse,
  QueryRequest,
  QueryResponse,
  ChatSession,
  ChatMessage,
  SystemHealth,
  IngestionStatus,
  RecoveryResponse,
} from "./types";

/** Response from listDocuments; when backend is unreachable (404) we return empty and set backendUnavailable. */
export type ListDocumentsResult = DocumentListResponse & { backendUnavailable?: boolean };

export async function listDocuments(params: {
  page?: number;
  page_size?: number;
  search?: string;
  status?: string;
  tag?: string;
}): Promise<ListDocumentsResult> {
  const qs = new URLSearchParams();
  if (params.page) qs.set("page", String(params.page));
  if (params.page_size) qs.set("page_size", String(params.page_size));
  if (params.search) qs.set("search", params.search);
  if (params.status) qs.set("status", params.status);
  if (params.tag) qs.set("tag", params.tag);
  try {
    return await request<DocumentListResponse>(`/api/documents?${qs}`);
  } catch (e) {
    const status = (e as Error & { status?: number })?.status;
    if (status === 404 || (e as Error).message?.includes("Failed to fetch")) {
      return {
        documents: [],
        total: 0,
        page: params.page || 1,
        page_size: params.page_size || 50,
        backendUnavailable: true,
      };
    }
    throw e;
  }
}

/** Fetch all document IDs matching current filters (for "select all" across pages). */
export async function listDocumentIds(params: {
  search?: string;
  status?: string;
  tag?: string;
  limit?: number;
}): Promise<{ ids: number[] }> {
  const qs = new URLSearchParams();
  if (params.search) qs.set("search", params.search);
  if (params.status) qs.set("status", params.status);
  if (params.tag) qs.set("tag", params.tag);
  if (params.limit != null) qs.set("limit", String(params.limit));
  return request<{ ids: number[] }>(`/api/documents/ids?${qs}`);
}

export async function getDocument(id: number): Promise<Document> {
  return request<Document>(`/api/documents/${id}`);
}

/** Fetch all storage-layer metadata JSON for a document (pageindex, graph, communities, docstore). */
export async function getDocumentMetadata(id: number): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/documents/${id}/metadata`);
}

export async function uploadDocuments(
  files: FileList | File[],
  pathTags?: string[],
): Promise<Document[]> {
  const form = new FormData();
  const fileList = Array.from(files);
  for (const file of fileList) {
    form.append("files", file);
  }
  if (pathTags && pathTags.length === fileList.length) {
    form.append("path_tags", JSON.stringify(pathTags));
  }
  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function importFolderFromPath(serverPath: string): Promise<Document[]> {
  const res = await fetch(`${API_BASE}/api/documents/import-folder`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: serverPath }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `Import failed: ${res.status}`);
  }
  return res.json();
}

export async function updateDocument(
  id: number,
  data: { tags?: string[]; notes?: string },
): Promise<Document> {
  return request<Document>(`/api/documents/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

/** Allowed document extensions (must match backend ALLOWED_EXTENSIONS). */
export const ALLOWED_DOCUMENT_EXTENSIONS = [
  ".pdf", ".doc", ".docx", ".xlsx", ".xlsm", ".xls",
  ".json", ".html", ".htm", ".png", ".jpg", ".jpeg",
];

/** Get the URL to serve the original document file (view or download). */
export function getDocumentPdfUrl(id: number): string {
  return `${API_BASE}/api/documents/${id}/pdf`;
}

/** Alias for viewing/downloading any document type. */
export function getDocumentFileUrl(id: number): string {
  return `${API_BASE}/api/documents/${id}/file`;
}

export async function deleteDocument(id: number): Promise<void> {
  await request(`/api/documents/${id}`, { method: "DELETE" });
}

export async function bulkDeleteDocuments(ids: number[]): Promise<{ deleted: number }> {
  return request(`/api/documents/bulk-delete`, {
    method: "POST",
    body: JSON.stringify({ document_ids: ids }),
  });
}

/** Metadata keys that can be selectively deleted (match backend METADATA_KEYS). */
export const METADATA_KEYS = [
  "pageindex",
  "graph",
  "communities",
  "chroma",
  "docstore",
  "figures",
] as const;

export type MetadataKey = (typeof METADATA_KEYS)[number];

/**
 * Delete selective document metadata or all metadata. Does NOT delete the document or file.
 * @param id - Document ID
 * @param keys - Optional list of metadata types to delete. If omitted or empty, deletes all.
 */
export async function deleteDocumentMetadata(
  id: number,
  keys?: MetadataKey[] | string[],
): Promise<Document> {
  return request<Document>(`/api/documents/${id}/metadata`, {
    method: "DELETE",
    body: JSON.stringify({ keys: keys ?? [] }),
  });
}

// ── Ingestion ──────────────────────────────────────────────────────────────

export async function startIngestion(
  documentIds: number[],
  pipelines: string[] = ["pageindex", "contextual", "graph"],
  options?: { onlyFailed?: boolean },
): Promise<unknown> {
  return request("/api/ingestion/start", {
    method: "POST",
    body: JSON.stringify({
      document_ids: documentIds,
      pipelines,
      only_failed: options?.onlyFailed ?? false,
    }),
  });
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  return request<IngestionStatus>("/api/ingestion/status");
}

export async function cancelIngestion(docId: number): Promise<void> {
  await request(`/api/ingestion/cancel/${docId}`, { method: "POST" });
}

export interface CancelAllResult {
  detail: string;
  cancelled_count: number;
  cancelled_documents: { document_id: number; filename: string }[];
}

export async function cancelAllIngestion(): Promise<CancelAllResult> {
  return request<CancelAllResult>("/api/ingestion/cancel-all", { method: "POST" });
}

export async function pauseIngestion(): Promise<void> {
  await request("/api/ingestion/pause", { method: "POST" });
}

export async function unpauseIngestion(): Promise<void> {
  await request("/api/ingestion/unpause", { method: "POST" });
}

// ── Ingestion Recovery ─────────────────────────────────────────────────────

export async function getRecoveryJobs(): Promise<RecoveryResponse> {
  return request<RecoveryResponse>("/api/ingestion/recovery");
}

export async function resumeIngestion(documentIds?: number[]): Promise<unknown> {
  return request("/api/ingestion/resume", {
    method: "POST",
    body: JSON.stringify({ document_ids: documentIds ?? null }),
  });
}

export async function dismissRecovery(documentIds?: number[]): Promise<void> {
  await request("/api/ingestion/dismiss-recovery", {
    method: "POST",
    body: JSON.stringify(documentIds ? { document_ids: documentIds } : {}),
  });
}

// ── Query ──────────────────────────────────────────────────────────────────

export async function queryRAG(req: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ── Chat ───────────────────────────────────────────────────────────────────

export async function listChatSessions(): Promise<ChatSession[]> {
  return request<ChatSession[]>("/api/chat/sessions");
}

export async function createChatSession(
  title?: string,
  documentId?: number,
): Promise<ChatSession> {
  return request<ChatSession>("/api/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title || "New Chat", document_id: documentId || null }),
  });
}

export async function getChatMessages(sessionId: number): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/api/chat/sessions/${sessionId}/messages`);
}

export interface ChatSessionState {
  last_selected_model: string | null;
}

export async function getChatSessionState(): Promise<ChatSessionState> {
  return request<ChatSessionState>("/api/chat/session-state");
}

export async function sendChatMessage(
  sessionId: number,
  content: string,
  forceRoute?: string | null,
  temperature?: number,
  model?: string | null,
): Promise<ChatMessage> {
  return request<ChatMessage>(`/api/chat/sessions/${sessionId}/message`, {
    method: "POST",
    body: JSON.stringify({ content, force_route: forceRoute, temperature, model: model ?? undefined }),
  });
}

export async function deleteChatSession(id: number): Promise<void> {
  await request(`/api/chat/sessions/${id}`, { method: "DELETE" });
}

// ── System ─────────────────────────────────────────────────────────────────

export async function getSystemHealth(): Promise<SystemHealth> {
  return request<SystemHealth>("/api/system/health");
}

export interface ResetVectorStoreResponse {
  success: boolean;
  chroma_reset: boolean;
  docstore_entries_cleared: number;
  documents_reset: number;
  message: string;
}

export async function resetVectorStore(): Promise<ResetVectorStoreResponse> {
  return request<ResetVectorStoreResponse>("/api/system/reset-vector-store", {
    method: "POST",
  });
}

// ── CAG ────────────────────────────────────────────────────────────────────

export async function loadCAG(docId: number): Promise<unknown> {
  return request(`/api/chat/cag/load/${docId}`, { method: "POST" });
}

export async function unloadCAG(): Promise<void> {
  await request("/api/chat/cag/unload", { method: "POST" });
}

export async function getCAGStatus(): Promise<Record<string, unknown>> {
  return request("/api/chat/cag/status");
}

// ── Hardware & Performance Options ──────────────────────────────────────

import type { HardwareProfile, PerformanceOptionsResponse } from "./types";

export async function getHardwareProfile(): Promise<HardwareProfile> {
  return request<HardwareProfile>("/api/hardware/profile");
}

export async function getPerformanceOptions(): Promise<PerformanceOptionsResponse> {
  return request<PerformanceOptionsResponse>("/api/hardware/options");
}

export async function updatePerformanceOption(
  key: string,
  enabled: boolean,
): Promise<PerformanceOptionsResponse> {
  return request<PerformanceOptionsResponse>("/api/hardware/options", {
    method: "PATCH",
    body: JSON.stringify({ key, enabled }),
  });
}
