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
  DocumentFacets,
  DocumentListResponse,
  QueryRequest,
  QueryResponse,
  ChatSession,
  ChatMessage,
  Workspace,
  SystemHealth,
  IngestionStatus,
  RecoveryResponse,
} from "./types";

/** Response from listDocuments; when backend is unreachable (404) we return empty and set backendUnavailable. */
export type ListDocumentsResult = DocumentListResponse & { backendUnavailable?: boolean };

export type DocumentSortBy = "name" | "created_at" | "page_count" | "status" | "chunk_count" | "ingestion_duration";
export type DocumentSortOrder = "asc" | "desc";

export async function listDocuments(params: {
  page?: number;
  page_size?: number;
  search?: string;
  status?: string;
  tag?: string;
  needs_ingestion?: boolean;
  sort_by?: DocumentSortBy;
  sort_order?: DocumentSortOrder;
}): Promise<ListDocumentsResult> {
  const qs = new URLSearchParams();
  if (params.page) qs.set("page", String(params.page));
  if (params.page_size) qs.set("page_size", String(params.page_size));
  if (params.search) qs.set("search", params.search);
  if (params.status) qs.set("status", params.status);
  if (params.tag) qs.set("tag", params.tag);
  if (params.needs_ingestion === true) qs.set("needs_ingestion", "true");
  if (params.sort_by) qs.set("sort_by", params.sort_by);
  if (params.sort_order) qs.set("sort_order", params.sort_order);
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
  needs_ingestion?: boolean;
  limit?: number;
}): Promise<{ ids: number[] }> {
  const qs = new URLSearchParams();
  if (params.search) qs.set("search", params.search);
  if (params.status) qs.set("status", params.status);
  if (params.tag) qs.set("tag", params.tag);
  if (params.needs_ingestion === true) qs.set("needs_ingestion", "true");
  if (params.limit != null) qs.set("limit", String(params.limit));
  return request<{ ids: number[] }>(`/api/documents/ids?${qs}`);
}

/** Fetch document counts by status, by tag, and needs_ingestion_count for filter UI. */
export async function getDocumentFacets(params?: {
  search?: string;
  status?: string;
  tag?: string;
}): Promise<DocumentFacets> {
  const qs = new URLSearchParams();
  if (params?.search) qs.set("search", params.search);
  if (params?.status) qs.set("status", params.status);
  if (params?.tag) qs.set("tag", params.tag);
  const suffix = qs.toString() ? `?${qs}` : "";
  return request<DocumentFacets>(`/api/documents/facets${suffix}`);
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
  data: { tags?: string[]; path_hierarchy?: string[]; notes?: string },
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

export interface ChatMessageOptions {
  force_route?: string | null;
  temperature?: number;
  model?: string | null;
  fast_response?: boolean;
  enable_vector?: boolean;
  enable_taxonomy?: boolean;
  enable_pageindex?: boolean;
  enable_graph?: boolean;
}

export async function sendChatMessage(
  sessionId: number,
  content: string,
  options?: ChatMessageOptions,
): Promise<ChatMessage> {
  const {
    force_route: forceRoute,
    temperature,
    model,
    fast_response: fastResponse,
    enable_vector: enableVector,
    enable_taxonomy: enableTaxonomy,
    enable_pageindex: enablePageIndex,
    enable_graph: enableGraph,
  } = options ?? {};
  return request<ChatMessage>(`/api/chat/sessions/${sessionId}/message`, {
    method: "POST",
    body: JSON.stringify({
      content,
      force_route: forceRoute ?? undefined,
      temperature,
      model: model ?? undefined,
      fast_response: fastResponse ?? false,
      enable_vector: enableVector ?? true,
      enable_taxonomy: enableTaxonomy ?? true,
      enable_pageindex: enablePageIndex ?? true,
      enable_graph: enableGraph ?? true,
    }),
  });
}

export async function setChatPreference(
  sessionId: number,
  optionText: string,
  accepted: boolean,
): Promise<{ detail: string }> {
  return request<{ detail: string }>(`/api/chat/sessions/${sessionId}/preference`, {
    method: "POST",
    body: JSON.stringify({ option_text: optionText, accepted }),
  });
}

export async function updateChatSession(id: number, title: string): Promise<ChatSession> {
  return request<ChatSession>(`/api/chat/sessions/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title: title.trim() || "New Chat" }),
  });
}

export async function deleteChatSession(id: number): Promise<void> {
  await request(`/api/chat/sessions/${id}`, { method: "DELETE" });
}

// ── Workspaces ──────────────────────────────────────────────────────────────

export async function listWorkspaces(): Promise<Workspace[]> {
  return request<Workspace[]>("/api/workspaces");
}

export async function createWorkspace(name?: string): Promise<Workspace> {
  return request<Workspace>("/api/workspaces", {
    method: "POST",
    body: JSON.stringify({ name: name || "New Workspace" }),
  });
}

export async function updateWorkspace(id: number, name: string): Promise<Workspace> {
  return request<Workspace>(`/api/workspaces/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

export async function deleteWorkspace(id: number): Promise<void> {
  await request(`/api/workspaces/${id}`, { method: "DELETE" });
}

export async function listWorkspaceSessions(workspaceId: number): Promise<ChatSession[]> {
  return request<ChatSession[]>(`/api/workspaces/${workspaceId}/sessions`);
}

export async function createWorkspaceSession(
  workspaceId: number,
  title?: string,
  documentId?: number,
): Promise<ChatSession> {
  return request<ChatSession>(`/api/workspaces/${workspaceId}/sessions`, {
    method: "POST",
    body: JSON.stringify({
      title: title || "New Chat",
      document_id: documentId || null,
      workspace_id: workspaceId,
    }),
  });
}

export async function getWorkspaceMessages(sessionId: number): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/api/workspaces/sessions/${sessionId}/messages`);
}

export async function setWorkspacePreference(
  sessionId: number,
  optionText: string,
  accepted: boolean,
): Promise<{ detail: string }> {
  return request<{ detail: string }>(`/api/workspaces/sessions/${sessionId}/preference`, {
    method: "POST",
    body: JSON.stringify({ option_text: optionText, accepted }),
  });
}

export async function sendWorkspaceMessage(
  sessionId: number,
  content: string,
  options?: ChatMessageOptions,
): Promise<ChatMessage> {
  const {
    force_route: forceRoute,
    temperature,
    model,
    fast_response: fastResponse,
    enable_vector: enableVector,
    enable_taxonomy: enableTaxonomy,
    enable_pageindex: enablePageIndex,
    enable_graph: enableGraph,
  } = options ?? {};
  return request<ChatMessage>(`/api/workspaces/sessions/${sessionId}/message`, {
    method: "POST",
    body: JSON.stringify({
      content,
      force_route: forceRoute ?? undefined,
      temperature,
      model: model ?? undefined,
      fast_response: fastResponse ?? false,
      enable_vector: enableVector ?? true,
      enable_taxonomy: enableTaxonomy ?? true,
      enable_pageindex: enablePageIndex ?? true,
      enable_graph: enableGraph ?? true,
    }),
  });
}

export async function updateWorkspaceSession(sessionId: number, title: string): Promise<ChatSession> {
  return request<ChatSession>(`/api/workspaces/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title: title.trim() || "New Chat" }),
  });
}

export async function deleteWorkspaceSession(sessionId: number): Promise<void> {
  await request(`/api/workspaces/sessions/${sessionId}`, { method: "DELETE" });
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

import type {
  HardwareProfile,
  PerformanceOptionsResponse,
  KnowledgeBaseStats,
  KnowledgeBaseRefreshResult,
  KnowledgeBaseEntity,
} from "./types";

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

// ── Knowledge Base ─────────────────────────────────────────────────────

export async function getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
  return request<KnowledgeBaseStats>("/api/system/knowledge-base/stats");
}

export async function refreshKnowledgeBase(model?: string): Promise<KnowledgeBaseRefreshResult> {
  return request<KnowledgeBaseRefreshResult>("/api/system/knowledge-base/refresh", {
    method: "POST",
    body: model && model.trim() ? JSON.stringify({ model: model.trim() }) : undefined,
  });
}

export async function listKnowledgeBaseEntities(params?: {
  entity_type?: string;
  search?: string;
  limit?: number;
}): Promise<{ entities: KnowledgeBaseEntity[] }> {
  const qs = new URLSearchParams();
  if (params?.entity_type) qs.set("entity_type", params.entity_type);
  if (params?.search) qs.set("search", params.search);
  if (params?.limit) qs.set("limit", String(params.limit));
  return request<{ entities: KnowledgeBaseEntity[] }>(`/api/system/knowledge-base/entities?${qs}`);
}
