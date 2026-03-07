"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Plus, Trash2, MessageSquare, AlertCircle, RefreshCw, FolderKanban } from "lucide-react";
import {
  createWorkspace,
  createWorkspaceSession,
  deleteWorkspace,
  deleteWorkspaceSession,
  getChatSessionState,
  getSystemHealth,
  getWorkspaceMessages,
  listWorkspaceSessions,
  listWorkspaces,
  sendWorkspaceMessage,
  setWorkspacePreference,
  updateWorkspace,
  updateWorkspaceSession,
  uploadDocuments,
  startIngestion,
} from "@/lib/api";
import type { ChatMessage, ChatSession, Workspace } from "@/lib/types";
import ChatPanel from "@/components/chat/ChatPanel";

function isNetworkError(e: unknown): boolean {
  const msg = (e as Error)?.message ?? "";
  return msg === "Failed to fetch" || msg.includes("NetworkError");
}

export default function WorkspacesPage() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeWorkspace, setActiveWorkspace] = useState<number | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState(false);

  const [forceRoute, setForceRoute] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.2);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const [responseMode, setResponseMode] = useState<"fast" | "deeper">("deeper");
  const [enableVector, setEnableVector] = useState(true);
  const [enableTaxonomy, setEnableTaxonomy] = useState(true);
  const [enablePageIndex, setEnablePageIndex] = useState(true);
  const [enableGraph, setEnableGraph] = useState(true);

  const [editingWorkspaceId, setEditingWorkspaceId] = useState<number | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<number | null>(null);
  const [editingValue, setEditingValue] = useState("");

  const activeWorkspaceObj = useMemo(
    () => workspaces.find((w) => w.id === activeWorkspace) ?? null,
    [workspaces, activeWorkspace],
  );

  const fetchWorkspaces = useCallback(async () => {
    try {
      setBackendUnavailable(false);
      const data = await listWorkspaces();
      setWorkspaces(data);
      if (data.length === 0) {
        setActiveWorkspace(null);
        setSessions([]);
        setActiveSession(null);
        setMessages([]);
        return;
      }
      if (!activeWorkspace || !data.some((w) => w.id === activeWorkspace)) {
        setActiveWorkspace(data[0].id);
      }
    } catch (err) {
      if (isNetworkError(err)) {
        setBackendUnavailable(true);
        setWorkspaces([]);
        setActiveWorkspace(null);
        setSessions([]);
        setActiveSession(null);
        setMessages([]);
      } else {
        console.error("Failed to fetch workspaces:", err);
      }
    }
  }, [activeWorkspace]);

  const fetchSessions = useCallback(async (workspaceId: number) => {
    try {
      const data = await listWorkspaceSessions(workspaceId);
      setSessions(data);
      if (data.length === 0) {
        setActiveSession(null);
        setMessages([]);
      } else if (!activeSession || !data.some((s) => s.id === activeSession)) {
        setActiveSession(data[0].id);
      }
    } catch (err) {
      console.error("Failed to fetch workspace sessions:", err);
    }
  }, [activeSession]);

  useEffect(() => {
    fetchWorkspaces();
  }, [fetchWorkspaces]);

  useEffect(() => {
    if (!activeWorkspace) return;
    fetchSessions(activeWorkspace);
  }, [activeWorkspace, fetchSessions]);

  useEffect(() => {
    const load = async () => {
      try {
        const [state, health] = await Promise.all([
          getChatSessionState(),
          getSystemHealth(),
        ]);
        if (state.last_selected_model) setSelectedModel(state.last_selected_model);
        if (health.ollama_models?.length) setOllamaModels(health.ollama_models);
      } catch {
        // ignore on backend issues
      }
    };
    load();
  }, []);

  const loadMessages = async (sessionId: number) => {
    setActiveSession(sessionId);
    try {
      const msgs = await getWorkspaceMessages(sessionId);
      setMessages(msgs);
    } catch (err) {
      console.error("Failed to load workspace messages:", err);
    }
  };

  const handleNewWorkspace = async () => {
    try {
      const workspace = await createWorkspace("New Workspace");
      await fetchWorkspaces();
      setActiveWorkspace(workspace.id);
      setSessions([]);
      setActiveSession(null);
      setMessages([]);
    } catch (err) {
      console.error("Failed to create workspace:", err);
    }
  };

  const handleDeleteWorkspace = async (workspaceId: number) => {
    try {
      await deleteWorkspace(workspaceId);
      await fetchWorkspaces();
    } catch (err) {
      console.error("Failed to delete workspace:", err);
    }
  };

  const handleNewSession = async () => {
    if (!activeWorkspace) return;
    try {
      const session = await createWorkspaceSession(activeWorkspace, "New Chat");
      await fetchSessions(activeWorkspace);
      await loadMessages(session.id);
      await fetchWorkspaces();
    } catch (err) {
      console.error("Failed to create workspace chat:", err);
    }
  };

  const handleDeleteSession = async (sessionId: number) => {
    try {
      await deleteWorkspaceSession(sessionId);
      if (activeSession === sessionId) {
        setActiveSession(null);
        setMessages([]);
      }
      if (activeWorkspace) {
        await fetchSessions(activeWorkspace);
      }
      await fetchWorkspaces();
    } catch (err) {
      console.error("Failed to delete workspace chat:", err);
    }
  };

  const handleSend = async (content: string) => {
    if (!activeSession) return;
    const sentAt = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
    const userMsg: ChatMessage = {
      role: "user",
      content,
      trace: [],
      metadata: {},
      route: "",
      sent_at: sentAt,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    try {
      const assistantMsg = await sendWorkspaceMessage(activeSession, content, {
        force_route: forceRoute,
        temperature,
        model: selectedModel,
        fast_response: responseMode === "fast",
        enable_vector: enableVector,
        enable_taxonomy: enableTaxonomy,
        enable_pageindex: enablePageIndex,
        enable_graph: enableGraph,
      });
      setMessages((prev) => [...prev, assistantMsg]);
      if (activeWorkspace) {
        await fetchSessions(activeWorkspace);
      }
      await fetchWorkspaces();
    } catch (err) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `Error: ${(err as Error).message}`,
        trace: [],
        metadata: {},
        route: "",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSetPreference = async (optionText: string, accepted: boolean) => {
    if (!activeSession) return;
    try {
      await setWorkspacePreference(activeSession, optionText, accepted);
    } catch (err) {
      console.error("Failed to save workspace preference:", err);
    }
  };

  const handleUploadFiles = async (files: File[]) => {
    if (!activeSession || files.length === 0) return;
    setUploadingFiles(true);
    try {
      const uploadedDocs = await uploadDocuments(files);
      const documentIds = uploadedDocs.map((d) => d.id);
      if (documentIds.length > 0) {
        await startIngestion(documentIds, ["pageindex", "contextual", "graph"]);
      }
      const filenames = uploadedDocs.map((d) => d.filename).slice(0, 3);
      const extraCount = Math.max(0, uploadedDocs.length - filenames.length);
      const msg: ChatMessage = {
        role: "assistant",
        content:
          uploadedDocs.length > 0
            ? `Uploaded ${uploadedDocs.length} file(s): ${filenames.join(", ")}${extraCount > 0 ? ` (+${extraCount} more)` : ""}. Ingestion started (PageIndex + Vector + Graph). Screenshot files will go through VLM UI analysis and WCAG compliance audit automatically.`
            : "No supported files were uploaded.",
        trace: [],
        metadata: {},
        route: "",
      };
      setMessages((prev) => [...prev, msg]);
      if (activeWorkspace) await fetchSessions(activeWorkspace);
      await fetchWorkspaces();
    } catch (err) {
      const msg: ChatMessage = {
        role: "assistant",
        content: `Upload failed: ${(err as Error).message}`,
        trace: [],
        metadata: {},
        route: "",
      };
      setMessages((prev) => [...prev, msg]);
    } finally {
      setUploadingFiles(false);
    }
  };

  const handleRenameWorkspace = async (workspaceId: number, newName: string) => {
    const name = newName.trim() || "New Workspace";
    setEditingWorkspaceId(null);
    setEditingValue("");
    try {
      await updateWorkspace(workspaceId, name);
      await fetchWorkspaces();
    } catch (err) {
      console.error("Failed to rename workspace:", err);
    }
  };

  const handleRenameSession = async (sessionId: number, newTitle: string) => {
    const title = newTitle.trim() || "New Chat";
    setEditingSessionId(null);
    setEditingValue("");
    try {
      await updateWorkspaceSession(sessionId, title);
      if (activeWorkspace) await fetchSessions(activeWorkspace);
      await fetchWorkspaces();
    } catch (err) {
      console.error("Failed to rename chat:", err);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {backendUnavailable && (
        <div className="alert-warning-banner flex items-center gap-3 px-4 py-2 rounded-lg mx-4 mt-2 shrink-0">
          <AlertCircle size={18} className="shrink-0" />
          <span className="text-sm">Backend not available. Start the backend to use workspaces.</span>
          <button
            type="button"
            onClick={() => fetchWorkspaces()}
            className="alert-warning-btn ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded text-sm"
          >
            <RefreshCw size={14} />
            Retry
          </button>
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <div className="w-72 border-r border-[var(--border)] bg-[var(--bg-secondary)] flex flex-col shrink-0">
          <div className="p-3 border-b border-[var(--border)] space-y-2">
            <button
              onClick={handleNewWorkspace}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--accent)] text-white rounded-lg text-sm hover:bg-[var(--accent-hover)]"
            >
              <FolderKanban size={16} />
              New Workspace
            </button>
            <button
              onClick={handleNewSession}
              disabled={!activeWorkspace}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--bg-tertiary)] text-[var(--text-primary)] rounded-lg text-sm border border-[var(--border)] hover:bg-[var(--bg-primary)] disabled:opacity-60"
            >
              <Plus size={16} />
              New Chat
            </button>
          </div>

          <div className="px-3 pt-3 text-xs uppercase tracking-wide text-[var(--text-secondary)]">Workspaces</div>
          <div className="max-h-56 overflow-y-auto border-b border-[var(--border)] mt-2">
            {workspaces.map((w) => (
              <div
                key={w.id}
                className={`flex items-center gap-2 px-3 py-2.5 cursor-pointer transition-colors ${
                  activeWorkspace === w.id ? "bg-[var(--bg-tertiary)]" : "hover:bg-[var(--bg-tertiary)]/50"
                }`}
                onClick={() => {
                  if (editingWorkspaceId !== w.id) {
                    setActiveWorkspace(w.id);
                    setActiveSession(null);
                    setMessages([]);
                  }
                }}
                onDoubleClick={(e) => {
                  e.stopPropagation();
                  setEditingWorkspaceId(w.id);
                  setEditingValue(w.name);
                  setEditingSessionId(null);
                }}
              >
                <FolderKanban size={14} className="text-[var(--text-secondary)] shrink-0" />
                <div className="flex-1 min-w-0">
                  {editingWorkspaceId === w.id ? (
                    <input
                      type="text"
                      value={editingValue}
                      onChange={(e) => setEditingValue(e.target.value)}
                      onBlur={() => handleRenameWorkspace(w.id, editingValue)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleRenameWorkspace(w.id, editingValue);
                        if (e.key === "Escape") {
                          setEditingWorkspaceId(null);
                          setEditingValue("");
                        }
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-full text-sm bg-[var(--bg-primary)] border border-[var(--accent)] rounded px-1.5 py-0.5 text-[var(--text-primary)] focus:outline-none"
                      autoFocus
                    />
                  ) : (
                    <div className="text-sm truncate">{w.name}</div>
                  )}
                  <div className="text-xs text-[var(--text-secondary)]">{w.chat_count} chats</div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteWorkspace(w.id);
                  }}
                  className="p-1 rounded hover:bg-[var(--bg-primary)] text-[var(--text-secondary)] hover:text-[var(--error)]"
                  title="Delete workspace and its chats"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
            {workspaces.length === 0 && (
              <div className="p-4 text-sm text-[var(--text-secondary)] text-center">No workspaces yet</div>
            )}
          </div>

          <div className="px-3 pt-3 text-xs uppercase tracking-wide text-[var(--text-secondary)]">Chats</div>
          <div className="flex-1 overflow-y-auto mt-2">
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`flex items-center gap-2 px-3 py-2.5 cursor-pointer border-b border-[var(--border)] transition-colors ${
                  activeSession === s.id ? "bg-[var(--bg-tertiary)]" : "hover:bg-[var(--bg-tertiary)]/50"
                }`}
                onClick={() => {
                  if (editingSessionId !== s.id) loadMessages(s.id);
                }}
                onDoubleClick={(e) => {
                  e.stopPropagation();
                  setEditingSessionId(s.id);
                  setEditingValue(s.title);
                  setEditingWorkspaceId(null);
                }}
              >
                <MessageSquare size={14} className="text-[var(--text-secondary)] shrink-0" />
                <div className="flex-1 min-w-0">
                  {editingSessionId === s.id ? (
                    <input
                      type="text"
                      value={editingValue}
                      onChange={(e) => setEditingValue(e.target.value)}
                      onBlur={() => handleRenameSession(s.id, editingValue)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleRenameSession(s.id, editingValue);
                        if (e.key === "Escape") {
                          setEditingSessionId(null);
                          setEditingValue("");
                        }
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-full text-sm bg-[var(--bg-primary)] border border-[var(--accent)] rounded px-1.5 py-0.5 text-[var(--text-primary)] focus:outline-none"
                      autoFocus
                    />
                  ) : (
                    <div className="text-sm truncate">{s.title}</div>
                  )}
                  <div className="text-xs text-[var(--text-secondary)]">{s.message_count} messages</div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(s.id);
                  }}
                  className="p-1 rounded hover:bg-[var(--bg-primary)] text-[var(--text-secondary)] hover:text-[var(--error)]"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && (
              <div className="p-4 text-sm text-[var(--text-secondary)] text-center">
                {activeWorkspaceObj ? "No chats in this workspace yet" : "Select or create a workspace"}
              </div>
            )}
          </div>
        </div>

        <div className="flex-1">
          {activeSession ? (
            <ChatPanel
              messages={messages}
              onSend={handleSend}
              onUploadFiles={handleUploadFiles}
              uploadingFiles={uploadingFiles}
              onSetPreference={handleSetPreference}
              loading={loading}
              forceRoute={forceRoute}
              onRouteChange={setForceRoute}
              temperature={temperature}
              onTemperatureChange={setTemperature}
              ollamaModels={ollamaModels}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
              responseMode={responseMode}
              onResponseModeChange={setResponseMode}
              enableVector={enableVector}
              enableTaxonomy={enableTaxonomy}
              enablePageIndex={enablePageIndex}
              enableGraph={enableGraph}
              onEnableVectorChange={setEnableVector}
              onEnableTaxonomyChange={setEnableTaxonomy}
              onEnablePageIndexChange={setEnablePageIndex}
              onEnableGraphChange={setEnableGraph}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-[var(--text-secondary)]">
              <div className="text-center">
                <FolderKanban size={48} className="mx-auto mb-3 opacity-30" />
                <p>Select a workspace chat to begin</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
