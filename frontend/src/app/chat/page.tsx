"use client";

import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, MessageSquare, AlertCircle, RefreshCw } from "lucide-react";
import {
  listChatSessions,
  createChatSession,
  getChatMessages,
  sendChatMessage,
  setChatPreference,
  deleteChatSession,
  updateChatSession,
  getChatSessionState,
  getSystemHealth,
  uploadDocuments,
  startIngestion,
  getDocument,
} from "@/lib/api";
import type { ChatSession, ChatMessage } from "@/lib/types";
import ChatPanel from "@/components/chat/ChatPanel";

function isNetworkError(e: unknown): boolean {
  const msg = (e as Error)?.message ?? "";
  return msg === "Failed to fetch" || msg.includes("NetworkError");
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [forceRoute, setForceRoute] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.2);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  // Response mode: fast = vector + LLM only; deeper = full adaptive RAG
  const [responseMode, setResponseMode] = useState<"fast" | "deeper">("deeper");
  // Deeper Thinking toggles (only apply when responseMode === "deeper")
  const [enableVector, setEnableVector] = useState(true);
  const [enableTaxonomy, setEnableTaxonomy] = useState(true);
  const [enablePageIndex, setEnablePageIndex] = useState(true);
  const [enableGraph, setEnableGraph] = useState(true);

  const [editingSessionId, setEditingSessionId] = useState<number | null>(null);
  const [editingValue, setEditingValue] = useState("");

  const CHAT_LAST_SESSION_KEY = "chat_last_session_id";

  const fetchSessions = useCallback(async (restoreSession = false) => {
    try {
      setBackendUnavailable(false);
      const data = await listChatSessions();
      setSessions(data);
      // Only restore last session on initial load (refresh / navigate back), not after every upload/send
      if (restoreSession && data.length > 0) {
        const raw = localStorage.getItem(CHAT_LAST_SESSION_KEY);
        const parsedId = raw ? parseInt(raw, 10) : NaN;
        const sessionId = Number.isFinite(parsedId) && data.some((s) => s.id === parsedId)
          ? parsedId
          : data[0].id;
        setActiveSession(sessionId);
        const msgs = await getChatMessages(sessionId);
        setMessages(msgs);
      }
    } catch (err) {
      if (isNetworkError(err)) {
        setSessions([]);
        setBackendUnavailable(true);
      } else {
        console.error("Failed to fetch sessions:", err);
      }
    }
  }, []);

  useEffect(() => {
    fetchSessions(true);
  }, [fetchSessions]);

  // Load session state (last selected model) and available Ollama models on chat page load
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
        // ignore; backend may be offline
      }
    };
    load();
  }, []);

  const loadMessages = async (sessionId: number) => {
    setActiveSession(sessionId);
    try {
      localStorage.setItem(CHAT_LAST_SESSION_KEY, String(sessionId));
      const msgs = await getChatMessages(sessionId);
      setMessages(msgs);
    } catch (err) {
      console.error("Failed to load messages:", err);
    }
  };

  const handleNewSession = async () => {
    try {
      const session = await createChatSession("New Chat");
      await fetchSessions();
      await loadMessages(session.id);
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  };

  const handleDeleteSession = async (id: number) => {
    try {
      await deleteChatSession(id);
      if (activeSession === id) {
        setActiveSession(null);
        setMessages([]);
        localStorage.removeItem(CHAT_LAST_SESSION_KEY);
      }
      await fetchSessions();
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const handleRenameSession = async (sessionId: number, newTitle: string) => {
    const title = newTitle.trim() || "New Chat";
    setEditingSessionId(null);
    setEditingValue("");
    try {
      await updateChatSession(sessionId, title);
      await fetchSessions();
    } catch (err) {
      console.error("Failed to rename chat:", err);
    }
  };

  const handleSend = async (content: string) => {
    if (!activeSession) return;
    // Time tracking: record when chat is sent (second precision)
    const sentAt = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
    // Optimistic: add user message
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
      const assistantMsg = await sendChatMessage(activeSession, content, {
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
      await fetchSessions(); // update message counts
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
      await setChatPreference(activeSession, optionText, accepted);
      await fetchSessions();
    } catch (err) {
      console.error("Failed to save chat preference:", err);
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
      const summary = filenames.join(", ");
      const isImage = files.some((f) =>
        /\.(png|jpg|jpeg)$/i.test(f.name)
      );
      const msg: ChatMessage = {
        role: "assistant",
        content:
          uploadedDocs.length > 0
            ? `Uploaded ${uploadedDocs.length} file(s): ${summary}${extraCount > 0 ? ` (+${extraCount} more)` : ""}. Ingestion started (PageIndex + Vector + Graph).${isImage ? " Running VLM UI analysis and WCAG compliance audit..." : ""}`
            : "No supported files were uploaded.",
        trace: [],
        metadata: {},
        route: "",
      };
      setMessages((prev) => [...prev, msg]);
      await fetchSessions();

      if (isImage && documentIds.length > 0) {
        pollForAuditResults(documentIds, uploadedDocs.map((d) => d.filename));
      }
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

  const pollForAuditResults = (docIds: number[], filenames: string[]) => {
    const POLL_INTERVAL = 6000;
    const MAX_POLLS = 80; // ~8 minutes max
    let polls = 0;

    const timer = setInterval(async () => {
      polls++;
      if (polls > MAX_POLLS) {
        clearInterval(timer);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "WCAG audit timed out waiting for results. You can ask about compliance anytime and the system will check for stored audit results.",
            trace: [],
            metadata: {},
            route: "",
          },
        ]);
        return;
      }

      try {
        for (let i = 0; i < docIds.length; i++) {
          const doc = await getDocument(docIds[i]);

          // Audit result available — render it
          if (doc.audit_result) {
            clearInterval(timer);
            const audit = doc.audit_result;
            const status = audit.audit_metadata?.overall_compliance_status ?? "Unknown";
            const violations = audit.detected_violations ?? [];
            const synthesis = audit.generative_synthesis?.compliance_summary ?? "";

            let content = `## WCAG Compliance Audit: ${filenames[i]}\n\n`;
            content += `**Overall Status: ${status}**\n\n`;

            if (violations.length > 0) {
              content += `### ${violations.length} Violation(s) Detected\n\n`;
              for (const v of violations) {
                const sev = v.severity ? ` [${v.severity.toUpperCase()}]` : "";
                content += `**${v.guideline_reference || "WCAG"}**${sev}\n`;
                if (v.issue_description) content += `${v.issue_description}\n`;
                if (v.element_type) content += `Element: ${v.element_type}\n`;
                if (v.suggested_fix) content += `Fix: ${v.suggested_fix}\n`;
                content += "\n";
              }
            } else {
              content += "No violations detected.\n\n";
            }

            if (synthesis) {
              content += `### Summary\n${synthesis}\n`;
            }

            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content,
                trace: [],
                metadata: { audit_document_id: docIds[i] },
                route: "WCAG_AUDIT",
              },
            ]);
            return;
          }

          // Ingestion finished but no audit result — audit failed or not applicable
          const done = doc.status === "ready" || doc.status === "error";
          if (done && !doc.audit_result) {
            clearInterval(timer);
            const errMsg = doc.error_message ? ` Error: ${doc.error_message}` : "";
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: `WCAG compliance audit for **${filenames[i]}** did not produce results. The LLM may have returned an empty response during the audit step. You can re-ingest the image from the Documents page to retry, or ask about WCAG compliance in chat and the system will use the stored UI description and WCAG knowledge base to answer.${errMsg}`,
                trace: [],
                metadata: {},
                route: "",
              },
            ]);
            return;
          }
        }
      } catch {
        // backend may be busy — just keep polling
      }
    }, POLL_INTERVAL);
  };

  return (
    <div className="flex h-full flex-col">
      {backendUnavailable && (
        <div className="alert-warning-banner flex items-center gap-3 px-4 py-2 rounded-lg mx-4 mt-2 shrink-0">
          <AlertCircle size={18} className="shrink-0" />
          <span className="text-sm">Backend not available. Start the backend to use chat.</span>
          <button
            type="button"
            onClick={() => fetchSessions()}
            className="alert-warning-btn ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded text-sm"
          >
            <RefreshCw size={14} />
            Retry
          </button>
        </div>
      )}
      <div className="flex flex-1 min-h-0">
      {/* Sessions sidebar */}
      <div className="w-64 border-r border-[var(--border)] bg-[var(--bg-secondary)] flex flex-col shrink-0">
        <div className="p-3 border-b border-[var(--border)]">
          <button
            onClick={handleNewSession}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--accent)] text-white rounded-lg text-sm hover:bg-[var(--accent-hover)]"
          >
            <Plus size={16} />
            New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
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
                onClick={(e) => { e.stopPropagation(); handleDeleteSession(s.id); }}
                className="p-1 rounded hover:bg-[var(--bg-primary)] text-[var(--text-secondary)] hover:text-[var(--error)]"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <div className="p-4 text-sm text-[var(--text-secondary)] text-center">
              No chat sessions yet
            </div>
          )}
        </div>
      </div>

      {/* Chat panel */}
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
              <MessageSquare size={48} className="mx-auto mb-3 opacity-30" />
              <p>Select or create a chat session</p>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
