"use client";

import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, MessageSquare, AlertCircle, RefreshCw } from "lucide-react";
import {
  listChatSessions,
  createChatSession,
  getChatMessages,
  sendChatMessage,
  deleteChatSession,
  getChatSessionState,
  getSystemHealth,
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

  const fetchSessions = useCallback(async () => {
    try {
      setBackendUnavailable(false);
      const data = await listChatSessions();
      setSessions(data);
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
    fetchSessions();
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
      }
      await fetchSessions();
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const handleSend = async (content: string) => {
    if (!activeSession) return;
    // Optimistic: add user message
    const userMsg: ChatMessage = { role: "user", content, trace: [], metadata: {}, route: "" };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    try {
      const assistantMsg = await sendChatMessage(
        activeSession,
        content,
        forceRoute,
        temperature,
        selectedModel,
      );
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
              onClick={() => loadMessages(s.id)}
            >
              <MessageSquare size={14} className="text-[var(--text-secondary)] shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate">{s.title}</div>
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
            loading={loading}
            forceRoute={forceRoute}
            onRouteChange={setForceRoute}
            temperature={temperature}
            onTemperatureChange={setTemperature}
            ollamaModels={ollamaModels}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
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
