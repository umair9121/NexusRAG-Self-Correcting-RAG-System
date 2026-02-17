"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import TraceViewer from "./TraceViewer";

interface ChatPanelProps {
  messages: ChatMessage[];
  onSend: (content: string) => Promise<void>;
  loading: boolean;
  forceRoute: string | null;
  onRouteChange: (route: string | null) => void;
  temperature: number;
  onTemperatureChange: (temp: number) => void;
  ollamaModels: string[];
  selectedModel: string | null;
  onModelChange: (model: string | null) => void;
}

const ROUTES = [
  { value: null, label: "Auto (Adaptive)" },
  { value: "PAGEINDEX", label: "PageIndex" },
  { value: "VECTOR", label: "Vector" },
  { value: "GRAPH", label: "Graph" },
  { value: "CAG", label: "CAG" },
];

export default function ChatPanel({
  messages,
  onSend,
  loading,
  forceRoute,
  onRouteChange,
  temperature,
  onTemperatureChange,
  ollamaModels,
  selectedModel,
  onModelChange,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const msg = input;
    setInput("");
    await onSend(msg);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Controls bar */}
      <div className="flex flex-wrap items-center gap-4 px-4 py-2 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--text-secondary)] whitespace-nowrap">Model:</span>
          <select
            value={selectedModel ?? ""}
            onChange={(e) => onModelChange(e.target.value || null)}
            className="text-xs bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)] min-w-[160px]"
            title="Ollama model for this chat (saved until server restart)"
          >
            <option value="">Default (server)</option>
            {selectedModel && !ollamaModels.includes(selectedModel) && (
              <option value={selectedModel}>{selectedModel} (last used)</option>
            )}
            {ollamaModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--text-secondary)] whitespace-nowrap">Route:</span>
          <select
            value={forceRoute ?? ""}
            onChange={(e) => onRouteChange(e.target.value || null)}
            className="text-xs bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)]"
          >
            {ROUTES.map((r) => (
              <option key={r.value ?? "auto"} value={r.value ?? ""}>
                {r.label}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--text-secondary)]">Temp:</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={temperature}
            onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
            className="w-20 accent-[var(--accent)]"
          />
          <span className="text-xs text-[var(--text-primary)] w-6">{temperature.toFixed(2)}</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-[var(--text-secondary)] text-sm">
            Start a conversation by typing a question about your documents.
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[75%] rounded-xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-[var(--accent)] text-white"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
              }`}
            >
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              {msg.role === "assistant" && (
                <>
                  {msg.route && (
                    <div className="mt-2 text-xs text-[var(--text-secondary)]">Route: {msg.route}</div>
                  )}
                  <TraceViewer trace={msg.trace} metadata={msg.metadata} />
                </>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-[var(--bg-tertiary)] rounded-xl px-4 py-3">
              <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            className="flex-1 px-4 py-2.5 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="px-4 py-2.5 bg-[var(--accent)] text-white rounded-lg hover:bg-[var(--accent-hover)] disabled:opacity-50 transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </form>
    </div>
  );
}
