"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, FileText, Zap, Brain } from "lucide-react";
import type { ChatMessage, SourceReference } from "@/lib/types";
import { getDocumentFileUrl, getDocumentPdfUrl } from "@/lib/api";
import TraceViewer from "./TraceViewer";
import PDFViewerModal from "@/components/documents/PDFViewerModal";
import SideBySideViewer from "@/components/documents/SideBySideViewer";

interface ChatPanelProps {
  messages: ChatMessage[];
  onSend: (content: string) => Promise<void>;
  onSetPreference?: (optionText: string, accepted: boolean) => Promise<void>;
  loading: boolean;
  forceRoute: string | null;
  onRouteChange: (route: string | null) => void;
  temperature: number;
  onTemperatureChange: (temp: number) => void;
  ollamaModels: string[];
  selectedModel: string | null;
  onModelChange: (model: string | null) => void;
  responseMode: "fast" | "deeper";
  onResponseModeChange: (mode: "fast" | "deeper") => void;
  enableVector: boolean;
  enableTaxonomy: boolean;
  enablePageIndex: boolean;
  enableGraph: boolean;
  onEnableVectorChange: (v: boolean) => void;
  onEnableTaxonomyChange: (v: boolean) => void;
  onEnablePageIndexChange: (v: boolean) => void;
  onEnableGraphChange: (v: boolean) => void;
}

const ROUTES = [
  { value: null, label: "Auto (Adaptive)" },
  { value: "PAGEINDEX", label: "PageIndex" },
  { value: "VECTOR", label: "Vector" },
  { value: "GRAPH", label: "Graph" },
  { value: "CAG", label: "CAG" },
];

/** Format ISO timestamp to local time HH:mm:ss for display. */
function formatTimeSeconds(iso: string | null | undefined): string | null {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return null;
  }
}

/** Normalize dashes: digit-dash-digit → decimal; space-dash-space → decimal point. */
function normalizeDashes(text: string): string {
  return text
    .replace(/(\d)-(\d)/g, "$1.$2")
    .replace(/\s-\s/g, " . ");
}

/**
 * Renders chat content with formatting:
 * - **text** → bold
 * - "text" or 'text' (phrase, 2+ chars) → italic
 * - Dash between digits (e.g. 4-2) or space-dash-space → displayed as decimal point
 */
function renderInlineFormattedContent(
  content: string,
  options?: {
    sources?: SourceReference[];
    onCitationClick?: (source: SourceReference) => void;
  }
): React.ReactNode[] {
  const normalized = normalizeDashes(content);
  let key = 0;
  const nextKey = () => key++;

  // 1. Split by **bold**
  const byBold = normalized.split(/\*\*(.+?)\*\*/g);
  const afterBold: (string | React.ReactNode)[] = byBold.map((seg, i) =>
    i % 2 === 1 ? <strong key={nextKey()}>{seg}</strong> : seg
  );

  // 2. Within each string segment, split by double-quoted "italic"
  type Segment = string | React.ReactElement;
  const afterDq: Segment[] = afterBold.flatMap((node): Segment[] => {
    if (typeof node !== "string") return [node as React.ReactElement];
    return node.split(/"([^"]*)"/g).map((seg, j) =>
      j % 2 === 1 ? <em key={nextKey()}>{seg}</em> : seg
    );
  });

  // 3. Within each string segment, replace single-quoted phrases (2+ chars) with italic
  const singleQuoteRe = /(^|\s)'([^']{2,})'(\s|$|[.,;:!?])/g;
  const result: Segment[] = [];
  for (const node of afterDq) {
    if (typeof node !== "string") {
      result.push(node);
      continue;
    }
    singleQuoteRe.lastIndex = 0;
    let lastEnd = 0;
    let m: RegExpExecArray | null;
    while ((m = singleQuoteRe.exec(node)) !== null) {
      result.push(node.slice(lastEnd, m.index + m[1].length));
      result.push(<em key={nextKey()}>{m[2]}</em>);
      lastEnd = m.index + m[0].length;
    }
    result.push(node.slice(lastEnd));
  }
  // 4. Render in-text citation markers [^n] as clickable superscripts.
  const withCitations = result.flatMap((node) => {
    if (typeof node !== "string") return [node];
    const parts = node.split(/\[\^(\d+)\]/g);
    return parts.map((seg, i) => {
      if (i % 2 !== 1) return seg;
      const idx = Number(seg) - 1;
      const src = options?.sources?.[idx];
      if (!src) return `[^${seg}]`;
      return (
        <button
          key={`cite-${nextKey()}`}
          type="button"
          onClick={() => options?.onCitationClick?.(src)}
          className="mx-0.5 align-super text-[10px] text-[var(--accent)] hover:underline"
          title={`Open ${src.filename}${src.primary_page ? ` · p. ${src.primary_page}` : ""}`}
        >
          {seg}
        </button>
      );
    });
  });

  return withCitations;
}

function renderFormattedContent(
  content: string,
  options?: {
    sources?: SourceReference[];
    onCitationClick?: (source: SourceReference) => void;
  }
): React.ReactNode[] {
  const lines = content.split(/\r?\n/);
  const out: React.ReactNode[] = [];
  const headingClassByLevel: Record<number, string> = {
    1: "text-xl font-semibold mt-2 mb-1",
    2: "text-lg font-semibold mt-2 mb-1",
    3: "text-base font-semibold mt-2 mb-1",
    4: "text-sm font-semibold mt-2 mb-1",
    5: "text-sm font-medium mt-1.5 mb-1",
    6: "text-xs font-medium mt-1.5 mb-1",
  };

  lines.forEach((line, idx) => {
    const m = line.match(/^\s*(#{1,6})\s+(.+)$/);
    if (m) {
      const level = m[1].length;
      const text = m[2].trim();
      const headingClass = headingClassByLevel[level] ?? headingClassByLevel[3];
      const headingContent = renderInlineFormattedContent(text, options);
      if (level === 1) out.push(<h1 key={`h-${idx}`} className={headingClass}>{headingContent}</h1>);
      else if (level === 2) out.push(<h2 key={`h-${idx}`} className={headingClass}>{headingContent}</h2>);
      else if (level === 3) out.push(<h3 key={`h-${idx}`} className={headingClass}>{headingContent}</h3>);
      else if (level === 4) out.push(<h4 key={`h-${idx}`} className={headingClass}>{headingContent}</h4>);
      else if (level === 5) out.push(<h5 key={`h-${idx}`} className={headingClass}>{headingContent}</h5>);
      else out.push(<h6 key={`h-${idx}`} className={headingClass}>{headingContent}</h6>);
      return;
    }

    // Preserve user-visible line breaks for non-heading text.
    out.push(
      <span key={`t-${idx}`}>
        {renderInlineFormattedContent(line, options)}
      </span>
    );
    if (idx < lines.length - 1) {
      out.push(<br key={`br-${idx}`} />);
    }
  });

  return out;
}

function getSourcesFromMetadata(metadata: Record<string, unknown>): SourceReference[] {
  const raw = metadata?.sources;
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (r): r is SourceReference =>
      r != null &&
      typeof r.document_id === "number" &&
      typeof r.filename === "string"
  );
}

function getReferencedPage(pages?: number[]): number | undefined {
  if (!pages || pages.length === 0) return undefined;
  const sortedUnique = [...new Set(pages)].sort((a, b) => a - b);
  return sortedUnique[0];
}

function isPdf(filename: string): boolean {
  return filename.toLowerCase().endsWith(".pdf");
}

export default function ChatPanel({
  messages,
  onSend,
  onSetPreference,
  loading,
  forceRoute,
  onRouteChange,
  temperature,
  onTemperatureChange,
  ollamaModels,
  selectedModel,
  onModelChange,
  responseMode,
  onResponseModeChange,
  enableVector,
  enableTaxonomy,
  enablePageIndex,
  enableGraph,
  onEnableVectorChange,
  onEnableTaxonomyChange,
  onEnablePageIndexChange,
  onEnableGraphChange,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [prefSaving, setPrefSaving] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [refViewer, setRefViewer] = useState<{
    documentId: number;
    filename: string;
    initialPage?: number;
  } | null>(null);
  const [sideBySideRef, setSideBySideRef] = useState<{
    id: number;
    name: string;
    initialPage?: number;
  } | null>(null);
  /** Message indices (in messages[]) for which "Refined queries" is expanded */
  const [showRefinedQueries, setShowRefinedQueries] = useState<Set<number>>(new Set());

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

  const getClarificationQuestions = (metadata: Record<string, unknown>): string[] => {
    const raw = metadata?.clarification_questions;
    if (!Array.isArray(raw)) return [];
    return raw
      .map((q) => (typeof q === "string" ? q.trim() : ""))
      .filter((q) => q.length > 0)
      .slice(0, 4);
  };

  /** Flatten taxonomy_options (e.g. { Component: ["A", "B"] }) to [{ type, name }] for selectable options */
  function getTaxonomyOptions(metadata: Record<string, unknown>): { type: string; name: string }[] {
    const raw = metadata?.taxonomy_options;
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return [];
    const out: { type: string; name: string }[] = [];
    for (const [type, names] of Object.entries(raw)) {
      if (!Array.isArray(names)) continue;
      for (const n of names) {
        if (typeof n === "string" && n.trim()) out.push({ type: String(type).replace(/_/g, " "), name: n.trim() });
      }
    }
    return out;
  }

  return (
    <div className="flex h-full">
      <div className="flex flex-col flex-1 min-w-0">
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
        {responseMode === "deeper" && (
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
        )}
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
              <div className="text-sm whitespace-pre-wrap">
                {renderFormattedContent(msg.content, {
                  sources: msg.role === "assistant" ? getSourcesFromMetadata(msg.metadata) : [],
                  onCitationClick: (ref) =>
                    setRefViewer({
                      documentId: ref.document_id,
                      filename: ref.filename,
                      initialPage: ref.primary_page ?? getReferencedPage(ref.pages),
                    }),
                })}
              </div>
              {(msg.sent_at || msg.response_received_at) && (() => {
                const sent = formatTimeSeconds(msg.sent_at);
                const received = formatTimeSeconds(msg.response_received_at);
                const parts: string[] = [];
                if (sent) parts.push(`Sent ${sent}`);
                if (received) parts.push(`Response ${received}`);
                if (parts.length === 0) return null;
                const duration =
                  msg.response_received_at && msg.sent_at
                    ? Math.round((new Date(msg.response_received_at).getTime() - new Date(msg.sent_at).getTime()) / 1000)
                    : null;
                return (
                  <div
                    className={
                      msg.role === "user"
                        ? "mt-2 pt-2 border-t border-white/20 text-[10px] text-white/80"
                        : "mt-2 pt-2 border-t border-[var(--border)]/40 text-[10px] text-[var(--text-secondary)]"
                    }
                  >
                    {parts.join(" · ")}
                    {duration != null && ` (${duration}s)`}
                  </div>
                );
              })()}
              {msg.role === "assistant" && (
                <>
                  {msg.route && (
                    <div className="mt-2 text-xs text-[var(--text-secondary)]">Route: {msg.route}</div>
                  )}
                  <TraceViewer trace={msg.trace} metadata={msg.metadata} />
                  {(() => {
                    const sources = getSourcesFromMetadata(msg.metadata);
                    if (sources.length === 0) return null;
                    return (
                      <div className="mt-3 pt-2 border-t border-[var(--border)]/50">
                        <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] mb-1.5">
                          <FileText size={12} />
                          <span>References</span>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {sources.map((ref, idx) => {
                            const pageLabel =
                              ref.pages && ref.pages.length > 0
                                ? ` · p. ${ref.pages.join(", ")}`
                                : ref.primary_page != null
                                  ? ` · p. ${ref.primary_page}`
                                  : "";
                            return (
                              <button
                                key={`${ref.document_id}-${idx}`}
                                type="button"
                                onClick={() =>
                                  setRefViewer({
                                    documentId: ref.document_id,
                                    filename: ref.filename,
                                    initialPage: ref.primary_page ?? getReferencedPage(ref.pages),
                                  })
                                }
                                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs bg-[var(--bg-primary)]/80 hover:bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--border)]/70 hover:border-[var(--accent)]/40 transition-colors"
                                title={ref.snippet ? `${ref.filename}${pageLabel}\n${ref.snippet}` : ref.filename + pageLabel}
                              >
                                <FileText size={11} />
                                <span className="truncate max-w-[140px]">{ref.filename}</span>
                                {pageLabel && <span className="text-[var(--text-secondary)]">{pageLabel}</span>}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })()}
                  {(() => {
                    const gapAlts = msg.metadata?.context_gap_alternatives;
                    const gapHeading = (msg.metadata?.context_gap_heading as string) || "Did you mean one of these?";
                    if (!Array.isArray(gapAlts) || gapAlts.length === 0) return null;
                    const options = gapAlts
                      .map((q) => (typeof q === "string" ? q.trim() : ""))
                      .filter((q) => q.length > 0);
                    if (options.length === 0) return null;
                    return (
                      <div className="mt-3 pt-2 border-t border-[var(--border)]/50">
                        <div className="text-xs text-[var(--text-secondary)] mb-2">{gapHeading}</div>
                        <div className="rounded-md border border-[var(--border)]/70 overflow-hidden">
                          <table className="w-full text-sm text-left">
                            <tbody>
                              {options.map((q, idx) => (
                                <tr key={idx}>
                                  <td>
                                    <button
                                      type="button"
                                      onClick={() => onSend(q)}
                                      className="w-full px-3 py-2 text-left text-[var(--text-primary)] hover:bg-[var(--accent)]/15 border-b border-[var(--border)]/50 last:border-b-0 transition-colors"
                                      title="Tap to search with this question"
                                    >
                                      {q}
                                    </button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <p className="text-[10px] text-[var(--text-secondary)] mt-1.5">Tap a row to re-run the search with that question.</p>
                      </div>
                    );
                  })()}
                  {(() => {
                    const taxonomyOptions = getTaxonomyOptions(msg.metadata ?? {});
                    const questions = getClarificationQuestions(msg.metadata ?? {});
                    const isTaxonomyDisambiguation = msg.metadata?.disambiguation_source === "taxonomy";
                    const hasClarification = taxonomyOptions.length > 0 || questions.length > 0;
                    if (!hasClarification) return null;

                    const userQuery =
                      msg.role === "assistant" && i > 0 && messages[i - 1]?.role === "user"
                        ? String(messages[i - 1].content ?? "").trim()
                        : "";

                    return (
                      <div className="mt-3 pt-2 border-t border-[var(--border)]/50">
                        {isTaxonomyDisambiguation && (
                          <div className="inline-flex items-center gap-1.5 px-2 py-0.5 mb-2 text-[10px] font-medium rounded-full bg-[var(--accent)]/10 text-[var(--accent)] border border-[var(--accent)]/20">
                            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
                            Verified from Knowledge Base
                          </div>
                        )}
                        {taxonomyOptions.length > 0 && (
                          <>
                            <div className="text-xs text-[var(--text-secondary)] mb-1.5">
                              Pick one category for a more specific answer:
                            </div>
                            <div className="flex flex-wrap gap-1.5 mb-2">
                              {taxonomyOptions.map((opt) => {
                                const refinedQuery = userQuery
                                  ? `${userQuery} — specifically for ${opt.name}`
                                  : `— specifically for ${opt.name}`;
                                return (
                                  <button
                                    key={`${opt.type}-${opt.name}`}
                                    type="button"
                                    onClick={() => {
                                      setInput(refinedQuery);
                                      inputRef.current?.focus();
                                    }}
                                    className="px-2.5 py-1.5 rounded-md text-xs bg-[var(--bg-primary)]/80 text-[var(--text-primary)] border border-[var(--border)]/70 hover:border-[var(--accent)]/40 hover:bg-[var(--accent)]/15 transition-colors"
                                    title={`Use: ${refinedQuery}`}
                                  >
                                    {opt.name}
                                  </button>
                                );
                              })}
                            </div>
                          </>
                        )}
                        {questions.length > 0 && (
                          <>
                            <button
                              type="button"
                              onClick={() =>
                                setShowRefinedQueries((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(i)) next.delete(i);
                                  else next.add(i);
                                  return next;
                                })
                              }
                              className="text-xs text-[var(--accent)] hover:underline mb-1.5 inline-flex items-center gap-1"
                            >
                              {showRefinedQueries.has(i) ? "Hide refined queries" : "Show refined queries"}
                            </button>
                            {showRefinedQueries.has(i) && (
                              <div className="grid grid-cols-1 gap-1.5">
                                {questions.map((q, idx) => (
                                  <div
                                    key={`${idx}-${q.slice(0, 20)}`}
                                    className="px-2.5 py-2 rounded-md text-xs bg-[var(--bg-primary)]/70 text-[var(--text-primary)] border border-[var(--border)]/70"
                                  >
                                    <div className="mb-2">{q}</div>
                                    <div className="flex items-center gap-1.5">
                                      <button
                                        type="button"
                                        onClick={async () => {
                                          if (!onSetPreference) return;
                                          setPrefSaving(`yes-${idx}`);
                                          try {
                                            await onSetPreference(q, true);
                                            setInput(q);
                                            inputRef.current?.focus();
                                          } finally {
                                            setPrefSaving(null);
                                          }
                                        }}
                                        disabled={prefSaving !== null}
                                        className="px-2 py-1 rounded-md border border-[var(--border)]/70 hover:border-[var(--accent)]/40 hover:bg-[var(--accent)]/15 transition-colors disabled:opacity-60"
                                        title="Use this disambiguation in this chat session"
                                      >
                                        Yes
                                      </button>
                                      <button
                                        type="button"
                                        onClick={async () => {
                                          if (!onSetPreference) return;
                                          setPrefSaving(`no-${idx}`);
                                          try {
                                            await onSetPreference(q, false);
                                          } finally {
                                            setPrefSaving(null);
                                          }
                                        }}
                                        disabled={prefSaving !== null}
                                        className="px-2 py-1 rounded-md border border-[var(--border)]/70 hover:border-[var(--border)] hover:bg-[var(--bg-secondary)] transition-colors disabled:opacity-60"
                                        title="Reject this disambiguation"
                                      >
                                        No
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => {
                                          setInput(q);
                                          inputRef.current?.focus();
                                        }}
                                        className="ml-auto px-2 py-1 rounded-md border border-[var(--border)]/70 hover:border-[var(--accent)]/35 hover:bg-[var(--accent)]/10 transition-colors"
                                        title="Use/edit this as your next query"
                                      >
                                        Use query
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })()}
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

      {/* Reference document viewer modal */}
      <PDFViewerModal
        open={refViewer !== null}
        onClose={() => setRefViewer(null)}
        pdfUrl={refViewer ? (isPdf(refViewer.filename) ? getDocumentPdfUrl(refViewer.documentId) : getDocumentFileUrl(refViewer.documentId)) : ""}
        documentName={refViewer?.filename ?? ""}
        initialPage={refViewer?.initialPage}
        onOpenSideBySide={
          refViewer
            ? () => {
                setSideBySideRef({
                  id: refViewer.documentId,
                  name: refViewer.filename,
                  initialPage: refViewer.initialPage,
                });
                setRefViewer(null);
              }
            : undefined
        }
      />
      <SideBySideViewer
        docId={sideBySideRef?.id ?? 0}
        docName={sideBySideRef?.name ?? ""}
        open={sideBySideRef !== null}
        onClose={() => setSideBySideRef(null)}
        initialPage={sideBySideRef?.initialPage}
      />

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex gap-2">
          <input
            ref={inputRef}
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

      {/* Right: mini options menu */}
      <div className="w-52 shrink-0 border-l border-[var(--border)] bg-[var(--bg-secondary)] flex flex-col p-3 gap-3">
        <span className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wide">Response</span>
        <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
          <button
            type="button"
            onClick={() => onResponseModeChange("fast")}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 text-xs font-medium transition-colors ${
              responseMode === "fast"
                ? "bg-[var(--accent)] text-white"
                : "bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-primary)] hover:text-[var(--text-primary)]"
            }`}
            title="Vector search + LLM only; quick answers"
          >
            <Zap size={14} />
            Fast
          </button>
          <button
            type="button"
            onClick={() => onResponseModeChange("deeper")}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 text-xs font-medium transition-colors ${
              responseMode === "deeper"
                ? "bg-[var(--accent)] text-white"
                : "bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-primary)] hover:text-[var(--text-primary)]"
            }`}
            title="Full adaptive RAG: router, taxonomy, PageIndex, Graph, grader"
          >
            <Brain size={14} />
            Deeper
          </button>
        </div>
        {responseMode === "deeper" && (
          <>
            <span className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wide mt-1">Retrievers</span>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enableVector}
                onChange={(e) => onEnableVectorChange(e.target.checked)}
                className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
              />
              <span className="text-xs text-[var(--text-primary)]">Vector</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enableTaxonomy}
                onChange={(e) => onEnableTaxonomyChange(e.target.checked)}
                className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
              />
              <span className="text-xs text-[var(--text-primary)]">Taxonomy</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enablePageIndex}
                onChange={(e) => onEnablePageIndexChange(e.target.checked)}
                className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
              />
              <span className="text-xs text-[var(--text-primary)]">PageIndex</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enableGraph}
                onChange={(e) => onEnableGraphChange(e.target.checked)}
                className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
              />
              <span className="text-xs text-[var(--text-primary)]">Graph</span>
            </label>
          </>
        )}
      </div>
    </div>
  );
}
