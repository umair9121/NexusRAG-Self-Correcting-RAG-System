"use client";

import { useState, useEffect, useCallback } from "react";
import { X, Copy, Check, ChevronDown, ChevronRight, Loader2, AlertCircle, Database, GitBranch, Network, FileText, Columns, Layers } from "lucide-react";
import { getDocumentMetadata } from "@/lib/api";

interface MetadataViewerProps {
  docId: number;
  docName: string;
  open: boolean;
  onClose: () => void;
  onOpenSideBySide?: () => void;
}

const TAB_CONFIG: {
  key: string;
  label: string;
  icon: React.ReactNode;
  description: string;
}[] = [
  { key: "pageindex", label: "Page Index", icon: <FileText size={14} />, description: "Hierarchical section tree with summaries" },
  { key: "docstore", label: "Doc Store", icon: <Database size={14} />, description: "Text chunks with contextual metadata" },
  { key: "chroma", label: "Vector Embeddings", icon: <Layers size={14} />, description: "ChromaDB vectors for this document" },
  { key: "graph", label: "Knowledge Graph", icon: <GitBranch size={14} />, description: "Entity-relationship graph structure" },
  { key: "communities", label: "Communities", icon: <Network size={14} />, description: "Graph community partitions and summaries" },
];

/* ── Expandable long string (show more / show less) ──────────────── */

const STRING_PREVIEW_LENGTH = 200;

function ExpandableString({ value }: { value: string }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = value.length > STRING_PREVIEW_LENGTH;
  const display = isLong && !expanded
    ? value.slice(0, STRING_PREVIEW_LENGTH)
    : value;

  return (
    <span className="inline">
      <span className="text-green-400 whitespace-pre-wrap break-words">
        &quot;{display}
        {isLong && !expanded && "…"}
        &quot;
      </span>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="ml-1.5 text-xs font-medium text-[var(--accent)] hover:underline focus:outline-none"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </span>
  );
}

/* ── Recursive collapsible JSON tree ─────────────────────────────── */

function JsonValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null) return <span className="text-gray-500 italic">null</span>;
  if (typeof value === "boolean")
    return <span className="text-amber-400">{String(value)}</span>;
  if (typeof value === "number")
    return <span className="text-cyan-400">{value}</span>;
  if (typeof value === "string") {
    return <ExpandableString value={value} />;
  }
  if (Array.isArray(value)) return <JsonArray items={value} depth={depth} />;
  if (typeof value === "object") return <JsonObject obj={value as Record<string, unknown>} depth={depth} />;
  return <span>{String(value)}</span>;
}

function JsonArray({ items, depth }: { items: unknown[]; depth: number }) {
  const [open, setOpen] = useState(depth < 2);

  if (items.length === 0) return <span className="text-[var(--text-secondary)]">[ ]</span>;

  return (
    <span>
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-0.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-xs opacity-70">Array[{items.length}]</span>
      </button>
      {open && (
        <div className="ml-4 border-l border-[var(--border)] pl-3">
          {items.map((item, i) => (
            <div key={i} className="py-0.5">
              <span className="text-[var(--text-secondary)] text-xs mr-2 select-none">{i}:</span>
              <JsonValue value={item} depth={depth + 1} />
            </div>
          ))}
        </div>
      )}
    </span>
  );
}

function JsonObject({ obj, depth }: { obj: Record<string, unknown>; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  const keys = Object.keys(obj);

  if (keys.length === 0) return <span className="text-[var(--text-secondary)]">{"{ }"}</span>;

  return (
    <span>
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-0.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-xs opacity-70">Object{`{${keys.length}}`}</span>
      </button>
      {open && (
        <div className="ml-4 border-l border-[var(--border)] pl-3">
          {keys.map((key) => (
            <div key={key} className="py-0.5">
              <span className="text-purple-400 font-medium text-sm">{key}</span>
              <span className="text-[var(--text-secondary)] mx-1">:</span>
              <JsonValue value={obj[key]} depth={depth + 1} />
            </div>
          ))}
        </div>
      )}
    </span>
  );
}

/* ── Raw JSON with copy button ───────────────────────────────────── */

function RawJsonView({ data }: { data: unknown }) {
  const [copied, setCopied] = useState(false);
  const text = JSON.stringify(data, null, 2);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);

  return (
    <div className="relative">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-[var(--bg-tertiary)] hover:bg-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors z-10"
        title="Copy JSON"
      >
        {copied ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
        {copied ? "Copied" : "Copy"}
      </button>
      <pre className="text-xs leading-relaxed overflow-auto max-h-[60vh] p-4 bg-[var(--bg-primary)] rounded-lg border border-[var(--border)] text-[var(--text-secondary)] font-mono whitespace-pre-wrap break-words">
        {text}
      </pre>
    </div>
  );
}

/* ── Main modal ──────────────────────────────────────────────────── */

export default function MetadataViewer({ docId, docName, open, onClose, onOpenSideBySide }: MetadataViewerProps) {
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("");
  const [viewMode, setViewMode] = useState<"tree" | "raw">("tree");

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setMetadata(null);
    getDocumentMetadata(docId)
      .then((data) => {
        setMetadata(data);
        // Auto-select first available tab
        const firstAvailable = TAB_CONFIG.find((t) => data[t.key]);
        if (firstAvailable) setActiveTab(firstAvailable.key);
      })
      .catch((err) => {
        const status = (err as Error & { status?: number })?.status;
        if (status === 404) {
          setError("No metadata files found. Ingest this document first to generate metadata.");
        } else {
          setError(err instanceof Error ? err.message : "Failed to load metadata");
        }
      })
      .finally(() => setLoading(false));
  }, [open, docId]);

  if (!open) return null;

  const availableTabs = TAB_CONFIG.filter((t) => metadata && metadata[t.key]);
  const currentData = metadata && activeTab ? metadata[activeTab] : null;
  const currentTabConfig = TAB_CONFIG.find((t) => t.key === activeTab);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-[90vw] max-w-5xl h-[85vh] bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] shrink-0">
          <div className="min-w-0 flex-1">
            <h2 className="text-lg font-semibold text-[var(--text-primary)] truncate">
              Storage Metadata
            </h2>
            <p className="text-sm text-[var(--text-secondary)] truncate mt-0.5">{docName}</p>
          </div>
          <div className="flex items-center gap-1 ml-4">
            {onOpenSideBySide && (
              <button
                onClick={onOpenSideBySide}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--accent)]/10 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors"
                title="View PDF and Metadata side by side"
              >
                <Columns size={13} />
                <span>Side by Side</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {loading && (
            <div className="flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3 text-[var(--text-secondary)]">
                <Loader2 size={28} className="animate-spin text-[var(--accent)]" />
                <span className="text-sm">Loading metadata...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="flex-1 flex items-center justify-center p-6">
              <div className="flex flex-col items-center gap-3 text-center max-w-md">
                <AlertCircle size={36} className="text-amber-400" />
                <p className="text-[var(--text-secondary)] text-sm">{error}</p>
              </div>
            </div>
          )}

          {metadata && !loading && !error && (
            <>
              {/* Sidebar tabs */}
              <div className="w-52 border-r border-[var(--border)] bg-[var(--bg-primary)] p-3 shrink-0 overflow-y-auto">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] mb-2 px-2">
                  Data Sources
                </p>
                {TAB_CONFIG.map((tab) => {
                  const available = metadata[tab.key] !== undefined;
                  const active = activeTab === tab.key;
                  return (
                    <button
                      key={tab.key}
                      onClick={() => available && setActiveTab(tab.key)}
                      disabled={!available}
                      className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-left text-sm transition-colors mb-1 ${
                        active
                          ? "bg-[var(--accent)]/15 text-[var(--accent)] font-medium"
                          : available
                          ? "text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]"
                          : "text-[var(--text-secondary)]/40 cursor-not-allowed"
                      }`}
                      title={!available ? "Not available — pipeline not run" : tab.description}
                    >
                      <span className={active ? "text-[var(--accent)]" : available ? "text-[var(--text-secondary)]" : "opacity-30"}>
                        {tab.icon}
                      </span>
                      <span className="truncate">{tab.label}</span>
                      {!available && (
                        <span className="ml-auto text-[10px] opacity-50">—</span>
                      )}
                    </button>
                  );
                })}

                {/* View mode toggle */}
                <div className="mt-4 pt-4 border-t border-[var(--border)]">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] mb-2 px-2">
                    View Mode
                  </p>
                  <div className="flex bg-[var(--bg-tertiary)] rounded-lg p-0.5">
                    <button
                      onClick={() => setViewMode("tree")}
                      className={`flex-1 px-3 py-1.5 text-xs rounded-md transition-colors ${
                        viewMode === "tree"
                          ? "bg-[var(--bg-secondary)] text-[var(--text-primary)] shadow-sm"
                          : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                      }`}
                    >
                      Tree
                    </button>
                    <button
                      onClick={() => setViewMode("raw")}
                      className={`flex-1 px-3 py-1.5 text-xs rounded-md transition-colors ${
                        viewMode === "raw"
                          ? "bg-[var(--bg-secondary)] text-[var(--text-primary)] shadow-sm"
                          : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                      }`}
                    >
                      Raw
                    </button>
                  </div>
                </div>

                {/* Quick stats */}
                {availableTabs.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-[var(--border)]">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] mb-2 px-2">
                      Available
                    </p>
                    <p className="text-xs text-[var(--text-secondary)] px-2">
                      {availableTabs.length} of {TAB_CONFIG.length} data sources
                    </p>
                  </div>
                )}
              </div>

              {/* Main content area */}
              <div className="flex-1 flex flex-col overflow-hidden">
                {currentTabConfig && (
                  <div className="px-5 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[var(--accent)]">{currentTabConfig.icon}</span>
                      <h3 className="font-medium text-sm text-[var(--text-primary)]">{currentTabConfig.label}</h3>
                    </div>
                    <p className="text-xs text-[var(--text-secondary)] mt-0.5">{currentTabConfig.description}</p>
                  </div>
                )}

                <div className="flex-1 overflow-auto p-5">
                  {currentData ? (
                    viewMode === "tree" ? (
                      <div className="text-sm font-mono">
                        <JsonValue value={currentData} depth={0} />
                      </div>
                    ) : (
                      <RawJsonView data={currentData} />
                    )
                  ) : (
                    <div className="flex items-center justify-center h-full text-[var(--text-secondary)] text-sm">
                      Select a data source from the sidebar
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
