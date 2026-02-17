"use client";

import { useState, useEffect, useCallback } from "react";
import {
  X,
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
  Loader2,
  AlertCircle,
  Database,
  GitBranch,
  Network,
  FileText,
  Download,
  ExternalLink,
  GripVertical,
  Columns,
  Layers,
} from "lucide-react";
import { getDocumentMetadata, getDocumentPdfUrl } from "@/lib/api";

/* ═══════════════════════════════════════════════════════════════════
   Props
   ═══════════════════════════════════════════════════════════════════ */

interface SideBySideViewerProps {
  docId: number;
  docName: string;
  open: boolean;
  onClose: () => void;
}

/* ═══════════════════════════════════════════════════════════════════
   Metadata tab config (mirrored from MetadataViewer)
   ═══════════════════════════════════════════════════════════════════ */

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

/* ═══════════════════════════════════════════════════════════════════
   Reusable JSON tree components
   ═══════════════════════════════════════════════════════════════════ */

const STRING_PREVIEW_LENGTH = 200;

function ExpandableString({ value }: { value: string }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = value.length > STRING_PREVIEW_LENGTH;
  const display = isLong && !expanded ? value.slice(0, STRING_PREVIEW_LENGTH) : value;

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

function JsonValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null) return <span className="text-gray-500 italic">null</span>;
  if (typeof value === "boolean") return <span className="text-amber-400">{String(value)}</span>;
  if (typeof value === "number") return <span className="text-cyan-400">{value}</span>;
  if (typeof value === "string") return <ExpandableString value={value} />;
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
      <pre className="text-xs leading-relaxed overflow-auto max-h-full p-4 bg-[var(--bg-primary)] rounded-lg border border-[var(--border)] text-[var(--text-secondary)] font-mono whitespace-pre-wrap break-words">
        {text}
      </pre>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Resizable divider handle
   ═══════════════════════════════════════════════════════════════════ */

function ResizeDivider({ onDrag }: { onDrag: (deltaX: number) => void }) {
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      let lastX = e.clientX;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const delta = moveEvent.clientX - lastX;
        lastX = moveEvent.clientX;
        onDrag(delta);
      };

      const handleMouseUp = () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [onDrag],
  );

  return (
    <div
      onMouseDown={handleMouseDown}
      className="w-2 shrink-0 cursor-col-resize flex items-center justify-center group hover:bg-[var(--accent)]/10 transition-colors"
      title="Drag to resize"
    >
      <GripVertical size={12} className="text-[var(--text-secondary)] opacity-40 group-hover:opacity-100 group-hover:text-[var(--accent)] transition-all" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Main component
   ═══════════════════════════════════════════════════════════════════ */

export default function SideBySideViewer({ docId, docName, open, onClose }: SideBySideViewerProps) {
  // Metadata state
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [metaLoading, setMetaLoading] = useState(false);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("");
  const [viewMode, setViewMode] = useState<"tree" | "raw">("tree");

  // PDF state
  const [pdfLoading, setPdfLoading] = useState(true);
  const pdfUrl = getDocumentPdfUrl(docId);

  // Layout state  –  percentage of container width allocated to PDF (left panel)
  const [splitPercent, setSplitPercent] = useState(50);

  // Escape key
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Fetch metadata
  useEffect(() => {
    if (!open) return;
    setMetaLoading(true);
    setMetaError(null);
    setMetadata(null);
    setPdfLoading(true);
    getDocumentMetadata(docId)
      .then((data) => {
        setMetadata(data);
        const firstAvailable = TAB_CONFIG.find((t) => data[t.key]);
        if (firstAvailable) setActiveTab(firstAvailable.key);
      })
      .catch((err) => {
        const status = (err as Error & { status?: number })?.status;
        if (status === 404) {
          setMetaError("No metadata files found. Ingest this document first to generate metadata.");
        } else {
          setMetaError(err instanceof Error ? err.message : "Failed to load metadata");
        }
      })
      .finally(() => setMetaLoading(false));
  }, [open, docId]);

  // Handle resize via divider drag
  const handleDrag = useCallback((deltaX: number) => {
    setSplitPercent((prev) => {
      const containerWidth = window.innerWidth * 0.96; // ~96vw modal width
      const deltaPercent = (deltaX / containerWidth) * 100;
      return Math.min(80, Math.max(20, prev + deltaPercent));
    });
  }, []);

  if (!open) return null;

  const availableTabs = TAB_CONFIG.filter((t) => metadata && metadata[t.key]);
  const currentData = metadata && activeTab ? metadata[activeTab] : null;
  const currentTabConfig = TAB_CONFIG.find((t) => t.key === activeTab);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-xl" onClick={onClose} />

      {/* Modal container */}
      <div
        className="relative w-[96vw] max-w-[1920px] h-[94vh] flex flex-col overflow-hidden rounded-2xl"
        style={{
          background: "linear-gradient(135deg, rgba(17,17,24,0.96) 0%, rgba(10,10,15,0.98) 100%)",
          border: "1px solid rgba(99,102,241,0.15)",
          boxShadow: "0 0 0 1px rgba(255,255,255,0.03), 0 25px 80px -12px rgba(0,0,0,0.8), 0 0 60px -15px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.05)",
        }}
      >
        {/* ── Header ────────────────────────────────────────── */}
        <div
          className="shrink-0 flex items-center justify-between px-5 py-3 select-none"
          style={{
            background: "linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            backdropFilter: "blur(20px)",
          }}
        >
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
              style={{
                background: "linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(99,102,241,0.08) 100%)",
                border: "1px solid rgba(99,102,241,0.2)",
              }}
            >
              <Columns size={16} className="text-indigo-400" />
            </div>
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-[var(--text-primary)] truncate">
                Side-by-Side Comparison
              </h2>
              <p className="text-xs text-[var(--text-secondary)] truncate">{docName}</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 ml-4">
            {/* Reset split */}
            <button
              onClick={() => setSplitPercent(50)}
              className="px-2.5 py-1 rounded-lg text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-white/5 transition-all"
              title="Reset to 50/50 split"
            >
              50 / 50
            </button>
            <div className="w-px h-5 bg-white/8 mx-1" />
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-red-500/10 transition-all active:scale-95 group"
              title="Close (Esc)"
            >
              <X size={15} className="text-[var(--text-secondary)] group-hover:text-red-400 transition-colors" />
            </button>
          </div>
        </div>

        {/* ── Split content ─────────────────────────────────── */}
        <div className="flex-1 flex min-h-0">
          {/* ── LEFT: PDF Viewer ────────────────────────────── */}
          <div
            className="flex flex-col min-w-0 overflow-hidden"
            style={{ width: `${splitPercent}%` }}
          >
            {/* Panel header */}
            <div className="shrink-0 flex items-center gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-primary)]/50">
              <div
                className="w-5 h-5 rounded flex items-center justify-center shrink-0"
                style={{
                  background: "linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(99,102,241,0.08) 100%)",
                  border: "1px solid rgba(99,102,241,0.15)",
                }}
              >
                <svg
                  width="11"
                  height="11"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-indigo-400"
                >
                  <path d="M14.5 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V7.5L14.5 2z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
              </div>
              <span className="text-xs font-semibold text-[var(--text-primary)]">Original PDF</span>
              <div className="flex-1" />
              <a
                href={pdfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="p-1 rounded hover:bg-white/5 transition-all"
                title="Open in new tab"
              >
                <ExternalLink size={12} className="text-[var(--text-secondary)]" />
              </a>
              <a
                href={pdfUrl}
                download={docName}
                className="p-1 rounded hover:bg-white/5 transition-all"
                title="Download PDF"
              >
                <Download size={12} className="text-[var(--text-secondary)]" />
              </a>
            </div>

            {/* PDF iframe */}
            <div
              className="flex-1 relative min-h-0"
              style={{
                background: "radial-gradient(ellipse at center, rgba(30,30,45,0.5) 0%, rgba(10,10,15,1) 70%)",
              }}
            >
              {pdfLoading && (
                <div className="absolute inset-0 flex items-center justify-center z-10">
                  <div className="flex flex-col items-center gap-3" style={{ background: "linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(99,102,241,0.05) 100%)", border: "1px solid rgba(99,102,241,0.15)", borderRadius: "1rem", padding: "1.5rem" }}>
                    <Loader2 size={22} className="animate-spin text-indigo-400" />
                    <span className="text-sm text-[var(--text-secondary)]">Loading PDF...</span>
                  </div>
                </div>
              )}
              <iframe
                src={pdfUrl}
                title={docName}
                className="w-full h-full border-0"
                onLoad={() => setPdfLoading(false)}
              />
            </div>
          </div>

          {/* ── Resize handle ───────────────────────────────── */}
          <ResizeDivider onDrag={handleDrag} />

          {/* ── RIGHT: Metadata Viewer ──────────────────────── */}
          <div
            className="flex flex-col min-w-0 overflow-hidden"
            style={{ width: `${100 - splitPercent}%` }}
          >
            {/* Panel header */}
            <div className="shrink-0 flex items-center gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-primary)]/50">
              <div
                className="w-5 h-5 rounded flex items-center justify-center shrink-0"
                style={{
                  background: "linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(99,102,241,0.08) 100%)",
                  border: "1px solid rgba(99,102,241,0.15)",
                }}
              >
                <Database size={11} className="text-indigo-400" />
              </div>
              <span className="text-xs font-semibold text-[var(--text-primary)]">Storage Metadata</span>

              {/* Tab selector (compact) */}
              {availableTabs.length > 0 && (
                <div className="flex items-center gap-0.5 ml-3">
                  {TAB_CONFIG.map((tab) => {
                    const available = metadata ? metadata[tab.key] !== undefined : false;
                    const active = activeTab === tab.key;
                    return (
                      <button
                        key={tab.key}
                        onClick={() => available && setActiveTab(tab.key)}
                        disabled={!available}
                        className={`px-2 py-1 rounded text-xs transition-colors ${
                          active
                            ? "bg-[var(--accent)]/15 text-[var(--accent)] font-medium"
                            : available
                            ? "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]"
                            : "text-[var(--text-secondary)]/30 cursor-not-allowed"
                        }`}
                        title={!available ? "Not available — pipeline not run" : tab.description}
                      >
                        {tab.label}
                      </button>
                    );
                  })}
                </div>
              )}

              <div className="flex-1" />

              {/* View mode toggle */}
              <div className="flex bg-[var(--bg-tertiary)] rounded-md p-0.5">
                <button
                  onClick={() => setViewMode("tree")}
                  className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
                    viewMode === "tree"
                      ? "bg-[var(--bg-secondary)] text-[var(--text-primary)] shadow-sm"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Tree
                </button>
                <button
                  onClick={() => setViewMode("raw")}
                  className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
                    viewMode === "raw"
                      ? "bg-[var(--bg-secondary)] text-[var(--text-primary)] shadow-sm"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Raw
                </button>
              </div>
            </div>

            {/* Metadata content area */}
            <div className="flex-1 overflow-auto">
              {metaLoading && (
                <div className="flex-1 flex items-center justify-center h-full">
                  <div className="flex flex-col items-center gap-3 text-[var(--text-secondary)]">
                    <Loader2 size={28} className="animate-spin text-[var(--accent)]" />
                    <span className="text-sm">Loading metadata...</span>
                  </div>
                </div>
              )}

              {metaError && (
                <div className="flex-1 flex items-center justify-center h-full p-6">
                  <div className="flex flex-col items-center gap-3 text-center max-w-md">
                    <AlertCircle size={36} className="text-amber-400" />
                    <p className="text-[var(--text-secondary)] text-sm">{metaError}</p>
                  </div>
                </div>
              )}

              {metadata && !metaLoading && !metaError && (
                <>
                  {currentTabConfig && (
                    <div className="px-4 py-2 border-b border-[var(--border)] bg-[var(--bg-secondary)]/50 shrink-0">
                      <div className="flex items-center gap-2">
                        <span className="text-[var(--accent)]">{currentTabConfig.icon}</span>
                        <h3 className="font-medium text-xs text-[var(--text-primary)]">{currentTabConfig.label}</h3>
                        <span className="text-[10px] text-[var(--text-secondary)]">— {currentTabConfig.description}</span>
                      </div>
                    </div>
                  )}
                  <div className="p-4">
                    {currentData ? (
                      viewMode === "tree" ? (
                        <div className="text-sm font-mono">
                          <JsonValue value={currentData} depth={0} />
                        </div>
                      ) : (
                        <RawJsonView data={currentData} />
                      )
                    ) : (
                      <div className="flex items-center justify-center h-32 text-[var(--text-secondary)] text-sm">
                        Select a data source tab above
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
