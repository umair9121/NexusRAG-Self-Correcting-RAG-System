"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Trash2, Play, RotateCcw, ChevronLeft, ChevronRight, ChevronDown, FileText, Braces, Eye, Eraser, ArrowUp, ArrowDown, ArrowUpDown, GitBranch } from "lucide-react";
import type { Document } from "@/lib/types";
import type { DocumentSortBy, DocumentSortOrder } from "@/lib/api";
import { deleteDocument, bulkDeleteDocuments, startIngestion, getDocumentPdfUrl, deleteDocumentMetadata, listDocumentIds } from "@/lib/api";
import ClearMetadataModal from "./ClearMetadataModal";
import MetadataViewer from "./MetadataViewer";
import PDFViewerModal from "./PDFViewerModal";
import SideBySideViewer from "./SideBySideViewer";

interface DocumentTableProps {
  documents: Document[];
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onRefresh: () => void;
  /** Current search/filters so "select all" can match the same set */
  search?: string;
  statusFilter?: string;
  tagFilter?: string;
  needsIngestionFilter?: boolean;
  sortBy?: DocumentSortBy | "";
  sortOrder?: DocumentSortOrder;
  onSortChange?: (sortBy: DocumentSortBy, sortOrder: DocumentSortOrder) => void;
  /** When > 0, suggest "Run without Graph" for docs with page_count >= this (e.g. long textbooks). */
  graphSkipPageThreshold?: number;
}

const STATUS_COLORS: Record<string, string> = {
  uploaded: "bg-gray-600",
  ingesting: "bg-yellow-600",
  ready: "bg-green-600",
  error: "bg-red-600",
};

const PIPELINE_COLORS: Record<string, string> = {
  pending: "text-gray-500",
  running: "text-yellow-400",
  done: "text-green-400",
  error: "text-red-400",
  skipped: "text-gray-600",
};

type DocTypeKey = "pdf" | "word" | "excel" | "image" | "json" | "html" | "other";

const TYPE_LABELS: Record<DocTypeKey, string> = {
  pdf: "PDF",
  word: "Word",
  excel: "Excel",
  image: "Image",
  json: "JSON",
  html: "HTML",
  other: "Other",
};

const TYPE_COLORS: Record<DocTypeKey, string> = {
  pdf: "bg-red-600 text-white",
  word: "bg-blue-600 text-white",
  excel: "bg-green-600 text-white",
  image: "bg-gray-600 text-white",
  json: "bg-amber-800 text-white",
  html: "bg-purple-600 text-white",
  other: "bg-[var(--bg-tertiary)] text-[var(--text-secondary)]",
};

/** Show only the file name (basename), not the full path. */
function getDisplayName(filename: string): string {
  if (!filename) return filename;
  const lastSlash = Math.max(filename.lastIndexOf("/"), filename.lastIndexOf("\\"));
  return lastSlash === -1 ? filename : filename.slice(lastSlash + 1);
}

function getDocType(filename: string): DocTypeKey {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "pdf") return "pdf";
  if (["doc", "docx"].includes(ext)) return "word";
  if (["xls", "xlsx", "csv"].includes(ext)) return "excel";
  if (["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "ico", "tiff", "tif"].includes(ext)) return "image";
  if (ext === "json") return "json";
  if (["html", "htm"].includes(ext)) return "html";
  return "other";
}

function hasFailedOrPending(doc: Document): boolean {
  return (
    doc.pageindex_status === "error" ||
    doc.pageindex_status === "pending" ||
    doc.contextual_status === "error" ||
    doc.contextual_status === "pending" ||
    doc.graph_status === "error" ||
    doc.graph_status === "pending"
  );
}

function formatUploadedDate(iso: string): string {
  if (!iso) return "—";
  try {
    // Backend may send naive ISO (no Z); treat as UTC so local display is correct
    const isoUtc = /Z|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso + "Z";
    const d = new Date(isoUtc);
    if (Number.isNaN(d.getTime())) return "—";
    const datePart = d.toLocaleDateString(undefined, { dateStyle: "medium" });
    const timePart = d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
    return `${datePart}, ${timePart}`;
  } catch {
    return "—";
  }
}

function formatIngestionTime(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s ? `${m}m ${s}s` : `${m}m`;
}

const DEFAULT_COLUMN_WIDTHS: Record<string, number> = {
  checkbox: 80,
  name: 220,
  type: 80,
  pages: 56,
  status: 96,
  pipelines: 128,
  chunks: 72,
  uploadedDate: 165,
  ingestionTime: 100,
  tags: 120,
  actions: 140,
};

export default function DocumentTable({
  documents,
  total,
  page,
  pageSize,
  onPageChange,
  onRefresh,
  search,
  statusFilter,
  tagFilter,
  needsIngestionFilter,
  sortBy = "",
  sortOrder = "desc",
  onSortChange,
  graphSkipPageThreshold = 0,
}: DocumentTableProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [selectAllDropdownOpen, setSelectAllDropdownOpen] = useState(false);
  const [selectAllLoading, setSelectAllLoading] = useState(false);
  const selectAllDropdownRef = useRef<HTMLDivElement>(null);
  const [runDropdownDocId, setRunDropdownDocId] = useState<number | null>(null);
  const runDropdownRef = useRef<HTMLDivElement>(null);
  const [showLogsHint, setShowLogsHint] = useState(false);
  const [metadataDoc, setMetadataDoc] = useState<{ id: number; name: string } | null>(null);
  const [pdfViewerDoc, setPdfViewerDoc] = useState<{ id: number; name: string } | null>(null);
  const [sideBySideDoc, setSideBySideDoc] = useState<{ id: number; name: string } | null>(null);
  const [clearMetadataDoc, setClearMetadataDoc] = useState<{ id: number; name: string } | null>(null);

  const [colWidths, setColWidths] = useState<Record<string, number>>(() => ({ ...DEFAULT_COLUMN_WIDTHS }));
  const [resizingCol, setResizingCol] = useState<string | null>(null);
  const resizeStartX = useRef(0);
  const resizeStartWidth = useRef(0);

  const totalPages = Math.ceil(total / pageSize);

  const handleResizeStart = (colKey: string, e: React.MouseEvent) => {
    e.preventDefault();
    setResizingCol(colKey);
    resizeStartX.current = e.clientX;
    resizeStartWidth.current = colWidths[colKey] ?? DEFAULT_COLUMN_WIDTHS[colKey];
  };

  useEffect(() => {
    if (!resizingCol) return;
    const onMove = (e: MouseEvent) => {
      const delta = e.clientX - resizeStartX.current;
      const newWidth = Math.max(40, resizeStartWidth.current + delta);
      setColWidths((w) => ({ ...w, [resizingCol]: newWidth }));
    };
    const onUp = () => setResizingCol(null);
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [resizingCol]);

  const handleSort = (col: DocumentSortBy) => {
    if (!onSortChange) return;
    if (sortBy === col) {
      onSortChange(col, sortOrder === "asc" ? "desc" : "asc");
    } else {
      const defaultOrder: DocumentSortOrder =
        col === "created_at" || col === "ingestion_duration" ? "desc" : "asc";
      onSortChange(col, defaultOrder);
    }
  };

  const SortIcon = ({ column }: { column: DocumentSortBy }) => {
    if (sortBy !== column) return <ArrowUpDown size={14} className="opacity-50" aria-hidden />;
    return sortOrder === "asc" ? <ArrowUp size={14} aria-hidden /> : <ArrowDown size={14} aria-hidden />;
  };

  // Close select-all dropdown when clicking outside
  useEffect(() => {
    if (!selectAllDropdownOpen) return;
    const onOutside = (e: MouseEvent) => {
      if (selectAllDropdownRef.current && !selectAllDropdownRef.current.contains(e.target as Node)) {
        setSelectAllDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, [selectAllDropdownOpen]);

  // Close run dropdown when clicking outside
  useEffect(() => {
    if (runDropdownDocId === null) return;
    const onOutside = (e: MouseEvent) => {
      if (runDropdownRef.current && !runDropdownRef.current.contains(e.target as Node)) {
        setRunDropdownDocId(null);
      }
    };
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, [runDropdownDocId]);

  const toggleSelect = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAllOnPage = () => {
    setSelectAllDropdownOpen(false);
    if (selected.size === documents.length && documents.length > 0) {
      setSelected(new Set());
    } else {
      setSelected(new Set(documents.map((d) => d.id)));
    }
  };

  const selectAllEntirely = async () => {
    setSelectAllDropdownOpen(false);
    if (total === 0) return;
    setSelectAllLoading(true);
    try {
      const { ids } = await listDocumentIds({
        search: search || undefined,
        status: statusFilter || undefined,
        tag: tagFilter || undefined,
        needs_ingestion: needsIngestionFilter || undefined,
      });
      setSelected(new Set(ids));
    } catch (e) {
      console.error("Failed to load all document IDs:", e);
    } finally {
      setSelectAllLoading(false);
    }
  };

  const allOnPageSelected = documents.length > 0 && selected.size === documents.length;

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this document and all its indexed data?")) return;
    await deleteDocument(id);
    onRefresh();
  };

  const handleBulkDelete = async () => {
    if (!confirm(`Delete ${selected.size} document(s)?`)) return;
    await bulkDeleteDocuments(Array.from(selected));
    setSelected(new Set());
    onRefresh();
  };

  const handleIngest = async (docId: number, pipelines: string[]) => {
    setRunDropdownDocId(null);
    await startIngestion([docId], pipelines);
    setShowLogsHint(true);
    onRefresh();
  };

  const handleRetryFailed = async (docId: number) => {
    await startIngestion([docId], ["pageindex", "contextual", "graph"], { onlyFailed: true });
    setShowLogsHint(true);
    onRefresh();
  };

  const handleBulkIngest = async () => {
    await startIngestion(Array.from(selected), ["pageindex", "contextual", "graph"]);
    setSelected(new Set());
    setShowLogsHint(true);
    onRefresh();
  };

  const handleBulkRetryFailed = async () => {
    await startIngestion(Array.from(selected), ["pageindex", "contextual", "graph"], { onlyFailed: true });
    setSelected(new Set());
    setShowLogsHint(true);
    onRefresh();
  };

  const handleBulkClearMetadata = async () => {
    if (!confirm(`Clear all metadata for ${selected.size} document(s)? The files will not be deleted.`)) return;
    for (const id of selected) {
      try {
        await deleteDocumentMetadata(id);
      } catch (e) {
        console.error("Failed to clear metadata for document", id, e);
      }
    }
    setSelected(new Set());
    onRefresh();
  };

  return (
    <div>
      {/* Ingestion started: point user to Logs */}
      {showLogsHint && (
        <div className="mb-3 flex items-center gap-2 px-3 py-2 bg-indigo-500/15 border border-indigo-500/40 rounded-lg text-sm">
          <FileText size={16} className="text-indigo-400 shrink-0" />
          <span className="text-[var(--text-primary)]">
            Ingestion queued. Open{" "}
            <Link href="/logs" className="text-indigo-400 hover:underline font-medium">
              Logs
            </Link>{" "}
            in the sidebar to watch live progress.
          </span>
          <button
            onClick={() => setShowLogsHint(false)}
            className="ml-auto text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      {/* Bulk actions */}
      {selected.size > 0 && (
        <div className="mb-3 flex items-center gap-2 px-3 py-2 bg-[var(--bg-tertiary)] rounded-lg border border-[var(--border)]">
          <span className="text-sm text-[var(--text-secondary)]">{selected.size} selected</span>
          <button onClick={handleBulkIngest} className="px-3 py-1 text-xs bg-[var(--accent)] text-white rounded hover:bg-[var(--accent-hover)]">
            <Play size={12} className="inline mr-1" /> Ingest All
          </button>
          <button
            onClick={() => {
              startIngestion(Array.from(selected), ["pageindex", "contextual"]);
              setSelected(new Set());
              setShowLogsHint(true);
              onRefresh();
            }}
            className="px-3 py-1 text-xs bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-primary)] rounded hover:bg-[var(--bg-tertiary)]"
            title="PageIndex + Vector only (no Graph); recommended for very long documents"
          >
            <GitBranch size={12} className="inline mr-1" /> Ingest without Graph
          </button>
          {documents.filter((d) => selected.has(d.id) && hasFailedOrPending(d)).length > 0 && (
            <button onClick={handleBulkRetryFailed} className="px-3 py-1 text-xs bg-amber-600 text-white rounded hover:bg-amber-700">
              <RotateCcw size={12} className="inline mr-1" /> Retry failed only
            </button>
          )}
          <button onClick={handleBulkClearMetadata} className="px-3 py-1 text-xs bg-amber-600 text-white rounded hover:bg-amber-700">
            <Eraser size={12} className="inline mr-1" /> Clear metadata
          </button>
          <button onClick={handleBulkDelete} className="px-3 py-1 text-xs bg-[var(--error)] text-white rounded hover:opacity-80">
            <Trash2 size={12} className="inline mr-1" /> Delete All
          </button>
        </div>
      )}

      {/* Table */}
      <div className="border border-[var(--border)] rounded-lg overflow-hidden overflow-x-auto">
        <table className="w-full text-sm" style={{ tableLayout: "fixed", minWidth: "100%" }}>
          <colgroup>
            <col style={{ width: colWidths.checkbox }} />
            <col style={{ width: colWidths.name }} />
            <col style={{ width: colWidths.type }} />
            <col style={{ width: colWidths.pages }} />
            <col style={{ width: colWidths.status }} />
            <col style={{ width: colWidths.pipelines }} />
            <col style={{ width: colWidths.chunks }} />
            <col style={{ width: colWidths.uploadedDate }} />
            <col style={{ width: colWidths.ingestionTime }} />
            <col style={{ width: colWidths.tags }} />
            <col style={{ width: colWidths.actions }} />
          </colgroup>
          <thead>
            <tr className="bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
              <th className="px-3 py-2.5 text-left pr-5">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={allOnPageSelected}
                    onChange={selectAllOnPage}
                    className="accent-[var(--accent)]"
                    title="Select all on this page"
                  />
                  <div className="relative" ref={selectAllDropdownRef}>
                    <button
                      type="button"
                      onClick={() => setSelectAllDropdownOpen((o) => !o)}
                      disabled={selectAllLoading || (documents.length === 0 && total === 0)}
                      className="flex items-center gap-1 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Select all options"
                    >
                      Select all
                      <ChevronDown size={14} className={selectAllDropdownOpen ? "rotate-180" : ""} />
                    </button>
                    {selectAllDropdownOpen && (
                      <div className="absolute left-0 top-full mt-1 z-10 min-w-[200px] py-1 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] shadow-lg">
                        <button
                          type="button"
                          onClick={selectAllOnPage}
                          className="w-full px-3 py-2 text-left text-xs hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
                        >
                          This page (up to {pageSize})
                        </button>
                        <button
                          type="button"
                          onClick={selectAllEntirely}
                          disabled={selectAllLoading || total === 0}
                          className="w-full px-3 py-2 text-left text-xs hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)] disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {selectAllLoading ? "Loading…" : `All documents (${total})`}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                <button
                  type="button"
                  onClick={() => onSortChange && handleSort("name")}
                  className="inline-flex items-center gap-1 text-left hover:text-[var(--text-primary)] cursor-pointer"
                  title="Sort by name"
                >
                  Name <SortIcon column="name" />
                </button>
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("name", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                Type
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("type", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                <button
                  type="button"
                  onClick={() => onSortChange && handleSort("page_count")}
                  className="inline-flex items-center gap-1 text-left hover:text-[var(--text-primary)] cursor-pointer"
                  title="Sort by pages"
                >
                  Pages <SortIcon column="page_count" />
                </button>
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("pages", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                <button
                  type="button"
                  onClick={() => onSortChange && handleSort("status")}
                  className="inline-flex items-center gap-1 text-left hover:text-[var(--text-primary)] cursor-pointer"
                  title="Sort by status"
                >
                  Status <SortIcon column="status" />
                </button>
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("status", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                Pipelines
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("pipelines", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                <button
                  type="button"
                  onClick={() => onSortChange && handleSort("chunk_count")}
                  className="inline-flex items-center gap-1 text-left hover:text-[var(--text-primary)] cursor-pointer"
                  title="Sort by chunks"
                >
                  Chunks <SortIcon column="chunk_count" />
                </button>
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("chunks", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                <button
                  type="button"
                  onClick={() => onSortChange && handleSort("created_at")}
                  className="inline-flex items-center gap-1 text-left hover:text-[var(--text-primary)] cursor-pointer"
                  title="Sort by uploaded date"
                >
                  Uploaded Date <SortIcon column="created_at" />
                </button>
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("uploadedDate", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                <button
                  type="button"
                  onClick={() => onSortChange && handleSort("ingestion_duration")}
                  className="inline-flex items-center gap-1 text-left hover:text-[var(--text-primary)] cursor-pointer"
                  title="Sort by ingestion time"
                >
                  Time Taken (Ingestion) <SortIcon column="ingestion_duration" />
                </button>
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("ingestionTime", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                Tags
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("tags", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
              <th className="px-3 py-2.5 text-left relative group">
                Actions
                <div
                  role="separator"
                  aria-orientation="vertical"
                  onMouseDown={(e) => handleResizeStart("actions", e)}
                  className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)]/30 group-hover:bg-[var(--accent)]/20"
                  title="Drag to resize column"
                />
              </th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => {
              const docType = getDocType(doc.filename);
              return (
              <tr
                key={doc.id}
                className="border-t border-[var(--border)] hover:bg-[var(--bg-tertiary)]/50 transition-colors"
              >
                <td className="px-3 py-2.5">
                  <input type="checkbox" checked={selected.has(doc.id)} onChange={() => toggleSelect(doc.id)} className="accent-[var(--accent)]" />
                </td>
                <td className="px-3 py-2.5 min-w-0">
                  <div className="font-medium truncate min-w-0" title={doc.filename}>{getDisplayName(doc.filename)}</div>
                  <div className="text-xs text-[var(--text-secondary)] truncate min-w-0" title={`${(doc.file_size / 1024 / 1024).toFixed(1)} MB`}>{(doc.file_size / 1024 / 1024).toFixed(1)} MB</div>
                </td>
                <td className="px-3 py-2.5">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${TYPE_COLORS[docType]}`}>
                    {TYPE_LABELS[docType]}
                  </span>
                </td>
                <td className="px-3 py-2.5">{docType === "pdf" ? doc.page_count : docType === "image" ? 1 : "—"}</td>
                <td className="px-3 py-2.5">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs text-white ${STATUS_COLORS[doc.status] || "bg-gray-600"}`}>
                    {doc.status}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex gap-1.5 text-xs">
                    <span className={PIPELINE_COLORS[doc.pageindex_status]} title={`PageIndex: ${doc.pageindex_status}`}>PI</span>
                    <span className={PIPELINE_COLORS[doc.contextual_status]} title={`Vector Embeddings Contextual Process: ${doc.contextual_status}`}>VE</span>
                    <span className={PIPELINE_COLORS[doc.graph_status]} title={`Graph: ${doc.graph_status}`}>GR</span>
                  </div>
                </td>
                <td className="px-3 py-2.5 text-[var(--text-secondary)]">{doc.chunk_count || "-"}</td>
                <td className="px-3 py-2.5 text-[var(--text-secondary)] min-w-0" title={formatUploadedDate(doc.created_at)}>
                  <span className="block truncate min-w-0">{formatUploadedDate(doc.created_at)}</span>
                </td>
                <td className="px-3 py-2.5 text-[var(--text-secondary)]">
                  {formatIngestionTime(doc.ingestion_duration_seconds)}
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex flex-col gap-1 min-w-0">
                    {doc.path_hierarchy && doc.path_hierarchy.length > 0 && (
                      <span className="text-xs text-[var(--text-secondary)] truncate" title={doc.path_hierarchy.join(" › ")}>
                        {doc.path_hierarchy.join(" › ")}
                      </span>
                    )}
                    <div className="flex flex-wrap gap-1">
                      {doc.tags.slice(0, 2).map((t) => (
                        <span key={t} className="text-xs px-1.5 py-0.5 bg-[var(--bg-tertiary)] rounded">{t}</span>
                      ))}
                    </div>
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setPdfViewerDoc({ id: doc.id, name: getDisplayName(doc.filename) })}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
                      title="View document"
                    >
                      <Eye size={14} className="text-indigo-300" />
                    </button>
                    <button
                      onClick={() => setClearMetadataDoc({ id: doc.id, name: getDisplayName(doc.filename) })}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
                      title="Clear metadata (keep document)"
                    >
                      <Eraser size={14} className="text-amber-400" />
                    </button>
                    <button
                      onClick={() => setMetadataDoc({ id: doc.id, name: getDisplayName(doc.filename) })}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
                      title="View storage metadata"
                    >
                      <Braces size={14} className="text-[var(--accent)]" />
                    </button>
                    <div className="relative inline-flex items-center gap-0" ref={runDropdownDocId === doc.id ? runDropdownRef : undefined}>
                      <button
                        onClick={() => {
                          if (doc.status === "ingesting") return;
                          handleIngest(doc.id, ["pageindex", "contextual", "graph"]);
                        }}
                        disabled={doc.status === "ingesting"}
                        className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed"
                        title={doc.status === "ingesting" ? "Ingestion in progress" : "Run all pipelines (PI + VE + GR)"}
                      >
                        <Play size={14} className="text-[var(--success)]" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (doc.status === "ingesting") return;
                          setRunDropdownDocId((id) => (id === doc.id ? null : doc.id));
                        }}
                        disabled={doc.status === "ingesting"}
                        className="p-0.5 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed"
                        title="More options (e.g. Run without Graph)"
                        aria-label="Ingestion options"
                      >
                        <ChevronDown size={14} className="text-[var(--text-secondary)]" />
                      </button>
                      {runDropdownDocId === doc.id && (
                        <div className="absolute right-0 bottom-full z-[100] mb-1 py-1 w-56 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg shadow-xl overflow-hidden">
                          {graphSkipPageThreshold > 0 && typeof doc.page_count === "number" && doc.page_count >= graphSkipPageThreshold && (
                            <p className="px-3 py-1.5 text-xs text-amber-400 border-b border-[var(--border)] whitespace-normal">
                              Long document — Graph RAG may take days. Use &quot;Run without Graph&quot; below.
                            </p>
                          )}
                          <button
                            onClick={() => handleIngest(doc.id, ["pageindex", "contextual"])}
                            className="w-full px-3 py-2 text-left text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] flex items-center gap-2 whitespace-nowrap"
                            title="PageIndex + Vector only; recommended for very long books"
                          >
                            <GitBranch size={14} className="text-amber-400 shrink-0" />
                            Run without GraphRAG
                          </button>
                          <button
                            onClick={() => handleIngest(doc.id, ["graph"])}
                            className="w-full px-3 py-2 text-left text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] flex items-center gap-2 whitespace-nowrap"
                            title="Run only Graph RAG (entity extraction, graph, communities)"
                          >
                            <GitBranch size={14} className="text-[var(--success)] shrink-0" />
                            Run only GraphRAG
                          </button>
                        </div>
                      )}
                    </div>
                    {hasFailedOrPending(doc) && (
                      <button
                        onClick={() => handleRetryFailed(doc.id)}
                        disabled={doc.status === "ingesting"}
                        className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Re-run only failed or pending pipelines (PI, VE, GR)"
                      >
                        <RotateCcw size={14} className="text-amber-400" />
                      </button>
                    )}
                    <button onClick={() => handleDelete(doc.id)} className="p-1 rounded hover:bg-[var(--bg-tertiary)]" title="Delete">
                      <Trash2 size={14} className="text-[var(--error)]" />
                    </button>
                  </div>
                </td>
              </tr>
              );
            })}
            {documents.length === 0 && (
              <tr>
                <td colSpan={11} className="px-3 py-12 text-center text-[var(--text-secondary)]">
                  No documents found. Upload documents to get started.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-3 flex items-center justify-between text-sm text-[var(--text-secondary)]">
          <span>
            {(page - 1) * pageSize + 1}-{Math.min(page * pageSize, total)} of {total}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page <= 1}
              className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-30"
            >
              <ChevronLeft size={18} />
            </button>
            <span>
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages}
              className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-30"
            >
              <ChevronRight size={18} />
            </button>
          </div>
        </div>
      )}

      {/* Clear Metadata Modal */}
      <ClearMetadataModal
        docId={clearMetadataDoc?.id ?? 0}
        docName={clearMetadataDoc?.name ?? ""}
        open={clearMetadataDoc !== null}
        onClose={() => setClearMetadataDoc(null)}
        onSuccess={onRefresh}
      />

      {/* Metadata Viewer Modal */}
      <MetadataViewer
        docId={metadataDoc?.id ?? 0}
        docName={metadataDoc?.name ?? ""}
        open={metadataDoc !== null}
        onClose={() => setMetadataDoc(null)}
        onOpenSideBySide={metadataDoc ? () => {
          const doc = metadataDoc;
          setMetadataDoc(null);
          setSideBySideDoc(doc);
        } : undefined}
      />

      {/* PDF Viewer Modal */}
      <PDFViewerModal
        open={pdfViewerDoc !== null}
        onClose={() => setPdfViewerDoc(null)}
        pdfUrl={pdfViewerDoc ? getDocumentPdfUrl(pdfViewerDoc.id) : ""}
        documentName={pdfViewerDoc?.name ?? ""}
        onOpenSideBySide={pdfViewerDoc ? () => {
          const doc = pdfViewerDoc;
          setPdfViewerDoc(null);
          setSideBySideDoc(doc);
        } : undefined}
      />

      {/* Side by Side Viewer Modal */}
      <SideBySideViewer
        docId={sideBySideDoc?.id ?? 0}
        docName={sideBySideDoc?.name ?? ""}
        open={sideBySideDoc !== null}
        onClose={() => setSideBySideDoc(null)}
      />
    </div>
  );
}
