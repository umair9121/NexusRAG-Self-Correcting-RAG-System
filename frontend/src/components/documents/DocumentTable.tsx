"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Trash2, Play, RotateCcw, ChevronLeft, ChevronRight, ChevronDown, FileText, Braces, Eye, Eraser } from "lucide-react";
import type { Document } from "@/lib/types";
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

function hasFailedOrPending(doc: Document): boolean {
  return (
    doc.extraction_status === "error" ||
    doc.extraction_status === "pending" ||
    doc.edi_status === "error" ||
    doc.edi_status === "pending" ||
    doc.graph_status === "error" ||
    doc.graph_status === "pending"
  );
}

export default function DocumentTable({
  documents,
  total,
  page,
  pageSize,
  onPageChange,
  onRefresh,
  search,
  statusFilter,
}: DocumentTableProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [selectAllDropdownOpen, setSelectAllDropdownOpen] = useState(false);
  const [selectAllLoading, setSelectAllLoading] = useState(false);
  const selectAllDropdownRef = useRef<HTMLDivElement>(null);
  const [ingestModalDoc, setIngestModalDoc] = useState<number | null>(null);
  const [showLogsHint, setShowLogsHint] = useState(false);
  const [metadataDoc, setMetadataDoc] = useState<{ id: number; name: string } | null>(null);
  const [pdfViewerDoc, setPdfViewerDoc] = useState<{ id: number; name: string } | null>(null);
  const [sideBySideDoc, setSideBySideDoc] = useState<{ id: number; name: string } | null>(null);
  const [clearMetadataDoc, setClearMetadataDoc] = useState<{ id: number; name: string } | null>(null);

  const totalPages = Math.ceil(total / pageSize);

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
    await startIngestion([docId], pipelines);
    setIngestModalDoc(null);
    setShowLogsHint(true);
    onRefresh();
  };

  const handleRetryFailed = async (docId: number) => {
    await startIngestion([docId], ["extraction", "edi_evaluation", "graph"], { onlyFailed: true });
    setShowLogsHint(true);
    onRefresh();
  };

  const handleBulkIngest = async () => {
    await startIngestion(Array.from(selected), ["extraction", "edi_evaluation", "graph"]);
    setSelected(new Set());
    setShowLogsHint(true);
    onRefresh();
  };

  const handleBulkRetryFailed = async () => {
    await startIngestion(Array.from(selected), ["extraction", "edi_evaluation", "graph"], { onlyFailed: true });
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
      <div className="border border-[var(--border)] rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
              <th className="px-3 py-2.5 text-left">
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
              <th className="px-3 py-2.5 text-left">Name</th>
              <th className="px-3 py-2.5 text-left w-16">Pages</th>
              <th className="px-3 py-2.5 text-left w-24">Status</th>
              <th className="px-3 py-2.5 text-left w-32">Pipelines</th>
              <th className="px-3 py-2.5 text-left w-20">EDI</th>
              <th className="px-3 py-2.5 text-left w-28">Tags</th>
              <th className="px-3 py-2.5 text-left w-24">Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr
                key={doc.id}
                className="border-t border-[var(--border)] hover:bg-[var(--bg-tertiary)]/50 transition-colors"
              >
                <td className="px-3 py-2.5">
                  <input type="checkbox" checked={selected.has(doc.id)} onChange={() => toggleSelect(doc.id)} className="accent-[var(--accent)]" />
                </td>
                <td className="px-3 py-2.5">
                  <div className="font-medium truncate max-w-xs">{doc.filename}</div>
                  <div className="text-xs text-[var(--text-secondary)]">{(doc.file_size / 1024 / 1024).toFixed(1)} MB</div>
                </td>
                <td className="px-3 py-2.5">{doc.page_count}</td>
                <td className="px-3 py-2.5">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs text-white ${STATUS_COLORS[doc.status] || "bg-gray-600"}`}>
                    {doc.status}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex gap-1.5 text-xs">
                    <span className={PIPELINE_COLORS[doc.extraction_status]} title={`Extraction: ${doc.extraction_status}`}>EX</span>
                    <span className={PIPELINE_COLORS[doc.edi_status]} title={`EDI: ${doc.edi_status}${doc.edi_rating ? ` (${doc.edi_rating})` : ""}`}>EDI</span>
                    <span className={PIPELINE_COLORS[doc.graph_status]} title={`Graph: ${doc.graph_status}`}>GR</span>
                  </div>
                </td>
                <td className="px-3 py-2.5 text-[var(--text-secondary)]">{doc.edi_rating || "-"}</td>
                <td className="px-3 py-2.5">
                  <div className="flex flex-wrap gap-1">
                    {doc.tags.slice(0, 2).map((t) => (
                      <span key={t} className="text-xs px-1.5 py-0.5 bg-[var(--bg-tertiary)] rounded">{t}</span>
                    ))}
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setPdfViewerDoc({ id: doc.id, name: doc.filename })}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
                      title="View document"
                    >
                      <Eye size={14} className="text-indigo-300" />
                    </button>
                    <button
                      onClick={() => setClearMetadataDoc({ id: doc.id, name: doc.filename })}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
                      title="Clear metadata (keep document)"
                    >
                      <Eraser size={14} className="text-amber-400" />
                    </button>
                    <button
                      onClick={() => setMetadataDoc({ id: doc.id, name: doc.filename })}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)]"
                      title="View storage metadata"
                    >
                      <Braces size={14} className="text-[var(--accent)]" />
                    </button>
                    <button
                      onClick={() => handleIngest(doc.id, ["extraction", "edi_evaluation", "graph"])}
                      disabled={doc.status === "ingesting"}
                      className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed"
                      title={doc.status === "ingesting" ? "Ingestion in progress" : "Run all pipelines"}
                    >
                      <Play size={14} className="text-[var(--success)]" />
                    </button>
                    {hasFailedOrPending(doc) && (
                      <button
                        onClick={() => handleRetryFailed(doc.id)}
                        disabled={doc.status === "ingesting"}
                        className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Re-run only failed or pending pipelines (EX, EDI, GR)"
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
            ))}
            {documents.length === 0 && (
              <tr>
                <td colSpan={8} className="px-3 py-12 text-center text-[var(--text-secondary)]">
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
