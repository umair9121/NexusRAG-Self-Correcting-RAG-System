"use client";

import { useEffect, useState, useCallback } from "react";
import { Upload, Search, Filter, AlertCircle, RefreshCw, DatabaseZap, Play, RotateCcw, Loader2 } from "lucide-react";
import { listDocuments, getDocumentFacets, getIngestionStatus, getRecoveryJobs, getSystemHealth, isBackendAvailable, startIngestion, listDocumentIds, API_BASE } from "@/lib/api";
import type { DocumentSortBy, DocumentSortOrder } from "@/lib/api";
import type { Document, DocumentFacets, IngestionStatus, RecoveryJob } from "@/lib/types";
import DocumentTable from "@/components/documents/DocumentTable";
import UploadModal from "@/components/documents/UploadModal";
import IngestionBanner from "@/components/documents/IngestionBanner";
import ResumeIngestionModal from "@/components/documents/ResumeIngestionModal";
import ResetVectorStoreModal from "@/components/documents/ResetVectorStoreModal";
import KnowledgeBasePanel from "@/components/documents/KnowledgeBasePanel";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [needsIngestionFilter, setNeedsIngestionFilter] = useState(false);
  const [facets, setFacets] = useState<DocumentFacets | null>(null);
  const [sortBy, setSortBy] = useState<DocumentSortBy | "">("");
  const [sortOrder, setSortOrder] = useState<DocumentSortOrder>("desc");
  const [showUpload, setShowUpload] = useState(false);
  const [showResetVectorStore, setShowResetVectorStore] = useState(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [recoveryJobs, setRecoveryJobs] = useState<RecoveryJob[]>([]);
  const [recoveryChecked, setRecoveryChecked] = useState(false);
  const [graphSkipPageThreshold, setGraphSkipPageThreshold] = useState(0);
  const [needsIngestionActionLoading, setNeedsIngestionActionLoading] = useState(false);
  const pageSize = 50;

  const fetchDocs = useCallback(async () => {
    try {
      const res = await listDocuments({
        page,
        page_size: pageSize,
        search: search || undefined,
        status: statusFilter || undefined,
        tag: tagFilter || undefined,
        needs_ingestion: needsIngestionFilter || undefined,
        sort_by: sortBy || undefined,
        sort_order: sortOrder,
      });
      setDocuments(res.documents);
      setTotal(res.total);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }, [page, search, statusFilter, tagFilter, needsIngestionFilter, sortBy, sortOrder]);

  // Decide "backend available" only from GET /api (so we don't show banner when backend is on but /api/documents 404s)
  useEffect(() => {
    let cancelled = false;
    isBackendAvailable().then((ok) => {
      if (!cancelled) setBackendOk(ok);
    });
    return () => { cancelled = true; };
  }, []);

  // Fetch health for graph_skip_page_threshold (suggest "Run without Graph" for long docs)
  useEffect(() => {
    if (!backendOk) return;
    let cancelled = false;
    getSystemHealth()
      .then((h) => {
        if (!cancelled && h.graph_skip_page_threshold != null && h.graph_skip_page_threshold > 0) {
          setGraphSkipPageThreshold(h.graph_skip_page_threshold);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [backendOk]);

  useEffect(() => {
    fetchDocs();
    const interval = setInterval(fetchDocs, backendOk ? 5000 : 10000);
    return () => clearInterval(interval);
  }, [fetchDocs, backendOk]);

  // Fetch facets for filter counts (status, tag, needs_ingestion)
  useEffect(() => {
    if (!backendOk) return;
    let cancelled = false;
    getDocumentFacets({
      search: search || undefined,
      status: statusFilter || undefined,
      tag: tagFilter || undefined,
    })
      .then((f) => {
        if (!cancelled) setFacets(f);
      })
      .catch(() => {
        if (!cancelled) setFacets(null);
      });
    return () => { cancelled = true; };
  }, [backendOk, search, statusFilter, tagFilter]);

  // Check for interrupted ingestion jobs on startup (show resume modal)
  useEffect(() => {
    if (!backendOk || recoveryChecked) return;
    let cancelled = false;
    getRecoveryJobs()
      .then((res) => {
        if (!cancelled && res.jobs.length > 0) {
          setRecoveryJobs(res.jobs);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setRecoveryChecked(true);
      });
    return () => { cancelled = true; };
  }, [backendOk, recoveryChecked]);

  // Poll ingestion status so banner and cancel survive page refresh
  useEffect(() => {
    if (!backendOk) return;
    const poll = async () => {
      try {
        const status = await getIngestionStatus();
        setIngestionStatus(status);
      } catch {
        setIngestionStatus(null);
      }
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [backendOk]);

  const needsIngestionCount = facets?.needs_ingestion_count ?? 0;
  const handleIngestAllNeeding = async () => {
    setNeedsIngestionActionLoading(true);
    try {
      const { ids } = await listDocumentIds({
        search: search || undefined,
        status: statusFilter || undefined,
        tag: tagFilter || undefined,
        needs_ingestion: true,
      });
      if (ids.length > 0) {
        await startIngestion(ids, ["pageindex", "contextual", "graph"]);
        fetchDocs();
      }
    } catch (e) {
      console.error("Failed to ingest documents needing ingestion:", e);
    } finally {
      setNeedsIngestionActionLoading(false);
    }
  };
  const handleRetryFailedNeeding = async () => {
    setNeedsIngestionActionLoading(true);
    try {
      const { ids } = await listDocumentIds({
        search: search || undefined,
        status: statusFilter || undefined,
        tag: tagFilter || undefined,
        needs_ingestion: true,
      });
      if (ids.length > 0) {
        await startIngestion(ids, ["pageindex", "contextual", "graph"], { onlyFailed: true });
        fetchDocs();
      }
    } catch (e) {
      console.error("Failed to retry failed documents:", e);
    } finally {
      setNeedsIngestionActionLoading(false);
    }
  };

  return (
    <div className="p-6">
      {/* Backend unavailable banner */}
      {backendOk === false && (
        <div className="alert-warning-banner mb-6 flex items-start gap-3 p-4 rounded-xl">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Backend not available</p>
            <p className="alert-warning-muted text-sm mt-1">
              Start the backend so documents and chat work. From the project root run:
            </p>
            <code className="alert-warning-code block mt-2 px-3 py-2 rounded text-sm font-mono">
              cd backend
            </code>
            <code className="alert-warning-code block mt-1 px-3 py-2 rounded text-sm font-mono">
              uvicorn app.main:app --reload --port 8000
            </code>
            <p className="alert-warning-subtle text-xs mt-2">
              Or from project root: <code className="alert-warning-code px-1 rounded">python backend/run.py</code>
            </p>
            <p className="alert-warning-subtle text-xs mt-1">
              Requests go to: {API_BASE}/api
            </p>
            <button
              type="button"
              onClick={() => { setBackendOk(null); fetchDocs(); }}
              className="alert-warning-btn mt-3 flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm"
            >
              <RefreshCw size={14} />
              Retry
            </button>
          </div>
        </div>
      )}
      {/* Ingestion in progress (persists across refresh; cancel available) */}
      <IngestionBanner
        status={ingestionStatus}
        onCancelDone={async () => {
          fetchDocs();
          try {
            setIngestionStatus(await getIngestionStatus());
          } catch {
            setIngestionStatus(null);
          }
        }}
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Documents</h1>
          <p className="text-sm text-[var(--text-secondary)] mt-1">
            Manage and ingest documents (PDF, Word, Excel, JSON, HTML, images) into the RAG system
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowResetVectorStore(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-[var(--btn-warning-bg)] border border-[var(--btn-warning-border)] text-[var(--btn-warning-text)] rounded-lg hover:bg-[var(--btn-warning-hover-bg)] transition-colors"
            title="Reset vector store to switch embedding models"
          >
            <DatabaseZap size={16} />
            Reset Vector Store
          </button>
          <button
            onClick={() => setShowUpload(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-[var(--accent)] text-white rounded-lg hover:bg-[var(--accent-hover)] transition-colors"
          >
            <Upload size={16} />
            Upload documents
          </button>
        </div>
      </div>

      {/* Knowledge Base panel */}
      <KnowledgeBasePanel />

      {/* Needs ingestion block: prominent when there are docs needing ingestion */}
      {needsIngestionCount > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3">
          <span className="text-sm font-medium text-[var(--text-primary)]">
            <strong>{needsIngestionCount}</strong> document{needsIngestionCount !== 1 ? "s" : ""} need ingestion
          </span>
          {!needsIngestionFilter && (
            <button
              type="button"
              onClick={() => { setNeedsIngestionFilter(true); setPage(1); }}
              className="text-sm text-amber-600 dark:text-amber-400 hover:underline font-medium"
            >
              Show only these
            </button>
          )}
          <button
            type="button"
            onClick={handleIngestAllNeeding}
            disabled={needsIngestionActionLoading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[var(--accent)] text-white rounded-lg hover:bg-[var(--accent-hover)] disabled:opacity-50"
          >
            {needsIngestionActionLoading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            Ingest all
          </button>
          <button
            type="button"
            onClick={handleRetryFailedNeeding}
            disabled={needsIngestionActionLoading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50"
          >
            {needsIngestionActionLoading ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
            Retry failed only
          </button>
        </div>
      )}

      {/* Search & Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="relative flex-1 min-w-[200px] max-w-md">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]" />
          <input
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            placeholder="Search documents..."
            className="w-full pl-9 pr-4 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
          />
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Filter size={16} className="text-[var(--text-secondary)] shrink-0" />
          {/* Status: clickable chips with counts from facets */}
          {facets && Object.keys(facets.by_status).length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() => { setStatusFilter(""); setPage(1); }}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium ${!statusFilter ? "bg-[var(--accent)] text-white" : "bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"}`}
              >
                All
              </button>
              {["ready", "uploaded", "ingesting", "error"].map((s) => {
                const count = facets.by_status[s] ?? 0;
                if (count === 0) return null;
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => { setStatusFilter(s); setPage(1); }}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium capitalize ${statusFilter === s ? "bg-[var(--accent)] text-white" : "bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]"}`}
                  >
                    {s} ({count})
                  </button>
                );
              })}
            </div>
          ) : (
            <select
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
              className="text-sm bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-2 text-[var(--text-primary)]"
            >
              <option value="">All Statuses</option>
              <option value="uploaded">Uploaded</option>
              <option value="ingesting">Ingesting</option>
              <option value="ready">Ready</option>
              <option value="error">Error</option>
            </select>
          )}
          {/* Tag filter */}
          <select
            value={tagFilter}
            onChange={(e) => { setTagFilter(e.target.value); setPage(1); }}
            className="text-sm bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-2 text-[var(--text-primary)] min-w-[120px]"
            title="Filter by tag"
          >
            <option value="">All tags</option>
            {facets && Object.entries(facets.by_tag).sort((a, b) => b[1] - a[1]).map(([t, count]) => (
              <option key={t} value={t}>{t} ({count})</option>
            ))}
          </select>
          {/* Needs ingestion filter toggle */}
          <label className="flex items-center gap-2 cursor-pointer text-sm text-[var(--text-secondary)]">
            <input
              type="checkbox"
              checked={needsIngestionFilter}
              onChange={(e) => { setNeedsIngestionFilter(e.target.checked); setPage(1); }}
              className="accent-[var(--accent)] rounded"
            />
            Needs ingestion
          </label>
        </div>
        <div className="text-sm text-[var(--text-secondary)] shrink-0">
          {total} document{total !== 1 ? "s" : ""}
        </div>
      </div>

      {/* Table */}
      <DocumentTable
        documents={documents}
        total={total}
        page={page}
        pageSize={pageSize}
        onPageChange={setPage}
        onRefresh={fetchDocs}
        search={search}
        statusFilter={statusFilter}
        tagFilter={tagFilter}
        needsIngestionFilter={needsIngestionFilter}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onSortChange={(by, order) => {
          setSortBy(by);
          setSortOrder(order);
          setPage(1);
        }}
        graphSkipPageThreshold={graphSkipPageThreshold}
      />

      {/* Upload Modal */}
      <UploadModal open={showUpload} onClose={() => setShowUpload(false)} onUploaded={fetchDocs} />

      {/* Reset Vector Store Modal */}
      <ResetVectorStoreModal
        open={showResetVectorStore}
        onClose={() => setShowResetVectorStore(false)}
        onSuccess={fetchDocs}
      />

      {/* Resume Ingestion Modal (shown on startup if interrupted jobs exist) */}
      {recoveryJobs.length > 0 && (
        <ResumeIngestionModal
          jobs={recoveryJobs}
          onDone={() => {
            setRecoveryJobs([]);
            fetchDocs();
          }}
        />
      )}
    </div>
  );
}
