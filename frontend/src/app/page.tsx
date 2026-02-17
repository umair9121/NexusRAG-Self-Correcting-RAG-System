"use client";

import { useEffect, useState, useCallback } from "react";
import { Upload, Search, Filter, AlertCircle, RefreshCw } from "lucide-react";
import { listDocuments, getIngestionStatus, getRecoveryJobs, isBackendAvailable, API_BASE } from "@/lib/api";
import type { Document, IngestionStatus, RecoveryJob } from "@/lib/types";
import DocumentTable from "@/components/documents/DocumentTable";
import UploadModal from "@/components/documents/UploadModal";
import IngestionBanner from "@/components/documents/IngestionBanner";
import ResumeIngestionModal from "@/components/documents/ResumeIngestionModal";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [recoveryJobs, setRecoveryJobs] = useState<RecoveryJob[]>([]);
  const [recoveryChecked, setRecoveryChecked] = useState(false);
  const pageSize = 50;

  const fetchDocs = useCallback(async () => {
    try {
      const res = await listDocuments({
        page,
        page_size: pageSize,
        search: search || undefined,
        status: statusFilter || undefined,
      });
      setDocuments(res.documents);
      setTotal(res.total);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }, [page, search, statusFilter]);

  // Decide "backend available" only from GET /api (so we don't show banner when backend is on but /api/documents 404s)
  useEffect(() => {
    let cancelled = false;
    isBackendAvailable().then((ok) => {
      if (!cancelled) setBackendOk(ok);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    fetchDocs();
    const interval = setInterval(fetchDocs, backendOk ? 5000 : 10000);
    return () => clearInterval(interval);
  }, [fetchDocs, backendOk]);

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
            onClick={() => setShowUpload(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-[var(--accent)] text-white rounded-lg hover:bg-[var(--accent-hover)] transition-colors"
          >
            <Upload size={16} />
            Upload documents
          </button>
        </div>
      </div>

      {/* Search & Filters */}
      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1 max-w-md">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]" />
          <input
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            placeholder="Search documents..."
            className="w-full pl-9 pr-4 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg text-sm text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter size={16} className="text-[var(--text-secondary)]" />
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
        </div>
        <div className="text-sm text-[var(--text-secondary)]">
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
      />

      {/* Upload Modal */}
      <UploadModal open={showUpload} onClose={() => setShowUpload(false)} onUploaded={fetchDocs} />

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
