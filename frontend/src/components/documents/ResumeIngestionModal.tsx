"use client";

import { useState } from "react";
import { AlertTriangle, Play, X, Check, FileText, RefreshCw } from "lucide-react";
import type { RecoveryJob } from "@/lib/types";
import { resumeIngestion, dismissRecovery } from "@/lib/api";

/** Human-readable pipeline label map. */
const PIPELINE_LABELS: Record<string, string> = {
  pageindex: "PageIndex",
  contextual: "Vector Embeddings",
  graph: "Graph RAG",
};

interface ResumeIngestionModalProps {
  jobs: RecoveryJob[];
  onDone: () => void;
}

export default function ResumeIngestionModal({ jobs, onDone }: ResumeIngestionModalProps) {
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(jobs.map((j) => j.document_id)),
  );
  const [loading, setLoading] = useState(false);

  if (jobs.length === 0) return null;

  const toggle = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === jobs.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(jobs.map((j) => j.document_id)));
    }
  };

  const handleResume = async () => {
    if (selected.size === 0) return;
    setLoading(true);
    try {
      const ids = Array.from(selected);
      // Resume selected, dismiss the rest
      await resumeIngestion(ids);
      const dismissed = jobs
        .filter((j) => !selected.has(j.document_id))
        .map((j) => j.document_id);
      if (dismissed.length > 0) {
        await dismissRecovery(dismissed);
      }
      onDone();
    } catch (e) {
      console.error("Resume failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleDismissAll = async () => {
    setLoading(true);
    try {
      await dismissRecovery();
      onDone();
    } catch (e) {
      console.error("Dismiss failed:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div
        className="w-full max-w-xl bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start gap-3 mb-5">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-amber-500/15 shrink-0">
            <AlertTriangle size={20} className="text-amber-400" />
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-semibold">Resume Interrupted Ingestion?</h2>
            <p className="text-sm text-[var(--text-secondary)] mt-1">
              The application was closed while {jobs.length === 1 ? "a document was" : `${jobs.length} documents were`} being
              ingested. You can resume from where each document left off.
            </p>
          </div>
          <button
            onClick={handleDismissAll}
            disabled={loading}
            className="text-[var(--text-secondary)] hover:text-white shrink-0"
            title="Dismiss all"
          >
            <X size={20} />
          </button>
        </div>

        {/* Document list */}
        <div className="max-h-64 overflow-y-auto mb-5 space-y-2 pr-1">
          {/* Select all */}
          <button
            onClick={toggleAll}
            className="flex items-center gap-2 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] mb-1 px-1"
          >
            <div
              className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                selected.size === jobs.length
                  ? "bg-[var(--accent)] border-[var(--accent)]"
                  : "border-[var(--border)]"
              }`}
            >
              {selected.size === jobs.length && <Check size={12} className="text-white" />}
            </div>
            {selected.size === jobs.length ? "Deselect all" : "Select all"}
          </button>

          {jobs.map((job) => {
            const isSelected = selected.has(job.document_id);
            return (
              <button
                key={job.document_id}
                onClick={() => toggle(job.document_id)}
                className={`w-full flex items-start gap-3 p-3 rounded-lg border text-left transition-colors ${
                  isSelected
                    ? "border-[var(--accent)] bg-[var(--accent)]/5"
                    : "border-[var(--border)] bg-[var(--bg-primary)] hover:border-[var(--text-secondary)]"
                }`}
              >
                {/* Checkbox */}
                <div
                  className={`mt-0.5 w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors ${
                    isSelected
                      ? "bg-[var(--accent)] border-[var(--accent)]"
                      : "border-[var(--border)]"
                  }`}
                >
                  {isSelected && <Check size={12} className="text-white" />}
                </div>

                {/* File info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <FileText size={14} className="text-[var(--text-secondary)] shrink-0" />
                    <span className="text-sm font-medium truncate">{job.filename}</span>
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {/* Show pipeline chips: completed in green, remaining in amber */}
                    {job.pipelines.map((p) => {
                      const completed = job.completed_pipelines.includes(p);
                      return (
                        <span
                          key={p}
                          className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                            completed
                              ? "bg-emerald-500/15 text-emerald-400"
                              : "bg-amber-500/15 text-amber-400"
                          }`}
                        >
                          {completed ? (
                            <Check size={10} />
                          ) : (
                            <RefreshCw size={10} />
                          )}
                          {PIPELINE_LABELS[p] ?? p}
                        </span>
                      );
                    })}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-[var(--border)]">
          <button
            onClick={handleDismissAll}
            disabled={loading}
            className="px-4 py-2 text-sm rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-primary)] transition-colors"
          >
            Dismiss
          </button>
          <button
            onClick={handleResume}
            disabled={loading || selected.size === 0}
            className="flex items-center gap-2 px-5 py-2 text-sm font-medium bg-[var(--accent)] text-white rounded-lg hover:bg-[var(--accent-hover)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <RefreshCw size={14} className="animate-spin" />
                Resuming…
              </>
            ) : (
              <>
                <Play size={14} />
                Resume {selected.size > 0 ? `(${selected.size})` : ""}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
