"use client";

import { useState } from "react";
import { X, DatabaseZap, Loader2, AlertTriangle, CheckCircle2 } from "lucide-react";
import { resetVectorStore, type ResetVectorStoreResponse } from "@/lib/api";

interface ResetVectorStoreModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function ResetVectorStoreModal({
  open,
  onClose,
  onSuccess,
}: ResetVectorStoreModalProps) {
  const [resetting, setResetting] = useState(false);
  const [result, setResult] = useState<ResetVectorStoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleReset = async () => {
    setResetting(true);
    setError(null);
    setResult(null);
    try {
      const res = await resetVectorStore();
      setResult(res);
      onSuccess();
    } catch (e) {
      console.error(e);
      setError(e instanceof Error ? e.message : "Failed to reset vector store");
    } finally {
      setResetting(false);
    }
  };

  const handleClose = () => {
    setResult(null);
    setError(null);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={handleClose} />
      <div className="relative w-full max-w-md bg-[var(--bg-secondary)] rounded-xl border border-[var(--border)] shadow-xl p-5">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-[var(--text-primary)] flex items-center gap-2">
            <DatabaseZap size={18} className="text-[var(--btn-warning-solid-bg)]" />
            Reset Vector Store
          </h3>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        {/* Result display */}
        {result && (
          <div className={`mb-4 p-3 rounded-lg border text-sm ${
            result.success
              ? "bg-[var(--success)]/10 border-[var(--success)]/30 text-[var(--success)]"
              : "bg-[var(--btn-warning-bg)] border-[var(--btn-warning-border)] text-[var(--btn-warning-text)]"
          }`}>
            <div className="flex items-start gap-2">
              <CheckCircle2 size={16} className="shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">{result.message}</p>
                <ul className="mt-2 space-y-0.5 text-xs opacity-80">
                  <li>ChromaDB collection: {result.chroma_reset ? "reset" : "failed"}</li>
                  <li>DocStore entries cleared: {result.docstore_entries_cleared}</li>
                  <li>Documents reset to pending: {result.documents_reset}</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="mb-4 p-3 rounded-lg border bg-[var(--error)]/10 border-[var(--error)]/30 text-[var(--error)] text-sm">
            {error}
          </div>
        )}

        {/* Explanation (only before action) */}
        {!result && (
          <>
            <div className="mb-4 p-3 rounded-lg bg-[var(--btn-warning-bg)] border border-[var(--btn-warning-border)]">
              <div className="flex items-start gap-2">
                <AlertTriangle size={16} className="text-[var(--btn-warning-solid-bg)] shrink-0 mt-0.5" />
                <div className="text-sm text-[var(--text-primary)]">
                  <p className="font-medium text-[var(--btn-warning-text)]">This will delete all vector embeddings</p>
                  <p className="mt-1 text-[var(--text-secondary)] text-xs leading-relaxed">
                    The ChromaDB collection and DocStore will be wiped. All documents&apos;
                    contextual status will be reset to &ldquo;pending&rdquo;. Document files
                    are <strong>not</strong> deleted.
                  </p>
                </div>
              </div>
            </div>

            <p className="text-xs text-[var(--text-secondary)] mb-4 leading-relaxed">
              Use this when switching embedding models (e.g. from <code className="px-1 py-0.5 bg-[var(--bg-tertiary)] rounded text-[var(--text-primary)]">nomic-embed-text</code> to{" "}
              <code className="px-1 py-0.5 bg-[var(--bg-tertiary)] rounded text-[var(--text-primary)]">qwen3-embedding:4b</code>).
              After resetting, re-ingest your documents to rebuild embeddings with the new model.
            </p>

            <div className="flex gap-2">
              <button
                onClick={handleClose}
                disabled={resetting}
                className="flex-1 px-4 py-2.5 rounded-lg bg-[var(--bg-tertiary)] hover:bg-[var(--border)] text-[var(--text-primary)] text-sm disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleReset}
                disabled={resetting}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-[var(--btn-warning-solid-bg)] text-white hover:bg-[var(--btn-warning-solid-hover)] disabled:opacity-50 text-sm font-medium"
              >
                {resetting ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <DatabaseZap size={16} />
                )}
                {resetting ? "Resetting..." : "Reset Vector Store"}
              </button>
            </div>
          </>
        )}

        {/* Close button after completion */}
        {result && (
          <button
            onClick={handleClose}
            className="w-full px-4 py-2.5 rounded-lg bg-[var(--bg-tertiary)] hover:bg-[var(--border)] text-[var(--text-primary)] text-sm"
          >
            Close
          </button>
        )}
      </div>
    </div>
  );
}
