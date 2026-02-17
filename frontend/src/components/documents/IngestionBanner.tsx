"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Loader2, FileText, XCircle, Pause, Play, CheckCircle2, Trash2 } from "lucide-react";
import type { IngestionStatus } from "@/lib/types";
import { cancelAllIngestion, pauseIngestion, unpauseIngestion } from "@/lib/api";

interface IngestionBannerProps {
  status: IngestionStatus | null;
  onCancelDone?: () => void;
}

type CancelState =
  | { phase: "idle" }
  | { phase: "cancelling" }
  | { phase: "cleaning" }
  | { phase: "done"; count: number; filenames: string[] };

export default function IngestionBanner({ status, onCancelDone }: IngestionBannerProps) {
  const [pauseScheduledMessage, setPauseScheduledMessage] = useState(false);
  const [cancelState, setCancelState] = useState<CancelState>({ phase: "idle" });

  useEffect(() => {
    if (status?.paused === true && status?.active_job == null) {
      setPauseScheduledMessage(false);
    }
  }, [status?.paused, status?.active_job]);

  // Auto-dismiss success message after 5 seconds
  useEffect(() => {
    if (cancelState.phase === "done") {
      const timer = setTimeout(() => {
        setCancelState({ phase: "idle" });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [cancelState]);

  if (!status && cancelState.phase === "idle") return null;
  const hasActive = status?.active_job != null;
  const queueCount = status?.queue_length ?? (status?.queue?.length ?? 0);
  const paused = status?.paused === true;

  // Show banner during cancel states even if no active jobs
  if (cancelState.phase !== "idle") {
    // Cancelling / cleaning / done states
    if (cancelState.phase === "cancelling") {
      return (
        <div className="mb-4 flex flex-col gap-2 rounded-lg border p-4 border-amber-500/40 bg-amber-500/10">
          <div className="flex items-center gap-2 text-amber-300">
            <Loader2 size={18} className="animate-spin shrink-0" />
            <span className="font-medium">Cancelling ingestion...</span>
          </div>
          <p className="text-sm text-amber-200/80">
            Stopping the current pipeline and cancelling all queued documents.
          </p>
        </div>
      );
    }
    if (cancelState.phase === "cleaning") {
      return (
        <div className="mb-4 flex flex-col gap-2 rounded-lg border p-4 border-amber-500/40 bg-amber-500/10">
          <div className="flex items-center gap-2 text-amber-300">
            <Trash2 size={18} className="animate-pulse shrink-0" />
            <span className="font-medium">Cleaning up pipeline data...</span>
          </div>
          <p className="text-sm text-amber-200/80">
            Removing all partial pipeline data (PageIndex, vector embeddings, graph) for cancelled documents.
          </p>
        </div>
      );
    }
    if (cancelState.phase === "done") {
      return (
        <div className="mb-4 flex flex-col gap-2 rounded-lg border p-4 border-green-500/40 bg-green-500/10">
          <div className="flex items-center gap-2 text-green-300">
            <CheckCircle2 size={18} className="shrink-0" />
            <span className="font-medium">Ingestion cancelled successfully</span>
          </div>
          <p className="text-sm text-green-200/80">
            {cancelState.count} document{cancelState.count !== 1 ? "s" : ""} cancelled and all pipeline data cleaned up
            {cancelState.filenames.length > 0 && (
              <span className="ml-1">
                ({cancelState.filenames.join(", ")})
              </span>
            )}
            .
          </p>
        </div>
      );
    }
  }

  if (!hasActive && queueCount === 0 && !paused && !pauseScheduledMessage) return null;

  const showPauseAfterStageMessage = pauseScheduledMessage || (paused && hasActive);

  const handleCancel = async () => {
    try {
      setCancelState({ phase: "cancelling" });
      // Small delay so the user sees the "cancelling" state
      await new Promise((r) => setTimeout(r, 400));
      setCancelState({ phase: "cleaning" });
      const result = await cancelAllIngestion();
      setPauseScheduledMessage(false);
      setCancelState({
        phase: "done",
        count: result.cancelled_count ?? 0,
        filenames: (result.cancelled_documents ?? []).map((d) => d.filename),
      });
      onCancelDone?.();
    } catch (e) {
      console.error("Cancel ingestion failed:", e);
      setCancelState({ phase: "idle" });
      setPauseScheduledMessage(false);
      onCancelDone?.();
    }
  };

  const handlePause = async () => {
    try {
      setPauseScheduledMessage(true);
      await pauseIngestion();
      onCancelDone?.();
    } catch (e) {
      console.error("Pause ingestion failed:", e);
      setPauseScheduledMessage(false);
    }
  };

  const handleUnpause = async () => {
    try {
      setPauseScheduledMessage(false);
      await unpauseIngestion();
      onCancelDone?.();
    } catch (e) {
      console.error("Unpause ingestion failed:", e);
    }
  };

  return (
    <div className="mb-4 flex flex-col gap-2 rounded-lg border p-4 border-[var(--ingestion-banner-border)] bg-[var(--ingestion-banner-bg)]">
      <div className="flex items-center gap-2 text-[var(--ingestion-banner-text)]">
        {paused && !showPauseAfterStageMessage ? (
          <span className="font-medium">Ingestion paused</span>
        ) : showPauseAfterStageMessage ? (
          <>
            <Loader2 size={18} className="animate-spin shrink-0" />
            <span className="font-medium">Pause will occur after current ingestion stage finishes</span>
          </>
        ) : (
          <>
            <Loader2 size={18} className="animate-spin shrink-0" />
            <span className="font-medium">Ingestion in progress</span>
          </>
        )}
      </div>
      {showPauseAfterStageMessage && (
        <p className="text-sm text-[var(--ingestion-banner-text)]">
          The current pipeline stage is still running; ingestion will pause before the next stage.
        </p>
      )}
      {status?.active_job && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="flex items-center gap-1.5 text-[var(--text-primary)]">
            <FileText size={14} className="text-[var(--ingestion-banner-accent)]" />
            {status.active_job.filename}
            {status.active_job.current_pipeline && !showPauseAfterStageMessage && (
              <span className="text-[var(--text-secondary)]">
                — {status.active_job.current_pipeline}
              </span>
            )}
          </span>
          {showPauseAfterStageMessage ? (
            <button
              type="button"
              onClick={handleUnpause}
              className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-[var(--ingestion-banner-text)] hover:bg-[var(--ingestion-banner-btn-hover)]"
              title="Cancel scheduled pause and keep running"
            >
              <Play size={14} />
              Resume
            </button>
          ) : !paused ? (
            <>
              <button
                type="button"
                onClick={handlePause}
                className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-[var(--ingestion-banner-text)] hover:bg-[var(--ingestion-banner-btn-hover)]"
                title="Pause after current pipeline (queue is also paused)"
              >
                <Pause size={14} />
                Pause
              </button>
              <button
                type="button"
                onClick={handleCancel}
                className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-[var(--ingestion-banner-text)] hover:bg-[var(--ingestion-banner-btn-hover)]"
                title="Cancel all ingestion (active + queued) and clean up"
              >
                <XCircle size={14} />
                Cancel
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleUnpause}
              className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-[var(--ingestion-banner-text)] hover:bg-[var(--ingestion-banner-btn-hover)]"
              title="Resume ingestion"
            >
              <Play size={14} />
              Resume
            </button>
          )}
        </div>
      )}
      {paused && !status?.active_job && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-[var(--text-secondary)]">
            {queueCount > 0 ? "Queue and worker paused." : "Worker paused."}
          </span>
          <button
            type="button"
            onClick={handleUnpause}
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-[var(--ingestion-banner-text)] hover:bg-[var(--ingestion-banner-btn-hover)]"
            title="Resume ingestion"
          >
            <Play size={14} />
            Resume
          </button>
        </div>
      )}
      {queueCount > 0 && (
        <div className="text-sm text-[var(--text-secondary)]">
          Queued: {queueCount} document{queueCount !== 1 ? "s" : ""}
          {status?.queue && status.queue.length > 0 && (
            <span className="ml-1">
              ({status.queue.map((j) => j.filename).join(", ")})
            </span>
          )}
        </div>
      )}
      <p className="text-xs text-[var(--text-secondary)]">
        This state is restored after refresh. Open{" "}
        <Link href="/logs" className="text-[var(--ingestion-banner-accent)] hover:underline">
          Logs
        </Link>{" "}
        for live progress.
      </p>
    </div>
  );
}
