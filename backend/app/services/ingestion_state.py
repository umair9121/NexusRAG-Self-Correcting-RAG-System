"""
Persistent ingestion state — survives application restarts.

Tracks every queued/active ingestion job and per-pipeline progress so that
when the application is closed mid-ingestion and re-opened, the user can
resume from the exact pipeline stage that was interrupted.

State file: storage/ingestion_state.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from app import config

logger = logging.getLogger(__name__)

STATE_FILE: Path = config.STORAGE_DIR / "ingestion_state.json"

# Pipeline execution order — used to determine "remaining" pipelines on resume
PIPELINE_ORDER = ["pageindex", "contextual", "graph"]


class _IngestionState:
    """Thread-safe, file-backed ingestion state store.

    Each entry is keyed by document_id (as string) and contains:
    {
        "document_id": int,
        "filename": str,
        "file_path": str,
        "pipelines": ["pageindex", "contextual", "graph"],   # originally requested
        "completed_pipelines": ["pageindex"],                 # finished successfully
        "current_pipeline": "contextual" | "",                # pipeline that was running
        "status": "queued" | "running" | "done" | "error" | "cancelled",
    }
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if STATE_FILE.exists():
            try:
                raw = STATE_FILE.read_text(encoding="utf-8")
                self._data = json.loads(raw) if raw.strip() else {}
                logger.info("Loaded ingestion state: %d entries", len(self._data))
            except Exception as exc:
                logger.warning("Could not load ingestion state: %s", exc)
                self._data = {}
        else:
            self._data = {}

    def _flush(self) -> None:
        """Write current state to disk."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to persist ingestion state: %s", exc)

    # ── Public API ───────────────────────────────────────────────────────

    def track_job(
        self,
        document_id: int,
        filename: str,
        file_path: str,
        pipelines: list[str],
    ) -> None:
        """Register a new ingestion job (called when enqueued)."""
        with self._lock:
            self._data[str(document_id)] = {
                "document_id": document_id,
                "filename": filename,
                "file_path": file_path,
                "pipelines": pipelines,
                "completed_pipelines": [],
                "current_pipeline": "",
                "status": "queued",
            }
            self._flush()

    def mark_running(self, document_id: int, pipeline: str) -> None:
        """Called when a pipeline stage begins for a document."""
        with self._lock:
            key = str(document_id)
            if key in self._data:
                self._data[key]["status"] = "running"
                self._data[key]["current_pipeline"] = pipeline
                self._flush()

    def mark_pipeline_done(self, document_id: int, pipeline: str) -> None:
        """Called when a single pipeline stage finishes successfully."""
        with self._lock:
            key = str(document_id)
            if key in self._data:
                completed = self._data[key].get("completed_pipelines", [])
                if pipeline not in completed:
                    completed.append(pipeline)
                self._data[key]["completed_pipelines"] = completed
                self._data[key]["current_pipeline"] = ""
                self._flush()

    def mark_done(self, document_id: int) -> None:
        """Called when the entire ingestion job completes — removes from state."""
        with self._lock:
            self._data.pop(str(document_id), None)
            self._flush()

    def mark_error(self, document_id: int) -> None:
        """Called when the entire ingestion job errors."""
        with self._lock:
            key = str(document_id)
            if key in self._data:
                self._data[key]["status"] = "error"
                self._flush()

    def mark_cancelled(self, document_id: int) -> None:
        """Called when the job is cancelled — removes from state."""
        with self._lock:
            self._data.pop(str(document_id), None)
            self._flush()

    def remove(self, document_id: int) -> None:
        """Remove entry (e.g. user dismisses recovery)."""
        with self._lock:
            self._data.pop(str(document_id), None)
            self._flush()

    def remove_all(self) -> None:
        """Clear all recovery entries (user dismisses all)."""
        with self._lock:
            self._data.clear()
            self._flush()

    # ── Recovery helpers ─────────────────────────────────────────────────

    def get_interrupted_jobs(self) -> list[dict[str, Any]]:
        """Return jobs that were interrupted (queued or running when the app shut down).

        For each job, compute the remaining pipelines that still need to run.
        """
        with self._lock:
            interrupted: list[dict[str, Any]] = []
            for entry in self._data.values():
                if entry["status"] in ("queued", "running"):
                    completed = set(entry.get("completed_pipelines", []))
                    requested = entry.get("pipelines", [])
                    # Remaining = requested pipelines that were not completed, in order
                    remaining = [p for p in PIPELINE_ORDER if p in requested and p not in completed]
                    interrupted.append({
                        **entry,
                        "remaining_pipelines": remaining,
                    })
            return interrupted

    def get_all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._data)


# Global singleton
ingestion_state = _IngestionState()
