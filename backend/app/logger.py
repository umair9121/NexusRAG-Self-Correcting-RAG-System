"""
LogBroadcaster – Rich terminal logging + WebSocket fan-out.

All module progress callbacks route through this broadcaster.
The backend terminal gets Rich-formatted output, and connected
WebSocket clients receive JSON log events in real time.

Logs can be emitted from the main async thread or from worker threads
(e.g. during ingestion). Thread-produced entries are queued and
broadcast by a background drain task on the main event loop.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from datetime import datetime, timezone
from queue import Empty
from typing import Any

from fastapi import WebSocket

# ── Rich console logging setup ──────────────────────────────────────────────

try:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("rag_system")


# ── WebSocket LogBroadcaster ────────────────────────────────────────────────

class LogBroadcaster:
    """
    Maintains a list of connected WebSocket clients and fans out
    log events to all of them.  Also logs to the terminal via
    Python's logging module. Supports log() from worker threads:
    entries are pushed to a queue and broadcast by a drain task.
    """

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._log_buffer: list[dict[str, Any]] = []
        self._max_buffer = 500  # keep last 500 logs in memory
        self._thread_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._drain_task: asyncio.Task[None] | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self._clients))
        # Send buffered recent logs to catch up
        for entry in self._log_buffer[-50:]:
            try:
                await ws.send_json(entry)
            except Exception:
                break
        # If buffer was empty, send one welcome so the UI does not show "Waiting for log events..."
        if not self._log_buffer:
            welcome = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "module": "system",
                "message": "Log stream connected. Backend is ready.",
                "document_id": None,
            }
            try:
                await ws.send_json(welcome)
            except Exception:
                pass

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

    def log(
        self,
        message: str,
        *,
        level: str = "INFO",
        module: str = "system",
        document_id: int | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """
        Synchronous log entry point – usable as a callback from modules.
        Logs to terminal and queues a broadcast to WebSocket clients.
        """
        # Terminal log
        log_func = getattr(logger, level.lower(), logger.info)
        prefix = f"[{module}]"
        if document_id is not None:
            prefix += f" [doc#{document_id}]"
        log_func("%s %s", prefix, message)

        # Build log entry
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "module": module,
            "message": message,
            "document_id": document_id,
        }
        if extra:
            entry["extra"] = extra

        # Buffer
        self._log_buffer.append(entry)
        if len(self._log_buffer) > self._max_buffer:
            self._log_buffer = self._log_buffer[-self._max_buffer:]

        # Schedule async broadcast: main thread = create_task; worker thread = queue for drain
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast(entry))
        except RuntimeError:
            # Called from a worker thread (e.g. ingestion callback) – queue for main-loop drain
            self._thread_queue.put_nowait(entry)

    async def _drain_loop(self):
        """Run on main event loop: periodically drain thread queue and broadcast to WebSocket clients."""
        while True:
            await asyncio.sleep(0.15)
            while True:
                try:
                    entry = self._thread_queue.get_nowait()
                except Empty:
                    break
                await self._broadcast(entry)

    async def flush_thread_queue(self):
        """Drain the thread log queue and broadcast all entries immediately.
        Call this after run_in_executor(...) returns so that thread-originated
        logs (e.g. 'Detecting communities...') are sent to WebSocket clients
        before the main thread logs completion (e.g. 'Graph complete')."""
        while True:
            try:
                entry = self._thread_queue.get_nowait()
            except Empty:
                break
            await self._broadcast(entry)

    def start_background_drain(self):
        """Start the drain task so logs from worker threads reach WebSocket clients. Call from app lifespan."""
        if self._drain_task is None or self._drain_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._drain_task = loop.create_task(self._drain_loop())
                logger.info("Log drain task started (ingestion progress will stream to UI)")
            except RuntimeError:
                pass

    async def _broadcast(self, entry: dict[str, Any]):
        """Send a log entry to all connected WebSocket clients."""
        dead: list[WebSocket] = []
        async with self._lock:
            for ws in self._clients:
                try:
                    await ws.send_json(entry)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.remove(ws)

    def get_recent_logs(self, count: int = 100) -> list[dict[str, Any]]:
        """Return recent log entries from the buffer."""
        return self._log_buffer[-count:]

    def make_callback(
        self,
        module: str,
        document_id: int | None = None,
    ):
        """
        Return a callback function suitable for module ingestion callbacks.

        Usage:
            callback = broadcaster.make_callback("extraction", doc_id=42)
            extract_image_description(path, callback=callback)
        """
        def _cb(message: str):
            self.log(message, module=module, document_id=document_id)
        return _cb


# ── Global broadcaster singleton ────────────────────────────────────────────

broadcaster = LogBroadcaster()
