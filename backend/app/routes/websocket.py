"""
WebSocket endpoint for live log streaming.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.logger import broadcaster

router = APIRouter()


@router.websocket("/ws/logs")
async def log_stream(ws: WebSocket):
    """Stream log events to connected clients."""
    await broadcaster.connect(ws)
    try:
        while True:
            # Keep connection alive; wait for client messages (e.g. pings)
            data = await ws.receive_text()
            # Could handle client-side filter requests here
    except WebSocketDisconnect:
        await broadcaster.disconnect(ws)
    except Exception:
        await broadcaster.disconnect(ws)
