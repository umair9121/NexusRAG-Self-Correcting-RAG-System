"""
SQLAlchemy ORM models for Document and ChatSession.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(512), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    status = Column(String(32), default="uploaded")  # uploaded|ingesting|ready|error
    tags = Column(Text, default="[]")                # JSON array
    path_hierarchy = Column(Text, default="[]")     # JSON array of path segments (e.g. ["Trader", "Financial", "MonthEnd"])
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    # Per-pipeline ingestion status
    pageindex_status = Column(String(32), default="pending")   # pending|running|done|error|skipped
    contextual_status = Column(String(32), default="pending")
    graph_status = Column(String(32), default="pending")
    # Stats
    chunk_count = Column(Integer, default=0)
    node_count = Column(Integer, default=0)
    edge_count = Column(Integer, default=0)
    error_message = Column(Text, default="")
    # Seconds to complete all 3 pipelines (set when status becomes "ready"); null if never completed
    ingestion_duration_seconds = Column(Integer, nullable=True)

    ingestion_type = Column(String(32), default="document")  # document | screenshot | regulatory
    # JSON blob for UI audit result (audit_metadata, detected_violations, generative_synthesis) when document is a screenshot
    audit_result = Column(Text, default=None)  # JSON

    chat_sessions = relationship("ChatSession", back_populates="document")


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, default="New Workspace")
    workspace_cache = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    last_used_at = Column(DateTime, default=_utcnow)

    chat_sessions = relationship("ChatSession", back_populates="workspace")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(512), default="New Chat")
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    messages = Column(Text, default="[]")  # JSON array of {role, content, trace, metadata}
    session_cache = Column(Text, default="")

    document = relationship("Document", back_populates="chat_sessions")
    workspace = relationship("Workspace", back_populates="chat_sessions")
