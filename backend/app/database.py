"""
SQLite database setup with async SQLAlchemy.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import DATABASE_PATH

DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables on startup and add any missing columns (migrations)."""
    async with engine.begin() as conn:
        from app.models import ChatSession, Document, Workspace  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)

    # Add ingestion_duration_seconds if missing (e.g. existing DBs created before this column)
    async with engine.begin() as conn:
        def _add_ingestion_duration_if_missing(sync_conn):
            cur = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('documents') WHERE name = 'ingestion_duration_seconds'")
            )
            if cur.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE documents ADD COLUMN ingestion_duration_seconds INTEGER NULL"))

        await conn.run_sync(_add_ingestion_duration_if_missing)

    # Add path_hierarchy if missing (for existing DBs).
    async with engine.begin() as conn:
        def _add_path_hierarchy_if_missing(sync_conn):
            cur = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('documents') WHERE name = 'path_hierarchy'")
            )
            if cur.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE documents ADD COLUMN path_hierarchy TEXT DEFAULT '[]'"))

        await conn.run_sync(_add_path_hierarchy_if_missing)

    # Add workspace columns if missing (for existing DBs).
    async with engine.begin() as conn:
        def _add_workspace_columns_if_missing(sync_conn):
            cur_workspace_id = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('chat_sessions') WHERE name = 'workspace_id'")
            )
            if cur_workspace_id.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN workspace_id INTEGER NULL"))

            cur_session_cache = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('chat_sessions') WHERE name = 'session_cache'")
            )
            if cur_session_cache.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN session_cache TEXT DEFAULT ''"))

            cur_workspace_cache = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('workspaces') WHERE name = 'workspace_cache'")
            )
            if cur_workspace_cache.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE workspaces ADD COLUMN workspace_cache TEXT DEFAULT ''"))

            cur_last_used = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('workspaces') WHERE name = 'last_used_at'")
            )
            if cur_last_used.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE workspaces ADD COLUMN last_used_at DATETIME"))

        await conn.run_sync(_add_workspace_columns_if_missing)

    # Add UI audit / screenshot columns if missing (for existing DBs).
    async with engine.begin() as conn:
        def _add_audit_columns_if_missing(sync_conn):
            cur_ingestion_type = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('documents') WHERE name = 'ingestion_type'")
            )
            if cur_ingestion_type.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE documents ADD COLUMN ingestion_type TEXT DEFAULT 'document'"))
            cur_audit_result = sync_conn.execute(
                text("SELECT 1 FROM pragma_table_info('documents') WHERE name = 'audit_result'")
            )
            if cur_audit_result.fetchone() is None:
                sync_conn.execute(text("ALTER TABLE documents ADD COLUMN audit_result TEXT NULL"))

        await conn.run_sync(_add_audit_columns_if_missing)


async def get_db() -> AsyncSession:
    """Dependency: yield a database session."""
    async with async_session() as session:
        yield session
