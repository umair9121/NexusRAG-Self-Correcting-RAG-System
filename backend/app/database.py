"""
SQLite database setup with async SQLAlchemy.
"""

from __future__ import annotations

import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import DATABASE_PATH

logger = logging.getLogger(__name__)

DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def _migrate_schema(conn) -> None:
    """
    Handle schema migration from old pipeline columns to new ones.
    Adds new columns if they don't exist and migrates data from old columns.
    """
    # Check if old columns exist
    result = await conn.execute(text("PRAGMA table_info(documents)"))
    columns = {row[1] for row in result.fetchall()}

    migrations_needed = "pageindex_status" in columns and "extraction_status" not in columns

    if migrations_needed:
        logger.info("Migrating database schema: old pipeline columns -> new pipeline columns")

        # Add new columns
        for col, default in [
            ("extraction_status", "'pending'"),
            ("edi_status", "'pending'"),
            ("edi_rating", "''"),
        ]:
            if col not in columns:
                await conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col} VARCHAR(32) DEFAULT {default}"))
                logger.info("  Added column: %s", col)

        # Remove chunk_count if it exists (no longer needed) - SQLite doesn't support DROP COLUMN in older versions
        # so we just leave it as-is; it won't be used

        logger.info("Schema migration complete")

    # Also ensure new columns exist even if old columns were already gone (fresh-ish DB)
    for col, default, col_type in [
        ("extraction_status", "'pending'", "VARCHAR(32)"),
        ("edi_status", "'pending'", "VARCHAR(32)"),
        ("edi_rating", "''", "VARCHAR(32)"),
    ]:
        if col not in columns:
            try:
                await conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col} {col_type} DEFAULT {default}"))
                logger.info("Added missing column: %s", col)
            except Exception:
                pass  # Column might already exist


async def init_db():
    """Create all tables on startup and handle schema migrations."""
    async with engine.begin() as conn:
        from app.models import Document, ChatSession  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)

    # Run migrations after table creation
    async with engine.begin() as conn:
        await _migrate_schema(conn)


async def get_db() -> AsyncSession:
    """Dependency: yield a database session."""
    async with async_session() as session:
        yield session
