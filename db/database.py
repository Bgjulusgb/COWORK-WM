from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.database_url, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


def _sqlite_type(col) -> str:
    """Map SQLAlchemy column types to SQLite-compatible declarations."""
    name = col.type.__class__.__name__.upper()
    if "INT" in name:
        return "INTEGER"
    if "FLOAT" in name or "NUMERIC" in name or "DECIMAL" in name:
        return "REAL"
    if "BOOL" in name:
        return "INTEGER"
    if "JSON" in name or "TEXT" in name:
        return "TEXT"
    if "DATE" in name or "TIME" in name:
        return "TEXT"
    return "TEXT"


def _add_missing_columns(sync_conn) -> None:
    """Lightweight migration: ALTER TABLE for any column declared in models
    but missing in the live schema. SQLite-friendly, only adds NULLABLE
    columns (which is all we ever add).
    """
    insp = inspect(sync_conn)
    existing_tables = set(insp.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        live_cols = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in live_cols:
                continue
            sql_type = _sqlite_type(col)
            stmt = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {sql_type}'
            sync_conn.execute(text(stmt))


async def init_db() -> None:
    from db import models  # noqa: F401 — register models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Auto-add columns that were introduced after the DB file was created.
        # SQLite only — Alembic would replace this for Postgres.
        if settings.database_url.startswith("sqlite"):
            await conn.run_sync(_add_missing_columns)
