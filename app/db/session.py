"""
db/session.py
---------------
SQLAlchemy engine/session for this service's OWN Postgres database.

This is intentionally independent of any other application's database
connection — Prescription Intelligence owns its `medicines` catalog and is
consumed purely over HTTP by whatever calls it.

The engine is created LAZILY, on first actual use — not at import time.
Mirrors the same pattern prod_app's own backend/src/db/pool.ts uses
(`getPool()` constructs on first call, not at module load) and exists for
the same reason: DATABASE_URL has no hardcoded production fallback and is
required (see app/core/config.py) — if engine creation happened eagerly at
import time, simply IMPORTING this module (which almost every other module
transitively does) would crash whenever DATABASE_URL is unset, including
for pure-unit tests that never touch a database at all. Deferring engine
creation until a session is actually requested means "the app refuses to
serve traffic without DATABASE_URL" and "you can import/collect tests
without one configured" are both true at once.
"""

from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def _require_database_url() -> str:
    url = settings.DATABASE_URL
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. This service cannot connect to its "
            "database without a PostgreSQL connection string — see .env.example."
        )
    return url


def get_engine() -> Engine:
    """The SQLAlchemy Engine, created on first call and cached thereafter."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            _require_database_url(),
            pool_pre_ping=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_POOL_MAX_OVERFLOW,
            future=True,
        )
    return _engine


def _get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, future=True)
    return _session_factory


class _LazySessionLocal:
    """
    Callable proxy standing in for what used to be a plain `sessionmaker()`
    instance — `SessionLocal()` still returns a Session exactly as before.
    The only difference is that the underlying engine/sessionmaker aren't
    built until the FIRST call, not at import time (see module docstring).
    """

    def __call__(self, *args, **kwargs) -> Session:
        return _get_session_factory()(*args, **kwargs)


SessionLocal = _LazySessionLocal()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
