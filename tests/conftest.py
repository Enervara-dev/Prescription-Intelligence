"""
tests/conftest.py
-------------------
DB-backed tests run against a REAL Postgres (they exercise actual pg_trgm
queries, not mocks). Point TEST_DATABASE_URL at a disposable database before
running them:

    docker compose up -d db
    TEST_DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/prescription_intelligence \
        pytest

Tests that need the DB request the `db` fixture, which is skipped (not
failed) when no test database is configured, so `pytest` still runs cleanly
with no setup.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
if TEST_DATABASE_URL:
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL


@pytest.fixture(scope="session", autouse=True)
def _migrated_schema():
    """Apply Alembic migrations once per test session, if a test DB is configured."""
    if not TEST_DATABASE_URL:
        yield
        return
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    yield


@pytest.fixture()
def db():
    """A DB session with the `medicines` table truncated before each test."""
    if not TEST_DATABASE_URL:
        pytest.skip("Set TEST_DATABASE_URL to a real Postgres to run DB-backed tests.")

    from sqlalchemy import text

    from app.db.session import SessionLocal

    session = SessionLocal()
    session.execute(text("TRUNCATE TABLE medicines"))
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
