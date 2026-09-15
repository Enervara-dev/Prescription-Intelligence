"""
main.py
-------
AI Prescription Intelligence — pure FastAPI backend (no frontend).

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs available at /docs (Swagger) and /redoc.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.v1.api import api_router
from app.api.v1.endpoints.health import health as health_check
from app.core.config import settings
from app.db.session import get_db, get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Boot-time DB check, before any request is served — mirrors prod_app's
    own backend/src/db/pool.ts assertDatabaseReachable(), called at boot
    before routes are mounted. DATABASE_URL has no fallback (see
    core/config.py): unset OR unreachable both fail startup loudly here,
    rather than letting the process come up and serve 500s from every
    request. Railway's restartPolicyType=ON_FAILURE is the intended
    recovery path for a database that's merely not up YET.
    """
    with get_engine().connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar_one()
        logger.info("[db] connected: %s", version.split(",")[0])
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", include_in_schema=False)
def root():
    """Redirect the API root to the interactive documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", include_in_schema=False)
def root_health(db: Session = Depends(get_db)):
    """Infra-friendly health check alias (load balancers, uptime probes)."""
    return health_check(db)
