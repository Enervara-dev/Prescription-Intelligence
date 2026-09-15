"""
api/v1/endpoints/health.py
---------------------------
Liveness / readiness check.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Returns API liveness, DB connectivity, and whether the Vision API key is configured."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        vision_api_configured=bool(settings.GOOGLE_VISION_API_KEY),
        database_connected=db_ok,
    )
