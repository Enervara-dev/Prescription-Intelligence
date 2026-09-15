"""
services/rxnorm_sync.py
--------------------------
Out-of-band sync job: populates the local `medicines` catalog with RxNorm
ingredient (generic-name) concepts. Like ABDM sync, this is NEVER called from
the OCR/prescription request path — matching happens against whatever is
already in Postgres; syncing is a separate, deliberately-triggered operation.

Run manually with:
    python -m scripts.sync_rxnorm_medicines

Safe to run at any time: disabled by default (RXNORM_SYNC_ENABLED=false); if
the upstream request fails for any reason (network, RxNav downtime, etc.) it
logs the error and returns 0 rather than raising into the caller.
"""

import logging

from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories import medicine_repository as repo
from app.services.rxnorm_client import RxNormClient, normalize_concept

logger = logging.getLogger(__name__)


def run_sync(db: Session) -> int:
    """Returns the number of records written. 0 if sync was skipped or failed."""
    if not settings.RXNORM_SYNC_ENABLED:
        logger.info("[rxnorm_sync] Skipped: RXNORM_SYNC_ENABLED is false.")
        return 0

    client = RxNormClient()
    if not client.is_configured():
        logger.warning("[rxnorm_sync] Skipped: RXNORM_BASE_URL is not set.")
        return 0

    try:
        concepts = list(client.fetch_ingredient_concepts())
    except Exception as exc:
        logger.error("[rxnorm_sync] Fetch failed: %s", exc, exc_info=True)
        return 0

    if not concepts:
        logger.warning("[rxnorm_sync] Upstream returned zero concepts — nothing written.")
        return 0

    batch = [normalize_concept(c) for c in concepts]
    written = repo.bulk_upsert(db, batch)
    logger.info("[rxnorm_sync] Upserted %d ingredient concepts from RxNorm.", written)
    return written
