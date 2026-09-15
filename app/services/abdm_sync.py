"""
services/abdm_sync.py
------------------------
Out-of-band sync job: populates the local `medicines` catalog from the ABDM
Drug Registry. This is intentionally NOT called from the OCR/prescription
request path — matching happens against whatever is already in Postgres;
syncing is a separate, deliberately-triggered operation (cron/manual).

Run manually with:
    python -m scripts.sync_abdm_medicines

Safe to run at any time: if ABDM_SYNC_ENABLED is false or credentials are
missing, it logs why and exits cleanly rather than failing — the app and the
legacy-seeded catalog keep working either way.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories import medicine_repository as repo
from app.services.abdm_client import ABDMDrugRegistryClient, normalize_record

logger = logging.getLogger(__name__)


def run_sync(db: Session) -> int:
    """Returns the number of records written. 0 if sync was skipped or is not yet implemented."""
    if not settings.ABDM_SYNC_ENABLED:
        logger.info("[abdm_sync] Skipped: ABDM_SYNC_ENABLED is false.")
        return 0

    client = ABDMDrugRegistryClient()
    if not client.is_configured():
        logger.warning(
            "[abdm_sync] Skipped: ABDM_SYNC_ENABLED is true but ABDM_BASE_URL / "
            "ABDM_CLIENT_ID / ABDM_CLIENT_SECRET are not fully set."
        )
        return 0

    try:
        records = list(client.fetch_records())
    except NotImplementedError as exc:
        # Expected today — see app/services/abdm_client.py for what's needed
        # to finish this once the real API contract is verified.
        logger.error("[abdm_sync] Not implemented yet: %s", exc)
        return 0

    written = 0
    now = datetime.now(timezone.utc)
    batch = []
    for record in records:
        data = normalize_record(record)
        data["source"] = "abdm"
        data.setdefault("external_id", record.external_id)
        data.setdefault("source_updated_at", now)
        batch.append(data)

    if batch:
        written = repo.bulk_upsert(db, batch)
        logger.info("[abdm_sync] Upserted %d records from ABDM.", written)

    return written
