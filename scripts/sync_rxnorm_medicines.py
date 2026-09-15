"""
scripts/sync_rxnorm_medicines.py
------------------------------------
Manual/cron entry point for the RxNorm sync. Never called from the app's
request path — see app/services/rxnorm_sync.py.

Usage:
    python -m scripts.sync_rxnorm_medicines
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)

from app.db.session import SessionLocal  # noqa: E402
from app.services.rxnorm_sync import run_sync  # noqa: E402


def run() -> int:
    db = SessionLocal()
    try:
        return run_sync(db)
    finally:
        db.close()


if __name__ == "__main__":
    run()
