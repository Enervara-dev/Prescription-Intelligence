"""
scripts/import_legacy_medicines.py
------------------------------------
One-off / idempotent migration: loads app/data/medicine_list.txt (the old
flat-file knowledgebase) into the Postgres `medicines` table, tagged
source="legacy_manual".

Safe to re-run — each row is upserted on (source, external_id), so running
this twice never duplicates rows. This is what keeps the catalog searchable
even before ABDM sync is configured (see app/services/abdm_sync.py).

Usage:
    python -m scripts.import_legacy_medicines
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal  # noqa: E402
from app.repositories import medicine_repository as repo  # noqa: E402

MEDICINE_LIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data", "medicine_list.txt"
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def load_legacy_names(path: str = MEDICINE_LIST_PATH) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run() -> int:
    names = load_legacy_names()
    if not names:
        print(f"[import_legacy_medicines] No entries found in {MEDICINE_LIST_PATH}")
        return 0

    db = SessionLocal()
    try:
        records = [
            {
                "external_id": f"legacy:{_slugify(name)}",
                "generic_name": name,
                "brand_name": None,
                "aliases": [],
                "source": "legacy_manual",
                # This flat file never distinguishes brand from generic
                # identity -- many of these names are actually brands
                # (Dolo, Crocin, Omez, ...) forced into generic_name purely
                # because the source data gives no other signal. Tagging
                # this explicitly (rather than asserting a generic identity
                # we don't actually know) lets the resolver report an
                # honest "unclassified catalog name" match type instead of
                # a confirmed-but-fabricated EXACT_GENERIC -- see
                # medicine_resolver.UNCLASSIFIED_NAME_SOURCE_VERSION. Rows
                # also curated in app/data/indian_brands.py (matched by
                # slug) get overwritten with a real brand/generic split by
                # scripts/seed_indian_brands.py and lose this tag.
                "source_version": "unclassified_name",
            }
            for name in names
        ]
        written = repo.bulk_upsert(db, records)
        print(f"[import_legacy_medicines] Upserted {written} / {len(names)} legacy medicines.")
        return written
    finally:
        db.close()


if __name__ == "__main__":
    run()
