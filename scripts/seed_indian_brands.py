"""
scripts/seed_indian_brands.py
--------------------------------
Idempotent seed of app/data/indian_brands.py into Postgres, tagged
source="legacy_manual" (source_version="hand_curated_indian_brands_v1" for
explicit provenance). Supplements the flat medicine_list.txt import with
structured brand/generic/strength/dosage-form/combination data.

Usage:
    python -m scripts.seed_indian_brands
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.indian_brands import INDIAN_BRANDS  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.repositories import medicine_repository as repo  # noqa: E402
from app.services.strength_parser import parse_strength  # noqa: E402

SOURCE = "legacy_manual"
SOURCE_VERSION = "hand_curated_indian_brands_v1"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def build_records() -> list[dict]:
    records = []
    for entry in INDIAN_BRANDS:
        combo = entry.get("combination_components") or []
        strength_value = strength_unit = None
        if not combo:
            info = parse_strength(entry.get("strength", ""))
            if info.single:
                strength_value = info.single.value
                strength_unit = info.single.unit

        records.append({
            # Same external_id scheme as scripts/import_legacy_medicines.py
            # ("legacy:<slug>"), DELIBERATELY: where a brand name here also
            # exists as a bare line in medicine_list.txt (e.g. "Dolo 650"),
            # this upserts onto that SAME row (enriching it with proper
            # brand/generic separation, strength, aliases, dosage form)
            # instead of creating a second, competing row for the same
            # real-world product under a different external_id namespace.
            "external_id": f"legacy:{_slugify(entry['brand_name'])}",
            "brand_name": entry["brand_name"],
            "generic_name": entry["generic_name"],
            "aliases": entry.get("aliases", []),
            "strength": entry.get("strength"),
            "strength_value": strength_value,
            "strength_unit": strength_unit,
            "dosage_form": entry.get("dosage_form"),
            "combination_components": combo,
            "source": SOURCE,
            "source_version": SOURCE_VERSION,
        })
    return records


def run() -> int:
    records = build_records()
    db = SessionLocal()
    try:
        written = repo.bulk_upsert(db, records)
        print(f"[seed_indian_brands] Upserted {written} / {len(records)} Indian brand records.")
        return written
    finally:
        db.close()


if __name__ == "__main__":
    run()
