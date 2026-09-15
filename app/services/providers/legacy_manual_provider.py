"""
services/providers/legacy_manual_provider.py
------------------------------------------------
Thin MedicineProvider adapter over the existing legacy-catalog sources
(flat medicine_list.txt names + the structured Indian-brands fixture).
Wraps scripts/import_legacy_medicines.py and scripts/seed_indian_brands.py's
existing, already-tested logic rather than duplicating it — this class exists
so the ingestion pipeline is describable uniformly as "providers", not to
replace those scripts.
"""

from typing import Iterator

from app.data.indian_brands import INDIAN_BRANDS
from scripts.import_legacy_medicines import _slugify, load_legacy_names
from scripts.seed_indian_brands import build_records as _build_indian_brand_records


class LegacyManualProvider:
    source_name = "legacy_manual"

    def is_configured(self) -> bool:
        return True  # always available; no external dependency

    def fetch_records(self) -> Iterator[dict]:
        for name in load_legacy_names():
            yield {
                "external_id": f"legacy:{_slugify(name)}",
                "generic_name": name,
                "source": self.source_name,
            }
        yield from _build_indian_brand_records()


def indian_brand_count() -> int:
    """Convenience for callers that just want a quick count (e.g. health checks)."""
    return len(INDIAN_BRANDS)
