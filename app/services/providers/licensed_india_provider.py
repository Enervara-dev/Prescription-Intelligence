"""
services/providers/licensed_india_provider.py
--------------------------------------------------
Documented, NOT-YET-IMPLEMENTED extension point for a licensed India-specific
dataset (MIMS India, CIMS, or similar) — the real path to comprehensive
Indian brand/strength/manufacturer coverage beyond the hand-curated
`legacy_manual` set. Same honest pattern as app/services/abdm_client.py: this
raises rather than guessing a contract, because there IS no contract yet —
licensing hasn't been arranged.

Deliberately never invented: no endpoint URLs, no field names, no auth flow.
Whoever arranges a license should implement `fetch_records()` against that
vendor's actual documented API/export format and remove this docstring's
placeholder note.

To wire in once a license + real integration details exist:
  1. Implement authentication/access per the vendor's actual contract.
  2. Implement `fetch_records()` to yield dicts shaped like
     medicine_repository.upsert_medicine() expects (generic_name, brand_name,
     aliases, strength, strength_value, strength_unit, dosage_form,
     manufacturer, combination_components, ...).
  3. Tag every record source="licensed_india" (add to SOURCE_PRIORITY in
     app/models/medicine.py, positioned above "rxnorm" — a licensed,
     purpose-built Indian dataset should outrank the free generic backbone,
     same as legacy_manual does).
  4. Add a feature gate (LICENSED_INDIA_SYNC_ENABLED, off by default) and a
     scripts/sync_licensed_india_medicines.py entry point, mirroring
     scripts/sync_rxnorm_medicines.py — sync stays out-of-band, never
     called from the OCR request path.
"""

from typing import Iterator


class LicensedIndiaProvider:
    source_name = "licensed_india"

    def is_configured(self) -> bool:
        return False  # no license/contract arranged yet

    def fetch_records(self) -> Iterator[dict]:
        raise NotImplementedError(
            "LicensedIndiaProvider is a documented extension point, not implemented: "
            "no India-specific dataset license/API contract has been arranged. "
            "See this module's docstring for what's needed to wire one in."
        )
