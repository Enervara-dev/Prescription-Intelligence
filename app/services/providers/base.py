"""
services/providers/base.py
------------------------------
Ingestion-side abstraction: every medicine source (legacy manual catalog,
RxNorm today; ABDM, MIMS India, CIMS later) implements this interface,
producing records already normalized to the `medicines` row shape defined in
app/models/medicine.py. The resolver (app/services/medicine_resolver.py)
never knows which provider a row came from beyond its `source` string — it
only ever reads the unified Postgres table. Adding a new provider means
writing one class here and a thin sync script; nothing in the resolver,
repository, or API changes.
"""

from typing import Iterator, Protocol


class MedicineProvider(Protocol):
    """
    A source of medicine records. `fetch_records()` yields dicts already
    shaped for `medicine_repository.upsert_medicine()` (same keys as the
    `medicines` table's writable columns) — providers own translating their
    own raw format into that shape; the resolver and repository never see
    provider-specific fields.
    """

    #: Value written to `medicines.source` for every record this provider produces.
    source_name: str

    def is_configured(self) -> bool:
        """False when required config (credentials, feature flag) is missing."""
        ...

    def fetch_records(self) -> Iterator[dict]:
        """Yield normalized records ready for upsert_medicine(). May be empty."""
        ...


class ProviderNotConfiguredError(Exception):
    """Raised by a provider's fetch_records() when called while not configured."""
