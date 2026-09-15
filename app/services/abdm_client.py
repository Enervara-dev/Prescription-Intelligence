"""
services/abdm_client.py
--------------------------
Client boundary for the ABDM (Ayushman Bharat Digital Mission) National Drug
Registry — the intended primary external source for the medicine catalog.

STATUS: the wire contract below is a documented EXTENSION POINT, not a
verified implementation. Before writing this client we tried to confirm the
actual API — auth flow, base URL, endpoints, request/response shapes —
against:
  - https://drugregistrybeta.abdm.gov.in/ (the Drug Registry portal itself)
  - https://devforum.abdm.gov.in/ (the ABDM developer forum, where other
    ABDM services' Swagger docs are usually linked/discussed)
Both were unreachable from this environment (connection refused), and no
public Swagger/OpenAPI spec for the Drug Registry specifically turned up via
search. Coverage we DID confirm: NHA + CDSCO + NRCeS launched a "National
Drug Registry" under ABDM adopting SNOMED CT for terminology — but not a
concrete REST contract.

Per the task's own instruction ("do not assume its API/authentication
contract"), this client deliberately does NOT guess endpoint paths, auth
flow, or payload shapes. `fetch_records()` raises NotImplementedError until
someone with sandbox access fills it in against the real, verified contract.

Everything around it — configuration, the "don't sync per-OCR-request" rule,
credential handling — IS real and usable today:
  - ABDM_SYNC_ENABLED (default false) gates whether sync runs at all.
  - ABDM_BASE_URL / ABDM_CLIENT_ID / ABDM_CLIENT_SECRET are read from env,
    never hard-coded, and are not required for local development — the app
    and the legacy-seeded catalog work fully without them.
  - Sync is invoked out-of-band (see services/abdm_sync.py), never from the
    prescription-processing request path.

To finish this integration once ABDM sandbox credentials + the actual API
spec are available:
  1. Get access at https://sandbox.abdm.gov.in/ and locate the Drug
     Registry's OpenAPI/Swagger definition (ask on devforum.abdm.gov.in if
     it isn't linked from the sandbox docs — that forum thread exists:
     "Need a centralised page with all swagger APIs").
  2. Implement `authenticate()` per whatever auth ABDM actually requires
     (likely an OAuth2 client-credentials flow, matching the pattern used by
     other ABDM services, but VERIFY this rather than assuming it).
  3. Implement `fetch_records()` to page through the registry and yield raw
     records.
  4. Implement `normalize_record()` to map a raw ABDM record onto our
     `Medicine` fields (generic_name, brand_name, strength, dosage_form,
     route, composition, manufacturer, atc_code, snomed_ct_code, rxcui,
     codes) — field names below are placeholders and MUST be checked against
     the real response shape.
"""

from dataclasses import dataclass
from typing import Any, Iterator

from app.core.config import settings


class ABDMNotConfiguredError(Exception):
    """Raised when a sync is attempted without ABDM_SYNC_ENABLED + credentials."""


@dataclass
class ABDMRecord:
    """Raw record as returned by the registry, before normalization."""

    external_id: str
    raw: dict[str, Any]


class ABDMDrugRegistryClient:
    def __init__(self) -> None:
        self.base_url = settings.ABDM_BASE_URL
        self.client_id = settings.ABDM_CLIENT_ID
        self.client_secret = settings.ABDM_CLIENT_SECRET

    def is_configured(self) -> bool:
        return bool(self.base_url and self.client_id and self.client_secret)

    def fetch_records(self, since: str | None = None) -> Iterator[ABDMRecord]:
        """
        Page through the ABDM Drug Registry and yield raw records.

        NOT IMPLEMENTED: the actual endpoint(s), auth flow, and pagination
        scheme could not be verified (see module docstring). Raises
        immediately rather than guessing a contract and silently returning
        wrong/empty data.
        """
        raise NotImplementedError(
            "ABDMDrugRegistryClient.fetch_records() is a documented extension "
            "point, not implemented: the Drug Registry's public API contract "
            "could not be verified (drugregistrybeta.abdm.gov.in and "
            "devforum.abdm.gov.in were both unreachable, no public Swagger "
            "found). Get sandbox access + the real OpenAPI spec from ABDM, "
            "then implement this against the verified contract. "
            "See app/services/abdm_client.py's module docstring."
        )


def normalize_record(record: ABDMRecord) -> dict[str, Any]:
    """
    Map a raw ABDM record onto our `medicines` row shape. Placeholder field
    names — MUST be revisited once the real response shape is known.
    """
    raise NotImplementedError(
        "normalize_record() depends on the real ABDM Drug Registry response "
        "shape, which has not been verified yet. See app/services/abdm_client.py."
    )
