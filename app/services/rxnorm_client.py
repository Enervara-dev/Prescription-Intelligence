"""
services/rxnorm_client.py
----------------------------
Client for RxNorm (NLM's RxNav REST API) — a free, public, UNAUTHENTICATED
API. No registration, no sandbox approval, no credentials: unlike the ABDM
Drug Registry, this was actually verified reachable and its response shape
confirmed live before writing this client:

    GET https://rxnav.nlm.nih.gov/REST/version.json
    -> {"version": "08-Sep-2026", "apiVersion": "3.1.355"}

    GET https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN
    -> {"minConceptGroup": {"minConcept": [
           {"rxcui": "435", "name": "albuterol", "tty": "IN"}, ...
       ]}}

SCOPE / LIMITATION — read this before assuming coverage:
RxNorm is a US-centric generic/ingredient nomenclature. `tty=IN` (Ingredient)
concepts give a clean generic-name backbone (e.g. "amoxicillin",
"paracetamol" -> RxNorm calls it "acetaminophen") but carry NO Indian brand
names — Dolo, Crocin, Meftal, Combiflam etc. simply don't exist in RxNorm.
For Indian OTC/branded coverage, keep growing the `legacy_manual` source (see
scripts/import_legacy_medicines.py) or license an India-specific dataset
(MIMS India, CIMS) — RxNorm does not replace that, it only adds a free,
automatically-syncable generic-name layer alongside it.
"""

from dataclasses import dataclass
from typing import Any, Iterator

import requests

from app.core.config import settings


@dataclass
class RxNormConcept:
    rxcui: str
    name: str
    tty: str  # RxNorm term type, e.g. "IN" (Ingredient)


class RxNormClient:
    def __init__(self) -> None:
        self.base_url = settings.RXNORM_BASE_URL
        self.timeout = settings.RXNORM_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        # No credentials needed — "configured" just means a base URL is set,
        # which it always is by default.
        return bool(self.base_url)

    def fetch_ingredient_concepts(self) -> Iterator[RxNormConcept]:
        """
        Fetch every RxNorm Ingredient (tty=IN) concept — the generic-name
        backbone. One request returns the full list (RxNorm has on the order
        of tens of thousands of ingredient concepts; NLM's own API is
        designed to return this in a single call, no pagination parameter
        exists for allconcepts.json).
        """
        url = f"{self.base_url}/allconcepts.json"
        resp = requests.get(url, params={"tty": "IN"}, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        group = (payload or {}).get("minConceptGroup") or {}
        concepts = group.get("minConcept") or []
        for c in concepts:
            rxcui = c.get("rxcui")
            name = c.get("name")
            if not rxcui or not name:
                continue
            yield RxNormConcept(rxcui=str(rxcui), name=str(name), tty=c.get("tty", "IN"))


def normalize_concept(concept: RxNormConcept) -> dict[str, Any]:
    """Map an RxNorm ingredient concept onto our `medicines` row shape."""
    return {
        "external_id": f"rxnorm:{concept.rxcui}",
        "generic_name": concept.name.strip().title(),
        "brand_name": None,
        "aliases": [],
        "rxcui": concept.rxcui,
        "codes": {"rxnorm_tty": concept.tty},
        "source": "rxnorm",
    }
