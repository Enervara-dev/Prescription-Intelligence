"""
services/providers/rxnorm_provider.py
-----------------------------------------
Thin MedicineProvider adapter over the existing, verified RxNorm client
(app/services/rxnorm_client.py) — wraps it rather than duplicating it.
"""

from typing import Iterator

from app.services.rxnorm_client import RxNormClient, normalize_concept


class RxNormProvider:
    source_name = "rxnorm"

    def __init__(self) -> None:
        self._client = RxNormClient()

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def fetch_records(self) -> Iterator[dict]:
        for concept in self._client.fetch_ingredient_concepts():
            yield normalize_concept(concept)
