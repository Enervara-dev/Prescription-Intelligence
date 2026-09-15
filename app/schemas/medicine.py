"""
schemas/medicine.py
--------------------
Models for the medicine catalog endpoints (Postgres-backed).
"""

from datetime import datetime

from pydantic import BaseModel, Field
from typing import List, Optional


class MedicineOut(BaseModel):
    id: str
    external_id: Optional[str] = None
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    strength: Optional[str] = None
    dosage_form: Optional[str] = None
    route: Optional[str] = None
    composition: Optional[str] = None
    manufacturer: Optional[str] = None
    atc_code: Optional[str] = None
    snomed_ct_code: Optional[str] = None
    rxcui: Optional[str] = None
    source: str
    source_version: Optional[str] = None
    source_updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MedicineListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    medicines: List[MedicineOut]


class MedicineSearchResult(BaseModel):
    id: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    strength: Optional[str] = None
    dosage_form: Optional[str] = None
    route: Optional[str] = None
    manufacturer: Optional[str] = None
    confidence: float = Field(description="Match confidence, 0-100.")
    match_type: str = Field(description="exact | alias | fuzzy")
    source: str


class MedicineSearchResponse(BaseModel):
    query: str
    results: List[MedicineSearchResult]
