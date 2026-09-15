"""
schemas/resolve.py
---------------------
Request/response models for POST /api/v1/medicines/resolve — the Indian
medicine normalization pipeline's API surface (app/services/medicine_resolver.py).
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.services.medicine_resolver import MatchStatus


class ResolveRequest(BaseModel):
    text: str = Field(min_length=1, description="OCR token, brand phrase, or short line to resolve.")


class ResolvedMedicineOut(BaseModel):
    id: str
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    strength: Optional[float] = Field(default=None, description="Parsed numeric strength, if a single value applies.")
    unit: Optional[str] = None
    dosage_form: Optional[str] = None
    manufacturer: Optional[str] = None
    combination_components: List[dict] = Field(default_factory=list)


class ResolveCandidateOut(BaseModel):
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    confidence: float
    match_type: str
    source: str


class ParsedStrengthOut(BaseModel):
    found: bool
    is_combination: bool
    components: List[dict] = Field(default_factory=list)


class ResolveResponse(BaseModel):
    input: str
    normalized_text: str
    status: MatchStatus
    match_type: Optional[str] = None
    medicine: Optional[ResolvedMedicineOut] = None
    source: Optional[str] = None
    confidence: Optional[float] = Field(default=None, description="0-100, consistent with the rest of this API.")
    candidates: List[ResolveCandidateOut] = Field(
        default_factory=list, description="Populated for REVIEW_REQUIRED (and diagnostically for HIGH/LOW_CONFIDENCE)."
    )
    parsed_strength: ParsedStrengthOut
    parsed_dosage_form: Optional[str] = None
    latency_ms: float
