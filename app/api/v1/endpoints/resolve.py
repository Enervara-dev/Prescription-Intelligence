"""
api/v1/endpoints/resolve.py
------------------------------
POST /api/v1/medicines/resolve — the Indian medicine normalization pipeline's
API surface. Distinct from GET /medicines/search (free-text catalog browsing):
this is the resolver downstream OCR processing should call to turn one piece
of OCR text into a structured, confidence-graded medicine entity — or an
explicit REVIEW_REQUIRED/NO_MATCH rather than a guess.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.resolve import (
    ParsedStrengthOut,
    ResolveCandidateOut,
    ResolveRequest,
    ResolveResponse,
    ResolvedMedicineOut,
)
from app.services.medicine_resolver import ResolutionResult, resolve

router = APIRouter(prefix="/medicines", tags=["medicines"])


def _medicine_out(result: ResolutionResult) -> ResolvedMedicineOut | None:
    if not result.matched:
        return None
    med = result.matched.medicine
    strength = float(med.strength_value) if med.strength_value is not None else None
    unit = med.strength_unit
    if strength is None and result.parsed_strength.single:
        # Catalog row has no structured strength of its own (e.g. a generic-
        # only entry) -- fall back to what was parsed straight from the
        # query text, clearly still sourced from the input, not invented.
        strength = float(result.parsed_strength.single.value)
        unit = result.parsed_strength.single.unit
    return ResolvedMedicineOut(
        id=str(med.id),
        brand_name=med.brand_name,
        generic_name=med.generic_name,
        strength=strength,
        unit=unit,
        dosage_form=med.dosage_form,
        manufacturer=med.manufacturer,
        combination_components=med.combination_components or [],
    )


def _to_response(result: ResolutionResult) -> ResolveResponse:
    return ResolveResponse(
        input=result.input_text,
        normalized_text=result.normalized_text,
        status=result.status,
        match_type=result.match_type,
        medicine=_medicine_out(result),
        source=result.source,
        confidence=result.confidence,
        candidates=[
            ResolveCandidateOut(
                brand_name=c.medicine.brand_name,
                generic_name=c.medicine.generic_name,
                confidence=c.confidence,
                match_type=c.match_type,
                source=c.medicine.source,
            )
            for c in result.candidates
        ],
        parsed_strength=ParsedStrengthOut(
            found=result.parsed_strength.found,
            is_combination=result.parsed_strength.is_combination,
            components=result.parsed_strength.as_dict().get(
                "components", [result.parsed_strength.as_dict()] if result.parsed_strength.found else []
            ),
        ),
        parsed_dosage_form=result.parsed_dosage_form,
        latency_ms=round(result.latency_ms, 2),
    )


@router.post("/resolve", response_model=ResolveResponse, summary="Resolve OCR text to a catalog medicine")
def resolve_medicine(payload: ResolveRequest, db: Session = Depends(get_db)) -> ResolveResponse:
    """
    Run the full Indian medicine normalization pipeline (text normalization
    -> strength/dosage-form extraction -> matching hierarchy -> confidence
    policy) on one piece of text. Never fabricates a match: uncertain input
    comes back as REVIEW_REQUIRED with candidates, not a forced pick.
    """
    result = resolve(db, payload.text)
    return _to_response(result)
