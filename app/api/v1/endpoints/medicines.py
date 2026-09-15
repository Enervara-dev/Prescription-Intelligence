"""
api/v1/endpoints/medicines.py
-------------------------------
Read-only access to the Postgres-backed medicine catalog used for OCR
matching (indexed pg_trgm search — see app.repositories.medicine_repository
and app.services.medicine_matching).
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories import medicine_repository as repo
from app.schemas.medicine import (
    MedicineListResponse,
    MedicineOut,
    MedicineSearchResponse,
    MedicineSearchResult,
)
from app.services.medicine_matching import search_many

router = APIRouter(prefix="/medicines", tags=["medicines"])


def _to_medicine_out(row) -> MedicineOut:
    return MedicineOut(
        id=str(row.id),
        external_id=row.external_id,
        generic_name=row.generic_name,
        brand_name=row.brand_name,
        aliases=row.aliases or [],
        strength=row.strength,
        dosage_form=row.dosage_form,
        route=row.route,
        composition=row.composition,
        manufacturer=row.manufacturer,
        atc_code=row.atc_code,
        snomed_ct_code=row.snomed_ct_code,
        rxcui=row.rxcui,
        source=row.source,
        source_version=row.source_version,
        source_updated_at=row.source_updated_at,
    )


@router.get("", response_model=MedicineListResponse, summary="List medicines in the catalog")
def list_medicines(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> MedicineListResponse:
    """Paginated listing of the local medicine catalog (all sources)."""
    rows, total = repo.list_medicines(db, limit=limit, offset=offset)
    return MedicineListResponse(
        total=total,
        limit=limit,
        offset=offset,
        medicines=[_to_medicine_out(r) for r in rows],
    )


@router.get("/search", response_model=MedicineSearchResponse, summary="Search medicines by name")
def search(
    q: str = Query(..., min_length=1, description="Free-text medicine name to search for."),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results to return."),
    db: Session = Depends(get_db),
) -> MedicineSearchResponse:
    """
    Indexed exact + fuzzy search (pg_trgm) against generic name, brand name,
    and aliases. An exact/alias hit is always ranked first.
    """
    matches = search_many(db, q, limit=limit)
    results = [
        MedicineSearchResult(
            id=str(m.medicine.id),
            generic_name=m.medicine.generic_name,
            brand_name=m.medicine.brand_name,
            strength=m.medicine.strength,
            dosage_form=m.medicine.dosage_form,
            route=m.medicine.route,
            manufacturer=m.medicine.manufacturer,
            confidence=round(m.confidence, 1),
            match_type=m.match_type,
            source=m.medicine.source,
        )
        for m in matches
    ]
    return MedicineSearchResponse(query=q, results=results)
