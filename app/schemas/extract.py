"""
schemas/extract.py
---------------------
Request/response models for POST /api/v1/prescriptions/extract — the
service-to-service adapter consumed by Enervara's
HttpPrescriptionProcessingService (prod_app/app/backend/src/prescriptions/
processing/httpProcessingService.ts).

Field names/shapes mirror prod_app's PrescriptionExtraction /
MedicationInput / PrescriptionMetadataInput TypeScript interfaces
(prod_app/app/backend/src/prescriptions/domain/types.ts) EXACTLY — camelCase
on the wire via `alias_generator=to_camel` — since that's what Enervara's
normaliser primarily reads (it also checks snake_case as a fallback, but
matching the primary contract directly is the correct behaviour, not a
reliance on that fallback).
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ExtractRequest(BaseModel):
    """Exactly what HttpPrescriptionProcessingService.extract() sends."""

    prescription_id: str
    user_id: str
    document_url: str
    mime_type: str
    file_name: str
    request_id: str


class _CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class MedicationOut(_CamelModel):
    name: str
    generic_name: Optional[str] = None
    strength: Optional[str] = None
    # Enervara's MedicationForm/MedicationRoute enums (MEDICATION_FORMS /
    # MEDICATION_ROUTES in domain/types.ts). Left null rather than guessed
    # when we don't have a confident mapping — Enervara's own normaliser
    # (`enumOr`) already defaults a null/unrecognized value to "OTHER" /
    # "ORAL" on its side, so there is no need (and no safety benefit) to
    # guess here.
    form: Optional[str] = None
    route: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    timing: Optional[str] = None
    duration_text: Optional[str] = None
    duration_days: Optional[int] = None
    instructions: Optional[str] = None
    common_use: Optional[str] = None
    confidence: Optional[float] = Field(default=None, description="0-1 (Enervara's scale, not this API's usual 0-100).")


class PrescriptionMetadataOut(_CamelModel):
    prescriber_name: Optional[str] = None
    prescriber_registration: Optional[str] = None
    clinic_name: Optional[str] = None
    patient_name: Optional[str] = None
    indication_notes: Optional[str] = None
    notes: Optional[str] = None


class PrescriptionExtractionOut(_CamelModel):
    prescribed_date: Optional[str] = None
    metadata: PrescriptionMetadataOut
    medications: List[MedicationOut] = Field(default_factory=list)
    # Left unset deliberately (see build_extraction) -- Enervara defaults a
    # missing slug to "general-medicine" itself; this service has no
    # capability to determine a speciality and won't pretend to.
    suggested_speciality_slug: Optional[str] = None
    suggested_speciality_reason: Optional[str] = None
