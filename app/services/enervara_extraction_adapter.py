"""
services/enervara_extraction_adapter.py
------------------------------------------
Pure transform: this service's own medicine-list shape (the same dicts
parse_medicines() has always returned) -> Enervara's PrescriptionExtraction
contract. No OCR, no matching, no I/O here — those are reused as-is from
ocr_service.py / extraction/__init__.py by the /extract endpoint; this
module ONLY reshapes their already-computed output.

Never invents data: a field Enervara wants that we genuinely don't extract
(prescribedDate, timing, instructions, commonUse, structured prescriber/
patient metadata) is sent as null, not guessed. The one exception is
`metadata.notes`, which carries our OCR's own free-text patient/doctor-info
block verbatim — real extracted text, not a fabrication, just unstructured.
"""

import re
from typing import Any, Optional

from app.schemas.extract import MedicationOut, PrescriptionExtractionOut, PrescriptionMetadataOut

# Only the forms/routes we can map with real confidence. Everything else is
# left null -- Enervara's own `enumOr()` already defaults an unrecognized/
# missing value to "OTHER" (form) or "ORAL" (route), so there is no safety
# reason to guess a mapping here (e.g. "suspension" -> "SYRUP" would be an
# approximation, not a fact we actually know).
_FORM_MAP = {
    "tablet": "TABLET",
    "capsule": "CAPSULE",
    "syrup": "SYRUP",
    "injection": "INJECTION",
    "drops": "DROPS",
}
_ROUTE_MAP = {
    "oral": "ORAL",
    "topical": "TOPICAL",
    "injection": "INJECTION",
    "inhaled": "INHALED",
    "ophthalmic": "OPHTHALMIC",
    "nasal": "NASAL",
}

_NO_PATIENT_INFO_TEXT = "Patient details not detected clearly."
_DURATION_DAYS_RE = re.compile(r"(\d+)\s*day", re.IGNORECASE)


def _clean(value: Optional[str]) -> Optional[str]:
    """Our regex extractors use the literal string "N/A" for "nothing found"
    — Enervara's contract represents that as null, not the string "N/A"."""
    if not value or value == "N/A":
        return None
    return value


def _parse_duration_days(duration_text: Optional[str]) -> Optional[int]:
    """Reformats information we already extracted ("5 days" -> 5); never
    invents a duration that wasn't in the text."""
    if not duration_text:
        return None
    m = _DURATION_DAYS_RE.search(duration_text)
    return int(m.group(1)) if m else None


def _medication_out(med: dict[str, Any]) -> MedicationOut:
    dosage = _clean(med.get("dosage"))
    duration_text = _clean(med.get("duration"))
    # Prefer the catalog's own structured strength (e.g. "650mg" on the
    # matched product) over the OCR-line-regex strength guess; fall back to
    # the latter only when the catalog row has none (e.g. a generic-only
    # entry) -- still real, extracted-from-the-line text, not invented.
    # `strength_text` (NOT `dosage`) is the strength-shaped regex detector —
    # they used to be the same field, which was a confirmed bug (a genuine
    # administration-dosage phrase like "1 tablet" and a strength phrase
    # like "500mg" are different concepts; see extraction/__init__.py).
    strength = med.get("strength") or _clean(med.get("strength_text"))

    confidence = med.get("overall_confidence", med.get("confidence"))
    # This API's own scale is 0-100 everywhere else; Enervara's is 0-1
    # (MEDICATION_REVIEW_CONFIDENCE = 0.7 in its domain/types.ts). Sending
    # our raw 0-100 value would have Enervara's own clamp
    # (Math.min(1, Math.max(0, n))) flatten every real match to 1.0,
    # destroying the signal entirely -- this conversion is required, not
    # cosmetic. Prefers `overall_confidence` (name-match fused with OCR read
    # certainty, when available) over plain `confidence` so Enervara's own
    # review-routing (MEDICATION_REVIEW_CONFIDENCE) actually reflects OCR
    # uncertainty instead of a clean-looking match hiding a poor OCR read;
    # identical to `confidence` whenever OCR-confidence context is absent.
    confidence_0_1 = round(confidence / 100.0, 3) if confidence is not None else None

    return MedicationOut(
        name=med["name"],
        generic_name=med.get("generic_name"),
        strength=strength,
        form=_FORM_MAP.get((med.get("dosage_form") or "").lower()),
        route=_ROUTE_MAP.get((med.get("route") or "").lower()),
        dosage=dosage,
        frequency=_clean(med.get("frequency")),
        timing=None,  # not a signal parse_medicines distinguishes from frequency
        duration_text=duration_text,
        duration_days=_parse_duration_days(duration_text),
        instructions=None,
        common_use=None,
        confidence=confidence_0_1,
    )


def build_extraction(patient_info: str, medicines: list[dict[str, Any]]) -> PrescriptionExtractionOut:
    """
    `patient_info` and `medicines` are exactly what ocr_service.extract_text_from_images()
    / extraction.parse_medicines() already produce for the existing
    POST /prescriptions/process endpoint -- this function is the ONLY new
    logic; everything upstream of it is reused verbatim.
    """
    notes = patient_info.strip() if patient_info and patient_info.strip() != _NO_PATIENT_INFO_TEXT else None

    return PrescriptionExtractionOut(
        prescribed_date=None,
        metadata=PrescriptionMetadataOut(notes=notes),
        medications=[_medication_out(m) for m in medicines],
        suggested_speciality_slug=None,
        suggested_speciality_reason=None,
    )
