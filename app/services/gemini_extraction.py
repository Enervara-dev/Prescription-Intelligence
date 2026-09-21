"""
services/gemini_extraction.py
---------------------------------
LLM-based prescription extraction using Google's Gemini API. Replaces the
Vision-OCR + Postgres-catalog-matching pipeline in the live request path
(see app/api/v1/endpoints/prescriptions.py) -- Gemini reads the prescription
image(s) directly and returns already-structured medicine data. There is no
catalog cross-reference, fuzzy matching, or confidence-gating against a
known-medicines table involved at all: Gemini's own extraction is the final
answer, per explicit instruction.

The catalog-matching code this replaces (app/services/medicine_resolver.py,
app/services/extraction/__init__.py, app/services/ocr_service.py,
app/services/enervara_extraction_adapter.py) is left in place, fully tested,
and importable -- just no longer called from the request path. See those
modules if this needs to be reverted or run alongside Gemini later.

Output is mapped directly into this service's EXISTING response schemas
(app/schemas/extract.py, app/schemas/prescription.py), so neither Enervara's
/extract contract nor this service's own /process response shape changes --
only how the data inside them gets produced.
"""

import io
import logging
import re
import time
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.extract import MedicationOut, PrescriptionExtractionOut, PrescriptionMetadataOut

logger = logging.getLogger(__name__)

# Bounded retry for the Gemini call itself -- covers a transient network
# blip or a momentary provider hiccup, not a permanent misconfiguration.
# Worst-case latency budget is deliberately kept well under Enervara's own
# hard deadline for the whole /extract call (RX_PROCESSING_TIMEOUT_MS,
# default 90s in prod_app/backend/src/prescriptions/processing/
# httpProcessingService.ts, via AbortSignal.timeout -- a total-elapsed-time
# deadline, not an idle-connection timeout, so it fires regardless of
# whether the response arrives all at once or streamed). A retry does not
# reduce latency for a call that's merely slow rather than failing outright
# -- it can only make total latency worse -- so this stays at 2 attempts,
# not more: 2 x _GEMINI_REQUEST_TIMEOUT_MS + backoff is the real worst case.
_MAX_GEMINI_ATTEMPTS = 2
_GEMINI_RETRY_BACKOFF_SECONDS = (1.0,)  # delay before attempt 2
# Per-attempt timeout: without this, a single slow/hanging call has no
# bound of its own and can consume the entire retry budget by itself
# (verified gap -- previously unset). 30s is generous for a single
# prescription image at the resolution cap below, while keeping
# 2 x 30s + 1s backoff = 61s comfortably under the 90s external deadline.
_GEMINI_REQUEST_TIMEOUT_MS = 30_000
# Caps worst-case generation time/cost and gives a second, independent
# bound alongside the request timeout above. Generous enough for a page
# with many medicines plus the full_text transcription; not unlimited.
_GEMINI_MAX_OUTPUT_TOKENS = 8192
# Long-edge cap before sending to Gemini. A modern phone photo is often
# 3000-4000px+ on the long edge (several MB) -- full resolution buys
# nothing for OCR-quality text legibility beyond this, but directly adds to
# upload time and Gemini's own processing time. 2048px keeps text easily
# legible (well above what a genuinely low-quality/blurry source needs)
# while cutting typical upload size substantially.
_GEMINI_MAX_IMAGE_DIMENSION = 2048


class GeminiProviderError(Exception):
    """
    Raised when Gemini could not be reached/used after retries were
    exhausted, or returned a response that couldn't be parsed as valid
    structured output. Distinct from:
      - a successful call that simply found no medicines on the page (not
        an error -- an empty `medications` list), and
      - a permanent configuration problem (missing API key), which is
        never retried and raises EnvironmentError immediately.
    Callers (see app/api/v1/endpoints/prescriptions.py) catch this
    specifically to return a distinct "extraction service unavailable, try
    again" response instead of a generic failure message -- mirrors the
    OCRProviderError pattern this replaces in the request path (see
    app/services/ocr_service.py, kept for reference/rollback).
    """


# ---------------------------------------------------------------------------
# Gemini-facing structured output schema. Deliberately a separate set of
# models from schemas/extract.py's wire contract (MedicationOut etc.) -- the
# prompt/response shape should be free to evolve independently of the wire
# contract's stability guarantees; _to_extraction_out()/_to_process_medicines()
# below do the mapping between the two.
# ---------------------------------------------------------------------------

class _GeminiMedication(BaseModel):
    name: str = Field(description="The medicine's name exactly as printed or written on the label.")
    generic_name: Optional[str] = Field(
        default=None,
        description="Generic/INN name, ONLY if explicitly printed (e.g. on a composition line) -- never guessed from the brand name.",
    )
    strength: Optional[str] = Field(default=None, description="e.g. '650mg', '500mg/5ml' -- exactly as stated.")
    form: Optional[str] = Field(
        default=None,
        description="One of TABLET, CAPSULE, SYRUP, INJECTION, DROPS, CREAM, OINTMENT, GEL, SPRAY, SUSPENSION, OTHER -- only if clearly stated or unambiguous from context (e.g. 'TAB.' prefix).",
    )
    route: Optional[str] = Field(
        default=None,
        description="One of ORAL, TOPICAL, INJECTION, INHALED, OPHTHALMIC, NASAL, OTHER -- only if clearly stated or unambiguous.",
    )
    dosage: Optional[str] = Field(
        default=None,
        description="How much to take PER ADMINISTRATION, e.g. '1 tablet'. This is NOT a total quantity to dispense -- a 'Total: 8 Tab' style note is a total, not a dosage; leave this null if only a total is stated.",
    )
    frequency: Optional[str] = Field(
        default=None,
        description=(
            "How often, described by WHICH TIMES OF DAY carry a nonzero dose -- e.g. "
            "'Twice Daily (Morning, Night)', 'Thrice Daily (Morning, Afternoon, Night)', "
            "'Once Daily (Night)'. For a dash/slash-separated dose pattern (e.g. '1-0-1', "
            "'1-1-1', '0-0-1', '1/2-0-1'), count ONLY the nonzero positions -- '0' means "
            "no dose at that time, it is not a third or fourth dose. See the Indian "
            "prescription conventions section of your instructions for full worked "
            "examples. Never report a meal-timing note (After Food, Before Food, etc.) "
            "as the frequency -- that belongs in `timing`; still report the actual "
            "frequency here even when a timing note is also present."
        ),
    )
    timing: Optional[str] = Field(
        default=None,
        description="Relative to meals, e.g. 'After Food', 'Before Food', 'Empty Stomach', 'Bedtime' -- distinct from frequency (how many times a day) and never a substitute for it.",
    )
    duration_text: Optional[str] = Field(
        default=None,
        description="e.g. '5 days', '2 weeks' -- only from an explicit unit (days/weeks/months) in the text. A bare number with no unit (e.g. a quantity in parentheses) is NOT a duration.",
    )
    duration_days: Optional[int] = Field(default=None, description="duration_text converted to a day count, only when unambiguous.")
    instructions: Optional[str] = Field(default=None, description="Any other explicit instruction specific to this medicine.")
    common_use: Optional[str] = Field(
        default=None,
        description="A short, general, well-known reason this TYPE of medicine is commonly used (e.g. 'commonly used to reduce fever') -- only for genuinely well-established, widely-known medicines; null otherwise. Never a clinical judgement about this specific patient's case.",
    )
    confidence: float = Field(description="Your own confidence, 0.0-1.0, that you read THIS MEDICINE'S NAME correctly from the image -- not how common or plausible the name is.")


class _GeminiMetadata(BaseModel):
    prescriber_name: Optional[str] = Field(
        default=None,
        description="The prescribing doctor's name only, WITHOUT a leading 'Dr.'/'Dr'/'Doctor' title or trailing qualifications like 'M.D.' -- e.g. 'A. Sharma', not 'Dr. A. Sharma, M.D.'.",
    )
    prescriber_registration: Optional[str] = None
    clinic_name: Optional[str] = None
    patient_name: Optional[str] = None
    indication_notes: Optional[str] = None
    notes: Optional[str] = None


class GeminiExtraction(BaseModel):
    prescribed_date: Optional[str] = Field(default=None, description="ISO 8601 date (YYYY-MM-DD) if a prescription date is visible, else null.")
    metadata: _GeminiMetadata = Field(default_factory=_GeminiMetadata)
    medications: List[_GeminiMedication] = Field(default_factory=list)
    full_text: str = Field(description="A plain-text transcription of everything legible on the page(s) -- for audit/debug purposes, not further parsed.")


_PROMPT = """You are reading a photograph or scan of a medical prescription, most likely from an Indian clinic or hospital. Extract exactly what is printed or written -- never invent, guess, or infer information that is not actually present on the page.

General rules:
- If a field is not clearly legible or not present, leave it null. Do not guess a plausible-looking value, and do not substitute a different, more "recognizable" medicine name for what is actually printed.
- "duration_text" must come from an explicit unit (days/weeks/months) in the text. Do not treat a bare number (e.g. a quantity written in parentheses with no unit) as a duration.
- List every distinct medicine on the page as a separate entry, even across multiple pages or a multi-page document, even if handwriting or print quality makes you uncertain -- report your actual confidence honestly instead of omitting an uncertain entry or inflating its confidence.
- confidence must reflect how sure YOU are that you read the name correctly, not how common or plausible the medicine name is as a real drug.
- Also provide a plain-text transcription of everything legible on the page(s) in `full_text`.

Indian prescription conventions -- these are required, not optional, and are the most common source of misreading. Apply them carefully:

1. Dash/slash-separated dose patterns (e.g. "1-0-1", "1-1-1", "0-0-1", "1/2-0-1") mean
   Morning-Afternoon-Night (occasionally a 4th position for a bedtime dose), where EACH
   NUMBER is the dose taken AT THAT SPECIFIC TIME -- it is a per-slot indicator, not a
   count of doses to add up, and "0" is an explicit "skip this time", not a dose.
   Count ONLY the nonzero positions when describing frequency, and name which times
   those are:
     "1-0-1"     -> one dose morning, none afternoon, one at night
                    = TWICE daily (Morning, Night) -- NOT three times daily.
     "1-1-1"     -> one dose at each of morning/afternoon/night
                    = THREE TIMES daily (Morning, Afternoon, Night).
     "0-0-1"     -> one dose at night only = ONCE daily (Night).
     "1/2-0-1"   -> half a dose in the morning, none in the afternoon, one at night
                    = TWICE daily (Morning, Night), with a half-dose in the morning --
                    mention the half-dose in `dosage` or `instructions`, not just `frequency`.
   A prescription stating separate lines like "1 Morning, 1 Night" (without dashes)
   follows the exact same logic: two stated times = twice daily, report as
   "Twice Daily (Morning, Night)".

2. Common frequency abbreviations: OD = once daily, BD or BID = twice daily,
   TDS or TID = thrice daily, QID = four times daily, HS = at bedtime/night,
   SOS or PRN = when required/as needed, STAT = immediately (one-time), AC = before
   meals, PC = after meals.

3. "AC", "PC", "before food", "after food", "empty stomach", and similar phrases
   describe TIMING RELATIVE TO A MEAL, not how many times a day the medicine is
   taken. Put these in `timing`. Never let a timing note stand in for `frequency` --
   if the text also states or implies an actual frequency (a dose pattern, an OD/BD/
   TDS-style abbreviation, or explicit times of day), report that real frequency in
   `frequency` even when a timing note is also present elsewhere in the same line.

4. "Tab.", "Cap.", "Syp.", "Inj.", "Susp." prefixes immediately before a medicine
   name indicate its dosage FORM (Tablet/Capsule/Syrup/Injection/Suspension) -- read
   them into `form`, not as part of the medicine's name.

5. A "Tot:" or "Total:" note (e.g. "Tot: 8 Tab", "Total: 16 Tab") states the TOTAL
   QUANTITY to dispense for the entire course (roughly frequency x duration), NOT a
   per-administration amount. Never report this number as `dosage` -- leave `dosage`
   null unless a genuine per-dose amount ("1 tablet", "2 tsp") is stated separately
   from any such total.

6. A composition/ingredient line beneath a brand name (e.g. "DOXYLAMINE 10MG +
   PYRIDOXINE 10MG + FOLIC ACID 2.5MG" printed under "TAB. VOMILAST") describes what
   that ONE branded medicine contains -- it is not a list of separate medicines to
   report individually; keep it associated with the brand name above it (e.g. as
   `generic_name` or `instructions` on that one entry), unless the brand name itself
   is illegible, in which case report what you can read honestly rather than
   inventing a brand name to attach it to.
"""


def _require_api_key() -> str:
    key = settings.GEMINI_API_KEY
    if not key:
        raise EnvironmentError(
            "Gemini_API_KEY is not set. Add it to the project root .env as: Gemini_API_KEY=your_key_here"
        )
    return key


def _downscale(img: Image.Image, max_dimension: int = _GEMINI_MAX_IMAGE_DIMENSION) -> Image.Image:
    """Shrinks (never enlarges) an image so its longer edge is at most
    `max_dimension`, preserving aspect ratio. A real, measured latency
    contributor: an unscaled multi-megapixel phone photo adds upload time
    and Gemini-side processing time with no legibility benefit beyond this
    resolution."""
    width, height = img.size
    longest = max(width, height)
    if longest <= max_dimension:
        return img
    scale = max_dimension / longest
    return img.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def _pil_images_to_parts(images: List[Image.Image]) -> List[types.Part]:
    parts = []
    for img in images:
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        rgb = _downscale(rgb)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    return parts


def _call_gemini_with_retry(images: List[Image.Image]) -> GeminiExtraction:
    """
    Calls Gemini with a small bounded retry for transient failures (network
    blip, momentary provider error, or a response that failed to parse as
    the requested structured schema). Raises GeminiProviderError once
    attempts are exhausted -- never surfaced to an external caller (see
    prescriptions.py), which only gets a generic "temporarily unavailable"
    message.
    """
    api_key = _require_api_key()  # permanent config error -- raises immediately, never retried
    parts = _pil_images_to_parts(images)
    parts.append(_PROMPT)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GeminiExtraction,
        max_output_tokens=_GEMINI_MAX_OUTPUT_TOKENS,
        http_options=types.HttpOptions(timeout=_GEMINI_REQUEST_TIMEOUT_MS),
    )

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_GEMINI_ATTEMPTS):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=parts,
                config=config,
            )
            parsed = response.parsed
            if parsed is None or not isinstance(parsed, GeminiExtraction):
                raise ValueError("Gemini response did not include valid structured output")
            return parsed
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: network/auth/parse errors all count
            last_exc = exc
            if attempt < _MAX_GEMINI_ATTEMPTS - 1:
                logger.warning(
                    "[gemini] extraction call failed (attempt %d/%d), retrying: %s",
                    attempt + 1, _MAX_GEMINI_ATTEMPTS, exc,
                )
                time.sleep(_GEMINI_RETRY_BACKOFF_SECONDS[attempt])

    raise GeminiProviderError(
        f"Gemini extraction unavailable after {_MAX_GEMINI_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


def extract_prescription(images: List[Image.Image]) -> GeminiExtraction:
    """
    images: PIL.Image pages already decoded from the uploaded file/PDF (see
    app/utils.py). All pages are sent in one request so Gemini can reason
    about a multi-page prescription holistically, rather than page-by-page.
    """
    return _call_gemini_with_retry(images)


# ---------------------------------------------------------------------------
# Mapping into this service's existing, unchanged response contracts.
# ---------------------------------------------------------------------------

_DURATION_DAYS_RE = re.compile(r"(\d+)\s*day", re.IGNORECASE)


def _derive_duration_days(duration_text: Optional[str], gemini_value: Optional[int]) -> Optional[int]:
    """Prefers a deterministic regex derivation from duration_text (same
    logic the previous adapter used) over trusting Gemini's own arithmetic,
    when duration_text explicitly says "N day(s)" -- falls back to
    Gemini's value for week/month-stated durations this regex doesn't cover."""
    if duration_text:
        m = _DURATION_DAYS_RE.search(duration_text)
        if m:
            return int(m.group(1))
    return gemini_value


def to_extraction_out(result: GeminiExtraction) -> PrescriptionExtractionOut:
    """Maps to Enervara's /extract contract -- unchanged field names/shape."""
    medications = [
        MedicationOut(
            name=med.name,
            generic_name=med.generic_name,
            strength=med.strength,
            form=med.form.upper() if med.form else None,
            route=med.route.upper() if med.route else None,
            dosage=med.dosage,
            frequency=med.frequency,
            timing=med.timing,
            duration_text=med.duration_text,
            duration_days=_derive_duration_days(med.duration_text, med.duration_days),
            instructions=med.instructions,
            common_use=med.common_use,
            confidence=med.confidence,
        )
        for med in result.medications
    ]
    return PrescriptionExtractionOut(
        prescribed_date=result.prescribed_date,
        metadata=PrescriptionMetadataOut(
            prescriber_name=result.metadata.prescriber_name,
            prescriber_registration=result.metadata.prescriber_registration,
            clinic_name=result.metadata.clinic_name,
            patient_name=result.metadata.patient_name,
            indication_notes=result.metadata.indication_notes,
            notes=result.metadata.notes,
        ),
        medications=medications,
        suggested_speciality_slug=None,
        suggested_speciality_reason=None,
    )


def to_process_medicines(result: GeminiExtraction) -> List[dict]:
    """
    Maps to this service's own /process response shape (schemas/prescription.py
    Medicine model) -- back-compat field names unchanged, but there is no
    catalog match behind any of these anymore: id/source/match_type/
    brand_name/manufacturer reflect that honestly rather than implying a
    catalog cross-reference that no longer happens.
    """
    return [
        {
            "name": med.name,
            "confidence": round(med.confidence * 100.0, 1),
            "dosage": med.dosage or "N/A",
            "frequency": med.frequency or "N/A",
            "duration": med.duration_text or "N/A",
            "raw_line": None,
            "id": None,
            "generic_name": med.generic_name,
            "brand_name": None,
            "strength": med.strength,
            "dosage_form": med.form.lower() if med.form else None,
            "route": med.route.lower() if med.route else None,
            "manufacturer": None,
            "match_type": "LLM_EXTRACTED",
            "source": "gemini",
        }
        for med in result.medications
    ]


_DR_PREFIX_RE = re.compile(r"^\s*dr\.?\s+", re.IGNORECASE)


def to_patient_info_text(result: GeminiExtraction) -> str:
    meta = result.metadata
    # Prompted to return the name without a "Dr." prefix (see
    # _GeminiMetadata.prescriber_name) -- stripped defensively here too, so a
    # prefix Gemini includes anyway never doubles up with the one added below.
    prescriber_name = _DR_PREFIX_RE.sub("", meta.prescriber_name).strip() if meta.prescriber_name else None
    lines = [
        v
        for v in (
            f"Dr. {prescriber_name}" if prescriber_name else None,
            f"Reg No: {meta.prescriber_registration}" if meta.prescriber_registration else None,
            meta.clinic_name,
            f"Patient: {meta.patient_name}" if meta.patient_name else None,
            meta.indication_notes,
            meta.notes,
        )
        if v
    ]
    return "\n".join(lines) if lines else "Patient details not detected clearly."
