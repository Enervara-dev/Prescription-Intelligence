"""
api/v1/endpoints/prescriptions.py
-----------------------------------
POST /process  — upload a prescription image/PDF, get back structured
                  patient info + medicine data.
POST /extract  — service-to-service adapter for Enervara's
                  HttpPrescriptionProcessingService: accepts Enervara's JSON
                  contract (a document_url, not file bytes) and returns
                  Enervara's PrescriptionExtraction shape.

Both now use Gemini (app/services/gemini_extraction.py) directly for image
-> structured-data extraction, with no Postgres-catalog cross-reference or
fuzzy matching involved — per explicit instruction, the previous Vision-OCR
+ medicine_resolver pipeline is no longer called from either endpoint. That
code (app/services/ocr_service.py, app/services/extraction/__init__.py,
app/services/medicine_resolver.py, app/services/enervara_extraction_adapter.py)
is left in place, fully tested, and importable if this needs to be reverted.
"""

import base64
import io
import logging
import secrets
import time

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from PIL import Image

from app.core.config import settings
from app.schemas.extract import ExtractRequest, PrescriptionExtractionOut
from app.schemas.prescription import PrescriptionProcessResponse
from app.services.gemini_extraction import (
    GeminiProviderError,
    extract_prescription as gemini_extract_prescription,
    to_extraction_out,
    to_patient_info_text,
    to_process_medicines,
)
from app.utils import download_document, file_bytes_to_pil_images, validate_file_size, validate_file_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])


def _pil_to_base64_jpeg(img: Image.Image, max_size=(900, 1200)) -> str:
    """Generate a lightweight JPEG data URL for the client-side image preview."""
    thumb = img.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def verify_service_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Service-to-service auth for POST /extract ONLY — does not gate /process
    or any other existing endpoint. Mirrors Enervara's own convention for
    this kind of upstream (an X-API-Key header, attached by the caller only
    when configured). If RX_PROCESSING_API_KEY isn't set on this side, the
    endpoint stays open (local dev needs no setup) — see the startup warning
    in app/core/config.py that makes that visible rather than silent.
    """
    expected = settings.RX_PROCESSING_API_KEY
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


@router.post(
    "/process",
    response_model=PrescriptionProcessResponse,
    summary="Extract structured data from a prescription image/PDF",
)
async def process_prescription(
    file: UploadFile = File(...),
) -> PrescriptionProcessResponse:
    """
    Upload a prescription image (JPEG/PNG/WEBP/BMP/TIFF) or PDF.

    Sends the image(s) directly to Gemini, which returns structured medicine
    data (name, strength, dosage, frequency, duration, confidence) — no
    Postgres-catalog cross-reference or fuzzy matching involved.
    """
    validate_file_type(file.content_type)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file is empty.")
    validate_file_size(image_bytes)

    pil_images = file_bytes_to_pil_images(image_bytes, file.content_type)

    start = time.perf_counter()
    try:
        result = gemini_extract_prescription(pil_images)
        medicines = to_process_medicines(result)
        patient_info = to_patient_info_text(result)

        image_preview = _pil_to_base64_jpeg(pil_images[0]) if pil_images else None

        elapsed = round(time.perf_counter() - start, 2)

        return PrescriptionProcessResponse(
            success=True,
            filename=file.filename,
            patient_info=patient_info,
            medicines=medicines,
            image_preview=image_preview,
            processing_time_sec=elapsed,
            debug={
                "full_text": result.full_text,
                "medicine_text": result.full_text,
                "word_count": len(result.full_text.split()),
            },
        )

    except HTTPException:
        raise
    except GeminiProviderError as exc:
        # Distinct from every other failure mode: the extraction provider
        # itself (Gemini) could not be reached/used after retries, as
        # opposed to a successful call finding nothing (not an error —
        # returns 200 with an empty medicines list) or some other bug in
        # this service's own processing. 503 signals "try again shortly" —
        # a 500 doesn't tell the caller whether retrying could help.
        logger.error(f"Extraction provider unavailable: {exc}", exc_info=True)
        raise HTTPException(503, "Prescription extraction service is temporarily unavailable. Please try again shortly.")
    except Exception as exc:
        # Full detail (which may reference internal hosts, provider error
        # bodies, etc.) goes to the server log only — never back to the
        # caller, who gets a generic message instead.
        logger.error(f"Processing failed: {exc}", exc_info=True)
        raise HTTPException(500, "Processing failed. Please try again.")


@router.post(
    "/extract",
    response_model=PrescriptionExtractionOut,
    response_model_by_alias=True,
    summary="Service-to-service adapter for Enervara's HttpPrescriptionProcessingService",
    dependencies=[Depends(verify_service_api_key)],
)
async def extract_prescription(
    payload: ExtractRequest,
) -> PrescriptionExtractionOut:
    """
    Enervara's RX_PROCESSING_PROVIDER=http contract: POST {RX_PROCESSING_API_URL}/extract
    with a JSON body naming a short-lived document_url (not file bytes).
    Sends the downloaded image(s) directly to Gemini — no Postgres-catalog
    cross-reference or fuzzy matching involved. Any failure here is
    surfaced as a non-2xx response; Enervara's own client treats any
    non-2xx uniformly as UPSTREAM_ERROR and never surfaces our response
    body to its end users, so no failure mode needs special-casing beyond
    returning the right status code.
    """
    logger.info(
        "[extract] request_id=%s prescription_id=%s mime_type=%s",
        payload.request_id, payload.prescription_id, payload.mime_type,
    )

    validate_file_type(payload.mime_type)
    document_bytes = download_document(payload.document_url, timeout=settings.EXTRACT_DOWNLOAD_TIMEOUT_SECONDS)
    if not document_bytes:
        raise HTTPException(422, "Downloaded document is empty.")
    validate_file_size(document_bytes)

    pil_images = file_bytes_to_pil_images(document_bytes, payload.mime_type)

    start = time.perf_counter()
    try:
        result = gemini_extract_prescription(pil_images)
        extraction = to_extraction_out(result)

        logger.info(
            "[extract] request_id=%s medications_found=%d elapsed_ms=%.1f",
            payload.request_id, len(extraction.medications), (time.perf_counter() - start) * 1000,
        )
        return extraction

    except HTTPException:
        raise
    except GeminiProviderError as exc:
        # See the matching handler in process_prescription() above — distinct
        # from a generic processing failure. Enervara's own client treats
        # any non-2xx uniformly (see httpProcessingService.ts), so this
        # doesn't change what Enervara's caller sees; it makes our own logs
        # and any other direct caller of this endpoint able to tell
        # "extraction provider down" apart from "something else broke."
        logger.error("[extract] request_id=%s extraction provider unavailable: %s", payload.request_id, exc, exc_info=True)
        raise HTTPException(503, "Prescription extraction service is temporarily unavailable. Please try again shortly.")
    except Exception as exc:
        logger.error("[extract] request_id=%s processing failed: %s", payload.request_id, exc, exc_info=True)
        raise HTTPException(502, "Processing failed.")
