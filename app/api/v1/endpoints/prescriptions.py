"""
api/v1/endpoints/prescriptions.py
-----------------------------------
POST /process  — the existing multipart OCR pipeline endpoint: upload a
                  prescription image/PDF, get back structured patient info +
                  medicine data. UNCHANGED by the addition below.
POST /extract  — service-to-service adapter for Enervara's
                  HttpPrescriptionProcessingService: accepts Enervara's JSON
                  contract (a document_url, not file bytes) and returns
                  Enervara's PrescriptionExtraction shape. Reuses the exact
                  same OCR + parse_medicines() pipeline as /process — see
                  app/services/enervara_extraction_adapter.py for the (only
                  new) output-shape transform.
"""

import base64
import io
import logging
import secrets
import time

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.extract import ExtractRequest, PrescriptionExtractionOut
from app.schemas.prescription import PrescriptionProcessResponse
from app.services.enervara_extraction_adapter import build_extraction
from app.services.extraction import parse_medicines
from app.services.ocr_service import OCRProviderError, extract_text_from_images
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


def _clean_patient_info(ocr_result: dict, medicines: list[dict]) -> str:
    """
    Drop lines from the OCR-detected patient-info block that actually turned
    out to be medicine lines (already captured in `medicines`), so they
    aren't duplicated. Shared by /process and /extract — was inline in
    /process only before /extract needed the identical logic.
    """
    raw_med_lines = {m.get("raw_line", "").strip() for m in medicines if m.get("raw_line")}
    clean_lines = []
    for line in ocr_result.get("patient_info", "").split("\n"):
        line_str = line.strip()
        if not line_str or line_str in raw_med_lines:
            continue
        if any(m["name"].split()[0].lower() in line_str.lower() for m in medicines):
            continue
        clean_lines.append(line_str)
    return "\n".join(clean_lines) if clean_lines else "Patient details not detected clearly."


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
    db: Session = Depends(get_db),
) -> PrescriptionProcessResponse:
    """
    Upload a prescription image (JPEG/PNG/WEBP/BMP/TIFF) or PDF.

    Runs OCR (Google Cloud Vision), classifies text into patient info vs.
    medicine lines, and matches medicine names against the Postgres-backed
    catalog (indexed pg_trgm search + OCR-confusable correction) to return
    structured medicine entries (name, strength, dosage form, match type,
    dosage/frequency/duration, confidence).
    """
    validate_file_type(file.content_type)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file is empty.")
    validate_file_size(image_bytes)

    pil_images = file_bytes_to_pil_images(image_bytes, file.content_type)

    start = time.perf_counter()
    try:
        # 1. OCR + Auto-Orientation + Categorisation
        ocr_result = extract_text_from_images(pil_images)

        # 2. Parse medicines from medicine text, fallback to full text if
        # needed. `medicine_ocr_lines` (when present) carries per-word
        # geometry/confidence through to the resolver -- see
        # app/services/extraction/__init__.py and ocr_geometry.py.
        medicines = parse_medicines(
            ocr_result["medicine_text"], db, ocr_lines=ocr_result.get("medicine_ocr_lines")
        )
        if not medicines and ocr_result["full_text"]:
            medicines = parse_medicines(ocr_result["full_text"], db, ocr_lines=ocr_result.get("all_ocr_lines"))

        final_patient_info = _clean_patient_info(ocr_result, medicines)

        # 3. Create visual preview from first properly oriented page
        image_preview = None
        if ocr_result.get("oriented_images"):
            image_preview = _pil_to_base64_jpeg(ocr_result["oriented_images"][0])

        elapsed = round(time.perf_counter() - start, 2)

        return PrescriptionProcessResponse(
            success=True,
            filename=file.filename,
            patient_info=final_patient_info,
            medicines=medicines,
            image_preview=image_preview,
            processing_time_sec=elapsed,
            debug={
                "full_text": ocr_result["full_text"],
                "medicine_text": ocr_result["medicine_text"],
                "word_count": ocr_result["word_count"],
            },
        )

    except HTTPException:
        raise
    except OCRProviderError as exc:
        # Distinct from every other failure mode: the OCR provider itself
        # (Google Vision) could not be reached/used after retries, as
        # opposed to a successful OCR call finding nothing (not an error —
        # returns 200 with an empty medicines list) or some other bug in
        # this service's own processing. 503 signals "try again shortly" —
        # a 500 doesn't tell the caller whether retrying could help.
        logger.error(f"OCR provider unavailable: {exc}", exc_info=True)
        raise HTTPException(503, "Prescription OCR service is temporarily unavailable. Please try again shortly.")
    except Exception as exc:
        # Full detail (which may reference internal hosts, DB/Vision error
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
    db: Session = Depends(get_db),
) -> PrescriptionExtractionOut:
    """
    Enervara's RX_PROCESSING_PROVIDER=http contract: POST {RX_PROCESSING_API_URL}/extract
    with a JSON body naming a short-lived document_url (not file bytes).
    Reuses the exact same OCR + parse_medicines() pipeline /process uses —
    only the input (download instead of multipart) and output shape
    (PrescriptionExtraction instead of PrescriptionProcessResponse) differ.
    Any failure here is surfaced as a non-2xx response; Enervara's own
    client treats any non-2xx uniformly as UPSTREAM_ERROR and never
    surfaces our response body to its end users, so no failure mode needs
    special-casing beyond returning the right status code.
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
        ocr_result = extract_text_from_images(pil_images)

        medicines = parse_medicines(
            ocr_result["medicine_text"], db, ocr_lines=ocr_result.get("medicine_ocr_lines")
        )
        if not medicines and ocr_result["full_text"]:
            medicines = parse_medicines(ocr_result["full_text"], db, ocr_lines=ocr_result.get("all_ocr_lines"))

        patient_info = _clean_patient_info(ocr_result, medicines)
        extraction = build_extraction(patient_info, medicines)

        logger.info(
            "[extract] request_id=%s medications_found=%d elapsed_ms=%.1f",
            payload.request_id, len(extraction.medications), (time.perf_counter() - start) * 1000,
        )
        return extraction

    except HTTPException:
        raise
    except OCRProviderError as exc:
        # See the matching handler in process_prescription() above — distinct
        # from a generic processing failure. Enervara's own client treats
        # any non-2xx uniformly (see httpProcessingService.ts), so this
        # doesn't change what Enervara's caller sees; it makes our own logs
        # and any other direct caller of this endpoint able to tell "OCR
        # provider down" apart from "something else broke."
        logger.error("[extract] request_id=%s OCR provider unavailable: %s", payload.request_id, exc, exc_info=True)
        raise HTTPException(503, "Prescription OCR service is temporarily unavailable. Please try again shortly.")
    except Exception as exc:
        logger.error("[extract] request_id=%s processing failed: %s", payload.request_id, exc, exc_info=True)
        raise HTTPException(502, "Processing failed.")
