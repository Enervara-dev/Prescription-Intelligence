"""
main.py
-------
FastAPI backend for AI Prescription OCR.
Single endpoint: POST /api/process
"""

import time
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging

from app.utils import validate_file_type, file_bytes_to_pil_images
from app.ocr_service import extract_text_from_images
from app.services.extraction import parse_medicines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Prescription OCR API",
    description="Upload a prescription image/PDF and get structured medicine data.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pil_to_base64_jpeg(img: Image.Image, max_size=(900, 1200)) -> str:
    """Generate lightweight JPEG data URL for frontend image preview."""
    thumb = img.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.post("/api/process")
async def process_prescription(file: UploadFile = File(...)):
    """
    Upload a prescription image or PDF.
    Returns structured medicine data + patient info + image preview.
    """
    validate_file_type(file.content_type)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file is empty.")

    pil_images = file_bytes_to_pil_images(image_bytes, file.content_type)

    start = time.perf_counter()
    try:
        # 1. OCR + Auto-Orientation + Categorisation
        ocr_result = extract_text_from_images(pil_images)

        # 2. Parse medicines from medicine text, fallback to full text if needed
        medicines = parse_medicines(ocr_result["medicine_text"])
        if not medicines and ocr_result["full_text"]:
            medicines = parse_medicines(ocr_result["full_text"])

        # Filter out lines that matched medicines from patient_info
        raw_med_lines = {m.get("raw_line", "").strip() for m in medicines if m.get("raw_line")}
        clean_patient_lines = []
        for line in ocr_result.get("patient_info", "").split("\n"):
            line_str = line.strip()
            if not line_str or line_str in raw_med_lines:
                continue
            if any(m["name"].split()[0].lower() in line_str.lower() for m in medicines):
                continue
            clean_patient_lines.append(line_str)
        
        final_patient_info = "\n".join(clean_patient_lines) if clean_patient_lines else "Patient details not detected clearly."

        # 3. Create visual preview from first properly oriented page
        image_preview = None
        if ocr_result.get("oriented_images"):
            image_preview = _pil_to_base64_jpeg(ocr_result["oriented_images"][0])

        elapsed = round(time.perf_counter() - start, 2)

        return {
            "success": True,
            "filename": file.filename,
            "patient_info": final_patient_info,
            "medicines": medicines,
            "image_preview": image_preview,
            "processing_time_sec": elapsed,
            "debug": {
                "full_text": ocr_result["full_text"],
                "medicine_text": ocr_result["medicine_text"],
                "word_count": ocr_result["word_count"],
            }
        }

    except Exception as exc:
        logger.error(f"Processing failed: {exc}", exc_info=True)
        raise HTTPException(500, f"Processing failed: {str(exc)}")


@app.get("/health")
def health():
    return {"status": "ok"}
