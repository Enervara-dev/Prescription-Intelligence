"""
utils.py
--------
Image validation and conversion helpers.
"""

from PIL import Image
from fastapi import HTTPException
import io
import logging

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

# Allowed MIME types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
    "application/pdf"
}


def validate_file_size(file_bytes: bytes) -> None:
    """Raises HTTP 413 if the upload exceeds MAX_UPLOAD_SIZE_MB."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: max {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

def validate_file_type(content_type: str) -> None:
    """
    Validate that the uploaded file is an accepted image or PDF type.
    Raises HTTP 400 if the content type is not allowed.
    """
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed types: JPEG, PNG, WEBP, BMP, TIFF, PDF."
            ),
        )


def download_document(url: str, timeout: int) -> bytes:
    """
    Fetch a document from a (presumed short-lived, presigned) URL — used by
    the /extract adapter, which receives a document_url rather than file
    bytes. Streams with a byte cap enforced DURING download (not just
    checked after the fact on a fully-buffered response), so a
    pathological/misbehaving URL can't exhaust memory before
    validate_file_size() would ever get a chance to reject it. Raises
    HTTPException (502) on any network failure or over-limit response —
    callers don't need their own try/except for this.
    """
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    try:
        with requests.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Document too large: max {settings.MAX_UPLOAD_SIZE_MB}MB.",
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except HTTPException:
        raise
    except requests.RequestException as exc:
        logger.error("[download_document] failed to fetch document: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Could not download the document.")


def file_bytes_to_pil_images(file_bytes: bytes, content_type: str) -> list[Image.Image]:
    """
    Convert raw file bytes into a list of PIL Image objects.
    If it's an image, returns a list with 1 image.
    If it's a PDF, uses PyMuPDF (fitz) to render each page to an image.
    Raises HTTP 422 if decoding fails.
    """
    if content_type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.page_count > settings.MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF has too many pages: max {settings.MAX_PDF_PAGES}.",
                )
            images = []
            # Render each page to an image (scaling up for better OCR resolution)
            zoom_matrix = fitz.Matrix(2.0, 2.0)
            for page in doc:
                pix = page.get_pixmap(matrix=zoom_matrix)
                # Convert fitz pixmap to PIL image
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                images.append(img)
            return images
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not decode PDF: {str(exc)}",
            )
    else:
        # It's a standard image
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()
            image = Image.open(io.BytesIO(file_bytes))
            return [image]
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not decode image: {str(exc)}",
            )
