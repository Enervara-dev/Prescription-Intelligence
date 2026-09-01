"""
utils.py
--------
Image validation and conversion helpers.
"""

from PIL import Image
from fastapi import HTTPException
import io

# Allowed MIME types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
    "application/pdf"
}

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
