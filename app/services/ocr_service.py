"""
ocr_service.py
--------------
Google Cloud Vision-based text extraction.
Categorises raw extracted words into patient_info vs medicine text sections.
Preserves line-level structure AND per-word geometry/confidence for accurate
medicine parsing -- see app/services/ocr_geometry.py for the resolution-
adaptive line-reconstruction algorithm. Plain-text fields (`patient_info`,
`medicine_text`, `full_text`) are still produced for backward compatibility
with every existing caller/test; `medicine_lines`/`patient_lines` additionally
expose the structured `OCRLine` objects (words, bounding box, confidence)
for callers that can use them (see app/services/extraction/__init__.py).
"""

import io
import os
import re
from typing import Any, Tuple
from PIL import Image
from dotenv import load_dotenv

from app.services.ocr_geometry import OCRLine, OCRWord, group_words_into_lines

# Load environment variables from the project root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

try:
    from google.cloud import vision
    from google.api_core.client_options import ClientOptions
except ImportError:
    vision = None
    ClientOptions = None

CONFIDENCE_THRESHOLD = 0.10

def _get_api_key() -> str:
    """Load Vision API key from environment. Raises clearly if missing."""
    key = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GOOGLE_VISION_API_KEY is not set. "
            "Add it to the project root .env as: GOOGLE_VISION_API_KEY=your_key_here"
        )
    return key

def _extract_page_words(image: Image.Image) -> list[OCRWord]:
    """
    Single-pass OCR on one page using Google Cloud Vision API. Returns the
    full per-word bounding box (not just its top-left corner) and its own
    confidence -- both required for resolution-adaptive line reconstruction
    (app/services/ocr_geometry.py) and for carrying OCR certainty downstream
    instead of discarding it at ingestion, which the previous implementation
    did (confidence was captured here and then never read again).
    """
    if vision is None:
        raise ImportError("google-cloud-vision is not installed. Run: pip install google-cloud-vision")

    # Initialize the client using the API key from .env
    client = vision.ImageAnnotatorClient(
        client_options={"api_key": _get_api_key()}  # type: ignore
    )

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    vision_image = vision.Image(content=content)

    # Restrict OCR to English only to prevent Cyrillic/non-English hallucinations on messy handwriting
    image_context = vision.ImageContext(language_hints=['en'])

    # Use document_text_detection for dense text / handwriting
    response = client.document_text_detection(
        image=vision_image,
        image_context=image_context
    )

    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors"
        )

    words: list[OCRWord] = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])

                    if not word_text:
                        continue

                    # Noise filter: skip single-character tokens (except medical markers like T, C)
                    if len(word_text) <= 1 and word_text.upper() not in ('T', 'C'):
                        continue
                    # Skip pure punctuation
                    if re.match(r'^[^\w\s]+$', word_text):
                        continue

                    # Full bounding box, not just the top-left corner -- width
                    # and height are what makes line grouping resolution-
                    # adaptive (see ocr_geometry.group_words_into_lines).
                    vertices = word.bounding_box.vertices
                    xs = [v.x for v in vertices] or [0]
                    ys = [v.y for v in vertices] or [0]
                    x_left, y_top = min(xs), min(ys)
                    width = max(xs) - x_left
                    height = max(ys) - y_top

                    confidence = word.confidence

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    words.append(OCRWord(
                        text=word_text,
                        confidence=round(float(confidence), 4),
                        x=x_left,
                        y=y_top,
                        width=width,
                        height=height,
                    ))

    return words


def _auto_orient_page(image: Image.Image) -> Tuple[Image.Image, list[OCRWord]]:
    """
    Extract page words. Google Cloud Vision automatically handles orientation,
    so we just pass the original image.
    """
    page_words = _extract_page_words(image)
    return image, page_words


# ---------------------------------------------------------------------------
# Categorisation & Line Grouping
# ---------------------------------------------------------------------------

_AD_FRAGMENTS = [
    "horlicks", "apollo247", "apolloz4z", "exclusiveoffer",
    "1800-572", "orderwia", "whatsepp",
    "upto 70% off", "upto 25% off", "free delivery",
    "flat 15% cashback", "free lab test", "buy smart health",
    "get free 1 year", "shop now", "order now", "book now",
    "home delivery", "home sample", "circle plan",
    "page 1 of", "page 2 of",
]


def _is_ad(text: str) -> bool:
    low = text.lower()
    return any(frag in low for frag in _AD_FRAGMENTS)


def _categorize(lines: list[OCRLine]) -> tuple[list[OCRLine], list[OCRLine]]:
    """
    Categorise reconstructed lines into patient_info and medicine sections.
    Operates on structured OCRLine objects (not flattened strings) so the
    caller can still reach each line's words/bounding-box/confidence after
    categorisation -- only the classification decision itself uses the
    line's plain text.
    """
    patient_lines: list[OCRLine] = []
    medicine_lines: list[OCRLine] = []
    in_medicine_section = False

    for line in lines:
        if _is_ad(line.text):
            continue

        low = line.text.lower()

        # Triggers for medicine section
        if (any(k in low for k in [" rx", "rx ", "r/", "treatment", "medicine", "medication", "adv:", "advise", "adv"])
                or re.search(r'\b(tab|cap|syp|inj|ointment)\b', low)
                or re.search(r'\b\d+\s*(mg|ml|mcg)\b', low)):
            in_medicine_section = True

        if in_medicine_section:
            medicine_lines.append(line)
        else:
            patient_lines.append(line)

    if not medicine_lines:
        medicine_lines = lines

    return patient_lines, medicine_lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_images(images: list[Image.Image]) -> dict[str, Any]:
    """
    Run OCR on one or more images. Returns categorised text (backward-
    compatible plain-text fields, unchanged shape for every existing
    caller/test) AND the structured OCRLine objects behind them
    (`medicine_ocr_lines` / `patient_ocr_lines`) for callers that can use
    per-word geometry and confidence -- see
    app/services/extraction/__init__.py, which uses these when present and
    falls back to the plain-text fields otherwise.
    """
    all_words: list[OCRWord] = []
    oriented_images = []

    for img in images:
        best_img, page_words = _auto_orient_page(img)
        oriented_images.append(best_img)
        all_words.extend(page_words)

    all_lines = group_words_into_lines(all_words)
    patient_lines, medicine_lines = _categorize(all_lines)

    return {
        "patient_info": "\n".join(l.text for l in patient_lines),
        "medicine_text": "\n".join(l.text for l in medicine_lines),
        "full_text": "\n".join(l.text for l in all_lines),
        "word_count": len(all_words),
        "oriented_images": oriented_images,
        # Structured, geometry/confidence-preserving representation of the
        # same lines above -- additive; nothing existing reads these keys.
        "medicine_ocr_lines": medicine_lines,
        "patient_ocr_lines": patient_lines,
        "all_ocr_lines": all_lines,
    }
