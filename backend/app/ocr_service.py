"""
ocr_service.py
--------------
Google Cloud Vision-based text extraction.
Categorises raw extracted words into patient_info vs medicine text sections.
Preserves line-level structure for accurate medicine parsing.
"""

import io
import os
import re
from typing import List, Dict, Any, Tuple
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

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
            "Add it to backend/.env as: GOOGLE_VISION_API_KEY=your_key_here"
        )
    return key

def _extract_page_details(image: Image.Image) -> list[dict]:
    """Single-pass OCR on one page using Google Cloud Vision API. Returns list of {word, confidence, y, x} dicts."""
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

    details = []
    
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
                        
                    # Get bounding box vertices
                    # A word's bounding box has 4 vertices. Let's find top-left to get x, y
                    vertices = word.bounding_box.vertices
                    x_left = min(v.x for v in vertices) if vertices else 0
                    y_top = min(v.y for v in vertices) if vertices else 0
                    
                    confidence = word.confidence
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    details.append({
                        "word": word_text,
                        "confidence": round(float(confidence), 4),
                        "y": y_top,
                        "x": x_left,
                    })

    # Sort reading order: top-to-bottom, left-to-right
    def reading_order(item):
        return (item["y"] // 20, item["x"])

    details.sort(key=reading_order)
    return details


def _auto_orient_page(image: Image.Image) -> Tuple[Image.Image, list[dict]]:
    """
    Extract page details. Google Cloud Vision automatically handles orientation,
    so we just pass the original image.
    """
    page_details = _extract_page_details(image)
    return image, page_details


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


def _details_to_lines(details: list[dict], line_y_threshold: int = 18) -> list[str]:
    """Group token dicts into line strings based on spatial y-coordinates."""
    if not details:
        return []
    sorted_items = sorted(details, key=lambda d: (d.get("y", 0), d.get("x", 0)))
    lines = []
    curr_line = []
    curr_y = None

    for d in sorted_items:
        y = d.get("y", 0)
        if curr_y is None or abs(y - curr_y) <= line_y_threshold:
            curr_line.append(d["word"])
            if curr_y is None:
                curr_y = y
        else:
            if curr_line:
                lines.append(" ".join(curr_line))
            curr_line = [d["word"]]
            curr_y = y

    if curr_line:
        lines.append(" ".join(curr_line))

    return lines


def _categorize(details: list[dict]) -> tuple[str, str]:
    """
    Categorise lines into patient_info and medicine_text.
    """
    lines = _details_to_lines(details)
    patient_lines = []
    medicine_lines = []
    in_medicine_section = False

    for line in lines:
        if _is_ad(line):
            continue

        low = line.lower()

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

    return "\n".join(patient_lines), "\n".join(medicine_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_images(images: list[Image.Image]) -> dict[str, Any]:
    """
    Run OCR on one or more images.
    Returns categorised text and oriented image references.
    """
    all_details = []
    oriented_images = []

    for img in images:
        best_img, page_details = _auto_orient_page(img)
        oriented_images.append(best_img)
        all_details.extend(page_details)

    patient_text, medicine_text = _categorize(all_details)
    all_lines = _details_to_lines(all_details)

    return {
        "patient_info": patient_text,
        "medicine_text": medicine_text,
        "full_text": "\n".join(all_lines),
        "word_count": len(all_details),
        "oriented_images": oriented_images,
    }
