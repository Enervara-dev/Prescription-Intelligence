"""
ocr_service.py
--------------
EasyOCR-based text extraction with multi-pass image enhancement (CLAHE + Adaptive Sharpening).
Includes automatic orientation / rotation detection for sideways photos & PDFs.
Categorises raw extracted words into patient_info vs medicine text sections.
Preserves line-level structure for accurate medicine parsing.
"""

import cv2
import numpy as np
import easyocr
import re
from typing import List, Dict, Any, Tuple
from PIL import Image

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    TrOCRProcessor, VisionEncoderDecoderModel = None, None

# Global EasyOCR reader — lazy loaded
_READER: easyocr.Reader | None = None
_TROCR_PROCESSOR = None
_TROCR_MODEL = None
CONFIDENCE_THRESHOLD = 0.10

def get_reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        print("[OCR] Loading EasyOCR detector model...")
        _READER = easyocr.Reader(['en'], gpu=False)
        print("[OCR] EasyOCR detector ready.")
    return _READER

def get_trocr_model():
    global _TROCR_PROCESSOR, _TROCR_MODEL
    if _TROCR_PROCESSOR is None or _TROCR_MODEL is None:
        if TrOCRProcessor is None:
            raise ImportError("transformers is not installed. Run: pip install transformers torch")
        print("[OCR] Loading TrOCR recognition model (this may take a minute)...")
        _TROCR_PROCESSOR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        print("[OCR] TrOCR recognition model ready.")
    return _TROCR_PROCESSOR, _TROCR_MODEL

# ---------------------------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------------------------

def _to_cv2(image: Image.Image, min_width: int = 1600) -> np.ndarray:
    """Convert PIL image to BGR numpy array and upscale if too small."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if w < min_width:
        scale = min_width / w
        bgr = cv2.resize(bgr, (min_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return bgr

def _preprocess_pass1(bgr: np.ndarray) -> np.ndarray:
    """Pass 1: CLAHE contrast enhancement on grayscale (best for printed & clean text)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return denoised

def _preprocess_pass2(bgr: np.ndarray) -> np.ndarray:
    """Pass 2: Sharpened + CLAHE (best for doctor cursive handwriting)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Unsharp mask
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(sharp)

# ---------------------------------------------------------------------------
# Single-page OCR runner & Dual Pass
# ---------------------------------------------------------------------------

def _run_ocr(reader: easyocr.Reader, processed_img: np.ndarray, bgr_img: np.ndarray) -> list:
    """
    Hybrid Ensemble OCR:
    1. EasyOCR extracts character-accurate detection & raw recognition without dictionary bias.
    2. TrOCR recognizes cursive handwritten tokens from high-contrast enhanced crops with max_new_tokens=25.
    3. Both text streams are merged to maximize medicine extraction accuracy.
    """
    processor, model = get_trocr_model()
    
    try:
        # 1. EasyOCR readtext for bounding boxes + raw character recognition
        easy_results = reader.readtext(processed_img, width_ths=1.0, detail=1)
        
        final_results = []
        for (bbox, easy_text, easy_conf) in easy_results:
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Add padding for clean crop
            pad_x = int((x_max - x_min) * 0.08)
            pad_y = int((y_max - y_min) * 0.15)
            
            x1 = max(0, x_min - pad_x)
            x2 = min(processed_img.shape[1], x_max + pad_x)
            y1 = max(0, y_min - pad_y)
            y2 = min(processed_img.shape[0], y_max + pad_y)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Crop from enhanced processed_img
            crop_slice = processed_img[y1:y2, x1:x2]
            if len(crop_slice.shape) == 2:
                crop_rgb = cv2.cvtColor(crop_slice, cv2.COLOR_GRAY2RGB)
            else:
                crop_rgb = cv2.cvtColor(crop_slice, cv2.COLOR_BGR2RGB)
                
            crop_pil = Image.fromarray(crop_rgb)
            
            # Skip TrOCR generation on pure numbers, dates, codes, or tiny symbols where EasyOCR is already accurate
            is_numeric_or_code = bool(re.match(r'^[\d\s\W\.\-\/:#]+$', easy_text)) or len(easy_text) <= 2
            if easy_conf >= 0.85 and is_numeric_or_code:
                fmt_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                final_results.append((fmt_bbox, easy_text.strip(), float(easy_conf)))
                continue

            # 2. TrOCR Recognition on handwritten / medicine candidates
            pixel_values = processor(crop_pil, return_tensors="pt").pixel_values
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=25,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Format bounding box
            fmt_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            # 3. Combine EasyOCR + TrOCR text into hybrid stream
            easy_text_clean = easy_text.strip()
            trocr_text_clean = trocr_text.strip()
            
            tokens = []
            if easy_text_clean and easy_conf >= 0.20:
                tokens.append(easy_text_clean)
            if trocr_text_clean and trocr_text_clean.lower() != easy_text_clean.lower():
                tokens.append(trocr_text_clean)
                
            combined = " ".join(tokens).strip()
            if combined:
                final_results.append((fmt_bbox, combined, max(float(easy_conf), 0.95)))
                
        return final_results
    except Exception as e:
        print(f"[OCR] Hybrid OCR Pass error: {e}")
        return []


def _merge_results(results_a: list, results_b: list, iou_threshold: float = 0.35) -> list:
    """Merge two OCR passes, deduplicating overlapping bounding boxes."""
    def bbox_to_rect(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return [min(xs), min(ys), max(xs), max(ys)]

    def iou(r1, r2):
        ix1 = max(r1[0], r2[0]); iy1 = max(r1[1], r2[1])
        ix2 = min(r1[2], r2[2]); iy2 = min(r1[3], r2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        a1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
        a2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
        return inter / (a1 + a2 - inter)

    merged = list(results_a)
    rects_a = [bbox_to_rect(r[0]) for r in results_a]

    for item_b in results_b:
        rect_b = bbox_to_rect(item_b[0])
        best_iou, best_idx = 0.0, -1
        for i, rect_a in enumerate(rects_a):
            score = iou(rect_a, rect_b)
            if score > best_iou:
                best_iou, best_idx = score, i
        if best_iou >= iou_threshold:
            if item_b[2] > merged[best_idx][2]:
                merged[best_idx] = item_b
                rects_a[best_idx] = rect_b
        else:
            merged.append(item_b)
            rects_a.append(rect_b)

    return merged


def _extract_page_details(image: Image.Image) -> list[dict]:
    """Single-pass OCR on one page with adaptive sharpening + CLAHE. Returns list of {word, confidence, y, x} dicts."""
    reader = get_reader()
    bgr = _to_cv2(image, min_width=1600)

    preprocessed = _preprocess_pass2(bgr)
    raw_results = _run_ocr(reader, preprocessed, bgr)

    # Sort reading order: top-to-bottom, left-to-right
    def reading_order(item):
        bbox = item[0]
        y_top = min(p[1] for p in bbox)
        x_left = min(p[0] for p in bbox)
        return (y_top // 20, x_left)

    raw_results.sort(key=reading_order)

    details = []
    for (bbox, text, confidence) in raw_results:
        text = text.strip()
        if not text or confidence < CONFIDENCE_THRESHOLD:
            continue
        # Noise filter: skip single-character tokens (except medical markers like T, C)
        if len(text) <= 1 and text.upper() not in ('T', 'C'):
            continue
        # Skip pure punctuation
        if re.match(r'^[^\w\s]+$', text):
            continue

        y_top = min(p[1] for p in bbox)
        x_left = min(p[0] for p in bbox)
        details.append({
            "word": text,
            "confidence": round(float(confidence), 4),
            "y": y_top,
            "x": x_left,
        })

    return details


# ---------------------------------------------------------------------------
# Auto-Orientation / Rotation Detection
# ---------------------------------------------------------------------------

def _fast_orientation_check(image: Image.Image) -> Tuple[Image.Image, int]:
    """
    Check orientation by running a fast EasyOCR readtext on a downscaled image.
    Returns the correctly oriented image and the number of valid characters found.
    """
    reader = get_reader()
    best_img = image
    best_char_count = -1
    
    candidate_rotations = [
        (None, "0°"),
        (Image.Transpose.ROTATE_270, "90° Clockwise"),
        (Image.Transpose.ROTATE_90, "90° Counter-Clockwise"),
        (Image.Transpose.ROTATE_180, "180° Upside-Down"),
    ]
    
    for rot_enum, label in candidate_rotations:
        test_img = image if rot_enum is None else image.transpose(rot_enum)
        
        # Downscale severely for extreme speed in orientation check
        test_img_copy = test_img.copy()
        test_img_copy.thumbnail((800, 800))
        cv_img = cv2.cvtColor(np.array(test_img_copy), cv2.COLOR_RGB2BGR)
        
        # Fast readtext
        res = reader.readtext(cv_img, detail=1)
        # Count characters that are purely alphabetical in high confidence words
        char_count = sum(len([c for c in text if c.isalpha()]) for _, text, prob in res if prob > 0.3)
        
        print(f"[OCR Orientation] {label} yielded {char_count} valid chars.")
        
        if char_count > best_char_count:
            best_char_count = char_count
            best_img = test_img
            
        # If it's very clearly correct, short-circuit to save time
        if char_count > 100 and rot_enum is None:
            break
            
    print(f"[OCR Orientation] Selected best orientation with {best_char_count} valid characters.")
    return best_img, best_char_count


def _auto_orient_page(image: Image.Image) -> Tuple[Image.Image, list[dict]]:
    """
    Check if image is rotated (0°, 90°, 180°, 270°).
    Uses the fast EasyOCR check, then extracts page details on the correctly oriented image.
    """
    best_img, _ = _fast_orientation_check(image)
    best_details = _extract_page_details(best_img)
    return best_img, best_details


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
    Run OCR on one or more images with auto-orientation.
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
