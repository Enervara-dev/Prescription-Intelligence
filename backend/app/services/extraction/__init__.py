"""
extraction/__init__.py
-----------------------
Medicine parsing with smart handwriting optical normalization, 
false-positive suppression, and field extraction (dosage, frequency, duration).
"""

import re
import os
from typing import List, Dict, Any, Optional
import logging
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frequency mapping
# ---------------------------------------------------------------------------

FREQ_MAP = {
    "OD":   "Once Daily",
    "BD":   "Twice Daily",
    "TDS":  "Thrice Daily",
    "QID":  "Four Times Daily",
    "HS":   "At Bedtime",
    "SOS":  "When Required",
    "STAT": "Immediately",
    "AC":   "Before Meals",
    "PC":   "After Meals",
    "PRN":  "When Required",
    "BBF":  "Before Breakfast",
    "AFTER FOOD": "After Meals",
    "BEFORE FOOD": "Before Meals",
}

# ---------------------------------------------------------------------------
# Medicine database — loaded once
# ---------------------------------------------------------------------------

_MEDICINE_DB: List[str] = []

def _load_medicine_db() -> List[str]:
    global _MEDICINE_DB
    if _MEDICINE_DB:
        return _MEDICINE_DB
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../data/medicine_list.txt"),
        "app/data/medicine_list.txt",
        "data/medicine_list.txt",
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _MEDICINE_DB = [line.strip() for line in f if line.strip()]
            print(f"[MedDB] Loaded {len(_MEDICINE_DB)} medicines from {path}")
            return _MEDICINE_DB
    print("[MedDB] WARNING: medicine_list.txt not found!")
    return []


# ---------------------------------------------------------------------------
# Common non-medicine words to ignore (English stopwords + Prescription boilerplate)
# ---------------------------------------------------------------------------

SKIP_WORDS = {
    "name", "address", "tablet", "take", "with", "hospital", "hospitals",
    "days", "day", "tab", "cap", "one", "two", "three", "four",
    "sig", "temp", "capsule", "syr", "syrup", "drop", "drops", "inj",
    "injection", "the", "and", "for", "after", "before", "food", "water",
    "morning", "night", "evening", "daily", "dose", "oral", "once", "twice",
    "times", "week", "weeks", "month", "months", "dr", "doctor", "consultant",
    "patient", "male", "female", "years", "yrs", "date", "time", "uhid", "reg",
    "appointment", "mobile", "online", "clinic", "department", "advice",
    "investigation", "weight", "height", "allergies", "diet", "diagnosis",
    "dale", "consullant", "mri", "brain", "plain", "teeth", "dental", "linked",
    "shelscar", "chandra", "consultant", "lame", "make", "care", "your", "ichandra",
    "adddivision", "smokycope", "headnight", "gizzlant", "uchphysical", "doffaking",
    "among", "duegsie", "curs-t-f-o", "cpping", "violoing", "apms", "life", "lift",
    "several", "other", "diseases", "stairs", "instead", "teeth", "dental",
    "former", "proper", "women", "school", "schools", "street", "streets",
    "protest", "protests", "terms", "announced", "experienced", "warranted",
    "princesses", "embassy", "emergency", "contact", "validity", "entries",
    "sign", "signature", "assessment", "record", "nationality", "religion",
    "student", "corporate", "treatment", "investigations", "clinical", "notes",
    "advised", "meditation", "exercises", "fruits", "vegetables", "fibre",
    "heaviness", "bloating", "nausea", "vomiting", "vomitings", "stools", "loss", "fever",
    "cough", "review", "candida", "effects", "mucas", "mucus", "srivathsa", "rivathsa",
    "kims", "medical",
}


def _clean_word(word: str) -> str:
    # Strip possessives like Polikarpon's -> Polikarpon
    w = re.sub(r"['’]s\b", '', word, flags=re.IGNORECASE)
    return re.sub(r'[^A-Za-z0-9\s\-]', '', w).strip()


def _is_pure_dosage_or_noise(word: str) -> bool:
    clean = word.strip(".,;:() ")
    if not clean:
        return True
    if re.match(r'^\d+(\.\d+)?(mg|ml|mcg|iu|gm?)$', clean, re.IGNORECASE):
        return True
    if re.match(r'^\d+(\.\d+)?/\d+(\.\d+)?$', clean):
        return True
    if clean.lower() in ("cd3", "d3", "b12", "am", "pm", "ep", "sos", "hs", "od", "bd", "tds", "qid"):
        return True
    return False


def _is_medicine_candidate(word: str) -> bool:
    if len(word) < 4:
        return False
    if not re.match(r'^[A-Za-z]', word):
        return False
    if word.lower() in SKIP_WORDS:
        return False
    if _is_pure_dosage_or_noise(word):
        return False
    digit_count = sum(1 for c in word if c.isdigit())
    if digit_count > len(word) * 0.3:
        return False
    # Reject TrOCR hallucinations: real words have at least ~15% vowels
    alpha_chars = [c for c in word.lower() if c.isalpha()]
    if alpha_chars:
        vowel_ratio = sum(1 for c in alpha_chars if c in 'aeiou') / len(alpha_chars)
        if vowel_ratio < 0.12:  # e.g. 'sphing', 'rrrr', 'xxxific'
            return False
    return True


def _clean_token_letters(token: str) -> str:
    return re.sub(r'[^a-zA-Z]', '', token).lower()


def _match_medicine(token: str, db: List[str]) -> Optional[Dict[str, Any]]:
    words = token.split()
    if any(_clean_token_letters(w) in SKIP_WORDS for w in words):
        return None

    token_clean = _clean_token_letters(token)
    if len(token_clean) < 4 or not db:
        return None

    best_match = None
    best_score = 0.0

    # Optical substitutions common in doctor handwriting OCR
    replacements = [
        ('m', 'n'), ('n', 'm'), ('1', 'l'), ('0', 'o'), ('5', 's'),
        ('q', 'g'), ('cl', 'd'), ('rn', 'm'), ('vv', 'w'),
        ('i', 'y'), ('y', 'i'), ('k', 'b'), ('b', 'k')
    ]
    variants = [token_clean]
    for old, new in replacements:
        if old in token_clean:
            variants.append(token_clean.replace(old, new))

    for var in variants:
        for med in db:
            med_clean = _clean_token_letters(med)
            med_base = _clean_token_letters(med.split()[0]) if ' ' in med else med_clean

            if len(med_base) < 4:
                continue

            # Stricter length disparity guard
            len_diff = abs(len(var) - len(med_base))
            if len_diff >= 4 and min(len(var), len(med_base)) < 7:
                continue

            # 1. Levenshtein ratio / WRatio & Token Set Ratio
            wr = float(fuzz.WRatio(var, med_base))
            token_ratio = float(fuzz.token_set_ratio(var, med_clean))

            # 2. Prefix similarity bonus (first 3-4 letters)
            prefix_len = min(4, len(var), len(med_base))
            prefix_match = (var[:prefix_len] == med_base[:prefix_len])

            score = max(wr, token_ratio)
            
            # Penalize length disparities to prevent "rivathsa" matching "riva"
            if len_diff >= 3:
                score -= (len_diff * 4)
                
            if prefix_match and score >= 70.0:
                score = min(100.0, score + 10.0)

            if score > best_score:
                best_score = score
                best_match = med

    # Increase threshold to 82 to reduce false positives
    if best_score >= 82.0 and best_match:
        return {"name": best_match, "confidence": round(min(100.0, best_score), 1)}

    return None



# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_dosage(line: str, med_name: str) -> str:
    # Regex search in line
    match = re.search(r'(\d{1,4}(?:\.\d+)?)\s?(mg|ml|mcg|iu|gm?)\b', line, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    # Bracketed dosage like (20) or (10) or [20mg]
    match_bracket_num = re.search(r'[\(\[]\s*(\d{1,4})\s*(?:mg|ml)?[\)\]]', line, re.IGNORECASE)
    if match_bracket_num:
        return f"{match_bracket_num.group(1)}mg"
    # Ratio like 1/2
    match_ratio = re.search(r'\(?\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*\)?', line)
    if match_ratio:
        return match_ratio.group(0).strip()
    match_form = re.search(r'\b(CD3|D3|Plus|NXT|SR|XL|Forte)\b', line, re.IGNORECASE)
    if match_form:
        return match_form.group(0).strip()
    return "N/A"


def _extract_duration(line: str, med_name: str) -> str:
    match = re.search(r'(\d+)\s?(day|days|week|weeks|month|months)\b', line, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    match_circle = re.search(r'[\(\[]\s*(\d{1,3})\s*[\)\]]', line)
    if match_circle:
        val = int(match_circle.group(1))
        if val in (1, 3, 5, 7, 10, 14, 15, 21, 30, 60, 90):
            return f"{val} days"
    return "N/A"


def _extract_frequency(line: str, med_name: str) -> str:
    upper = line.upper()
    for key, value in FREQ_MAP.items():
        if re.search(r'\b' + key + r'\b', upper):
            return value
    m3 = re.search(r'\b(\d|\-)\s*[-/]\s*(\d|\-)\s*[-/]\s*(\d|\-)\b', line)
    if m3:
        return f"{m3.group(0)} (M-A-N)"
        
    # Cursive optical misreadings of 1-0-1 or 1-0-0 (e.g. t-0-0, 1-o-1, l-0-l)
    m_opt = re.search(r'\b([01tli])\s*[-/]\s*([01tlio])\s*[-/]\s*([01tlio])\b', line, re.IGNORECASE)
    if m_opt:
        def norm_d(c):
            return '1' if c.lower() in ('1', 't', 'l', 'i') else '0'
        return f"{norm_d(m_opt.group(1))}-{norm_d(m_opt.group(2))}-{norm_d(m_opt.group(3))} (M-A-N)"

    m_time = re.search(r'\b(\d{1,2}\s*(?:AM|PM)|bedtime|morning|night|ep|evening)\b', line, re.IGNORECASE)
    if m_time:
        t = m_time.group(0).strip()
        if t.lower() == 'ep':
            return "Evening"
        return t.capitalize()
    
    # Typical 2-slot pattern like 1-0 or 0-1
    m_dashes = re.search(r'\b\d\s*-\s*\d\b', line)
    if m_dashes:
        return m_dashes.group(0).replace(" ", "")
        
    return "N/A"


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

def parse_medicines(text: str) -> List[Dict[str, Any]]:
    """
    Parse raw text into a structured list of medicines.
    Each item: {name, confidence, dosage, frequency, duration, raw_line}
    """
    if not text or not text.strip():
        return []

    db = _load_medicine_db()
    lines = text.split("\n") if "\n" in text else _split_into_lines(text)

    results = []
    seen_names = set()

    for raw_line in lines:
        tokens = raw_line.split()
        if not tokens:
            continue

        line_clean = re.sub(r'[^A-Za-z0-9\s/\-\(\)\.]', ' ', raw_line)
        line_clean = re.sub(r'\s+', ' ', line_clean).strip()

        if len(line_clean) < 3:
            continue

        words = line_clean.split()
        matched_medicine = None

        # 1. Try single word tokens
        for word in words:
            word_clean = _clean_word(word)
            match = _match_medicine(word_clean, db)
            if match:
                matched_medicine = match
                break

        # 2. Try bigrams if no single word matched
        if not matched_medicine:
            for idx in range(len(words) - 1):
                bigram = _clean_word(words[idx]) + " " + _clean_word(words[idx + 1])
                match = _match_medicine(bigram, db)
                if match:
                    matched_medicine = match
                    break

        if matched_medicine:
            name = matched_medicine["name"]
            root_name = name.split()[0].lower()
            if root_name in seen_names or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            seen_names.add(root_name)

            results.append({
                "name": name,
                "confidence": min(100.0, matched_medicine["confidence"]),
                "dosage": _extract_dosage(line_clean, name),
                "frequency": _extract_frequency(line_clean, name),
                "duration": _extract_duration(line_clean, name),
                "raw_line": raw_line.strip(),
            })

    return results


def _split_into_lines(text: str, words_per_line: int = 8) -> List[str]:
    words = text.split()
    lines = []
    current = []

    for word in words:
        current.append(word)
        chunk = " ".join(current)
        has_dose = bool(re.search(r'\d+\s*(mg|ml|mcg)', chunk, re.IGNORECASE))
        has_freq = any(re.search(r'\b' + k + r'\b', chunk.upper()) for k in FREQ_MAP)

        if len(current) >= words_per_line or (has_dose and has_freq):
            lines.append(chunk)
            current = []

    if current:
        lines.append(" ".join(current))

    return lines
