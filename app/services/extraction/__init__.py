"""
extraction/__init__.py
-----------------------
Medicine parsing: turns raw OCR text into structured medicine entries.

Field extraction (dosage/frequency/duration) is pure regex over the OCR line
and unrelated to the catalog. Name resolution (which medicine a token refers
to) is delegated to `app.services.medicine_resolver.resolve()` — the Indian
medicine normalization pipeline (text normalization -> strength/dosage-form
extraction -> exact/fuzzy matching hierarchy -> confidence policy). Only
EXACT and HIGH_CONFIDENCE results are accepted here; REVIEW_REQUIRED,
LOW_CONFIDENCE, and NO_MATCH are all treated the same way this pipeline
always has -- silently skipped, never forced into the result. This is the
same safety behaviour `medicine_matching.match_token()` had before it, just
backed by the richer resolver.

`medicine_matching.match_token()` itself is unchanged and left in place
(still covered by its own tests) even though this module no longer calls
it — `GET /api/v1/medicines/search` uses `medicine_matching.search_many()`
instead, which was already independent of `match_token()`.
"""

import logging
import re
import time
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.services.medicine_resolver import MatchStatus
from app.services.medicine_resolver import resolve as resolve_medicine

logger = logging.getLogger(__name__)

# Statuses accepted as a confident match for the prescription pipeline.
# Everything else (REVIEW_REQUIRED, LOW_CONFIDENCE, NO_MATCH) maps to the
# pipeline's existing "unmatched" behaviour: the line/token is simply
# omitted from the result, exactly as before.
_CONFIDENT_STATUSES = frozenset({MatchStatus.EXACT, MatchStatus.HIGH_CONFIDENCE})

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


def _clean_token_letters(token: str) -> str:
    return re.sub(r'[^a-zA-Z]', '', token).lower()


def _clean_token_alnum(token: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', token).lower()


def _is_skip_token(token: str) -> bool:
    """Cheap pre-filter so we don't hit the DB for obvious boilerplate words."""
    words = token.split()
    if any(_clean_token_letters(w) in SKIP_WORDS for w in words):
        return True
    # Length is measured on the alnum-cleaned token, NOT letters-only: an
    # OCR-corrupted brand like "D0LO" is 4 real characters but only 3
    # letters once the garbled digit is stripped for a letters-only count --
    # that used to make this filter reject it before it ever reached the
    # matcher (real bug, found via a full OCR-pipeline integration test:
    # "D0LO 650" resolved to nothing at all, not even the wrong thing).
    if len(_clean_token_alnum(token)) < 4:
        return True
    return False


# ---------------------------------------------------------------------------
# Field extractors (pure regex, unrelated to the medicine catalog)
# ---------------------------------------------------------------------------

def _extract_dosage(line: str, med_name: str) -> str:
    match = re.search(r'(\d{1,4}(?:\.\d+)?)\s?(mg|ml|mcg|iu|gm?)\b', line, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    match_bracket_num = re.search(r'[\(\[]\s*(\d{1,4})\s*(?:mg|ml)?[\)\]]', line, re.IGNORECASE)
    if match_bracket_num:
        return f"{match_bracket_num.group(1)}mg"
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

    m_dashes = re.search(r'\b\d\s*-\s*\d\b', line)
    if m_dashes:
        return m_dashes.group(0).replace(" ", "")

    return "N/A"


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

def _confident_resolve(db: Session, term: str, stats: Dict[str, int]):
    """
    Resolve one candidate term via the medicine_resolver pipeline, accepting
    only EXACT/HIGH_CONFIDENCE. Returns the matched candidate (same shape the
    old `MedicineMatch` had: `.medicine` / `.confidence` / `.match_type`) or
    None. `stats` is mutated in place purely for the one summary log line
    `parse_medicines` emits per call — never logs per-token, to avoid
    flooding logs (a single line can attempt a dozen+ word/bigram lookups).
    """
    result = resolve_medicine(db, term)
    stats[result.status.value] = stats.get(result.status.value, 0) + 1
    if result.status in _CONFIDENT_STATUSES and result.matched is not None:
        return result.matched
    return None


def parse_medicines(text: str, db: Session) -> List[Dict[str, Any]]:
    """
    Parse raw OCR text into a structured list of medicines, matching names
    against the Postgres-backed catalog via the Indian medicine
    normalization pipeline (see `app.services.medicine_resolver.resolve`).

    Only confident matches (resolver status EXACT / HIGH_CONFIDENCE) are
    emitted — REVIEW_REQUIRED, LOW_CONFIDENCE, and NO_MATCH tokens are all
    silently skipped here, same as before, so uncertain matches are never
    forced into the result.
    """
    if not text or not text.strip():
        return []

    start = time.perf_counter()
    lines = text.split("\n") if "\n" in text else _split_into_lines(text)

    results = []
    seen_names = set()
    stats: Dict[str, int] = {}

    for raw_line in lines:
        if not raw_line.split():
            continue

        line_clean = re.sub(r'[^A-Za-z0-9\s/\-\(\)\.]', ' ', raw_line)
        line_clean = re.sub(r'\s+', ' ', line_clean).strip()

        if len(line_clean) < 3:
            continue

        words = line_clean.split()
        match = None

        # For each starting word, try it together with the NEXT word first
        # (e.g. "Dolo" + "680mg" -> "Dolo 680mg"), then the bare word alone.
        # Order matters: resolving "Dolo" in isolation is blind to a
        # strength sitting right next to it in the OCR text, which defeats
        # the resolver's own strength-conflict safety check (found via a
        # full-pipeline integration test: "Tab Dolo 680mg" was word-scanned
        # as "Dolo" alone -> no strength to conflict with -> wrongly
        # accepted as the 650mg product). If the combined phrase comes back
        # REVIEW_REQUIRED with real candidates, that's a meaningful signal
        # ("this clearly names something, but not safely") -- don't then
        # fall back to the bare word, which would silently lose that signal
        # and risk exactly the wrong-strength acceptance this exists to stop.
        for idx, word in enumerate(words):
            word_clean = _clean_word(word)
            if _is_skip_token(word_clean):
                continue

            blocked = False
            if idx + 1 < len(words):
                combined = f"{word_clean} {_clean_word(words[idx + 1])}".strip()
                if combined != word_clean and not _is_skip_token(combined):
                    result = resolve_medicine(db, combined)
                    stats[result.status.value] = stats.get(result.status.value, 0) + 1
                    if result.status in _CONFIDENT_STATUSES and result.matched is not None:
                        match = result.matched
                        break
                    if result.status == MatchStatus.REVIEW_REQUIRED and result.candidates:
                        blocked = True

            if blocked:
                continue

            candidate = _confident_resolve(db, word_clean, stats)
            if candidate is not None:
                match = candidate
                break

        if match is not None:
            med = match.medicine
            # Invariant: `match` is only ever assigned from a candidate whose
            # `.medicine` was already checked non-None (see the loops above).
            assert med is not None
            name = med.brand_name or med.generic_name or (med.aliases[0] if med.aliases else None)
            if not name:
                # Malformed catalog row (no generic/brand name or alias to
                # display) — nothing to safely render, treat as unmatched
                # rather than crash.
                continue
            root_name = (med.generic_name or name).split()[0].lower()
            if root_name in seen_names or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            seen_names.add(root_name)

            results.append({
                # Back-compat fields (previous API shape)
                "name": name,
                "confidence": round(match.confidence, 1),
                "dosage": _extract_dosage(line_clean, name),
                "frequency": _extract_frequency(line_clean, name),
                "duration": _extract_duration(line_clean, name),
                "raw_line": raw_line.strip(),
                # Structured catalog fields
                "id": str(med.id),
                "generic_name": med.generic_name,
                "brand_name": med.brand_name,
                "strength": med.strength,
                "dosage_form": med.dosage_form,
                "route": med.route,
                "manufacturer": med.manufacturer,
                "match_type": match.match_type,
                "source": med.source,
            })

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[parse_medicines] medicines_found=%d review_required=%d no_match=%d "
        "low_confidence=%d lines=%d elapsed_ms=%.1f",
        len(results),
        stats.get(MatchStatus.REVIEW_REQUIRED.value, 0),
        stats.get(MatchStatus.NO_MATCH.value, 0),
        stats.get(MatchStatus.LOW_CONFIDENCE.value, 0),
        len(lines),
        elapsed_ms,
    )
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
