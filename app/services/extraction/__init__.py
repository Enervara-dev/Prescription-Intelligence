"""
extraction/__init__.py
-----------------------
Medicine parsing: turns raw OCR text (or, when available, structured
OCRLine objects — see app/services/ocr_geometry.py) into structured
medicine entries.

Name resolution (which medicine a token/phrase refers to) is delegated to
`app.services.medicine_resolver.resolve()` — the Indian medicine
normalization pipeline. Only EXACT and HIGH_CONFIDENCE results are ever
accepted here; REVIEW_REQUIRED, LOW_CONFIDENCE, and NO_MATCH are all
silently skipped — never forced into the result.

Candidate generation is CATALOG-AWARE and SPAN-BASED rather than a fixed
minimum-length pre-filter: a token is only skipped up front when its SHAPE
makes it obviously non-medicine (a bare number, a frequency/duration
pattern, a dosage-form word alone, a known stopword) — never because it is
merely short. A short real brand name (e.g. "Pan" in "Pan 40") must still
reach the resolver when combined with an adjacent token — a fixed length
floor applied before combination was a confirmed bug (see the OCR/matching
forensic audit, section 3).

The scanner also does not stop at the first confident match on a line: it
advances a cursor token-by-token, so a line/row that legitimately contains
more than one medicine (or that OCR reconstruction happened to merge two
rows into one) can still yield more than one result, each with its own
dosage/frequency/duration extracted only from the text BETWEEN it and the
next medicine on the line (never the whole line, which would risk
cross-assigning one medicine's instructions to another — see audit
section 13).
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.services.dosage_form import normalize_dosage_form
from app.services.medicine_resolver import MatchStatus, fuse_confidence
from app.services.medicine_resolver import resolve as resolve_medicine
from app.services.ocr_geometry import OCRLine

logger = logging.getLogger(__name__)

# Statuses accepted as a confident match for the prescription pipeline.
# Everything else (REVIEW_REQUIRED, LOW_CONFIDENCE, NO_MATCH) maps to the
# pipeline's existing "unmatched" behaviour: the span is simply omitted from
# the result, exactly as before.
_CONFIDENT_STATUSES = frozenset({MatchStatus.EXACT, MatchStatus.HIGH_CONFIDENCE})

# Longest medicine-name span (in tokens) the scanner will try at any one
# starting position: covers "medicine + strength" ("Dolo 650"), "medicine +
# form" or brand names with an internal space, and the rare three-token
# case ("Vitamin B 12", combination names). Kept deliberately small — this
# is a candidate-generation bound, not a claim about real name lengths; the
# resolver's own catalog lookup is what actually decides a match, not this
# number.
_MAX_SPAN_TOKENS = 3

# How many consecutive medicine-less lines after a matched medicine are
# still eligible to supply its dosage/frequency/duration (e.g. printed on
# their own row/line beneath the name). Bounded so association can't drift
# arbitrarily far from the medicine it's meant to describe.
_MAX_INSTRUCTION_LOOKAHEAD_LINES = 2

# ---------------------------------------------------------------------------
# Frequency mapping
# ---------------------------------------------------------------------------

FREQ_MAP = {
    "OD":   "Once Daily",
    "BD":   "Twice Daily",
    "BID":  "Twice Daily",
    "TDS":  "Thrice Daily",
    "TID":  "Thrice Daily",
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
# Common non-medicine words to ignore (English stopwords + Prescription
# boilerplate). Shape-based noise filtering, never a length rule — see
# module docstring and audit section 3/4.
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
    "kims", "medical", "as", "directed", "instructed", "advised",
}

_FREQ_KEYWORDS = set(FREQ_MAP.keys())
_DURATION_RE = re.compile(r"\b\d+\s?(day|days|week|weeks|month|months)\b", re.IGNORECASE)
_FREQ_FRACTIONAL_TRIAD_RE = re.compile(r"\b(\d/\d|\d)\s*-\s*(\d/\d|\d)\s*-\s*(\d/\d|\d)\b")
_FREQ_TRIAD_RE = re.compile(r"\b(\d|\-)\s*[-/]\s*(\d|\-)\s*[-/]\s*(\d|\-)\b")
_FREQ_OPTICAL_TRIAD_RE = re.compile(r"\b([01tli])\s*[-/]\s*([01tlio])\s*[-/]\s*([01tlio])\b", re.IGNORECASE)
_PURE_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
# Genuine administration-dose phrasing — how MUCH to take per dose, distinct
# from strength (how much is IN one unit) and frequency (how often). Only
# fires on real amount+unit phrasing; never inferred otherwise (audit
# section 10).
_DOSAGE_RE = re.compile(
    r"\b(\d+/\d+|\d+(?:\.\d+)?|half|one|two|three)\s*-?\s*"
    r"(tablets?|tabs?|capsules?|caps?|drops?|puffs?|spoons?|tsp|teaspoons?)\b",
    re.IGNORECASE,
)


def _clean_word(word: str) -> str:
    # Strip possessives like Polikarpon's -> Polikarpon
    w = re.sub(r"['’]s\b", '', word, flags=re.IGNORECASE)
    return re.sub(r'[^A-Za-z0-9\s\-]', '', w).strip()


def _clean_token_letters(token: str) -> str:
    return re.sub(r'[^a-zA-Z]', '', token).lower()


def _is_noise_token(raw_word: str) -> bool:
    """
    SHAPE-based pre-filter, never a length rule: a token is skipped up front
    only when it obviously cannot be (the start of) a medicine name — a
    stopword, a bare number, a frequency/duration pattern, or a dosage-form
    word alone. A short but real brand name (e.g. "Pan") is NOT filtered
    here; it reaches the span scanner like anything else and is only ever
    accepted if the catalog actually resolves it (see module docstring).
    """
    w = _clean_word(raw_word)
    if not w:
        return True
    if _clean_token_letters(w) in SKIP_WORDS:
        return True
    if _PURE_NUMBER_RE.match(w):
        return True
    if w.upper() in _FREQ_KEYWORDS:
        return True
    if _FREQ_TRIAD_RE.fullmatch(w) or _FREQ_FRACTIONAL_TRIAD_RE.fullmatch(w):
        return True
    if _DURATION_RE.fullmatch(w):
        return True
    if normalize_dosage_form(w) is not None:
        # A bare dosage-form word/abbreviation ("Tab", "Cap", "Syrup", ...)
        # is never itself a medicine name.
        return True
    return False


# ---------------------------------------------------------------------------
# Field extractors (pure regex, unrelated to the medicine catalog)
# ---------------------------------------------------------------------------

def _extract_strength_text(text: str) -> str:
    """
    Detects a strength-SHAPED token in free text (e.g. "500mg") — how much
    is in one unit of the medicine. Distinct from `_extract_dosage` (how
    much to take per administration) — conflating the two was a confirmed
    bug (audit section 10): the field previously named "dosage" actually
    re-detected strength. Kept as its own function because
    enervara_extraction_adapter.py still needs a text-derived strength
    fallback for catalog rows with no structured strength_value.
    """
    match = re.search(r'(\d{1,4}(?:\.\d+)?)\s?(mg|ml|mcg|iu|gm?)\b', text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    match_bracket_num = re.search(r'[\(\[]\s*(\d{1,4})\s*(?:mg|ml)?[\)\]]', text, re.IGNORECASE)
    if match_bracket_num:
        return f"{match_bracket_num.group(1)}mg"
    match_ratio = re.search(r'\(?\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*\)?', text)
    if match_ratio:
        return match_ratio.group(0).strip()
    match_form = re.search(r'\b(CD3|D3|Plus|NXT|SR|XL|Forte)\b', text, re.IGNORECASE)
    if match_form:
        return match_form.group(0).strip()
    return "N/A"


def _extract_dosage(text: str) -> str:
    """
    Genuine administration dosage — how much to take per dose (e.g. "1
    tablet", "2 drops"). Only extracted when the text actually contains
    amount+unit phrasing; never inferred from a bare number or from
    strength (audit section 10). Real prescriptions very often never state
    this explicitly (the frequency triad implies "one dose" by convention)
    — returning "N/A" in that case is the honest answer, not a gap to guess
    around.
    """
    match = _DOSAGE_RE.search(text)
    return match.group(0).strip() if match else "N/A"


def _extract_duration(text: str) -> str:
    """
    Only extracts duration when the text contains an explicit unit (day(s)/
    week(s)/month(s)). A bare bracketed number like "(5)" is NOT treated as
    "5 days" — that was a confirmed hallucination (audit section 12): there
    is no unit anywhere in the source text to justify it.
    """
    match = _DURATION_RE.search(text)
    return match.group(0).strip() if match else "N/A"


def _extract_frequency(text: str) -> str:
    upper = text.upper()
    for key, value in FREQ_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', upper):
            return value

    # Fractional triad first ("1/2-0-1") -- a segment may itself be a
    # fraction; the separator between segments is always '-' here (never
    # '/', which would be ambiguous with a fraction inside a segment).
    mf = _FREQ_FRACTIONAL_TRIAD_RE.search(text)
    if mf:
        return f"{mf.group(1)}-{mf.group(2)}-{mf.group(3)} (M-A-N)"

    m3 = _FREQ_TRIAD_RE.search(text)
    if m3:
        return f"{m3.group(0)} (M-A-N)"

    # Cursive optical misreadings of 1-0-1 or 1-0-0 (e.g. t-0-0, 1-o-1, l-0-l)
    m_opt = _FREQ_OPTICAL_TRIAD_RE.search(text)
    if m_opt:
        def norm_d(c):
            return '1' if c.lower() in ('1', 't', 'l', 'i') else '0'
        return f"{norm_d(m_opt.group(1))}-{norm_d(m_opt.group(2))}-{norm_d(m_opt.group(3))} (M-A-N)"

    m_time = re.search(r'\b(\d{1,2}\s*(?:AM|PM)|bedtime|morning|night|ep|evening)\b', text, re.IGNORECASE)
    if m_time:
        t = m_time.group(0).strip()
        if t.lower() == 'ep':
            return "Evening"
        return t.capitalize()

    m_dashes = re.search(r'\b\d\s*-\s*\d\b', text)
    if m_dashes:
        return m_dashes.group(0).replace(" ", "")

    return "N/A"


# ---------------------------------------------------------------------------
# Span-based candidate scanning
# ---------------------------------------------------------------------------

def _tokenize(line: str) -> List[str]:
    line_clean = re.sub(r'[^A-Za-z0-9\s/\-\(\)\.]', ' ', line)
    line_clean = re.sub(r'\s+', ' ', line_clean).strip()
    return line_clean.split() if line_clean else []


def _scan_line_for_medicines(
    db: Session, words: List[str], ocr_confidence: Optional[float], stats: Dict[str, int]
) -> List[tuple]:
    """
    Forward-scanning span search: tries spans of decreasing length (3, 2, 1
    tokens) at each cursor position, accepting the LONGEST confident match
    found (more specific beats less specific — "Dolo 650" over bare "Dolo").
    Does not stop after the first match: the cursor advances past whatever
    was consumed (a matched span, a genuinely conflicting span, or one
    token) and scanning continues, so a line with more than one medicine is
    not structurally limited to reporting only the first.

    A REVIEW_REQUIRED result at a given span is only allowed to block
    retrying a SHORTER span at the same position when the query itself
    stated a strength (`result.parsed_strength.found`) — that is exactly
    the "Dolo 680mg must never silently become Dolo 650" case: falling back
    to the bare name would discard the stated, conflicting strength. A
    REVIEW_REQUIRED for any other reason (generic ambiguity, an
    uncorroborated fuzzy score) does NOT block shorter spans — there is no
    conflicting evidence to protect, just insufficient evidence at the
    longer span, and a shorter span may still resolve cleanly (e.g. combined
    "amoxicillin as" being too weak a fuzzy match must not prevent bare
    "amoxicillin" from resolving on its own).

    Returns a list of (start, end_exclusive, ResolvedCandidate) tuples, in
    line order, never overlapping.
    """
    matches: List[tuple] = []
    i = 0
    n = len(words)
    while i < n:
        if _is_noise_token(words[i]):
            i += 1
            continue

        max_len = min(_MAX_SPAN_TOKENS, n - i)
        accepted = None
        strength_conflict_len = None

        for span_len in range(max_len, 0, -1):
            phrase = " ".join(_clean_word(w) for w in words[i : i + span_len]).strip()
            if not phrase:
                continue
            result = resolve_medicine(db, phrase, ocr_confidence=ocr_confidence)
            stats[result.status.value] = stats.get(result.status.value, 0) + 1
            if result.status in _CONFIDENT_STATUSES and result.matched is not None:
                accepted = (span_len, result.matched)
                break
            if (
                result.status == MatchStatus.REVIEW_REQUIRED
                and result.parsed_strength.found
                and result.candidates
                and result.candidates[0].match_type
                in ("EXACT_ALIAS", "EXACT_BRAND", "EXACT_CATALOG_NAME")
            ):
                # Genuine strength conflict on an otherwise-exact NAME match
                # (e.g. "Dolo 680mg" hitting the 650mg product's alias
                # exactly, but with a conflicting stated strength) -- stop
                # here. Trying a SHORTER span next (bare "Dolo") would
                # silently discard the very strength that caused the
                # conflict and resolve to the wrong product with full
                # confidence ("Dolo 680 must never become Dolo 650").
                #
                # Deliberately narrower than "any REVIEW_REQUIRED that
                # mentions a strength": a FUZZY-stage candidate downgraded by
                # the identity-corroboration gate (see medicine_resolver.py)
                # also reports parsed_strength.found=True whenever the query
                # text happens to contain a strength number, even with no
                # real conflict — that is "not sure this is the right
                # medicine at all", not "conflicting strength on a confirmed
                # identity", and must NOT block a shorter span from cleanly
                # resolving (e.g. a misspelled "Amoxycillin 500mg BD" must
                # still let bare "Amoxycillin" reach and OCR-correct to the
                # real "Amoxicillin").
                strength_conflict_len = span_len
                break

        if accepted is not None:
            span_len, matched = accepted
            matches.append((i, i + span_len, matched))
            i += span_len
        elif strength_conflict_len is not None:
            # Genuine strength conflict: skip past the whole conflicting
            # span without asserting anything, and without retrying a
            # shorter span here that would discard the stated strength.
            i += strength_conflict_len
        else:
            i += 1

    return matches


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

def _line_texts_and_confidence(
    text: str, ocr_lines: Optional[Sequence[OCRLine]]
) -> List[tuple]:
    """
    Returns [(line_text, ocr_confidence_or_None), ...]. Prefers structured
    OCRLine data (real per-line OCR confidence) when the caller has it;
    falls back to splitting the plain-text `text` (ocr_confidence=None for
    every line) otherwise — the path every existing caller/test without
    OCR geometry context takes, unchanged from previous behaviour.
    """
    if ocr_lines:
        return [(line.text, line.confidence) for line in ocr_lines]
    if not text or not text.strip():
        return []
    # `text.split("\n")` on a string with no newline at all naturally
    # returns a single-element list -- exactly right, since a real OCR
    # result with only one detected row/line produces exactly this
    # (ocr_service.py joins lines with "\n", so a 1-line result has none).
    # Treating it as one line, rather than re-chunking it by word count via
    # _split_into_lines, is what lets the span scanner see the whole line at
    # once — needed for multi-medicine detection and correct instruction
    # windowing, both of which an arbitrary mid-line cut would break.
    return [(line, None) for line in text.split("\n")]


def _merge_instruction_only_line(pending: Dict[str, Any], line_text: str) -> bool:
    """
    Fills in whichever of dosage/frequency/duration `pending` is still
    missing, from a line that itself contained no resolvable medicine —
    the common "medicine name on one line, dosage/frequency/duration on the
    next" layout. Never overwrites a field already found on the medicine's
    own line. Returns True if anything was actually merged (used to decide
    whether continuing the bounded lookahead is still worthwhile).
    """
    merged = False
    if pending["dosage"] == "N/A":
        val = _extract_dosage(line_text)
        if val != "N/A":
            pending["dosage"] = val
            merged = True
    if pending["frequency"] == "N/A":
        val = _extract_frequency(line_text)
        if val != "N/A":
            pending["frequency"] = val
            merged = True
    if pending["duration"] == "N/A":
        val = _extract_duration(line_text)
        if val != "N/A":
            pending["duration"] = val
            merged = True
    if pending.get("strength_text", "N/A") == "N/A":
        val = _extract_strength_text(line_text)
        if val != "N/A":
            pending["strength_text"] = val
            merged = True
    return merged


def parse_medicines(
    text: str, db: Session, ocr_lines: Optional[Sequence[OCRLine]] = None
) -> List[Dict[str, Any]]:
    """
    Parse raw OCR text (or structured OCRLine data, when available) into a
    structured list of medicines, matching names against the Postgres-
    backed catalog via the Indian medicine normalization pipeline.

    Only confident matches (resolver status EXACT / HIGH_CONFIDENCE) are
    emitted — REVIEW_REQUIRED, LOW_CONFIDENCE, and NO_MATCH spans are all
    silently skipped, never forced into the result.
    """
    lines = _line_texts_and_confidence(text, ocr_lines)
    if not lines:
        return []

    start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    seen_names = set()
    stats: Dict[str, int] = {}
    pending: Optional[Dict[str, Any]] = None
    pending_lookahead_left = 0

    for raw_line, line_confidence in lines:
        if not raw_line or not raw_line.split():
            continue

        words = _tokenize(raw_line)
        if len(" ".join(words)) < 3:
            continue

        line_matches = _scan_line_for_medicines(db, words, line_confidence, stats)

        if not line_matches:
            if pending is not None and pending_lookahead_left > 0:
                _merge_instruction_only_line(pending, raw_line)
                pending_lookahead_left -= 1
            continue

        for idx, (start_tok, end_tok, matched) in enumerate(line_matches):
            med = matched.medicine
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

            window_end = line_matches[idx + 1][0] if idx + 1 < len(line_matches) else len(words)
            window_text = " ".join(words[end_tok:window_end])
            # A strength can legitimately be absorbed INTO the matched span
            # itself (e.g. "Amoxycillin 500mg" resolves as one 2-token
            # phrase via OCR correction) rather than trailing it -- check
            # the matched span's own text too, not just what follows it.
            # Dosage/frequency/duration are not at risk of this: those
            # tokens (BD, 1-0-1, 5 days, ...) are noise-filtered and would
            # never be absorbed into a resolved medicine-name span.
            span_text = " ".join(words[start_tok:end_tok])
            strength_source_text = f"{span_text} {window_text}".strip()

            result = {
                # Back-compat fields (previous API shape)
                "name": name,
                "confidence": round(matched.confidence, 1),
                "dosage": _extract_dosage(window_text),
                "frequency": _extract_frequency(window_text),
                "duration": _extract_duration(window_text),
                "raw_line": raw_line.strip(),
                # Structured catalog fields
                "id": str(med.id),
                "generic_name": med.generic_name,
                "brand_name": med.brand_name,
                "strength": med.strength,
                "dosage_form": med.dosage_form,
                "route": med.route,
                "manufacturer": med.manufacturer,
                "match_type": matched.match_type,
                "source": med.source,
                # New: strength genuinely re-detected from the OCR text
                # (distinct from "dosage" — see _extract_strength_text),
                # used by enervara_extraction_adapter.py as a fallback when
                # the catalog row has no structured strength of its own.
                "strength_text": _extract_strength_text(strength_source_text),
                "ocr_confidence": line_confidence,
                "overall_confidence": fuse_confidence(matched.confidence, line_confidence),
            }
            results.append(result)

        last_result = results[-1] if results else None
        if last_result is not None and last_result["dosage"] == "N/A" and last_result["frequency"] == "N/A" and last_result["duration"] == "N/A":
            pending = last_result
            pending_lookahead_left = _MAX_INSTRUCTION_LOOKAHEAD_LINES
        else:
            pending = None
            pending_lookahead_left = 0

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
