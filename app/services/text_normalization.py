"""
services/text_normalization.py
----------------------------------
Deterministic text normalization — the FIRST stage of the resolution
pipeline, run before any database lookup or fuzzy matching.

Deliberately conservative: this only does things that are unambiguously safe
(case, whitespace, punctuation, alnum-boundary spacing). It does NOT do
OCR-confusable substitution (0<->O, 1<->I/l, 5<->S, 8<->B, ...) — that stays
scoped to candidate generation/ranking (see generate_ocr_variants below and
medicine_matching.generate_optical_variants), never applied blindly here,
because blindly rewriting characters at the normalization stage is exactly
what turns "Dolo 680" into a silent, wrong "Dolo 650".

    normalize_text("Dolo 650")   == "dolo 650"
    normalize_text("Dolo-650")   == "dolo 650"
    normalize_text("DOLO650")    == "dolo650"   -- see split_alnum_boundaries
    normalize_text("D0LO 650")   == "d0lo 650"  -- '0' is NOT corrected here
"""

import re

# Common OCR single-character confusions, used ONLY for controlled candidate
# generation (never applied during canonical normalization/storage). Kept
# here (not duplicated in medicine_matching.py) as the single source of truth
# for this table; medicine_matching.generate_optical_variants covers
# multi-character sequences (rn<->m, cl<->d) which are a different, broader
# concern from this module's single-character substitution table.
OCR_CHAR_CONFUSIONS: dict[str, list[str]] = {
    "0": ["o"],
    "o": ["0"],
    "1": ["i", "l"],
    "i": ["1", "l"],
    "l": ["1", "i"],
    "5": ["s"],
    "s": ["5"],
    "8": ["b"],
    "b": ["8"],
}


def normalize_text(text: str) -> str:
    """
    Deterministic, lossless-intent normalization for search/comparison.
    Safe to apply to both incoming OCR text and catalog data.
    """
    if not text:
        return ""

    t = text.strip().lower()
    # Hyphens between tokens act as separators here (Dolo-650 -> Dolo 650),
    # not as part of a chemical name — medicine names in this catalog don't
    # rely on hyphens for meaning.
    t = t.replace("-", " ")
    # Collapse punctuation noise (keep alphanumerics, spaces, '+' and '/' —
    # '+' matters for combinations like "500mg + 125mg", '/' for "100mg/5ml").
    t = re.sub(r"[^\w\s+/]", " ", t)
    t = split_alnum_boundaries(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_TRAILING_DIGITS_RE = re.compile(r"^([a-zA-Z]+)(\d{2,})$")


def split_alnum_boundaries(text: str) -> str:
    """
    Split a token that is CLEANLY letters-then-digits ("dolo650" -> "dolo
    650", "pan40" -> "pan 40"), so it's comparable to the spaced form.

    Deliberately narrow: only whole tokens matching letters+2-or-more-digits
    qualify. A token with a digit embedded mid-word (e.g. "d0lo", where the
    '0' is a probable OCR misread of 'o', not a genuine trailing strength
    number) is left untouched — inserting a space there would make things
    worse ("d 0lo"), not better. OCR-confusable digits inside a word are
    handled later, at candidate-generation time, by generate_ocr_variants().
    """
    return " ".join(
        f"{m.group(1)} {m.group(2)}" if (m := _TRAILING_DIGITS_RE.match(tok)) else tok
        for tok in text.split()
    )


def generate_ocr_variants(token: str) -> list[str]:
    """
    Controlled, single-substitution OCR-confusable variants for candidate
    generation ONLY (never for canonical storage/normalization — see module
    docstring). Generates one variant per confusable character occurrence,
    same conservative "one substitution at a time" approach as
    medicine_matching.generate_optical_variants, applied to the
    single-character confusion table above.
    """
    variants = [token]
    for i, ch in enumerate(token):
        for repl in OCR_CHAR_CONFUSIONS.get(ch, []):
            variants.append(token[:i] + repl + token[i + 1 :])
    return variants
