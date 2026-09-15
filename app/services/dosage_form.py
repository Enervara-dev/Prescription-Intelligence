"""
services/dosage_form.py
--------------------------
Deterministic dosage-form normalization. A formulation mismatch (tablet vs.
syrup) must influence ranking — "Paracetamol 500 mg tablet" must not
automatically match "Paracetamol 500 mg/5 ml syrup" just because the name and
strength agree.
"""

import re

_CANONICAL_FORMS: dict[str, str] = {
    "tab": "tablet", "tabs": "tablet", "tablet": "tablet", "tablets": "tablet",
    "cap": "capsule", "caps": "capsule", "capsule": "capsule", "capsules": "capsule",
    "syp": "syrup", "syr": "syrup", "syrup": "syrup",
    "inj": "injection", "injection": "injection", "injections": "injection",
    "susp": "suspension", "suspension": "suspension",
    "drop": "drops", "drops": "drops",
    "oint": "ointment", "ointment": "ointment",
    "gel": "gel",
    "cream": "cream",
    "lotion": "lotion",
    "spray": "spray",
}

CANONICAL_FORMS = frozenset(_CANONICAL_FORMS.values())

_WORD_RE = re.compile(r"[a-zA-Z]+")


def normalize_dosage_form(text: str) -> str | None:
    """
    Find and canonicalize a dosage-form word anywhere in `text`. Returns None
    if no recognizable form is present — never guesses a form.
    """
    if not text:
        return None
    for word in _WORD_RE.findall(text.lower()):
        canonical = _CANONICAL_FORMS.get(word)
        if canonical:
            return canonical
    return None


def strip_form_words(text: str) -> str:
    """Remove recognized dosage-form words from `text`, leaving the rest intact."""
    if not text:
        return text
    stripped = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in sorted(_CANONICAL_FORMS, key=len, reverse=True)) + r")\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", stripped).strip()


def dosage_forms_compatible(a: str | None, b: str | None) -> bool:
    """
    Two forms are "compatible" for ranking purposes if either is unknown
    (no formulation asserted — don't penalize for missing data) or they're
    the same canonical form. A tablet is never compatible with a syrup.
    """
    if not a or not b:
        return True
    return a == b
