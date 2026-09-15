"""
services/strength_parser.py
------------------------------
Deterministic strength/unit parsing — a SEPARATE stage from name matching.
Never mixed with fuzzy medicine-name matching: this module only ever looks
at digits + unit tokens, regardless of what drug name (if any) is nearby.

Handles:
    "500 mg"            -> single: 500 mg
    "500mg"              -> single: 500 mg
    "650 MG"              -> single: 650 mg
    "40mg"                -> single: 40 mg
    "5 ml"                 -> single: 5 ml
    "100 mg/5 ml"           -> single: 100 mg per 5 ml  (per_value/per_unit set)
    "250 mg + 125 mg"        -> combination: [250 mg, 125 mg]
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional

# Canonical unit set. Anything not in here after normalization is rejected
# (better to report "no strength found" than silently accept a bogus unit).
_UNIT_ALIASES: dict[str, str] = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "mcg": "mcg", "microgram": "mcg", "micrograms": "mcg", "ug": "mcg",
    "g": "g", "gm": "g", "gram": "g", "grams": "g",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml", "milliliters": "ml",
    "l": "l", "litre": "l", "liter": "l",
    "iu": "iu",
    "%": "%",
}
_UNIT_PATTERN = "|".join(sorted((re.escape(u) for u in _UNIT_ALIASES), key=len, reverse=True))

_SINGLE_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b", re.IGNORECASE)
_PER_VOLUME_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\s*/\s*(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b",
    re.IGNORECASE,
)


@dataclass
class StrengthComponent:
    value: Decimal
    unit: str


@dataclass
class StrengthInfo:
    components: list[StrengthComponent] = field(default_factory=list)
    # Set only for a "X unit / Y unit2" single strength, e.g. "100mg/5ml".
    per_value: Optional[Decimal] = None
    per_unit: Optional[str] = None

    @property
    def is_combination(self) -> bool:
        return len(self.components) > 1

    @property
    def single(self) -> Optional[StrengthComponent]:
        return self.components[0] if len(self.components) == 1 else None

    @property
    def found(self) -> bool:
        return bool(self.components)

    def as_dict(self) -> dict:
        if self.is_combination:
            return {"components": [{"strength": float(c.value), "unit": c.unit} for c in self.components]}
        if self.single:
            d: dict = {"strength": float(self.single.value), "unit": self.single.unit}
            if self.per_value is not None:
                d["per_value"] = float(self.per_value)
                d["per_unit"] = self.per_unit
            return d
        return {}


def _to_decimal(raw: str) -> Optional[Decimal]:
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _normalize_unit(raw: str) -> Optional[str]:
    return _UNIT_ALIASES.get(raw.strip().lower())


def _parse_single(text: str) -> Optional[StrengthComponent]:
    per = _PER_VOLUME_RE.search(text)
    if per:
        value = _to_decimal(per.group(1))
        unit = _normalize_unit(per.group(2))
        if value is not None and unit is not None:
            return StrengthComponent(value=value, unit=unit)

    m = _SINGLE_RE.search(text)
    if not m:
        return None
    value = _to_decimal(m.group(1))
    unit = _normalize_unit(m.group(2))
    if value is None or unit is None:
        return None
    return StrengthComponent(value=value, unit=unit)


def strip_strength_tokens(text: str) -> str:
    """
    Remove NUMBER+UNIT tokens (e.g. "500 mg", "100mg/5ml") from `text`,
    leaving the rest (name words, bare numbers with no unit) intact. Used to
    isolate the "name portion" of an OCR string before exact-name lookup —
    never removes a bare number, since for this catalog's convention a bare
    number is often part of the brand identity itself (e.g. "Dolo 650").
    """
    if not text:
        return text
    t = _PER_VOLUME_RE.sub(" ", text)
    t = _SINGLE_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_strength(text: str) -> StrengthInfo:
    """
    Parse strength information out of free text. Returns an empty
    StrengthInfo (`.found == False`) if nothing recognizable is present —
    never guesses.
    """
    if not text or not text.strip():
        return StrengthInfo()

    # Combination check first: split on '+' and try to parse each side
    # independently. Only treated as a combination if >= 2 sides each
    # yield a real component — a stray '+' with no second dose doesn't
    # count.
    if "+" in text:
        parts = [p.strip() for p in text.split("+") if p.strip()]
        components = []
        for part in parts:
            comp = _parse_single(part)
            if comp:
                components.append(comp)
        if len(components) >= 2:
            return StrengthInfo(components=components)

    per = _PER_VOLUME_RE.search(text)
    if per:
        value = _to_decimal(per.group(1))
        unit = _normalize_unit(per.group(2))
        per_value = _to_decimal(per.group(3))
        per_unit = _normalize_unit(per.group(4))
        if value is not None and unit is not None:
            return StrengthInfo(
                components=[StrengthComponent(value=value, unit=unit)],
                per_value=per_value,
                per_unit=per_unit,
            )

    single = _parse_single(text)
    if single:
        return StrengthInfo(components=[single])

    return StrengthInfo()
