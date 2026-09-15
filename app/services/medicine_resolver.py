"""
services/medicine_resolver.py
--------------------------------
The Indian medicine normalization pipeline — resolves free-text OCR input
(brand or generic, with or without strength/dosage-form, with or without
common OCR corruption) to a catalog entity, or explicitly declines to guess.

    OCR text
      -> text normalization (deterministic, conservative)
      -> strength extraction (separate, deterministic stage)
      -> dosage-form extraction (separate, deterministic stage)
      -> matching hierarchy:
           1. exact canonical brand         (source-priority ordered)
           2. exact alias                   (source-priority ordered)
           3. exact generic (+ strength/form disambiguation if >1 row)
           4-5. (folded into 1-3: this catalog's brand_name convention
                 already embeds strength, e.g. "Dolo 650" as one string —
                 see module note below)
           6. RxNorm exact ingredient match  (naturally reached by stage 3
                 when only an RxNorm row exists for that generic identity —
                 source-priority ordering means an Indian-catalog row always
                 wins if one exists)
           7. controlled fuzzy candidate generation (indexed pg_trgm
                 retrieval + OCR-confusable variants, single-substitution
                 only)
           8. candidate ranking (name similarity + strength agreement +
                 dosage-form compatibility + source reliability)
      -> confidence policy -> MatchStatus
      -> ResolutionResult (auditable: original text, normalized text,
           matched entity, confidence, match_type, source, and the
           candidate set when the result is REVIEW_REQUIRED)

SAFETY RULE: never invent a generic name, strength, dosage form,
manufacturer, or brand mapping from fuzzy similarity alone. "Dolo 680" must
never silently become "Dolo 650" — a strength mismatch is a ranking penalty,
not something the exact-match stages can paper over (they only ever fire on
a literal string match), and when the top two candidates are too close, the
result is REVIEW_REQUIRED rather than a forced pick.

Note on stages 4-5 (brand + strength): this catalog stores Indian brand
identity WITH strength baked into `brand_name` (e.g. "Dolo 650", "Pan 40") —
matching Indian real-world labeling convention — so an exact/fuzzy match on
the full brand string already resolves brand+strength together; there's no
separate "brand alone" vs "brand+strength" distinction to make at the DB
level. What stage 5 adds here is a RANKING check: if the OCR text also
carried an explicit, separately-stated strength (e.g. "Dolo 650mg" or a
strength mentioned elsewhere on the line), that parsed strength is compared
against the matched row's strength_value as a confirmation/penalty signal.
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.medicine import Medicine
from app.repositories import medicine_repository as repo
from app.services import text_normalization as tn
from app.services.dosage_form import dosage_forms_compatible, normalize_dosage_form, strip_form_words
from app.services.medicine_matching import _rerank, generate_optical_variants
from app.services.strength_parser import StrengthInfo, parse_strength, strip_strength_tokens

logger = logging.getLogger(__name__)


class MatchStatus(str, enum.Enum):
    EXACT = "EXACT"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NO_MATCH = "NO_MATCH"


@dataclass
class ResolvedCandidate:
    medicine: Medicine
    confidence: float  # 0-100
    match_type: str


@dataclass
class ResolutionResult:
    # -- Auditability (section 12): every field needed to reconstruct why a
    # decision was made, without re-running anything. --
    input_text: str
    normalized_text: str
    status: MatchStatus
    parsed_strength: StrengthInfo
    parsed_dosage_form: Optional[str]
    matched: Optional[ResolvedCandidate] = None
    candidates: list[ResolvedCandidate] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def match_type(self) -> Optional[str]:
        return self.matched.match_type if self.matched else None

    @property
    def confidence(self) -> Optional[float]:
        return self.matched.confidence if self.matched else None

    @property
    def source(self) -> Optional[str]:
        return self.matched.medicine.source if self.matched else None


def _strength_agreement_delta(query_strength: StrengthInfo, med: Medicine) -> float:
    """
    Ranking adjustment (not a hard gate) for how well a candidate's stored
    strength agrees with what was parsed from the query text. Neutral (0) if
    either side has no strength info — absence of data is never penalized,
    only a genuine mismatch is.
    """
    if not query_strength.found or med.strength_value is None:
        return 0.0
    if query_strength.is_combination:
        # Combination-vs-combination: at least loosely compare component
        # count as a signal, without over-claiming precision here.
        if med.combination_components and len(med.combination_components) == len(query_strength.components):
            return settings.MEDICINE_STRENGTH_MATCH_BONUS
        return -settings.MEDICINE_STRENGTH_MISMATCH_PENALTY if med.combination_components else 0.0

    single = query_strength.single
    if single is None:
        return 0.0
    try:
        med_value = float(med.strength_value)
    except (TypeError, ValueError):
        return 0.0
    same_unit = (med.strength_unit or "").lower() == single.unit.lower()
    if same_unit and abs(med_value - float(single.value)) < 0.01:
        return settings.MEDICINE_STRENGTH_MATCH_BONUS
    if same_unit:
        # Same unit, different number -- e.g. query says 680mg, row is
        # 650mg. This is exactly the "Dolo 680 must not become Dolo 650"
        # case: penalize, don't ignore.
        return -settings.MEDICINE_STRENGTH_MISMATCH_PENALTY
    return 0.0


def _form_agreement_delta(query_form: Optional[str], med: Medicine) -> float:
    if not query_form or not med.dosage_form:
        return 0.0
    return 0.0 if dosage_forms_compatible(query_form, med.dosage_form) else -settings.MEDICINE_FORM_MISMATCH_PENALTY


def _source_bonus(med: Medicine) -> float:
    from app.models.medicine import SOURCE_PRIORITY

    return settings.MEDICINE_SOURCE_PRIORITY_BONUS if SOURCE_PRIORITY.get(med.source, 99) == 0 else 0.0


def _score_candidate(med: Medicine, base_score: float, query_strength: StrengthInfo, query_form: Optional[str]) -> float:
    score = base_score
    score += _strength_agreement_delta(query_strength, med)
    score += _form_agreement_delta(query_form, med)
    score += _source_bonus(med)
    return max(0.0, min(100.0, score))


def _classify_match_type(med: Medicine, stage: str) -> str:
    """UPPER_SNAKE match-type vocabulary for the resolver's richer contract
    (distinct from the lowercase vocabulary medicine_matching.match_token
    uses, kept for backward compatibility on the existing /prescriptions
    endpoint)."""
    is_rxnorm = med.source == "rxnorm"
    if stage == "exact_brand":
        return "EXACT_BRAND"
    if stage == "exact_alias":
        return "EXACT_ALIAS"
    if stage == "exact_generic":
        return "EXACT_GENERIC_RXNORM" if is_rxnorm else "EXACT_GENERIC"
    if stage == "ocr_corrected":
        return "OCR_NORMALIZED_BRAND" if med.brand_name else "OCR_NORMALIZED_GENERIC"
    if stage == "fuzzy":
        return "FUZZY_BRAND" if med.brand_name else "FUZZY_GENERIC"
    return "UNKNOWN"


def _strength_matches(med: Medicine, query_strength: StrengthInfo) -> bool:
    if not query_strength.found or med.strength_value is None:
        return False
    single = query_strength.single
    if single is None:
        return False
    try:
        med_value = float(med.strength_value)
    except (TypeError, ValueError):
        return False
    same_unit = (med.strength_unit or "").lower() == single.unit.lower()
    return same_unit and abs(med_value - float(single.value)) < 0.01


def _disambiguate_by_strength_form(
    rows: list[Medicine], query_strength: StrengthInfo, query_form: Optional[str]
) -> tuple[str, list[Medicine]]:
    """
    Narrow a set of same-generic rows using whatever strength/form
    information the query actually gave — never guesses, never silently
    drops an explicit constraint in favour of a different one. Returns
    ("unique", [row]) when exactly one row fits, else ("ambiguous", rows)
    with the best-effort filtered set (for a REVIEW_REQUIRED candidate list).
    """
    if len(rows) == 1:
        return "unique", rows

    # Source-priority pre-filter: rows here are already source-priority
    # ordered (find_by_canonical_name). If only ONE row sits at the best
    # (highest-priority) source tier, the rest are lower-priority-source
    # duplicates describing the SAME generic identity (e.g. a bare RxNorm
    # ingredient stub alongside the Indian legacy_manual entry for it) --
    # that is NOT the real-world brand ambiguity REVIEW_REQUIRED exists for
    # (Dolo vs Crocin vs Calpol are genuinely different products). Per
    # section 13: differing source naming/coverage is not a duplicate error.
    from app.models.medicine import DEFAULT_SOURCE_PRIORITY, SOURCE_PRIORITY

    best_priority = min(SOURCE_PRIORITY.get(r.source, DEFAULT_SOURCE_PRIORITY) for r in rows)
    top_tier = [r for r in rows if SOURCE_PRIORITY.get(r.source, DEFAULT_SOURCE_PRIORITY) == best_priority]
    if len(top_tier) == 1:
        return "unique", top_tier
    if len(top_tier) < len(rows):
        rows = top_tier  # narrow away lower-priority-source duplicates; keep resolving among the real candidates

    if query_strength.found and query_form:
        both = [r for r in rows if _strength_matches(r, query_strength) and dosage_forms_compatible(query_form, r.dosage_form)]
        if len(both) == 1:
            return "unique", both
        if both:
            return "ambiguous", both

    if query_strength.found:
        by_strength = [r for r in rows if _strength_matches(r, query_strength)]
        # Only trust a strength-only match if the query didn't also name a
        # form that contradicts it (checked above) -- if it reaches here
        # with a form given, the strength+form combo above found nothing,
        # so don't silently ignore the stated form.
        if len(by_strength) == 1 and not query_form:
            return "unique", by_strength
        if by_strength:
            return "ambiguous", by_strength

    if query_form and not query_strength.found:
        by_form = [r for r in rows if dosage_forms_compatible(query_form, r.dosage_form)]
        if len(by_form) == 1:
            return "unique", by_form
        if by_form:
            return "ambiguous", by_form

    return "ambiguous", rows


def _status_for(confidence: float, margin_ok: bool, is_exact: bool) -> MatchStatus:
    if is_exact:
        return MatchStatus.EXACT
    if not margin_ok:
        return MatchStatus.REVIEW_REQUIRED
    if confidence >= settings.MEDICINE_MATCH_THRESHOLD:
        return MatchStatus.HIGH_CONFIDENCE
    if confidence >= settings.MEDICINE_LOW_CONFIDENCE_THRESHOLD:
        return MatchStatus.LOW_CONFIDENCE
    return MatchStatus.NO_MATCH


def resolve(db: Session, input_text: str) -> ResolutionResult:
    """
    Resolve one piece of OCR text (a token, brand phrase, or short line) to
    a catalog entity. Never raises on bad/empty input or on "no match" — the
    caller always gets a well-formed ResolutionResult with an explicit
    status instead.
    """
    start = time.perf_counter()
    normalized = tn.normalize_text(input_text or "")

    if not normalized:
        result = ResolutionResult(
            input_text=input_text or "",
            normalized_text="",
            status=MatchStatus.NO_MATCH,
            parsed_strength=StrengthInfo(),
            parsed_dosage_form=None,
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    query_strength = parse_strength(normalized)
    query_form = normalize_dosage_form(normalized)
    name_portion = strip_form_words(strip_strength_tokens(normalized)) or normalized

    result = _resolve_normalized(db, input_text or "", normalized, name_portion, query_strength, query_form)
    result.latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "[medicine_resolver] status=%s match_type=%s source=%s confidence=%s "
        "candidate_count=%d latency_ms=%.1f review_required=%s",
        result.status.value,
        result.match_type,
        result.source,
        f"{result.confidence:.1f}" if result.confidence is not None else None,
        len(result.candidates),
        result.latency_ms,
        result.status == MatchStatus.REVIEW_REQUIRED,
    )
    return result


def _resolve_normalized(
    db: Session,
    input_text: str,
    normalized: str,
    name_portion: str,
    query_strength: StrengthInfo,
    query_form: Optional[str],
) -> ResolutionResult:
    # Stages 1+2: exact BRAND or ALIAS match — a single, unambiguous product
    # identity by construction (this catalog's brand_name convention already
    # embeds strength, e.g. "Dolo 650"), so these are always accepted
    # directly. Try the full normalized text first (covers "dolo 650"
    # matching brand_name="Dolo 650" verbatim, hyphen/no-space variants
    # after normalize_text), then the strength/form-stripped name portion.
    for candidate_text in ({normalized, name_portion} if name_portion != normalized else {normalized}):
        exact = repo.find_exact(db, candidate_text)
        if exact is None:
            continue
        med = exact.medicine
        alias_hit = repo.find_alias_exact(db, candidate_text)
        if alias_hit and alias_hit.medicine.id == med.id:
            stage = "exact_alias"
        elif med.brand_name and tn.normalize_text(med.brand_name) == candidate_text:
            stage = "exact_brand"
        else:
            # Exact hit landed on generic_name specifically -- do NOT accept
            # it yet: multiple products can share one generic identity (e.g.
            # four different Paracetamol products), and find_exact()'s
            # LIMIT 1 would otherwise silently pick an arbitrary one. Handle
            # this properly via stage 3 below instead of trusting this hit.
            continue

        if query_strength.found and med.strength_value is not None and not _strength_matches(med, query_strength):
            # The name matched exactly (often via a "bare" alias like "dolo"
            # that carries no strength of its own), but the query separately
            # stated a strength that CONFLICTS with this product's actual
            # strength -- e.g. "Dolo 680mg" hitting the "dolo" alias on the
            # 650mg product. This is exactly the case the safety rule names:
            # never silently turn "Dolo 680" into "Dolo 650". Flag it for
            # review instead of accepting a blind exact match.
            conflict = ResolvedCandidate(medicine=med, confidence=70.0, match_type=_classify_match_type(med, stage))
            return ResolutionResult(
                input_text=input_text,
                normalized_text=normalized,
                status=MatchStatus.REVIEW_REQUIRED,
                parsed_strength=query_strength,
                parsed_dosage_form=query_form,
                matched=None,
                candidates=[conflict],
            )

        match_type = _classify_match_type(med, stage)
        matched = ResolvedCandidate(medicine=med, confidence=100.0, match_type=match_type)
        return ResolutionResult(
            input_text=input_text,
            normalized_text=normalized,
            status=MatchStatus.EXACT,
            parsed_strength=query_strength,
            parsed_dosage_form=query_form,
            matched=matched,
            candidates=[matched],
        )

    # Stage 3 (+6, RxNorm exact ingredient): exact GENERIC identity match,
    # disambiguated by strength/dosage-form when more than one product
    # shares that generic name. Never picks an arbitrary product among
    # several equally-named candidates.
    generic_rows = repo.find_by_canonical_name(db, name_portion)
    if generic_rows:
        outcome, rows = _disambiguate_by_strength_form(generic_rows, query_strength, query_form)
        if outcome == "unique":
            med = rows[0]
            matched = ResolvedCandidate(medicine=med, confidence=100.0, match_type=_classify_match_type(med, "exact_generic"))
            return ResolutionResult(
                input_text=input_text,
                normalized_text=normalized,
                status=MatchStatus.EXACT,
                parsed_strength=query_strength,
                parsed_dosage_form=query_form,
                matched=matched,
                candidates=[matched],
            )
        # Generic identity is certain, but which specific product isn't --
        # never force a pick between them.
        candidates = [
            ResolvedCandidate(medicine=r, confidence=85.0, match_type=_classify_match_type(r, "exact_generic"))
            for r in rows
        ]
        return ResolutionResult(
            input_text=input_text,
            normalized_text=normalized,
            status=MatchStatus.REVIEW_REQUIRED,
            parsed_strength=query_strength,
            parsed_dosage_form=query_form,
            matched=None,
            candidates=candidates,
        )

    # Stages 7+8: controlled fuzzy candidate generation + ranking.
    all_scored: dict = {}  # medicine.id -> (Medicine, score, match_type)

    def consider(term: str, stage: str) -> None:
        candidates = repo.search_candidates(db, term)
        for med, base_score in _rerank(term, candidates):
            score = _score_candidate(med, base_score, query_strength, query_form)
            prev = all_scored.get(med.id)
            if prev is None or score > prev[1]:
                all_scored[med.id] = (med, score, _classify_match_type(med, stage))

    consider(normalized, "fuzzy")
    if name_portion != normalized:
        consider(name_portion, "fuzzy")

    # Controlled OCR-confusable variants (single substitution at a time) —
    # medicine-aware, only used here at candidate-generation time, never
    # applied to canonical storage/normalization.
    for variant in generate_optical_variants(name_portion)[1:]:
        consider(variant, "ocr_corrected")
    for variant in tn.generate_ocr_variants(name_portion)[1:]:
        consider(variant, "ocr_corrected")

    ranked = sorted(all_scored.values(), key=lambda triple: triple[1], reverse=True)
    candidates = [
        ResolvedCandidate(medicine=med, confidence=round(score, 1), match_type=mtype)
        for med, score, mtype in ranked[: settings.MEDICINE_CANDIDATE_LIMIT]
    ]

    if not candidates:
        return ResolutionResult(
            input_text=input_text,
            normalized_text=normalized,
            status=MatchStatus.NO_MATCH,
            parsed_strength=query_strength,
            parsed_dosage_form=query_form,
        )

    top = candidates[0]
    margin_ok = len(candidates) == 1 or (top.confidence - candidates[1].confidence) >= settings.MEDICINE_AMBIGUITY_MARGIN
    status = _status_for(top.confidence, margin_ok, is_exact=False)

    if status in (MatchStatus.HIGH_CONFIDENCE, MatchStatus.LOW_CONFIDENCE):
        return ResolutionResult(
            input_text=input_text,
            normalized_text=normalized,
            status=status,
            parsed_strength=query_strength,
            parsed_dosage_form=query_form,
            matched=top,
            candidates=candidates,
        )

    # REVIEW_REQUIRED (ambiguous) or NO_MATCH (below the low-confidence
    # floor): never force a pick — surface the candidate set instead so a
    # human/downstream step can decide, per the safety rule.
    return ResolutionResult(
        input_text=input_text,
        normalized_text=normalized,
        status=status,
        parsed_strength=query_strength,
        parsed_dosage_form=query_form,
        matched=None,
        candidates=candidates if status == MatchStatus.REVIEW_REQUIRED else [],
    )
