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

SAFETY RULE (identity corroboration): a high raw fuzzy/trigram string-
similarity score is NEVER, on its own, sufficient evidence for
HIGH_CONFIDENCE. Two clinically unrelated medicines can be close as strings
purely by coincidence (verified: "Asthalin" scores ~69 against "Aspirin"
with no correction or context involved at all). HIGH_CONFIDENCE from the
fuzzy stage (stage 7-8) requires one of:
  (a) the query, or a principled single-character OCR-confusable
      correction of it, is an EXACT match against a catalog name/alias
      (see `_exact_after_correction` below) -- this is what legitimizes
      "D0LO 650" -> "Dolo 650" and "Crocln 650" -> "Crocin 650": a
      structural, explainable correction landing on a real identity, not
      mere resemblance; or
  (b) the query separately stated a strength or dosage form that agrees
      with this specific candidate (see `_has_identity_corroboration`) --
      independent evidence beyond string shape alone.
A candidate meeting neither bar is capped at REVIEW_REQUIRED regardless of
its numeric score, however high — see `_resolve_normalized`'s fuzzy-stage
acceptance gate. This is a property of the EVIDENCE the resolver has, not a
name-specific rule, so it generalizes to any future lookalike pair without
needing a new special case.

CONFIDENCE SEMANTICS (documented once, here, as the single source of truth
— every field below is read by multiple call sites; see the module-level
grep audit in the OCR/matching forensic review for the full consumer list):
  - `ResolutionResult.confidence` (delegates to `matched.confidence`):
    0-100. PURE medicine-matching confidence (name similarity + strength/
    form/source ranking adjustments). Never incorporates OCR read quality.
    This is the pre-existing, unchanged-meaning compatibility field —
    consumed by `/api/v1/medicines/resolve`, `/process`'s per-medicine
    dict, and (divided by 100) Enervara's `/extract` adapter historically.
  - `ResolutionResult.ocr_confidence` (new, optional): 0.0-1.0, the OCR
    provider's own certainty about the text span this result was resolved
    from — set by the caller (see `resolve()`'s new parameter) when that
    information is available from structured OCR data; `None` when it
    isn't (e.g. a direct API call with no OCR context), in which case
    nothing about existing behavior changes.
  - `ResolutionResult.overall_confidence` (new, computed property): fuses
    `confidence` with `ocr_confidence` when both are present — a
    dampening-only combination (never inflates a score), reflecting that a
    clean-looking match is only as trustworthy as the OCR read it came
    from. Equals `confidence` exactly when `ocr_confidence` is `None`.
  - `match_type`: which resolution stage/strategy produced the match
    (UPPER_SNAKE vocabulary — EXACT_BRAND/EXACT_ALIAS/EXACT_GENERIC/
    EXACT_GENERIC_RXNORM/EXACT_CATALOG_NAME/OCR_NORMALIZED_*/FUZZY_*).
    Distinct from, and never mixed with, `medicine_matching.py`'s own
    lowercase vocabulary for the unrelated `/medicines/search` code path.
  - `matched`: the ONE candidate a caller should treat as "this is the
    identified medicine" — populated ONLY for EXACT and (corroborated)
    HIGH_CONFIDENCE. Never populated for LOW_CONFIDENCE, REVIEW_REQUIRED,
    or NO_MATCH: a wrong medicine is worse than an unresolved one.
  - `source`: provenance tag of the matched row (`legacy_manual`/`rxnorm`/
    etc.), `None` whenever `matched` is `None`.

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
    # OCR provider's own certainty (0.0-1.0) about the text this was resolved
    # from, when the caller had structured OCR data to supply it — see the
    # CONFIDENCE SEMANTICS note in the module docstring. None when unknown.
    ocr_confidence: Optional[float] = None

    @property
    def match_type(self) -> Optional[str]:
        return self.matched.match_type if self.matched else None

    @property
    def confidence(self) -> Optional[float]:
        return self.matched.confidence if self.matched else None

    @property
    def source(self) -> Optional[str]:
        return self.matched.medicine.source if self.matched else None

    @property
    def overall_confidence(self) -> Optional[float]:
        """See `fuse_confidence` — the single source of truth for this
        formula, also used outside this dataclass (extraction/__init__.py)
        for per-medicine results built from a matched-and-then-consumed
        candidate rather than a live ResolutionResult."""
        return fuse_confidence(self.confidence, self.ocr_confidence)


def fuse_confidence(match_confidence: Optional[float], ocr_confidence: Optional[float]) -> Optional[float]:
    """
    Fuses match confidence (0-100, pure name/identity evidence) with OCR
    read confidence (0.0-1.0, the provider's own certainty about the source
    text) into one combined signal. Dampens only — never inflates a score
    above what the name-matching evidence alone supports — and is identical
    to `match_confidence` whenever `ocr_confidence` is unavailable, so every
    caller that never supplies OCR context sees no behavior change at all.
    """
    if match_confidence is None:
        return None
    if ocr_confidence is None:
        return match_confidence
    ocr_factor = 0.5 + 0.5 * max(0.0, min(1.0, ocr_confidence))
    return round(match_confidence * ocr_factor, 1)


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


# Provenance tag written by scripts/import_legacy_medicines.py for flat-file
# rows whose brand/generic identity the source data never actually
# established (see catalog audit section 8) — a name forced into the
# `generic_name` column purely because the schema has nowhere else to put
# an unannotated string. `_classify_match_type` checks this so a match
# against one of these rows is never reported as a confirmed generic
# identity it doesn't actually have.
UNCLASSIFIED_NAME_SOURCE_VERSION = "unclassified_name"


def _classify_match_type(med: Medicine, stage: str) -> str:
    """UPPER_SNAKE match-type vocabulary for the resolver's richer contract
    (distinct from the lowercase vocabulary medicine_matching.match_token
    uses, kept for backward compatibility on the existing /prescriptions
    endpoint)."""
    if med.source_version == UNCLASSIFIED_NAME_SOURCE_VERSION:
        # Brand-vs-generic was never established for this row -- say so,
        # rather than implying a confirmed generic identity just because
        # the string happens to live in the generic_name column.
        if stage in ("exact_brand", "exact_alias", "exact_generic"):
            return "EXACT_CATALOG_NAME"
        if stage == "ocr_corrected":
            return "OCR_NORMALIZED_CATALOG_NAME"
        if stage == "fuzzy":
            return "FUZZY_CATALOG_NAME"
        return "UNKNOWN"

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


def _has_identity_corroboration(
    med: Medicine, query_strength: StrengthInfo, query_form: Optional[str]
) -> bool:
    """
    Independent evidence, beyond raw name-string similarity, that this
    specific candidate is genuinely what the query means — never a check
    against a specific medicine name (that would be a special case that
    stops generalizing the moment a new lookalike pair is found); always a
    property of the evidence itself:
      - the query separately stated a strength that agrees with this row's
        own strength, or
      - the query separately stated a dosage form compatible with this
        row's own dosage form.
    Absence of stated strength/form is NOT corroboration either way (it's
    simply no evidence) — see the module-level SAFETY RULE note for why a
    fuzzy stage match with neither of these is capped below HIGH_CONFIDENCE.
    """
    if query_strength.found and med.strength_value is not None and _strength_matches(med, query_strength):
        return True
    if query_form and med.dosage_form and dosage_forms_compatible(query_form, med.dosage_form):
        return True
    return False


def _exact_after_correction(db: Session, variant: str) -> Optional[Medicine]:
    """
    A principled, explainable identity check for one OCR-confusable
    variant: does it land EXACTLY on a real catalog name/alias? This is
    categorically stronger evidence than "this fuzzy-scores similarly" —
    it's the same kind of check stage 1 performs on the raw text, just
    applied to a single, deliberate character correction instead. Used to
    gate HIGH_CONFIDENCE for the fuzzy stage without relying on string
    similarity alone (see module SAFETY RULE).
    """
    hit = repo.find_exact(db, variant)
    return hit.medicine if hit else None


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


def resolve(
    db: Session, input_text: str, ocr_confidence: Optional[float] = None
) -> ResolutionResult:
    """
    Resolve one piece of OCR text (a token, brand phrase, or short line) to
    a catalog entity. Never raises on bad/empty input or on "no match" — the
    caller always gets a well-formed ResolutionResult with an explicit
    status instead.

    `ocr_confidence` (0.0-1.0, optional): the OCR provider's own certainty
    about this text span, when the caller has structured OCR data to supply
    it (see app/services/ocr_geometry.py). Purely informational — exposed
    via `ResolutionResult.overall_confidence` — and never affects the
    MatchStatus decision itself, so omitting it (every pre-existing call
    site does) changes nothing about existing behavior.
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
            ocr_confidence=ocr_confidence,
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    query_strength = parse_strength(normalized)
    query_form = normalize_dosage_form(normalized)
    name_portion = strip_form_words(strip_strength_tokens(normalized)) or normalized

    result = _resolve_normalized(db, input_text or "", normalized, name_portion, query_strength, query_form)
    result.ocr_confidence = ocr_confidence
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
    # Each entry: medicine.id -> (Medicine, score, match_type, identity_verified).
    # `identity_verified` marks a candidate reached via _exact_after_correction
    # (a literal hit after a principled single-character correction) rather
    # than raw fuzzy resemblance — see the module SAFETY RULE. It can only be
    # upgraded (False -> True), never downgraded, if the same medicine is
    # reached both ways across different variants.
    all_scored: dict = {}

    def consider(term: str, stage: str) -> None:
        candidates = repo.search_candidates(db, term)
        for med, base_score in _rerank(term, candidates):
            score = _score_candidate(med, base_score, query_strength, query_form)
            prev = all_scored.get(med.id)
            if prev is None or score > prev[1]:
                verified = prev[3] if prev else False
                all_scored[med.id] = (med, score, _classify_match_type(med, stage), verified)

    def consider_verified_exact(variant: str) -> bool:
        """Returns True if `variant` is a literal catalog hit (and records
        it), so the caller can skip the weaker fuzzy pass for it entirely."""
        med = _exact_after_correction(db, variant)
        if med is None:
            return False
        prev = all_scored.get(med.id)
        if prev is None or not prev[3]:
            all_scored[med.id] = (med, 100.0, _classify_match_type(med, "ocr_corrected"), True)
        return True

    consider(normalized, "fuzzy")
    if name_portion != normalized:
        consider(name_portion, "fuzzy")

    # OCR-confusable variants: check for a literal EXACT hit first (a
    # principled correction landing on a real identity — strong evidence),
    # only falling back to fuzzy-scoring the variant like anything else when
    # it isn't. Never applied to canonical storage/normalization, only here.
    for variant in generate_optical_variants(name_portion)[1:]:
        if not consider_verified_exact(variant):
            consider(variant, "ocr_corrected")
    for variant in tn.generate_ocr_variants(name_portion)[1:]:
        if not consider_verified_exact(variant):
            consider(variant, "ocr_corrected")

    ranked = sorted(all_scored.values(), key=lambda row: row[1], reverse=True)
    candidates = [
        ResolvedCandidate(medicine=med, confidence=round(score, 1), match_type=mtype)
        for med, score, mtype, _verified in ranked[: settings.MEDICINE_CANDIDATE_LIMIT]
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
    top_medicine, _, _, top_verified = ranked[0]
    margin_ok = len(candidates) == 1 or (top.confidence - candidates[1].confidence) >= settings.MEDICINE_AMBIGUITY_MARGIN
    raw_status = _status_for(top.confidence, margin_ok, is_exact=False)

    # Acceptance gate: a high fuzzy score alone is never sufficient identity
    # evidence for HIGH_CONFIDENCE (see module SAFETY RULE). Require either a
    # verified exact-after-correction hit, or independent strength/form
    # corroboration from the query text. Anything else that scored high
    # enough to look like HIGH_CONFIDENCE is downgraded to REVIEW_REQUIRED —
    # never silently accepted, never silently dropped to NO_MATCH either
    # (there IS a real candidate worth a human look, just not enough to
    # assert on its own).
    if raw_status == MatchStatus.HIGH_CONFIDENCE and not (
        top_verified or _has_identity_corroboration(top_medicine, query_strength, query_form)
    ):
        status = MatchStatus.REVIEW_REQUIRED
    else:
        status = raw_status

    if status == MatchStatus.HIGH_CONFIDENCE:
        return ResolutionResult(
            input_text=input_text,
            normalized_text=normalized,
            status=status,
            parsed_strength=query_strength,
            parsed_dosage_form=query_form,
            matched=top,
            candidates=candidates,
        )

    # LOW_CONFIDENCE, REVIEW_REQUIRED (ambiguous or ungated), or NO_MATCH
    # (below the low-confidence floor): never force a pick — a wrong medicine
    # is worse than an unresolved one. `matched` stays None for ALL of these;
    # candidates are still surfaced for LOW_CONFIDENCE/REVIEW_REQUIRED so a
    # human/downstream step has something to review, never NO_MATCH.
    return ResolutionResult(
        input_text=input_text,
        normalized_text=normalized,
        status=status,
        parsed_strength=query_strength,
        parsed_dosage_form=query_form,
        matched=None,
        candidates=candidates if status in (MatchStatus.REVIEW_REQUIRED, MatchStatus.LOW_CONFIDENCE) else [],
    )
