"""
services/medicine_matching.py
--------------------------------
OCR-aware medicine matching against the Postgres-backed catalog.

Retrieval is index-backed (pg_trgm via medicine_repository.search_candidates),
not a linear scan. The OCR-specific correction logic that used to run over an
in-memory Python list — optical-confusable substitutions (m<->n, rn<->m,
1<->l, 0<->o, ...) plus RapidFuzz re-ranking with a prefix bonus and a
length-disparity penalty — is preserved here, just applied to the small
candidate set Postgres returns instead of the whole catalog.
"""

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.medicine import Medicine
from app.repositories import medicine_repository as repo

# Optical substitutions common in doctor-handwriting OCR misreads. Same table
# as the previous in-memory matcher.
_OPTICAL_REPLACEMENTS = [
    ("m", "n"), ("n", "m"), ("1", "l"), ("0", "o"), ("5", "s"),
    ("q", "g"), ("cl", "d"), ("rn", "m"), ("vv", "w"),
    ("i", "y"), ("y", "i"), ("k", "b"), ("b", "k"),
]

MatchType = str  # "exact" | "alias" | "ocr_corrected" | "fuzzy" | "ambiguous" | "unresolved"


@dataclass
class MedicineMatch:
    medicine: Optional[Medicine]
    confidence: float  # 0-100
    match_type: MatchType


def generate_optical_variants(token: str) -> list[str]:
    """Same optical-confusable variant generation the old in-memory matcher used."""
    variants = [token]
    for old, new in _OPTICAL_REPLACEMENTS:
        if old in token:
            variants.append(token.replace(old, new))
    return variants


def _score_against_name(term: str, name: str) -> float:
    wr = float(fuzz.WRatio(term, name))
    token_ratio = float(fuzz.token_set_ratio(term, name))
    score = max(wr, token_ratio)

    len_diff = abs(len(term) - len(name))
    if len_diff >= 3:
        score -= len_diff * 4

    prefix_len = min(4, len(term), len(name))
    if prefix_len and term[:prefix_len] == name[:prefix_len] and score >= 70.0:
        score = min(100.0, score + 10.0)

    return max(0.0, min(100.0, score))


def _rerank(term: str, candidates: list[repo.Candidate]) -> list[tuple[Medicine, float]]:
    """
    Re-score a small candidate pool with the same RapidFuzz logic the old
    matcher used (WRatio / token_set_ratio, prefix bonus, length-disparity
    penalty), rather than trusting the raw trigram similarity as the final
    score. Returns (medicine, score) sorted best-first.

    Scores against BOTH brand_name and generic_name (whichever are present)
    and keeps the best — NOT just generic_name. A query that looks like a
    brand (e.g. "D0LO 650") must be compared against brand_name="Dolo 650",
    not silently against generic_name="Paracetamol": scoring generic-name-
    only was a real bug (verified: WRatio("d0lo 650", "paracetamol") == 10.5
    vs WRatio("d0lo 650", "dolo 650") == 87.5 — comparing against the wrong
    string entirely missed an otherwise-obvious OCR-corrupted brand match).
    """
    scored: list[tuple[Medicine, float]] = []
    for cand in candidates:
        med = cand.medicine
        names = {n.lower() for n in (med.generic_name, med.brand_name) if n}
        if not names:
            continue

        best = max(_score_against_name(term, name) for name in names)
        scored.append((med, best))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def match_token(db: Session, token: str) -> MedicineMatch:
    """
    Resolve a single OCR token/bigram against the catalog.

    Order of attempts, cheapest/most-certain first:
      1. Exact match on the raw token (generic/brand name)          -> "exact"
      2. Exact match on the raw token against an alias               -> "alias"
      3. Fuzzy (indexed) search on the raw token                     -> "fuzzy"
      4. Fuzzy (indexed) search on each optical-confusable variant    -> "ocr_corrected"
      Ambiguity check applies to steps 3-4: if the best two candidates
      are within MEDICINE_AMBIGUITY_MARGIN of each other, report
      "ambiguous" rather than forcing a pick.
    """
    token_norm = token.strip().lower()
    if not token_norm:
        return MedicineMatch(medicine=None, confidence=0.0, match_type="unresolved")

    exact = repo.find_exact(db, token_norm)
    if exact:
        alias_hit = repo.find_alias_exact(db, token_norm)
        match_type = "alias" if alias_hit and alias_hit.medicine.id == exact.medicine.id else "exact"
        return MedicineMatch(medicine=exact.medicine, confidence=100.0, match_type=match_type)

    threshold = settings.MEDICINE_MATCH_THRESHOLD
    margin = settings.MEDICINE_AMBIGUITY_MARGIN

    def best_from(term: str) -> list[tuple[Medicine, float]]:
        candidates = repo.search_candidates(db, term)
        return _rerank(term, candidates)

    # 1. Raw token, fuzzy.
    ranked = best_from(token_norm)
    if ranked and ranked[0][1] >= threshold:
        if len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < margin:
            return MedicineMatch(medicine=None, confidence=ranked[0][1], match_type="ambiguous")
        return MedicineMatch(medicine=ranked[0][0], confidence=ranked[0][1], match_type="fuzzy")

    # 2. Optical-confusable variants.
    best_variant_result: Optional[tuple[Medicine, float]] = None
    best_variant_ranked: list[tuple[Medicine, float]] = []
    for variant in generate_optical_variants(token_norm)[1:]:  # skip index 0 == raw token, already tried
        variant_ranked = best_from(variant)
        if variant_ranked and (best_variant_result is None or variant_ranked[0][1] > best_variant_result[1]):
            best_variant_result = variant_ranked[0]
            best_variant_ranked = variant_ranked

    if best_variant_result and best_variant_result[1] >= threshold:
        if len(best_variant_ranked) > 1 and (best_variant_result[1] - best_variant_ranked[1][1]) < margin:
            return MedicineMatch(medicine=None, confidence=best_variant_result[1], match_type="ambiguous")
        return MedicineMatch(medicine=best_variant_result[0], confidence=best_variant_result[1], match_type="ocr_corrected")

    # Nothing cleared the threshold — do not force a match.
    best_score = ranked[0][1] if ranked else (best_variant_result[1] if best_variant_result else 0.0)
    return MedicineMatch(medicine=None, confidence=best_score, match_type="unresolved")


def search_many(db: Session, term: str, limit: int = 10) -> list[MedicineMatch]:
    """
    Free-text lookup used by the /medicines/search endpoint: returns multiple
    ranked candidates (unlike match_token, which resolves to at most one).
    An exact/alias hit is always ranked first; the rest are indexed fuzzy
    candidates re-ranked with RapidFuzz.
    """
    term_norm = term.strip().lower()
    if not term_norm:
        return []

    results: list[MedicineMatch] = []
    seen_ids = set()

    exact = repo.find_exact(db, term_norm)
    if exact:
        alias_hit = repo.find_alias_exact(db, term_norm)
        match_type = "alias" if alias_hit and alias_hit.medicine.id == exact.medicine.id else "exact"
        results.append(MedicineMatch(medicine=exact.medicine, confidence=100.0, match_type=match_type))
        seen_ids.add(exact.medicine.id)

    candidates = repo.search_candidates(db, term_norm, limit=limit)
    for med, score in _rerank(term_norm, candidates):
        if med.id in seen_ids or len(results) >= limit:
            continue
        results.append(MedicineMatch(medicine=med, confidence=score, match_type="fuzzy"))
        seen_ids.add(med.id)

    return results[:limit]
