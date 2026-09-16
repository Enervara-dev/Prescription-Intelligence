"""
services/ocr_geometry.py
---------------------------
Pure, OCR-provider-agnostic spatial reconstruction. Groups OCR words into
lines using RELATIVE vertical overlap (proportional to word height), not an
absolute pixel constant -- so the same grouping rule behaves consistently at
any image resolution/DPI. A fixed-pixel threshold (the previous approach)
silently merges two real rows on a lower-resolution image and can silently
split one real row on a higher-resolution scan of the exact same document --
verified during the OCR audit. This module is DB-free and provider-agnostic
so it can be unit-tested directly, without mocking Google Vision.
"""

from dataclasses import dataclass


@dataclass
class OCRWord:
    """One word as reported by the OCR provider, with its full bounding box
    (not just the top-left corner) and its own confidence -- both required
    for resolution-adaptive row grouping and for carrying OCR certainty
    downstream instead of discarding it at ingestion."""

    text: str
    confidence: float  # 0.0-1.0, as reported by the OCR provider
    x: int
    y: int
    width: int
    height: int

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def x2(self) -> int:
        return self.x + self.width


@dataclass
class OCRLine:
    """One reconstructed line: the words that were grouped into it (in
    correct left-to-right order), the joined text, its union bounding box,
    and two DISTINCT confidence signals -- the mean (overall certainty) and
    the minimum (whether any single word on this line was hard to read, a
    signal the mean can hide)."""

    words: list[OCRWord]
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    min_confidence: float

    @property
    def y2(self) -> int:
        return self.y + self.height


# A word must share at least this fraction of its vertical extent with a
# row's running band to belong to that row. Expressed relative to word
# HEIGHT rather than an absolute pixel count, so the same ratio groups
# correctly whether the source scan is 150dpi or 600dpi, or a downscaled
# phone photo of the same page.
_ROW_OVERLAP_RATIO = 0.4
# Guards against a zero/negative-height word from a degenerate bounding box
# (never actually divide by zero when computing the overlap ratio).
_MIN_WORD_DIM = 1


def _vertical_overlap(a_y: int, a_y2: int, b_y: int, b_y2: int) -> int:
    return max(0, min(a_y2, b_y2) - max(a_y, b_y))


def group_words_into_lines(words: list[OCRWord]) -> list[OCRLine]:
    """
    Cluster words into lines by vertical position (relative overlap, not a
    fixed pixel gap), then sort each line's words left-to-right by x --
    decoupled from the initial y-sort used only to walk words top-to-bottom.
    This is deliberate: per-word y-jitter within one printed row (ordinary
    OCR bounding-box imprecision) previously scrambled reading order because
    the old implementation sorted by (y, x) globally and read words off in
    that same order with no row-scoped re-sort. Grouping first, then sorting
    x within each group, fixes that independently of the grouping decision
    itself.

    Never merges two rows just because their y-values are numerically close
    -- the test is vertical overlap relative to word height, so a genuinely
    different row stays separate even when the raw pixel gap between them is
    small (which happens routinely on lower-resolution images).
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w.y, w.x))

    rows: list[list[OCRWord]] = []
    current: list[OCRWord] = [ordered[0]]
    band_y, band_y2 = ordered[0].y, ordered[0].y2

    for w in ordered[1:]:
        word_h = max(w.height, _MIN_WORD_DIM)
        band_h = max(band_y2 - band_y, _MIN_WORD_DIM)
        reference_h = min(word_h, band_h)
        overlap = _vertical_overlap(w.y, w.y2, band_y, band_y2)
        if overlap >= _ROW_OVERLAP_RATIO * reference_h:
            current.append(w)
            band_y = min(band_y, w.y)
            band_y2 = max(band_y2, w.y2)
        else:
            rows.append(current)
            current = [w]
            band_y, band_y2 = w.y, w.y2
    rows.append(current)

    lines: list[OCRLine] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda w: w.x)
        confidences = [w.confidence for w in row_sorted]
        xs, ys = [w.x for w in row_sorted], [w.y for w in row_sorted]
        x2s, y2s = [w.x2 for w in row_sorted], [w.y2 for w in row_sorted]
        lines.append(
            OCRLine(
                words=row_sorted,
                text=" ".join(w.text for w in row_sorted),
                x=min(xs),
                y=min(ys),
                width=max(x2s) - min(xs),
                height=max(y2s) - min(ys),
                confidence=sum(confidences) / len(confidences),
                min_confidence=min(confidences),
            )
        )
    return lines
