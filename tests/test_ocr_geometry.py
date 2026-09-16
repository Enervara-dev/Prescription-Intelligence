"""
tests/test_ocr_geometry.py
------------------------------
Unit tests for the resolution-adaptive OCR line-grouping algorithm
(app/services/ocr_geometry.py). No DB, no Google Vision -- pure geometry.

These directly target the two verified bugs in the previous fixed-18px-
threshold implementation: (1) two real rows merging into one on a lower-
resolution image, and (2) word order within a line scrambling due to
ordinary per-word y-jitter.
"""

from app.services.ocr_geometry import OCRWord, group_words_into_lines


def _word(text, x, y, width=None, height=20, confidence=0.95):
    # Default width is just enough to be nonzero/plausible; grouping only
    # depends on x for ordering (not width), and on y/height for row tests.
    return OCRWord(text=text, confidence=confidence, x=x, y=y, width=width or len(text) * 10, height=height)


def test_empty_input_returns_no_lines():
    assert group_words_into_lines([]) == []


def test_two_rows_with_normal_gap_stay_separate():
    words = [
        _word("DOLO", x=10, y=100, height=20),
        _word("650", x=60, y=100, height=20),
        _word("AMOXICILLIN", x=10, y=140, height=20),  # 20px gap: no vertical overlap at all
        _word("500mg", x=200, y=140, height=20),
    ]
    lines = group_words_into_lines(words)
    assert len(lines) == 2
    assert lines[0].text == "DOLO 650"
    assert lines[1].text == "AMOXICILLIN 500mg"


def test_small_absolute_gap_does_not_merge_two_real_rows_on_a_low_res_image():
    """
    The previous fixed-18px-threshold implementation merged two distinct
    medicine rows into one garbled line when they were only 15px apart --
    verified during the audit. This must no longer happen: at height=20,
    a word starting 15px below the previous row's top has only
    (20-15)=5px of vertical overlap with it -- 25% of its own height, well
    under the 40% overlap ratio required to count as the same row.
    """
    words = [
        _word("DOLO", x=10, y=100, height=20),
        _word("650", x=60, y=102, height=20),
        _word("1-0-1", x=120, y=101, height=20),
        _word("AMOXICILLIN", x=10, y=115, height=20),
        _word("500mg", x=90, y=116, height=20),
    ]
    lines = group_words_into_lines(words)
    assert len(lines) == 2, [l.text for l in lines]
    assert "AMOXICILLIN" not in lines[0].text
    assert "DOLO" not in lines[1].text


def test_same_row_word_order_survives_realistic_y_jitter():
    """
    Ordinary per-word bounding-box y-jitter within ONE printed row must not
    scramble left-to-right reading order. Verified bug: the previous
    implementation sorted purely by (y, x), so a word positioned further
    left but reported at a slightly higher y than its neighbour could be
    read out of order.
    """
    words = [
        _word("DOLO", x=10, y=100, height=20),   # leftmost, top y
        _word("650", x=60, y=102, height=20),    # middle, slightly lower y
        _word("1-0-1", x=120, y=101, height=20),  # rightmost, in between
    ]
    lines = group_words_into_lines(words)
    assert len(lines) == 1
    assert lines[0].text == "DOLO 650 1-0-1"


def test_grouping_is_resolution_invariant():
    """
    Scaling every coordinate and dimension by the same factor (simulating a
    higher-DPI scan of the identical page) must not change how many lines
    are produced, or which words end up in which line -- the whole point of
    using a relative overlap ratio instead of an absolute pixel constant.
    """
    base = [
        ("DOLO", 10, 100, 20),
        ("650", 60, 102, 20),
        ("1-0-1", 120, 101, 20),
        ("AMOXICILLIN", 10, 115, 20),
        ("500mg", 90, 116, 20),
    ]
    low_res = [_word(t, x=x, y=y, height=h) for t, x, y, h in base]
    scale = 4
    high_res = [_word(t, x=x * scale, y=y * scale, height=h * scale) for t, x, y, h in base]

    low_lines = group_words_into_lines(low_res)
    high_lines = group_words_into_lines(high_res)

    assert [l.text for l in low_lines] == [l.text for l in high_lines]


def test_confidence_signals_are_distinct():
    words = [
        _word("DOLO", x=10, y=100, confidence=0.95),
        _word("650", x=60, y=100, confidence=0.40),  # one badly-read word
    ]
    lines = group_words_into_lines(words)
    assert len(lines) == 1
    line = lines[0]
    assert line.min_confidence == 0.40
    assert line.confidence == (0.95 + 0.40) / 2
    assert line.min_confidence < line.confidence  # mean hides the weak word; min doesn't


def test_bounding_box_is_the_union_of_its_words():
    words = [
        _word("DOLO", x=10, y=100, width=40, height=20),
        _word("650", x=60, y=105, width=30, height=15),
    ]
    line = group_words_into_lines(words)[0]
    assert line.x == 10
    assert line.y == 100
    assert line.width == (60 + 30) - 10
    assert line.height == (105 + 15) - 100
