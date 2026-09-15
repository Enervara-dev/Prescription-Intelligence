"""
tests/test_text_normalization.py
------------------------------------
Pure unit tests (no DB) for deterministic text normalization.
"""

from app.services.text_normalization import generate_ocr_variants, normalize_text, split_alnum_boundaries


def test_normalize_comparable_forms():
    # All four spellings from the spec must normalize to the same string.
    assert normalize_text("Dolo 650") == "dolo 650"
    assert normalize_text("Dolo-650") == "dolo 650"
    assert normalize_text("DOLO650") == "dolo 650"  # split_alnum_boundaries splits the clean trailing digits
    assert normalize_text("Dolo650") == "dolo 650"


def test_normalize_does_not_correct_ocr_confusables():
    # '0' is NOT blindly turned into 'o' at normalization time -- that's
    # deferred to candidate generation (see generate_ocr_variants).
    assert normalize_text("D0LO 650") == "d0lo 650"


def test_normalize_whitespace_and_punctuation():
    assert normalize_text("  Dolo   650  ") == "dolo 650"
    assert normalize_text("Dolo, 650!") == "dolo 650"


def test_normalize_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_split_alnum_boundaries_only_splits_clean_trailing_digits():
    assert split_alnum_boundaries("dolo650") == "dolo 650"
    assert split_alnum_boundaries("pan40") == "pan 40"
    # A digit embedded mid-word (likely OCR noise, not a genuine trailing
    # strength) must NOT be split -- that would make it worse, not better.
    assert split_alnum_boundaries("d0lo") == "d0lo"
    assert split_alnum_boundaries("d0lo650") == "d0lo650"  # not a clean letters+digits token


def test_generate_ocr_variants_covers_target_confusions():
    variants = generate_ocr_variants("d0lo")
    assert "dolo" in variants  # 0 -> o
    variants2 = generate_ocr_variants("cr0cln")
    assert "crocln" in variants2  # 0 -> o
    assert "cr0cin" in variants2  # l -> i


def test_generate_ocr_variants_includes_original_first():
    variants = generate_ocr_variants("dolo")
    assert variants[0] == "dolo"
