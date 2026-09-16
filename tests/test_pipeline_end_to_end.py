"""
tests/test_pipeline_end_to_end.py
-------------------------------------
Full-pipeline test: synthetic OCR words -> geometry reconstruction -> OCR
lines -> candidate extraction -> normalization -> medicine resolver ->
dosage/frequency/duration -> final API response (through the real FastAPI
app and Pydantic response models, not just the internal dict).

Unlike every other test in this suite, this one builds its OCR input at the
WORD level (OCRWord, with real bounding boxes and per-word confidence) and
runs it through group_words_into_lines() for real, rather than starting
from an already-flattened string. The point is to catch bugs where each
stage passes its own unit tests but information is silently lost crossing
the boundary between stages -- exactly the class of bug the OCR/matching
forensic audit found (confidence captured then discarded, spatial structure
flattened too early, medicine-name-to-instruction association done
globally instead of per-span).
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.repositories import medicine_repository as repo
from app.services.extraction import parse_medicines
from app.services.ocr_geometry import OCRWord, group_words_into_lines


def _word(text, x, y, width=None, height=20, confidence=0.95):
    return OCRWord(text=text, confidence=confidence, x=x, y=y, width=width or len(text) * 10, height=height)


def _tiny_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (20, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def two_row_words():
    """
    Two distinct medicine rows, built at the WORD level with real bounding
    boxes -- one word ("650") deliberately reported at a slightly different
    y than its row-mates (ordinary OCR jitter), and the two rows separated
    by only 15px (the exact gap verified during the audit to merge under
    the old fixed-18px-threshold implementation). One word on the second
    row is given a poor confidence (0.35) to verify that signal survives.
    """
    return [
        _word("Dolo", x=10, y=100, height=20, confidence=0.97),
        _word("650", x=60, y=102, height=20, confidence=0.96),
        _word("1-0-1", x=120, y=101, height=20, confidence=0.95),
        _word("5", x=220, y=100, height=20, confidence=0.94),
        _word("days", x=250, y=101, height=20, confidence=0.93),
        _word("Pan", x=10, y=115, height=20, confidence=0.35),  # hard-to-read
        _word("40", x=60, y=116, height=20, confidence=0.90),
        _word("1-0-0", x=100, y=115, height=20, confidence=0.90),
        _word("7", x=200, y=116, height=20, confidence=0.90),
        _word("days", x=220, y=115, height=20, confidence=0.90),
    ]


@pytest.fixture()
def catalog(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:dolo-650",
        "brand_name": "Dolo 650", "generic_name": "Paracetamol",
        "aliases": ["dolo", "dolo650", "dolo 650", "dolo-650"],
        "strength": "650mg", "strength_value": 650, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {
        "external_id": "legacy:pan-40",
        "brand_name": "Pan 40", "generic_name": "Pantoprazole",
        "aliases": ["pan", "pan40", "pan 40", "pan-40"],
        "strength": "40mg", "strength_value": 40, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })
    db.commit()
    return db


def test_words_to_lines_to_medicines_full_chain(catalog, two_row_words):
    # Stage 1: geometry reconstruction -- must NOT merge the two rows
    # despite the 15px gap (the verified OCR bug this whole effort started
    # from), and must order "650" correctly despite its y-jitter.
    lines = group_words_into_lines(two_row_words)
    assert len(lines) == 2, [l.text for l in lines]
    assert lines[0].text == "Dolo 650 1-0-1 5 days"
    assert lines[1].text == "Pan 40 1-0-0 7 days"

    # Stage 2: candidate extraction + resolution, fed the STRUCTURED lines
    # (not a pre-flattened string) -- this is what actually exercises the
    # OCR-confidence plumbing end to end.
    results = parse_medicines("\n".join(l.text for l in lines), catalog, ocr_lines=lines)

    names = {r["name"]: r for r in results}
    assert set(names) == {"Dolo 650", "Pan 40"}

    dolo = names["Dolo 650"]
    pan = names["Pan 40"]

    # Medicine <-> instruction association survived per-span, not globally.
    assert dolo["frequency"] == "1-0-1 (M-A-N)"
    assert dolo["duration"] == "5 days"
    assert pan["frequency"] == "1-0-0 (M-A-N)"
    assert pan["duration"] == "7 days"
    assert dolo["duration"] != pan["duration"]  # never cross-assigned

    # Strength was never accidentally placed into dosage.
    assert dolo["dosage"] != dolo["strength"]
    assert dolo["dosage"] == "N/A"  # no genuine per-dose amount stated
    assert dolo["strength"] == "650mg"

    # OCR confidence survived the entire pipeline: the low-confidence "Pan"
    # word's line-level confidence (mean of its row, pulling the average
    # down) is reflected in Pan's ocr_confidence and dampens its
    # overall_confidence below its pure match confidence -- while Dolo's
    # clean row (no weak words) shows no such dampening.
    assert dolo["ocr_confidence"] is not None
    assert pan["ocr_confidence"] is not None
    assert pan["ocr_confidence"] < dolo["ocr_confidence"]
    assert pan["overall_confidence"] < pan["confidence"]
    assert pan["overall_confidence"] < dolo["overall_confidence"]


def test_unresolved_medicine_stays_unresolved_through_full_chain(catalog):
    """A word with no catalog match anywhere must never surface as a
    medicine, however it's threaded through the structured-line path."""
    words = [
        _word("Xyzcompletelyfakemedicine", x=10, y=100, height=20, confidence=0.9),
        _word("1-0-1", x=300, y=100, height=20, confidence=0.9),
    ]
    lines = group_words_into_lines(words)
    results = parse_medicines(lines[0].text, catalog, ocr_lines=lines)
    assert results == []


def test_final_api_shape_compatible_with_structured_ocr_input(catalog, monkeypatch, two_row_words):
    """
    Runs the real FastAPI app (not just parse_medicines() in isolation) with
    a stubbed OCR call returning structured OCRLine data, confirming the
    Pydantic response model still validates and the existing API shape
    (name/confidence/dosage/frequency/duration/strength/...) is unchanged.
    """
    from app.db.session import get_db
    from app.main import app
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    lines = group_words_into_lines(two_row_words)

    def _fake_ocr(images):
        text = "\n".join(l.text for l in lines)
        return {
            "patient_info": "",
            "medicine_text": text,
            "full_text": text,
            "word_count": sum(len(l.words) for l in lines),
            "oriented_images": images,
            "medicine_ocr_lines": lines,
            "patient_ocr_lines": [],
            "all_ocr_lines": lines,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)

    def _override_get_db():
        yield catalog

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/prescriptions/process",
                files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    names = {m["name"] for m in body["medicines"]}
    assert names == {"Dolo 650", "Pan 40"}
    for med in body["medicines"]:
        for key in (
            "name", "confidence", "dosage", "frequency", "duration", "raw_line",
            "id", "generic_name", "brand_name", "strength", "dosage_form",
            "route", "manufacturer", "match_type", "source",
        ):
            assert key in med, key
