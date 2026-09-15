"""
tests/test_prescription_resolver_integration.py
----------------------------------------------------
Integration tests for POST /api/v1/prescriptions/process now that
parse_medicines() is backed by medicine_resolver.resolve() instead of
medicine_matching.match_token() (see app/services/extraction/__init__.py).

Covers the four required resolver scenarios in the context of the FULL
prescription pipeline (OCR text -> parse_medicines -> HTTP response), not
just the standalone resolver/resolve-endpoint, plus a response-shape
regression check confirming the existing API contract (keys/types) is
unchanged by the swap.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.repositories import medicine_repository as repo


def _tiny_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (20, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(db):
    from app.db.session import get_db
    from app.main import app

    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed_dolo(db):
    repo.upsert_medicine(db, {
        "external_id": "legacy:dolo-650",
        "brand_name": "Dolo 650", "generic_name": "Paracetamol",
        "aliases": ["dolo", "dolo650", "dolo 650", "dolo-650"],
        "strength": "650mg", "strength_value": 650, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })


def _seed_paracetamol_family(db):
    # Multiple products under one generic identity, no single one of them
    # "the" Paracetamol -- used for the ambiguous-generic case.
    _seed_dolo(db)
    repo.upsert_medicine(db, {
        "external_id": "legacy:crocin-650",
        "brand_name": "Crocin 650", "generic_name": "Paracetamol",
        "aliases": ["crocin", "crocin650", "crocin 650"],
        "strength": "650mg", "strength_value": 650, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })
    repo.upsert_medicine(db, {
        "external_id": "legacy:calpol-500",
        "brand_name": "Calpol 500", "generic_name": "Paracetamol",
        "aliases": ["calpol", "calpol500", "calpol 500"],
        "strength": "500mg", "strength_value": 500, "strength_unit": "mg",
        "dosage_form": "tablet", "source": "legacy_manual",
    })


def _process_with_ocr_text(client, prescriptions_module, monkeypatch, text: str) -> dict:
    def _fake_ocr(images):
        return {
            "patient_info": "",
            "medicine_text": text,
            "full_text": text,
            "word_count": len(text.split()),
            "oriented_images": images,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)
    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    return resp.json()


def test_ocr_variant_resolves_to_correct_medicine_in_full_pipeline(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()

    body = _process_with_ocr_text(client, prescriptions_module, monkeypatch, "Tab D0LO 650 1-0-1 x 5 days")
    assert body["success"] is True
    names = {m["name"] for m in body["medicines"]}
    assert "Dolo 650" in names


def test_ambiguous_generic_maps_to_existing_unmatched_behaviour(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_paracetamol_family(db)
    db.commit()

    # "Paracetamol" alone (no strength/brand to disambiguate) is
    # REVIEW_REQUIRED at the resolver level -- must map to the SAME
    # behaviour the pipeline always had for anything it isn't confident
    # about: simply not included in `medicines`, never a forced/arbitrary
    # pick among Dolo/Crocin/Calpol.
    body = _process_with_ocr_text(client, prescriptions_module, monkeypatch, "Tab Paracetamol 1-0-1 x 5 days")
    assert body["success"] is True
    assert body["medicines"] == []


def test_strength_conflict_never_auto_accepted_in_full_pipeline(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()

    # "Dolo 680mg" must NEVER silently resolve to the 650mg product, even
    # once it's flowed all the way through OCR -> parse_medicines -> the
    # HTTP response.
    body = _process_with_ocr_text(client, prescriptions_module, monkeypatch, "Tab Dolo 680mg 1-0-1 x 5 days")
    assert body["success"] is True
    assert body["medicines"] == []  # REVIEW_REQUIRED -> not forced into the result
    for m in body["medicines"]:
        assert m["name"] != "Dolo 650"  # belt-and-suspenders: never present, under any name


def test_fake_medicine_maps_to_existing_unmatched_behaviour(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()

    body = _process_with_ocr_text(
        client, prescriptions_module, monkeypatch, "Tab Xyzcompletelyfakemedicinename 1-0-1 x 5 days"
    )
    assert body["success"] is True
    assert body["medicines"] == []


def test_response_shape_unchanged_by_the_resolver_swap(client, db, monkeypatch):
    """
    Regression check: every key the API contract has always had is still
    present, with the same types, after parse_medicines() switched from
    medicine_matching.match_token() to medicine_resolver.resolve().
    """
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()

    body = _process_with_ocr_text(client, prescriptions_module, monkeypatch, "Tab Dolo 650 1-0-1 x 5 days")

    # Top-level shape, unchanged.
    for key in ("success", "filename", "patient_info", "medicines", "image_preview", "processing_time_sec", "debug"):
        assert key in body, key
    assert isinstance(body["medicines"], list)
    assert isinstance(body["debug"], dict)
    for key in ("full_text", "medicine_text", "word_count"):
        assert key in body["debug"], key

    # Per-medicine shape, unchanged (back-compat fields + structured fields).
    assert len(body["medicines"]) == 1
    med = body["medicines"][0]
    for key in (
        "name", "confidence", "dosage", "frequency", "duration", "raw_line",
        "id", "generic_name", "brand_name", "strength", "dosage_form",
        "route", "manufacturer", "match_type", "source",
    ):
        assert key in med, key
    assert isinstance(med["name"], str)
    assert isinstance(med["confidence"], (int, float))
    assert isinstance(med["match_type"], str)


def test_multiple_confident_medicines_still_all_returned(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()

    text = "Tab Dolo 650 1-0-1 x 5 days\nTab Amoxicillin 500mg BD 7 days"
    body = _process_with_ocr_text(client, prescriptions_module, monkeypatch, text)
    names = {m["name"] for m in body["medicines"]}
    assert names == {"Dolo 650", "Amoxicillin"}
