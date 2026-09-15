"""
tests/test_resolve_api.py
-----------------------------
HTTP-level tests for POST /api/v1/medicines/resolve.
"""

import pytest
from fastapi.testclient import TestClient

from app.repositories import medicine_repository as repo


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


def test_resolve_exact_brand(client, db):
    _seed_dolo(db)
    db.commit()
    resp = client.post("/api/v1/medicines/resolve", json={"text": "Dolo 650"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "EXACT"
    assert body["medicine"]["brand_name"] == "Dolo 650"
    assert body["medicine"]["generic_name"] == "Paracetamol"
    assert body["source"] == "legacy_manual"
    assert body["confidence"] == 100.0


def test_resolve_ocr_variant(client, db):
    _seed_dolo(db)
    db.commit()
    resp = client.post("/api/v1/medicines/resolve", json={"text": "D0LO 650"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("HIGH_CONFIDENCE", "EXACT")
    assert body["medicine"]["brand_name"] == "Dolo 650"
    assert body["normalized_text"] == "d0lo 650"


def test_resolve_review_required_never_fabricates(client, db):
    _seed_dolo(db)
    db.commit()
    resp = client.post("/api/v1/medicines/resolve", json={"text": "Dolo 680mg"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] != "EXACT"
    assert body["medicine"] is None  # never silently substitutes


def test_resolve_unknown_medicine(client, db):
    resp = client.post("/api/v1/medicines/resolve", json={"text": "Totallyunknownxyz"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "NO_MATCH"
    assert body["medicine"] is None
    assert body["candidates"] == []


def test_resolve_empty_text_is_validation_error(client):
    resp = client.post("/api/v1/medicines/resolve", json={"text": ""})
    assert resp.status_code == 422  # min_length=1


def test_resolve_missing_field_is_validation_error(client):
    resp = client.post("/api/v1/medicines/resolve", json={})
    assert resp.status_code == 422


def test_resolve_response_carries_parsed_strength(client, db):
    _seed_dolo(db)
    db.commit()
    resp = client.post("/api/v1/medicines/resolve", json={"text": "Paracetamol 650 mg tablet"})
    body = resp.json()
    assert body["parsed_strength"]["found"] is True
    assert body["parsed_dosage_form"] == "tablet"
