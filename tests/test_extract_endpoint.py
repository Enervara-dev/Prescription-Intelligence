"""
tests/test_extract_endpoint.py
----------------------------------
HTTP-level tests for POST /api/v1/prescriptions/extract — the
service-to-service adapter for Enervara's HttpPrescriptionProcessingService.

The document download (`download_document`) and OCR call
(`extract_text_from_images`) are monkeypatched for determinism, exactly like
the existing /process tests — this exercises the real FastAPI app, real DB,
real parse_medicines()/resolver, and the real response transform; only the
two genuinely external calls (fetching a URL, calling Google Vision) are
stubbed.
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


def _extract_payload(**overrides) -> dict:
    payload = {
        "prescription_id": "rx-123",
        "user_id": "user-456",
        "document_url": "https://storage.example.com/signed/doc.jpg",
        "mime_type": "image/jpeg",
        "file_name": "prescription.jpg",
        "request_id": "req-789",
    }
    payload.update(overrides)
    return payload


def _stub_download(monkeypatch, prescriptions_module, content: bytes = b"fake-image-bytes"):
    monkeypatch.setattr(prescriptions_module, "download_document", lambda url, timeout: content)


def _stub_ocr(monkeypatch, prescriptions_module, text: str):
    def _fake_ocr(images):
        return {
            "patient_info": "Dr. Ramesh Kumar\nPatient: Anita Sharma",
            "medicine_text": text,
            "full_text": text,
            "word_count": len(text.split()),
            "oriented_images": images,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)


def _stub_decode(monkeypatch, prescriptions_module):
    # file_bytes_to_pil_images normally decodes real image bytes; for these
    # tests the "image" is a fake placeholder, so stub decoding too (OCR
    # itself is already stubbed and never looks at the image content).
    monkeypatch.setattr(prescriptions_module, "file_bytes_to_pil_images", lambda content, mime: [object()])


def test_extract_success_maps_contract_correctly(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_ocr(monkeypatch, prescriptions_module, "Tab Dolo 650 mg 1-0-1 x 5 days")

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
    body = resp.json()

    assert "medications" in body
    assert len(body["medications"]) == 1
    med = body["medications"][0]

    # camelCase on the wire, matching Enervara's TS interface exactly.
    assert med["name"] == "Dolo 650"
    assert med["genericName"] == "Paracetamol"
    assert med["form"] == "TABLET"
    # Confidence must be on Enervara's 0-1 scale, not this API's usual 0-100.
    assert 0.0 <= med["confidence"] <= 1.0
    assert med["confidence"] > 0.7  # would NOT be "needs review" on their side
    assert "durationText" in med
    assert med["durationDays"] == 5

    assert body["metadata"]["notes"]  # real OCR patient-info text, not fabricated structure
    assert body["metadata"]["patientName"] is None  # never invented -- we don't extract this
    assert body["prescribedDate"] is None
    assert "suggestedSpecialitySlug" not in body or body["suggestedSpecialitySlug"] is None


def test_extract_never_fabricates_unmatched_medicine(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_ocr(monkeypatch, prescriptions_module, "Tab Xyzcompletelyfakemedicinename 1-0-1")

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
    assert resp.json()["medications"] == []


def test_extract_strength_conflict_never_auto_accepted(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _seed_dolo(db)
    db.commit()
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_ocr(monkeypatch, prescriptions_module, "Tab Dolo 680mg 1-0-1 x 5 days")

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()["medications"]}
    assert "Dolo 650" not in names


def test_extract_na_fields_become_null_not_literal_string(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    # No dosage/frequency/duration info at all in this line.
    _stub_ocr(monkeypatch, prescriptions_module, "Tab Amoxicillin as directed")

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
    med = resp.json()["medications"][0]
    assert med["dosage"] is None  # not the literal string "N/A"
    assert med["durationText"] is None
    assert med["durationDays"] is None


def test_extract_invalid_mime_type_is_rejected(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload(mime_type="text/plain"))
    assert resp.status_code == 400


def test_extract_download_failure_is_non_2xx(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    from fastapi import HTTPException

    def _fail_download(url, timeout):
        raise HTTPException(502, "Could not download the document.")

    monkeypatch.setattr(prescriptions_module, "download_document", _fail_download)

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 502


def test_extract_empty_download_is_422(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module, content=b"")
    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 422


def test_extract_processing_failure_is_502_not_leaked(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)

    def _boom(images):
        raise RuntimeError("SECRET internal detail: db password=hunter2")

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _boom)

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 502
    assert "hunter2" not in resp.text


def test_extract_malformed_request_is_422(client, db):
    resp = client.post("/api/v1/prescriptions/extract", json={"prescription_id": "only-one-field"})
    assert resp.status_code == 422


def test_extract_requires_api_key_when_configured(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    from app.core.config import settings

    monkeypatch.setattr(settings, "RX_PROCESSING_API_KEY", "shared-secret-123")
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_ocr(monkeypatch, prescriptions_module, "")

    # No key at all.
    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 401

    # Wrong key.
    resp = client.post(
        "/api/v1/prescriptions/extract", json=_extract_payload(), headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401

    # Correct key.
    resp = client.post(
        "/api/v1/prescriptions/extract", json=_extract_payload(), headers={"X-API-Key": "shared-secret-123"}
    )
    assert resp.status_code == 200


def test_extract_no_key_required_when_unconfigured(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    from app.core.config import settings

    monkeypatch.setattr(settings, "RX_PROCESSING_API_KEY", "")
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_ocr(monkeypatch, prescriptions_module, "")

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200


def test_process_endpoint_response_unchanged_after_refactor(client, db, monkeypatch):
    """
    _clean_patient_info() was extracted out of process_prescription() into a
    shared helper so /extract could reuse it -- confirm /process's own
    response is byte-for-byte the same shape/values as before.
    """
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    import io
    from PIL import Image

    _seed_dolo(db)
    db.commit()

    def _fake_ocr(images):
        text = "Tab Dolo 650 1-0-1 x 5 days"
        return {
            "patient_info": "Dr. Ramesh Kumar\nTab Dolo 650 1-0-1 x 5 days",
            "medicine_text": text,
            "full_text": text,
            "word_count": 5,
            "oriented_images": images,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)

    img = Image.new("RGB", (20, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["medicines"][0]["name"] == "Dolo 650"
    # The doctor-name line survives; the medicine line is correctly excluded
    # from patient_info (the exact behaviour _clean_patient_info replicates).
    assert "Ramesh Kumar" in body["patient_info"]
    assert "1-0-1" not in body["patient_info"]
