"""
tests/test_extract_endpoint.py
----------------------------------
HTTP-level tests for POST /api/v1/prescriptions/extract — the
service-to-service adapter for Enervara's HttpPrescriptionProcessingService.

The document download (`download_document`) and Gemini extraction call are
monkeypatched for determinism — this exercises the real FastAPI app and the
real response transform (app/services/gemini_extraction.to_extraction_out);
only the two genuinely external calls (fetching a URL, calling Gemini) are
stubbed.
"""

import pytest
from fastapi.testclient import TestClient

from app.services.gemini_extraction import (
    GeminiExtraction,
    GeminiProviderError,
    _GeminiMedication,
    _GeminiMetadata,
)


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


def _stub_decode(monkeypatch, prescriptions_module):
    # file_bytes_to_pil_images normally decodes real image bytes; for these
    # tests the "image" is a fake placeholder, so stub decoding too (Gemini
    # extraction itself is already stubbed and never looks at the image).
    monkeypatch.setattr(prescriptions_module, "file_bytes_to_pil_images", lambda content, mime: [object()])


def _stub_gemini(monkeypatch, prescriptions_module, result: GeminiExtraction):
    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", lambda images: result)


def test_extract_success_maps_contract_correctly(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_gemini(monkeypatch, prescriptions_module, GeminiExtraction(
        metadata=_GeminiMetadata(prescriber_name="Ramesh Kumar", patient_name=None),
        medications=[
            _GeminiMedication(
                name="Dolo 650", generic_name="Paracetamol", strength="650mg", form="TABLET",
                frequency="1-0-1 (M-A-N)", duration_text="5 days", duration_days=5,
                confidence=0.95,
            )
        ],
        full_text="Tab Dolo 650 mg 1-0-1 x 5 days",
    ))

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
    # Confidence must be on Enervara's 0-1 scale, not this API's usual 0-100
    # -- Gemini's own confidence is already on that scale, passed straight
    # through (see to_extraction_out()).
    assert 0.0 <= med["confidence"] <= 1.0
    assert med["confidence"] > 0.7  # would NOT be "needs review" on their side
    assert "durationText" in med
    assert med["durationDays"] == 5

    assert body["metadata"]["notes"] is None
    assert body["metadata"]["prescriberName"] == "Ramesh Kumar"
    assert body["metadata"]["patientName"] is None  # never invented -- Gemini didn't state one
    assert body["prescribedDate"] is None
    assert "suggestedSpecialitySlug" not in body or body["suggestedSpecialitySlug"] is None


def test_extract_no_medicines_is_empty_list_not_fabricated(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_gemini(monkeypatch, prescriptions_module, GeminiExtraction(full_text="illegible"))

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
    assert resp.json()["medications"] == []


def test_extract_null_fields_stay_null_not_placeholder_strings(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_gemini(monkeypatch, prescriptions_module, GeminiExtraction(
        medications=[_GeminiMedication(name="Amoxicillin", confidence=0.8)],  # nothing else stated
        full_text="Tab Amoxicillin as directed",
    ))

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
    med = resp.json()["medications"][0]
    assert med["dosage"] is None
    assert med["durationText"] is None
    assert med["durationDays"] is None


def test_extract_invalid_mime_type_is_rejected(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload(mime_type="text/plain"))
    assert resp.status_code == 400


def test_extract_download_failure_is_non_2xx(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    from fastapi import HTTPException

    def _fail_download(url, timeout):
        raise HTTPException(502, "Could not download the document.")

    monkeypatch.setattr(prescriptions_module, "download_document", _fail_download)

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 502


def test_extract_empty_download_is_422(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module, content=b"")
    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 422


def test_extract_provider_unavailable_is_503(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)

    def _fail(images):
        raise GeminiProviderError("Gemini unreachable after retries")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _fail)

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 503


def test_extract_processing_failure_is_502_not_leaked(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)

    def _boom(images):
        raise RuntimeError("SECRET internal detail: db password=hunter2")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _boom)

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 502
    assert "hunter2" not in resp.text


def test_extract_malformed_request_is_422(client):
    resp = client.post("/api/v1/prescriptions/extract", json={"prescription_id": "only-one-field"})
    assert resp.status_code == 422


def test_extract_requires_api_key_when_configured(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    from app.core.config import settings

    monkeypatch.setattr(settings, "RX_PROCESSING_API_KEY", "shared-secret-123")
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_gemini(monkeypatch, prescriptions_module, GeminiExtraction(full_text=""))

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


def test_extract_no_key_required_when_unconfigured(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module
    from app.core.config import settings

    monkeypatch.setattr(settings, "RX_PROCESSING_API_KEY", "")
    _stub_download(monkeypatch, prescriptions_module)
    _stub_decode(monkeypatch, prescriptions_module)
    _stub_gemini(monkeypatch, prescriptions_module, GeminiExtraction(full_text=""))

    resp = client.post("/api/v1/prescriptions/extract", json=_extract_payload())
    assert resp.status_code == 200
