"""
tests/test_api.py
--------------------
HTTP-level tests for POST /api/v1/prescriptions/process: request
validation, status codes, error handling, and the response contract.
Gemini extraction (app/services/gemini_extraction.py) is monkeypatched for
determinism, no external cost/credentials, and no live-API flakiness —
these test the ENDPOINT's contract (auth, validation, error mapping,
response shape), not Gemini's actual extraction quality.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services.gemini_extraction import (
    GeminiExtraction,
    GeminiProviderError,
    _GeminiMedication,
    _GeminiMetadata,
)


def _tiny_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (20, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _tiny_pdf_bytes(pages: int = 1) -> bytes:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    buf = doc.tobytes()
    doc.close()
    return buf


@pytest.fixture()
def client(db):
    """A TestClient wired to the same DB session/transaction as the `db` fixture."""
    from app.db.session import get_db
    from app.main import app

    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _stub_gemini(monkeypatch, prescriptions_module, result: GeminiExtraction):
    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", lambda images: result)


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database_connected"] is True


def test_missing_file_is_422(client):
    resp = client.post("/api/v1/prescriptions/process")
    assert resp.status_code == 422  # FastAPI request-validation error


def test_invalid_file_type_is_400(client):
    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_empty_file_is_400(client):
    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_corrupt_image_bytes_is_422(client):
    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("broken.jpg", b"not-a-real-jpeg-just-garbage-bytes", "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "Could not decode image" in resp.json()["detail"]


def test_oversized_file_is_413(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)  # anything nonzero is "too big"
    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 413


def test_pdf_with_too_many_pages_is_422(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 1)
    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("rx.pdf", _tiny_pdf_bytes(pages=3), "application/pdf")},
    )
    assert resp.status_code == 422
    assert "too many pages" in resp.json()["detail"].lower()


def test_internal_error_never_leaks_exception_detail_to_caller(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _boom(images):
        raise RuntimeError("SECRET internal detail: db password=hunter2 host=internal-db.local")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _boom)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert "hunter2" not in body["detail"]
    assert "internal-db.local" not in body["detail"]
    assert "SECRET" not in body["detail"]


def test_medicine_extracted_end_to_end(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    result = GeminiExtraction(
        metadata=_GeminiMetadata(prescriber_name="Ramesh Kumar"),
        medications=[
            _GeminiMedication(
                name="Dolo 650", generic_name="Paracetamol", strength="650mg",
                frequency="1-0-1 (M-A-N)", duration_text="5 days", duration_days=5,
                confidence=0.95,
            )
        ],
        full_text="Tab Dolo 650 1-0-1 x 5 days",
    )
    _stub_gemini(monkeypatch, prescriptions_module, result)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["medicines"]) == 1
    med = body["medicines"][0]
    assert med["name"] == "Dolo 650"
    assert med["match_type"] == "LLM_EXTRACTED"
    assert med["source"] == "gemini"
    assert med["duration"] == "5 days"
    assert med["confidence"] == 95.0  # Gemini's 0-1 scale converted to this API's 0-100


def test_medicine_not_found_returns_empty_list_not_error(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    result = GeminiExtraction(full_text="Completely illegible / no medicines found")
    _stub_gemini(monkeypatch, prescriptions_module, result)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["medicines"] == []


def test_multiple_medicines_all_returned(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    result = GeminiExtraction(
        medications=[
            _GeminiMedication(name="Paracetamol", frequency="Once Daily", duration_text="5 days", confidence=0.9),
            _GeminiMedication(name="Amoxicillin", frequency="Twice Daily", duration_text="7 days", confidence=0.85),
        ],
        full_text="Tab Paracetamol 500mg OD 5 days\nTab Amoxicillin 500mg BD 7 days",
    )
    _stub_gemini(monkeypatch, prescriptions_module, result)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()["medicines"]}
    assert names == {"Paracetamol", "Amoxicillin"}


def test_extraction_provider_unavailable_is_503(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _fail(images):
        raise GeminiProviderError("Gemini unreachable after retries")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _fail)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"].lower()


def test_unrelated_processing_failure_is_500_not_503(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _fail(images):
        raise Exception("some other unexpected error")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _fail)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 500
    assert "detail" in resp.json()
