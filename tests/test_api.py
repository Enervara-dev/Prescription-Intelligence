"""
tests/test_api.py
--------------------
HTTP-level tests for the prescription-processing API: request validation,
status codes, error handling, and the full response contract. The Google
Vision call itself is monkeypatched for these (deterministic, no external
cost/credentials) — see test_real_vision_ocr.py for the credential-backed
end-to-end OCR test.
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

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _boom)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert "hunter2" not in body["detail"]
    assert "internal-db.local" not in body["detail"]
    assert "SECRET" not in body["detail"]


def test_medicine_successfully_matched_end_to_end(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    # Seed one medicine and stub OCR to return a deterministic line for it.
    repo.upsert_medicine(db, {
        "external_id": "legacy:paracetamol",
        "generic_name": "Paracetamol",
        "brand_name": "Dolo 650",
        "aliases": ["Dolo", "Crocin", "Calpol"],
        "source": "legacy_manual",
    })
    db.commit()

    def _fake_ocr(images):
        text = "Tab Dolo 650 1-0-1 x 5 days"
        return {
            "patient_info": "",
            "medicine_text": text,
            "full_text": text,
            "word_count": 5,
            "oriented_images": images,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)

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
    # parse_medicines is now backed by medicine_resolver.resolve() (richer
    # UPPER_SNAKE match_type vocabulary than the old medicine_matching
    # lowercase one). The pipeline looks ahead to combine "Dolo"+"650" before
    # trying "Dolo" alone, so this resolves via the full brand string
    # directly (EXACT_BRAND) rather than needing the bare "dolo" alias.
    assert med["match_type"] == "EXACT_BRAND"
    assert med["duration"] == "5 days"


def test_medicine_not_found_returns_empty_list_not_error(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _fake_ocr(images):
        text = "Completelyunknownxyzabc 1-0-1"
        return {
            "patient_info": "",
            "medicine_text": text,
            "full_text": text,
            "word_count": 2,
            "oriented_images": images,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["medicines"] == []


def test_multiple_medicines_all_returned(client, db, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    repo.upsert_medicine(db, {"external_id": "legacy:paracetamol", "generic_name": "Paracetamol", "source": "legacy_manual"})
    repo.upsert_medicine(db, {"external_id": "legacy:amoxicillin", "generic_name": "Amoxicillin", "source": "legacy_manual"})
    db.commit()

    def _fake_ocr(images):
        text = "Tab Paracetamol 500mg OD 5 days\nTab Amoxicillin 500mg BD 7 days"
        return {
            "patient_info": "",
            "medicine_text": text,
            "full_text": text,
            "word_count": 10,
            "oriented_images": images,
        }

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fake_ocr)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()["medicines"]}
    assert names == {"Paracetamol", "Amoxicillin"}


def test_malformed_ocr_result_missing_keys_is_500_not_unhandled_crash(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _malformed_ocr(images):
        return {}  # missing every expected key

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _malformed_ocr)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    # Must fail as a clean, structured 500 — not an unhandled server crash
    # that closes the connection.
    assert resp.status_code == 500
    assert "detail" in resp.json()


def test_ocr_failure_is_structured_500(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _fail(images):
        raise Exception("Vision API quota exceeded")

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _fail)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 500
    assert "detail" in resp.json()
