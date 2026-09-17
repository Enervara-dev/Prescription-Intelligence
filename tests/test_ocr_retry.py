"""
tests/test_ocr_retry.py
---------------------------
Tests for the bounded Vision-call retry and the distinct OCRProviderError
signal (app/services/ocr_service.py), and its mapping to a 503 response
distinct from every other kind of processing failure
(app/api/v1/endpoints/prescriptions.py).
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services.ocr_service import OCRProviderError, _call_vision_with_retry


class _FakeErrorField:
    def __init__(self, message=""):
        self.message = message


class _FakeResponse:
    def __init__(self, message=""):
        self.error = _FakeErrorField(message)


class _FlakyClient:
    """Fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times: int, exc: Exception = None):
        self.fail_times = fail_times
        self.calls = 0
        self.exc = exc or ConnectionError("simulated transient network failure")

    def document_text_detection(self, image, image_context):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return _FakeResponse()


class _AlwaysFailsClient:
    def __init__(self, exc: Exception = None):
        self.calls = 0
        self.exc = exc or ConnectionError("simulated persistent network failure")

    def document_text_detection(self, image, image_context):
        self.calls += 1
        raise self.exc


class _ResponseErrorFieldClient:
    """Simulates Vision returning HTTP 200 with its own embedded error."""

    def __init__(self):
        self.calls = 0

    def document_text_detection(self, image, image_context):
        self.calls += 1
        return _FakeResponse(message="RESOURCE_EXHAUSTED: quota exceeded")


def test_retry_succeeds_after_transient_failures():
    client = _FlakyClient(fail_times=2)
    response = _call_vision_with_retry(client, object(), object())
    assert response is not None
    assert client.calls == 3  # 2 failures + 1 success, within _MAX_OCR_ATTEMPTS


def test_retry_gives_up_after_max_attempts_and_raises_distinct_error():
    client = _AlwaysFailsClient()
    with pytest.raises(OCRProviderError) as exc_info:
        _call_vision_with_retry(client, object(), object())
    assert client.calls == 3  # bounded, does not retry forever
    assert "simulated persistent network failure" in str(exc_info.value.__cause__)


def test_response_embedded_error_field_also_triggers_provider_error():
    client = _ResponseErrorFieldClient()
    with pytest.raises(OCRProviderError):
        _call_vision_with_retry(client, object(), object())
    assert client.calls == 3


def test_permanent_config_errors_are_not_retried():
    """Missing API key / package not installed are config problems, not
    transient failures -- they must fail immediately, not burn through
    retries or backoff sleep."""
    from app.services import ocr_service

    # _get_api_key() raises before any client/network call happens at all,
    # so there is nothing to retry -- it runs before _extract_page_words
    # ever enters the retry loop. Verified here that it still raises
    # EnvironmentError directly, not wrapped as OCRProviderError.
    import os
    old = os.environ.pop("GOOGLE_VISION_API_KEY", None)
    try:
        with pytest.raises(EnvironmentError):
            ocr_service._get_api_key()
    finally:
        if old is not None:
            os.environ["GOOGLE_VISION_API_KEY"] = old


# -- Endpoint-level mapping to a distinct 503 --------------------------------

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


def test_process_returns_503_when_ocr_provider_unavailable(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _raise_provider_error(images):
        raise OCRProviderError("simulated: Vision unreachable after retries")

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _raise_provider_error)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"].lower()


def test_process_returns_500_for_unrelated_failures_not_503(client, monkeypatch):
    """A bug elsewhere in processing must NOT be reported as an OCR-provider
    problem -- only OCRProviderError maps to 503."""
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _raise_something_else(images):
        raise RuntimeError("unrelated internal bug")

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _raise_something_else)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 500


def test_extract_returns_503_when_ocr_provider_unavailable(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    monkeypatch.setattr(prescriptions_module, "download_document", lambda url, timeout: b"fake-bytes")
    monkeypatch.setattr(prescriptions_module, "file_bytes_to_pil_images", lambda content, mime: [object()])

    def _raise_provider_error(images):
        raise OCRProviderError("simulated: Vision unreachable after retries")

    monkeypatch.setattr(prescriptions_module, "extract_text_from_images", _raise_provider_error)

    resp = client.post(
        "/api/v1/prescriptions/extract",
        json={
            "prescription_id": "rx-1", "user_id": "u-1",
            "document_url": "https://storage.example.com/doc.jpg",
            "mime_type": "image/jpeg", "file_name": "doc.jpg", "request_id": "req-1",
        },
    )
    assert resp.status_code == 503
