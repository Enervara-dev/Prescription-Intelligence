"""
tests/test_ocr_retry.py
---------------------------
Unit tests for the bounded Vision-call retry and the distinct
OCRProviderError signal (app/services/ocr_service.py).

This module is no longer called from the live request path (see
app/api/v1/endpoints/prescriptions.py and app/services/gemini_extraction.py
-- extraction now goes through Gemini) but is left in place, fully tested,
in case it needs to be reverted to or run alongside Gemini later. The
endpoint-level 503-mapping coverage this file used to include has a Gemini
equivalent in tests/test_gemini_extraction.py.
"""

import pytest

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
