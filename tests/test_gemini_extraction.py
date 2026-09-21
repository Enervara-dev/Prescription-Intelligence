"""
tests/test_gemini_extraction.py
-----------------------------------
Unit tests for app/services/gemini_extraction.py: the bounded retry and
distinct GeminiProviderError signal (mirroring the Vision-retry pattern in
test_ocr_retry.py), the mapping into this service's existing response
schemas, and the endpoint-level 503 mapping.

The real Gemini client is never called here -- `genai.Client` is
monkeypatched with a fake that returns/raises whatever each test needs,
exactly like test_ocr_retry.py fakes the Vision client. None of this
requires GEMINI_API_KEY to be set.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services import gemini_extraction
from app.services.gemini_extraction import (
    GeminiExtraction,
    GeminiProviderError,
    _GeminiMedication,
    _GeminiMetadata,
    to_extraction_out,
    to_patient_info_text,
    to_process_medicines,
)


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class _FakeModels:
    """Returns/raises each item in `responses` in order, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _install_fake_client(monkeypatch, responses):
    models = _FakeModels(responses)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = models

    monkeypatch.setattr(gemini_extraction.genai, "Client", _FakeClient)
    return models


@pytest.fixture(autouse=True)
def _require_api_key(monkeypatch):
    """Every test in this file needs SOME key set so _require_api_key()
    doesn't short-circuit before ever reaching the fake client -- the
    permanent-config-error case is tested separately by unsetting it."""
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test-key-not-real")


def _tiny_image():
    return Image.new("RGB", (10, 10), color="white")


# -- Single attempt / error signal -------------------------------------------
# No retry -- see _MAX_GEMINI_ATTEMPTS docstring: verified empirically
# against the live deployment that a short per-attempt timeout combined
# with a retry made things WORSE (Gemini's own servers started returning
# 504 DEADLINE_EXCEEDED consistently for a call that had previously
# succeeded, once a tight timeout was introduced). Retrying a call that
# needs more time doesn't help -- it just repeats the same wait -- so a
# single, more generously-bounded attempt is the safer trade: predictable
# worst-case latency over resilience to a rare transient blip.

def test_single_failure_raises_distinct_error_with_no_retry(monkeypatch):
    models = _install_fake_client(monkeypatch, [
        ConnectionError("simulated failure"),
    ])
    with pytest.raises(GeminiProviderError) as exc_info:
        gemini_extraction.extract_prescription([_tiny_image()])
    assert models.calls == 1  # no second attempt
    assert "simulated failure" in str(exc_info.value.__cause__)


def test_unparseable_response_is_treated_as_a_failure(monkeypatch):
    models = _install_fake_client(monkeypatch, [
        _FakeResponse(parsed=None),
    ])
    with pytest.raises(GeminiProviderError):
        gemini_extraction.extract_prescription([_tiny_image()])
    assert models.calls == 1


def test_permanent_config_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "")
    calls = []

    class _ShouldNeverBeConstructed:
        def __init__(self, api_key=None):
            calls.append(1)

    monkeypatch.setattr(gemini_extraction.genai, "Client", _ShouldNeverBeConstructed)

    with pytest.raises(EnvironmentError):
        gemini_extraction.extract_prescription([_tiny_image()])
    assert calls == []  # never even tried to construct a client


# -- Latency: per-request timeout, output cap, image downscaling ------------

def test_request_has_a_per_attempt_timeout_and_output_cap(monkeypatch):
    """
    A single hung/slow attempt must not be able to consume the whole retry
    budget unbounded -- verified gap, previously unset entirely.
    """
    captured_configs = []

    class _CapturingModels:
        def generate_content(self, model, contents, config):
            captured_configs.append(config)
            return _FakeResponse(parsed=GeminiExtraction(full_text="ok"))

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = _CapturingModels()

    monkeypatch.setattr(gemini_extraction.genai, "Client", _FakeClient)
    gemini_extraction.extract_prescription([_tiny_image()])

    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.http_options is not None
    assert config.http_options.timeout == gemini_extraction._GEMINI_REQUEST_TIMEOUT_MS
    assert config.max_output_tokens == gemini_extraction._GEMINI_MAX_OUTPUT_TOKENS


def test_worst_case_retry_latency_budget_is_conservative():
    """
    Guards the actual latency math, not just the individual numbers: total
    worst-case time (attempts x per-attempt timeout + backoff) must stay
    comfortably under Enervara's 90s deadline (see module comment on
    _MAX_GEMINI_ATTEMPTS) -- with real margin for image download and other
    endpoint overhead that happens outside this function.
    """
    worst_case_seconds = (
        gemini_extraction._MAX_GEMINI_ATTEMPTS * (gemini_extraction._GEMINI_REQUEST_TIMEOUT_MS / 1000)
        + sum(gemini_extraction._GEMINI_RETRY_BACKOFF_SECONDS)
    )
    # Real margin under the 90s external deadline for image download and
    # other endpoint overhead (not just this function's own call).
    assert worst_case_seconds <= 80


def test_large_image_is_downscaled():
    big = Image.new("RGB", (4000, 3000), color="white")
    small = gemini_extraction._downscale(big)
    assert max(small.size) == gemini_extraction._GEMINI_MAX_IMAGE_DIMENSION
    # Aspect ratio preserved.
    assert small.size[0] / small.size[1] == pytest.approx(4000 / 3000, rel=0.01)


def test_small_image_is_not_upscaled():
    small = Image.new("RGB", (400, 300), color="white")
    result = gemini_extraction._downscale(small)
    assert result.size == (400, 300)


# -- Mapping into existing response schemas ----------------------------------

def test_to_extraction_out_maps_fields_and_passes_confidence_through():
    result = GeminiExtraction(
        prescribed_date="2026-01-15",
        metadata=_GeminiMetadata(prescriber_name="A. Sharma", clinic_name="City Clinic"),
        medications=[
            _GeminiMedication(
                name="Dolo 650", generic_name="Paracetamol", strength="650mg",
                form="tablet", route="oral", dosage="1 tablet", frequency="Twice Daily",
                timing="After Food", duration_text="5 days", duration_days=5,
                instructions="Take with water", common_use="Fever relief",
                confidence=0.93,
            )
        ],
        full_text="...",
    )
    out = to_extraction_out(result)
    assert out.prescribed_date == "2026-01-15"
    assert out.metadata.prescriber_name == "A. Sharma"
    assert out.metadata.clinic_name == "City Clinic"
    assert len(out.medications) == 1
    med = out.medications[0]
    assert med.name == "Dolo 650"
    assert med.form == "TABLET"  # uppercased for Enervara's enum
    assert med.route == "ORAL"
    assert med.confidence == 0.93  # Gemini's 0-1 scale, unchanged
    assert med.duration_days == 5
    assert out.suggested_speciality_slug is None
    assert out.suggested_speciality_reason is None


def test_to_extraction_out_null_fields_stay_null():
    result = GeminiExtraction(medications=[_GeminiMedication(name="Amoxicillin", confidence=0.7)], full_text="")
    out = to_extraction_out(result)
    med = out.medications[0]
    assert med.dosage is None
    assert med.form is None
    assert med.duration_text is None
    assert med.duration_days is None


def test_duration_days_derived_deterministically_from_duration_text():
    """Prefers the regex-derived day count over Gemini's own arithmetic when
    duration_text explicitly says "N day(s)" -- catches a wrong duration_days
    Gemini might compute itself."""
    result = GeminiExtraction(
        medications=[_GeminiMedication(name="X", duration_text="5 days", duration_days=99, confidence=0.9)],
        full_text="",
    )
    out = to_extraction_out(result)
    assert out.medications[0].duration_days == 5  # not Gemini's wrong 99


def test_to_process_medicines_shape_and_confidence_scale():
    result = GeminiExtraction(
        medications=[
            _GeminiMedication(name="Dolo 650", generic_name="Paracetamol", strength="650mg", confidence=0.8)
        ],
        full_text="",
    )
    meds = to_process_medicines(result)
    assert len(meds) == 1
    med = meds[0]
    assert med["name"] == "Dolo 650"
    assert med["confidence"] == 80.0  # 0-1 -> 0-100
    assert med["dosage"] == "N/A"  # nothing stated, honest sentinel (back-compat)
    assert med["match_type"] == "LLM_EXTRACTED"
    assert med["source"] == "gemini"
    assert med["id"] is None  # no catalog row behind this anymore


def test_to_patient_info_text_composes_available_fields():
    result = GeminiExtraction(metadata=_GeminiMetadata(prescriber_name="A. Sharma", clinic_name="City Clinic"), full_text="")
    text = to_patient_info_text(result)
    assert "A. Sharma" in text
    assert "City Clinic" in text


def test_to_patient_info_text_falls_back_when_nothing_found():
    result = GeminiExtraction(full_text="")
    assert to_patient_info_text(result) == "Patient details not detected clearly."


def test_to_patient_info_text_never_doubles_dr_prefix():
    """
    Verified live-test bug: Gemini sometimes includes "Dr." in
    prescriber_name despite being asked not to (e.g. "Dr. A. Sharma, M.D.")
    -- must not become "Dr. Dr. A. Sharma, M.D." when this function adds its
    own "Dr." prefix.
    """
    for name in ("Dr. A. Sharma, M.D.", "Dr A. Sharma", "A. Sharma"):
        result = GeminiExtraction(metadata=_GeminiMetadata(prescriber_name=name), full_text="")
        text = to_patient_info_text(result)
        assert text.count("Dr") == 1, (name, text)


# -- Prompt content: guards against silently losing the Indian-convention ---
# -- guidance that fixed the verified "1-0-1" -> "Three Times Daily" misread -

def test_prompt_explains_dash_pattern_convention():
    prompt = gemini_extraction._PROMPT
    assert "1-0-1" in prompt
    assert "TWICE daily" in prompt or "Twice daily" in prompt.lower()
    assert "not a count of doses" in prompt or "not a dose" in prompt


def test_prompt_explains_total_quantity_is_not_dosage():
    prompt = gemini_extraction._PROMPT
    assert "Tot" in prompt
    assert "not a per-administration amount" in prompt or "NOT a" in prompt


def test_prompt_distinguishes_timing_from_frequency():
    prompt = gemini_extraction._PROMPT
    assert "before food" in prompt.lower() or "AC" in prompt
    assert "timing" in prompt.lower()


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


def test_process_returns_503_when_gemini_unavailable(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    def _raise_provider_error(images):
        raise GeminiProviderError("simulated: Gemini unreachable after retries")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _raise_provider_error)

    resp = client.post(
        "/api/v1/prescriptions/process",
        files={"file": ("prescription.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"].lower()


def test_extract_returns_503_when_gemini_unavailable(client, monkeypatch):
    import app.api.v1.endpoints.prescriptions as prescriptions_module

    monkeypatch.setattr(prescriptions_module, "download_document", lambda url, timeout: b"fake-bytes")
    monkeypatch.setattr(prescriptions_module, "file_bytes_to_pil_images", lambda content, mime: [object()])

    def _raise_provider_error(images):
        raise GeminiProviderError("simulated: Gemini unreachable after retries")

    monkeypatch.setattr(prescriptions_module, "gemini_extract_prescription", _raise_provider_error)

    resp = client.post(
        "/api/v1/prescriptions/extract",
        json={
            "prescription_id": "rx-1", "user_id": "u-1",
            "document_url": "https://storage.example.com/doc.jpg",
            "mime_type": "image/jpeg", "file_name": "doc.jpg", "request_id": "req-1",
        },
    )
    assert resp.status_code == 503
