"""
core/config.py
---------------
Centralised application settings, loaded from environment variables / .env.
"""

import logging
import os
from dotenv import load_dotenv

# Load .env from the project root, once, before anything reads os.environ.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)


class Settings:
    PROJECT_NAME: str = "AI Prescription Intelligence API"
    DESCRIPTION: str = (
        "Upload a handwritten or printed prescription (image/PDF) and receive "
        "structured patient info + medicine data. Pure REST API — no frontend."
    )
    VERSION: str = "3.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # CORS: comma separated list of allowed origins, "*" for all (default, since
    # this is a headless API meant to be consumed by any client).
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    GOOGLE_VISION_API_KEY: str = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()

    # --- Gemini LLM-based prescription extraction (app/services/gemini_extraction.py) ---
    # Replaces the Vision-OCR + Postgres-catalog-matching pipeline in the
    # live request path (/process, /extract) -- Gemini reads the
    # prescription image(s) directly and returns structured medicine data,
    # with no catalog cross-reference involved. Env var name is
    # deliberately "Gemini_API_KEY" (not the all-caps convention the rest
    # of this file uses) to match what's actually set in the deployment
    # environment -- env var names are case-sensitive; do not "fix" the
    # casing here without also changing it wherever the value is set.
    GEMINI_API_KEY: str = os.environ.get("Gemini_API_KEY", "").strip()
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

    # --- Upload limits (unbounded uploads are a resource-exhaustion risk:
    # unlimited file size, or a pathological many-page PDF, both turn one
    # request into a large amount of memory/CPU/Vision-API work) ---
    MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "15"))
    MAX_PDF_PAGES: int = int(os.environ.get("MAX_PDF_PAGES", "20"))

    # --- Database (own Postgres instance; independent of any other app's DB) ---
    # postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
    #
    # REQUIRED, no hardcoded fallback of any kind (matches prod_app's own
    # backend/src/db/pool.ts: connectionString() throws rather than default
    # to something that merely "looks local"). Left "" here rather than
    # raising immediately is deliberate and safe: the engine is created
    # LAZILY (see app/db/session.py), so simply importing settings — which
    # nearly every module does, including pure-unit tests that never touch
    # a database — never requires DATABASE_URL. The hard requirement is
    # enforced at the two places that actually need a connection: engine
    # creation (app/db/session.get_engine) and app startup (app/main.py).
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "").strip()
    DB_POOL_SIZE: int = int(os.environ.get("DB_POOL_SIZE", "5"))
    DB_POOL_MAX_OVERFLOW: int = int(os.environ.get("DB_POOL_MAX_OVERFLOW", "10"))

    # --- Medicine matching (pg_trgm-backed) ---
    # RapidFuzz score (0-100) a candidate must clear to be treated as a confident match.
    MEDICINE_MATCH_THRESHOLD: float = float(os.environ.get("MEDICINE_MATCH_THRESHOLD", "82.0"))
    # If the top two candidates' scores are within this margin, the match is reported
    # as "ambiguous" instead of being forced.
    MEDICINE_AMBIGUITY_MARGIN: float = float(os.environ.get("MEDICINE_AMBIGUITY_MARGIN", "4.0"))
    # How many trigram-similarity candidates to pull from Postgres per token/variant
    # before re-ranking them in Python with RapidFuzz. Keeps the fuzzy pass cheap
    # regardless of catalog size.
    MEDICINE_CANDIDATE_LIMIT: int = int(os.environ.get("MEDICINE_CANDIDATE_LIMIT", "8"))
    # Minimum pg_trgm similarity (0-1) for a row to be considered a candidate at all.
    MEDICINE_MIN_TRIGRAM_SIMILARITY: float = float(os.environ.get("MEDICINE_MIN_TRIGRAM_SIMILARITY", "0.2"))

    # --- Resolver confidence policy (app/services/medicine_resolver.py) ---
    # Score (0-100) below MEDICINE_MATCH_THRESHOLD but at/above this is still
    # reported (status=LOW_CONFIDENCE, still auditable) rather than treated
    # as no match at all. Below this -> NO_MATCH; nothing is fabricated.
    MEDICINE_LOW_CONFIDENCE_THRESHOLD: float = float(os.environ.get("MEDICINE_LOW_CONFIDENCE_THRESHOLD", "60.0"))
    # Ranking bonus/penalty magnitudes (0-100 scale, tuned conservatively;
    # exact name-match stages are never affected by these — only fuzzy
    # candidate ranking is).
    MEDICINE_STRENGTH_MATCH_BONUS: float = float(os.environ.get("MEDICINE_STRENGTH_MATCH_BONUS", "8.0"))
    MEDICINE_STRENGTH_MISMATCH_PENALTY: float = float(os.environ.get("MEDICINE_STRENGTH_MISMATCH_PENALTY", "25.0"))
    MEDICINE_FORM_MISMATCH_PENALTY: float = float(os.environ.get("MEDICINE_FORM_MISMATCH_PENALTY", "15.0"))
    MEDICINE_SOURCE_PRIORITY_BONUS: float = float(os.environ.get("MEDICINE_SOURCE_PRIORITY_BONUS", "2.0"))

    # --- ABDM Drug Registry sync (optional, off by default) ---
    # See README "ABDM Drug Registry integration" for status/caveats. Sync populates
    # the local `medicines` table out-of-band; it is never called from the OCR
    # request path.
    ABDM_SYNC_ENABLED: bool = os.environ.get("ABDM_SYNC_ENABLED", "false").strip().lower() == "true"
    ABDM_BASE_URL: str = os.environ.get("ABDM_BASE_URL", "").strip()
    ABDM_CLIENT_ID: str = os.environ.get("ABDM_CLIENT_ID", "").strip()
    ABDM_CLIENT_SECRET: str = os.environ.get("ABDM_CLIENT_SECRET", "").strip()

    # --- RxNorm sync (optional, off by default) ---
    # Free, public, unauthenticated REST API (api.bioontology... no — NLM's own
    # RxNav service) — verified reachable, no registration needed, unlike ABDM.
    # Generic/ingredient-name backbone only (US-centric) — does NOT cover
    # Indian brand names (Dolo, Crocin, ...); keep growing `legacy_manual` for
    # those. See README "RxNorm sync" for details. Never called from the OCR
    # request path — out-of-band only, same rule as ABDM.
    RXNORM_SYNC_ENABLED: bool = os.environ.get("RXNORM_SYNC_ENABLED", "false").strip().lower() == "true"
    RXNORM_BASE_URL: str = os.environ.get("RXNORM_BASE_URL", "https://rxnav.nlm.nih.gov/REST").strip().rstrip("/")
    RXNORM_TIMEOUT_SECONDS: int = int(os.environ.get("RXNORM_TIMEOUT_SECONDS", "30"))

    # --- Enervara /extract adapter (POST /api/v1/prescriptions/extract) ---
    # Service-to-service endpoint consumed by Enervara's
    # HttpPrescriptionProcessingService (RX_PROCESSING_PROVIDER=http). Same
    # env var NAME as Enervara's own RX_PROCESSING_API_KEY, deliberately —
    # whoever wires the two together copies one secret value into both
    # sides' env, same pattern Enervara itself uses for its per-speciality
    # chat backends and lab-processing provider (an X-API-Key header, sent
    # only if configured on the caller's side; here, required only if set
    # on ours — see api_key_auth.py). Not required for local development or
    # for the existing multipart /process endpoint, which this does not
    # touch or gate in any way.
    RX_PROCESSING_API_KEY: str = os.environ.get("RX_PROCESSING_API_KEY", "").strip()
    # Timeout for downloading the document from Enervara's short-lived
    # signed document_url — separate from RXNORM_TIMEOUT_SECONDS (unrelated
    # concerns) and comfortably inside Enervara's own 90s default
    # RX_PROCESSING_TIMEOUT_MS budget for the whole call.
    EXTRACT_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.environ.get("EXTRACT_DOWNLOAD_TIMEOUT_SECONDS", "30"))


settings = Settings()

if not settings.RX_PROCESSING_API_KEY:
    # Visible, not silent. The /extract endpoint still works unauthenticated
    # when this is unset (so local testing needs no setup), but a production
    # deploy that forgot to set it should be obvious in the logs, not
    # silently open.
    logger.warning(
        "RX_PROCESSING_API_KEY is not set — POST /api/v1/prescriptions/extract "
        "is running WITHOUT service-to-service authentication. Set it before "
        "exposing this endpoint outside local development."
    )

if not settings.GEMINI_API_KEY:
    # /process and /extract both depend on this now (see
    # app/services/gemini_extraction.py) -- a soft warning here (not a
    # startup failure), same pattern as GOOGLE_VISION_API_KEY: this module
    # must stay importable with no key configured so pure-unit tests keep
    # collecting cleanly. The actual endpoints fail clearly, at request
    # time, if this is missing when they're called.
    logger.warning(
        "Gemini_API_KEY is not set. POST /api/v1/prescriptions/process and "
        "/extract will fail at request time until it's configured."
    )

if not settings.DATABASE_URL:
    # No fallback exists to silently mask this, unlike before — see the
    # DATABASE_URL comment above. Still just a warning here (not a raise):
    # this module must stay importable with no DB configured so pure-unit
    # tests keep collecting cleanly. app/main.py's startup check is what
    # actually refuses to SERVE traffic without one.
    logger.warning("DATABASE_URL is not set. This service will fail to start without it.")
