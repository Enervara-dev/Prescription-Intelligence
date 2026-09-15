# AI Prescription Intelligence 🏥

A pure **FastAPI** backend that accepts **handwritten or printed doctor prescription images or PDFs** and extracts structured medicine data, powered by **Google Cloud Vision API** for handwriting recognition and a **Postgres-backed medicine catalog** (indexed `pg_trgm` fuzzy search) for name resolution.

There is no bundled frontend — this is a headless REST API. Consume it from any HTTP client (curl, Postman, a web/mobile app, another service, etc.). Interactive documentation is auto-generated at `/docs` (Swagger UI) and `/redoc`.

This service owns its **own** Postgres database — it is not wired into any other application's database. It is meant to be consumed purely over HTTP.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Client["Any HTTP Client"]
        C["curl / Postman / web app / mobile app / service"]
    end

    subgraph API["FastAPI Backend (app/)"]
        R1["POST /api/v1/prescriptions/process"]
        R2["GET /api/v1/medicines"]
        R3["GET /api/v1/medicines/search"]
        R4["GET /api/v1/health"]
        PDFConv["PyMuPDF (fitz)\nPDF to Image Conversion"]
    end

    subgraph OCR["OCR Engine (Google Cloud Vision)"]
        VisionAPI["Google Cloud Vision API\n(document_text_detection)"]
    end

    subgraph Extraction["Information Extraction & Matching Engine"]
        LineGroup["Spatial Line Grouping\n(Y-Coordinate Clustering)"]
        Classifier["Section Categorizer\n(Ad Filtering, Patient Info vs. Rx Lines)"]
        RegexEngine["Regex Field Parser\n(Dosage, Frequency: 1-0-1/TDS, Duration)"]
        OpticalVariants["OCR-confusable Variant Generator\n(m<->n, rn<->m, 1<->l, 0<->o, ...)"]
        Matcher["medicine_matching.match_token()\nexact -> alias -> fuzzy -> ocr_corrected"]
    end

    subgraph DB["Postgres (this service's own DB)"]
        MedTable[("medicines table\npg_trgm GIN index on search_text")]
    end

    subgraph Sync["Catalog sync (out-of-band, never per-request)"]
        Legacy["scripts/import_legacy_medicines.py\n(one-off, source=legacy_manual)"]
        RxNorm["scripts/sync_rxnorm_medicines.py\n(source=rxnorm, WORKING — free, no signup)"]
        ABDM["scripts/sync_abdm_medicines.py\n(source=abdm, NOT YET IMPLEMENTED\nsee ABDM section below)"]
    end

    C -->|multipart upload| R1
    R1 --> PDFConv
    PDFConv --> VisionAPI
    R1 --> VisionAPI
    VisionAPI --> LineGroup
    LineGroup --> Classifier
    Classifier --> RegexEngine
    Classifier --> OpticalVariants
    OpticalVariants --> Matcher
    Matcher <-->|indexed pg_trgm search| MedTable
    RegexEngine & Matcher --> C
    R2 & R3 --> MedTable
    Legacy -.->|upsert| MedTable
    RxNorm -.->|upsert, disabled by default| MedTable
    ABDM -.->|upsert, disabled by default| MedTable
```

---

## 📁 Project Structure

```text
Prescription-Intelligence/
├── app/
│   ├── main.py                          # FastAPI app instance, CORS, router mounting
│   ├── core/config.py                   # Settings (env vars: DB, CORS, matching, ABDM)
│   ├── api/v1/
│   │   ├── api.py                       # Aggregates all v1 routers
│   │   └── endpoints/
│   │       ├── health.py                # GET /api/v1/health (incl. DB connectivity)
│   │       ├── prescriptions.py         # POST /api/v1/prescriptions/process
│   │       ├── medicines.py             # GET /api/v1/medicines, /medicines/search
│   │       └── resolve.py               # POST /api/v1/medicines/resolve
│   ├── schemas/                         # Pydantic request/response models
│   ├── db/                              # SQLAlchemy engine/session (this service's own DB)
│   ├── models/medicine.py               # `medicines` ORM model (+ SOURCE_PRIORITY)
│   ├── repositories/medicine_repository.py  # ALL Postgres access (exact + pg_trgm search, upsert)
│   ├── services/
│   │   ├── ocr_service.py               # Google Cloud Vision integration & categorisation
│   │   ├── extraction/                  # Field extraction (regex) + parse_medicines() pipeline
│   │   ├── medicine_matching.py         # OCR-aware name resolution against the catalog
│   │   ├── medicine_resolver.py         # Indian medicine normalization pipeline (see below)
│   │   ├── text_normalization.py        # Deterministic text normalization
│   │   ├── strength_parser.py           # Deterministic strength/unit parsing
│   │   ├── dosage_form.py               # Deterministic dosage-form normalization
│   │   ├── providers/                   # MedicineProvider interface + legacy/RxNorm/licensed-India adapters
│   │   ├── rxnorm_client.py             # RxNorm (NLM RxNav) client — free, public, WORKING
│   │   ├── rxnorm_sync.py               # Out-of-band sync orchestration (never per-request)
│   │   ├── abdm_client.py               # ABDM Drug Registry client (NOT YET IMPLEMENTED — see below)
│   │   └── abdm_sync.py                 # Out-of-band sync orchestration (never per-request)
│   ├── data/
│   │   ├── medicine_list.txt            # Legacy flat-file catalog — import SOURCE only, not read at runtime
│   │   └── indian_brands.py             # Hand-curated Indian brand/generic/strength fixture
│   └── utils.py                         # Image & PDF decoding/validation helpers, upload limits
├── alembic/                             # DB migrations (versioned schema history)
├── scripts/
│   ├── import_legacy_medicines.py       # Idempotent: medicine_list.txt -> Postgres (source=legacy_manual)
│   ├── seed_indian_brands.py            # Idempotent: indian_brands.py -> Postgres (enriches the same rows)
│   ├── sync_rxnorm_medicines.py         # Manual/cron entry point for RxNorm sync
│   └── sync_abdm_medicines.py           # Manual/cron entry point for ABDM sync
├── tests/                               # Real-Postgres tests (skip cleanly if no test DB configured)
├── docker-compose.yml                   # Local Postgres for this service ONLY
├── .env.example
├── requirements.txt / requirements-dev.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Start this service's own Postgres

```bash
docker compose up -d db
```

This starts a dedicated Postgres 16 container (`prescription_intelligence` DB) — independent of any other application's database. You can point `DATABASE_URL` at any Postgres 13+ instead; the only requirement is that the `pg_trgm` extension is available (the migration enables it automatically).

### 3. Configure environment

Create a `.env` file in the project root (see `.env.example` for the full list):
```env
GOOGLE_VISION_API_KEY=your_google_cloud_vision_api_key_here
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/prescription_intelligence
```

### 4. Apply migrations and seed the legacy catalog

```bash
python -m alembic upgrade head
python -m scripts.import_legacy_medicines
python -m scripts.seed_indian_brands
```

Both imports are idempotent — safe to re-run any time. The first loads every distinct name from `app/data/medicine_list.txt` into Postgres tagged `source="legacy_manual"`; the second enriches the same rows (where they overlap) with structured brand/generic/strength/dosage-form/combination data for representative Indian medicines (see "Indian medicine normalization pipeline" below). The catalog is fully searchable before ABDM/RxNorm sync is ever configured.

### 5. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for interactive Swagger docs, or **http://localhost:8000/redoc**.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
TEST_DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/prescription_intelligence pytest
```

Tests run against a **real** Postgres (they exercise actual `pg_trgm` queries — nothing is mocked at the DB layer) and are skipped, not failed, when `TEST_DATABASE_URL` isn't set. 69 tests covering: legacy import with zero data loss, exact/alias/fuzzy/ocr_corrected/ambiguous/unresolved matching, strength/form/route/manufacturer pass-through, the full `parse_medicines()` pipeline, the Indian medicine resolver (exact/OCR-variant/strength-disambiguation/ambiguous/no-match/null-safety/RxNorm-supplementary), the deterministic strength & dosage-form parsers, and index-usage regressions (pg_trgm, expression, and GIN indexes all verified via `EXPLAIN`).

---

## 🇮🇳 Indian medicine normalization pipeline

`app/services/medicine_resolver.py` — resolves free-text OCR input (brand or generic, with/without strength/dosage-form, with/without common OCR corruption) to a catalog entity, or explicitly declines to guess:

```
OCR text → text normalization → strength extraction → dosage-form extraction
  → matching hierarchy (exact brand/alias → exact generic+strength → RxNorm
    exact ingredient → controlled fuzzy candidates → ranking)
  → confidence policy → EXACT | HIGH_CONFIDENCE | LOW_CONFIDENCE
                          | REVIEW_REQUIRED | NO_MATCH
```

- **Text normalization** (`text_normalization.py`) — deterministic only (case, whitespace, hyphens, clean alnum-boundary splitting). OCR-confusable substitution (`0↔o`, `1↔i/l`, `5↔s`, `8↔b`) is deliberately NOT applied here — only at controlled candidate-generation time — so it can never silently corrupt a stored value.
- **Strength parsing** (`strength_parser.py`) — a separate deterministic stage: `"500mg"`, `"100 mg/5 ml"`, `"250mg + 125mg"` (combinations) all parse to structured `{value, unit}`, never mixed with name matching.
- **Dosage-form normalization** (`dosage_form.py`) — `tab/tabs/tablet/tablets → tablet`, etc.; a tablet never fuzzy-matches a syrup.
- **Safety rule, enforced in code, not just docs**: an exact name match with a *conflicting* stated strength (`"Dolo 680mg"` hitting the "dolo" alias on the 650mg product) is never silently accepted — it's downgraded to `REVIEW_REQUIRED`. A generic name shared by multiple products with no strength given to disambiguate is `REVIEW_REQUIRED`, never an arbitrary pick.
- **Providers** (`app/services/providers/`) — `MedicineProvider` interface; `LegacyManualProvider` and `RxNormProvider` wrap the existing, already-tested ingestion scripts; `LicensedIndiaProvider` is a documented, not-yet-implemented extension point (same honest pattern as ABDM) for MIMS India/CIMS once licensed.
- **Indian brand fixture** (`app/data/indian_brands.py`, seeded via `scripts/seed_indian_brands.py`) — hand-curated, provenance-tagged (`source_version="hand_curated_indian_brands_v1"`) brand→generic→strength→form data for representative analgesics, antibiotics, antihistamines, PPIs, antihypertensives, antidiabetics, and combination products (Combiflam, Meftal-P, Augmentin 625, ...). Deliberately merges into the same rows `medicine_list.txt` already created for the same brand (same `external_id` scheme) rather than creating parallel duplicate rows.

---

## 🚀 API Endpoints

| Method | Path                              | Description                                                        |
|--------|-----------------------------------|----------------------------------------------------------------------|
| GET    | `/health`                         | Infra-friendly health check alias                                  |
| GET    | `/api/v1/health`                  | API status + DB connectivity + whether the Vision key is configured |
| POST   | `/api/v1/prescriptions/process`   | Upload a prescription image/PDF (multipart) → structured patient + medicine data |
| POST   | `/api/v1/prescriptions/extract`   | Service-to-service adapter for Enervara's `HttpPrescriptionProcessingService` — see below |
| GET    | `/api/v1/medicines`               | Paginated listing of the medicine catalog                          |
| GET    | `/api/v1/medicines/search?q=...`  | Indexed exact + fuzzy search by name                                |
| POST   | `/api/v1/medicines/resolve`       | Indian medicine normalization pipeline — `{"text": "D0LO 650"}` → structured, confidence-graded result |

### Enervara integration (`POST /api/v1/prescriptions/extract`)

Matches the contract `HttpPrescriptionProcessingService` already expects (`prod_app/app/backend/src/prescriptions/processing/httpProcessingService.ts`) — accepts a JSON body naming a `document_url` (not a file upload) and returns Enervara's `PrescriptionExtraction` shape verbatim (camelCase field names, `confidence` on a 0-1 scale). It reuses the exact same OCR + `parse_medicines()` pipeline `/process` uses — see `app/services/enervara_extraction_adapter.py` for the (only new) output-shape transform. Never fabricates: a medicine the resolver isn't confident about is simply absent from `medications`, and fields this service doesn't actually extract (`prescribedDate`, `timing`, `instructions`, structured prescriber/patient metadata) are sent as `null`, not guessed.

**To connect Enervara to this service** (config only — no code changes needed on Enervara's side, `HttpPrescriptionProcessingService` already implements this contract exactly):
```env
RX_PROCESSING_PROVIDER=http
RX_PROCESSING_API_URL=https://<this-service-host>/api/v1/prescriptions   # Enervara appends "/extract" itself
RX_PROCESSING_API_KEY=<shared secret — same value as this service's RX_PROCESSING_API_KEY>
```
`RX_PROCESSING_API_KEY` is optional on this side (unauthenticated if unset — a startup log warns loudly when that's the case) but should be set before this endpoint is reachable outside local development.

### Example: resolve OCR text to a medicine

```bash
curl -X POST http://localhost:8000/api/v1/medicines/resolve \
  -H "Content-Type: application/json" -d '{"text": "D0LO 650"}'
```
```json
{
  "input": "D0LO 650", "normalized_text": "d0lo 650",
  "status": "HIGH_CONFIDENCE", "match_type": "OCR_NORMALIZED_BRAND",
  "medicine": {
    "brand_name": "Dolo 650", "generic_name": "Paracetamol",
    "strength": 650.0, "unit": "mg", "dosage_form": "tablet"
  },
  "source": "legacy_manual", "confidence": 100.0
}
```

An uncertain input never fabricates a result:
```bash
curl -X POST http://localhost:8000/api/v1/medicines/resolve \
  -H "Content-Type: application/json" -d '{"text": "Dolo 680mg"}'
```
```json
{ "input": "Dolo 680mg", "status": "REVIEW_REQUIRED", "medicine": null,
  "candidates": [{"brand_name": "Dolo 650", "confidence": 70.0}] }
```

### Example: process a prescription

```bash
curl -X POST http://localhost:8000/api/v1/prescriptions/process \
  -F "file=@/path/to/prescription.jpg"
```

Response (truncated):
```json
{
  "success": true,
  "filename": "prescription.jpg",
  "patient_info": "John Doe, 34/M, Dr. Smith",
  "medicines": [
    {
      "name": "Dolo 650",
      "confidence": 100.0,
      "dosage": "N/A",
      "frequency": "1-0-1 (M-A-N)",
      "duration": "5 days",
      "raw_line": "Tab Dolo 650 1-0-1 x 5 days",
      "id": "371a370a-75c6-48cd-abeb-c38287c45d2c",
      "generic_name": "Paracetamol",
      "brand_name": "Dolo 650",
      "strength": null,
      "dosage_form": null,
      "route": null,
      "manufacturer": null,
      "match_type": "alias",
      "source": "legacy_manual"
    }
  ],
  "image_preview": "data:image/jpeg;base64,...",
  "processing_time_sec": 1.42,
  "debug": { "full_text": "...", "medicine_text": "...", "word_count": 87 }
}
```

`match_type` is one of `exact | alias | fuzzy | ocr_corrected`. Ambiguous (two close candidates) and unresolved (nothing confident) tokens are never included in `medicines` — matches are never forced.

---

## 🗃️ Medicine catalog architecture

Replaces the old `medicine_list.txt` + in-memory RapidFuzz-over-a-Python-list approach:

```
medicine_list.txt (one-off import) ──┐
                                      ├──> Postgres `medicines` table (pg_trgm GIN index)
ABDM Drug Registry (sync, optional) ─┘              │
                                                      ▼
                                    indexed pg_trgm search (small candidate set)
                                                      │
                                                      ▼
                            RapidFuzz re-rank + OCR-confusable variants
                            (m<->n, rn<->m, 1<->l, 0<->o, ...) — same logic
                            as before, just applied to a handful of DB
                            candidates instead of the whole catalog
                                                      │
                                                      ▼
                    { id, brand_name, generic_name, strength, dosage_form,
                      route, manufacturer, match_type, source, confidence }
```

- **Table**: `medicines` (see `app/models/medicine.py` / `alembic/versions/0001_create_medicines_table.py`) — `generic_name`, `brand_name`, `aliases text[]`, `strength`, `dosage_form`, `route`, `composition`, `manufacturer`, `atc_code`, `snomed_ct_code`, `rxcui`, `codes jsonb`, `source`, `source_version`, `source_updated_at`.
- **Retrieval**: `app/repositories/medicine_repository.py` is the only place that queries the table — exact/alias lookups hit btree/GIN indexes, fuzzy lookups hit a `pg_trgm` GIN index on a precomputed `search_text` column, so search stays fast regardless of catalog size (no Python-side linear scan). Fuzzy queries use the `%` similarity operator with `pg_trgm.similarity_threshold` set per-query — a bare `similarity(col, term) >= x` predicate is **not** index-aware and silently falls back to a sequential scan even with the index present (found via `EXPLAIN ANALYZE` at ~14.7k rows: seq scan ~38ms vs. the `%`-operator form ~1ms; regression-tested in `tests/test_medicine_matching.py`). `bulk_upsert()` runs `ANALYZE` after writing so the planner's stats don't go stale after a large sync.
- **Matching**: `app/services/medicine_matching.py` resolves a token to `exact | alias | fuzzy | ocr_corrected | ambiguous | unresolved`, preserving the original optical-confusable correction logic.
- **Upserts are idempotent**: keyed on `(source, external_id)` — re-running the legacy import or a future ABDM sync never duplicates rows.

---

## 🌐 RxNorm sync — status

**Working and verified**, unlike ABDM below. RxNorm ([NLM's RxNav REST API](https://rxnav.nlm.nih.gov/REST/)) is free, public, and requires **no signup, no credentials, no sandbox approval** — its contract was confirmed live before writing the client (`GET /allconcepts.json?tty=IN` returns `{"minConceptGroup": {"minConcept": [{"rxcui": "435", "name": "albuterol", "tty": "IN"}, ...]}}`).

Run it with:
```bash
RXNORM_SYNC_ENABLED=true python -m scripts.sync_rxnorm_medicines
```
Verified end-to-end against real data: fetched **14,689 real RxNorm ingredient concepts** in one call, upserted all of them (idempotent on re-run — a second run upserts the same 14,689, no duplicates), and confirmed real matches against them (`Amoxicillin`, `Acetaminophen`, `Azithromycin`, `Ibuprofen` all resolved as `exact` matches, `source="rxnorm"`).

**Important limitation — this does NOT cover Indian brand names.** RxNorm's `tty=IN` (Ingredient) concepts are a generic/INN-name backbone only (e.g. RxNorm calls paracetamol "acetaminophen", its US name) — it has no entries at all for Dolo, Crocin, Meftal, Combiflam, or any other Indian OTC/branded product. For Indian brand coverage, keep growing `legacy_manual` (`app/data/medicine_list.txt` / direct upserts), or license an India-specific dataset (MIMS India, CIMS) — RxNorm is a free supplementary generic-name layer alongside that, not a replacement for it.

Off by default (`RXNORM_SYNC_ENABLED=false`) and, like ABDM, never called from the prescription-processing request path — it's a deliberately-triggered, out-of-band sync.

---

## 🔌 ABDM Drug Registry integration — status

**Not yet implemented against a verified contract**, by design (see `app/services/abdm_client.py`'s docstring for the full account). Before writing a client we tried to confirm ABDM's actual Drug Registry API — auth flow, base URL, endpoints, payload shapes — against the Drug Registry portal (`drugregistrybeta.abdm.gov.in`) and the ABDM developer forum; both were unreachable from this environment, and no public Swagger/OpenAPI spec for the Drug Registry specifically was found. We did confirm that NHA + CDSCO + NRCeS launched a "National Drug Registry" under ABDM adopting SNOMED CT — but that's not a concrete REST contract to build against.

What **is** real and working today:
- `ABDM_SYNC_ENABLED` (default `false`), `ABDM_BASE_URL`, `ABDM_CLIENT_ID`, `ABDM_CLIENT_SECRET` — read from env, never hard-coded, **not required** for local development or for the catalog to work (the legacy-seeded data is fully searchable without any of this).
- `scripts/sync_abdm_medicines.py` — a manual/cron entry point, **never called from the prescription-processing request path**. Running it today logs a clear "not implemented yet" message and exits cleanly (verified: exit code 0 in all three states — disabled, enabled-without-credentials, enabled-with-credentials).

**To finish this once you have ABDM sandbox access + the real API spec:**
1. Get sandbox access at `sandbox.abdm.gov.in` and locate the Drug Registry's OpenAPI/Swagger definition (ask on `devforum.abdm.gov.in` if it isn't linked — there's an existing thread there asking for a centralised Swagger index).
2. Implement authentication in `ABDMDrugRegistryClient` per whatever ABDM actually requires — verify this rather than assuming an OAuth2 client-credentials flow just because other ABDM services use one.
3. Implement `fetch_records()` (pagination) and `normalize_record()` (map the real response fields onto `generic_name` / `brand_name` / `strength` / `dosage_form` / `route` / `composition` / `manufacturer` / `atc_code` / `snomed_ct_code` / `rxcui` / `codes`).
4. Set `ABDM_SYNC_ENABLED=true` + the three credential vars, then run `python -m scripts.sync_abdm_medicines` (manually or on a schedule) — it upserts into the same `medicines` table, tagged `source="abdm"`, alongside the `legacy_manual` rows.

---

## 🔑 Key Features

1. **Google Cloud Vision OCR**: Leverages `document_text_detection` to accurately read terrible doctor handwriting, even heavily cursive scripts.
2. **Robust Multi-format Support**: Seamlessly processes PDF uploads (multi-page, rendered via `PyMuPDF`) as well as standard images (JPG, PNG, WEBP, BMP, TIFF).
3. **Indexed, structured medicine catalog**: Postgres + `pg_trgm`, not a flat file scanned in Python — see "Medicine catalog architecture" above.
4. **OCR-aware fuzzy matching**: RapidFuzz re-ranking plus optical-confusable substitutions (`1 ↔ l`, `rn ↔ m`, `m ↔ n`, `0 ↔ o`, ...) applied to indexed candidates, never a full-catalog scan.
5. **Never forces a match**: ambiguous (two close candidates) and unresolved tokens are excluded from results rather than guessed.
6. **Structured, versioned JSON API**: All endpoints live under `/api/v1`, are fully typed with Pydantic response models, and are self-documented via OpenAPI (`/docs`, `/redoc`).
7. **Bounded resource use per request**: uploads are capped at `MAX_UPLOAD_SIZE_MB` (default 15MB, → HTTP 413) and PDFs at `MAX_PDF_PAGES` (default 20, → HTTP 422), so one request can't turn into unbounded memory/CPU/Vision-API work.
8. **No internal detail leaked on error**: an unexpected failure returns a generic message to the caller; the real exception (which may reference internal hosts, DB errors, etc.) goes to the server log only.
