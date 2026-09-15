"""
schemas/prescription.py
------------------------
Response/request models for the prescription-processing endpoint.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Medicine(BaseModel):
    # Back-compat fields (previous API shape)
    name: str = Field(description="Matched medicine name (brand name if known, else generic name).")
    confidence: float = Field(description="Match confidence, 0-100.")
    dosage: str = Field(description="Regex-extracted dosage from the OCR line, e.g. '500mg', or 'N/A'.")
    frequency: str = Field(description="Regex-extracted dosing frequency, e.g. 'Twice Daily', or 'N/A'.")
    duration: str = Field(description="Regex-extracted treatment duration, e.g. '5 days', or 'N/A'.")
    raw_line: Optional[str] = Field(default=None, description="The raw OCR line this medicine was parsed from.")

    # Structured catalog fields (from the Postgres `medicines` table)
    id: Optional[str] = Field(default=None, description="Catalog row id of the matched medicine.")
    generic_name: Optional[str] = Field(default=None, description="Generic/INN name, if known.")
    brand_name: Optional[str] = Field(default=None, description="Brand name, if known.")
    strength: Optional[str] = Field(default=None, description="Catalog strength, e.g. '500mg', if known.")
    dosage_form: Optional[str] = Field(default=None, description="e.g. 'tablet', 'syrup', if known.")
    route: Optional[str] = Field(default=None, description="e.g. 'oral', if known.")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer, if known.")
    match_type: Optional[str] = Field(
        default=None,
        description="How the match was resolved: exact | alias | fuzzy | ocr_corrected. "
        "Ambiguous/unresolved tokens are never included in this list.",
    )
    source: Optional[str] = Field(
        default=None, description="Catalog provenance, e.g. 'legacy_manual' or 'abdm'."
    )


class ProcessingDebugInfo(BaseModel):
    full_text: str = Field(description="Full raw OCR text extracted from the document.")
    medicine_text: str = Field(description="OCR text classified as belonging to the medicine/Rx section.")
    word_count: int = Field(description="Total number of OCR word tokens detected.")


class PrescriptionProcessResponse(BaseModel):
    success: bool
    filename: str
    patient_info: str = Field(description="Detected patient/doctor details as free text.")
    medicines: List[Medicine]
    image_preview: Optional[str] = Field(
        default=None, description="Base64 data URL JPEG preview of the (auto-oriented) first page."
    )
    processing_time_sec: float
    debug: ProcessingDebugInfo
