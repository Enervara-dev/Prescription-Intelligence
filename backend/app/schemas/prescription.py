from pydantic import BaseModel, Field
from typing import List, Optional, Any

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class ExtractedField(BaseModel):
    value: Optional[str] = None
    source: str = Field(description="One of 'ocr', 'normalized', 'database_match', 'human_verified'")
    confidence: float
    bbox: Optional[BoundingBox] = None
    verified: bool = False

class MedicineEntry(BaseModel):
    id: Optional[str] = None
    medicine_name: ExtractedField
    strength: ExtractedField
    dosage: ExtractedField
    frequency: ExtractedField
    duration: ExtractedField
    timing: ExtractedField
    
class PatientInfo(BaseModel):
    raw_text: str

class PrescriptionResponse(BaseModel):
    success: bool
    prescription_id: str
    filename: str
    patient_info: PatientInfo
    medicines: List[MedicineEntry]
    processing_time_sec: float
