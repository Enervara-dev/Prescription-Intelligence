from pydantic import BaseModel
from typing import Optional

class VerifiedField(BaseModel):
    value: Optional[str]
    is_verified: bool

class VerifyMedicineEntry(BaseModel):
    id: str
    medicine_name: VerifiedField
    strength: VerifiedField
    dosage: VerifiedField
    frequency: VerifiedField
    duration: VerifiedField
    timing: VerifiedField

class VerifyPrescriptionRequest(BaseModel):
    medicines: list[VerifyMedicineEntry]
