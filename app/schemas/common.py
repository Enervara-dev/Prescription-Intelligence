"""
schemas/common.py
------------------
Shared/misc response models.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    vision_api_configured: bool
    database_connected: bool


class ErrorResponse(BaseModel):
    detail: str
