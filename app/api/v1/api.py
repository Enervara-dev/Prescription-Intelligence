"""
api/v1/api.py
--------------
Aggregates all v1 endpoint routers into a single APIRouter.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, medicines, prescriptions, resolve

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(prescriptions.router)
api_router.include_router(medicines.router)
api_router.include_router(resolve.router)
