from fastapi import APIRouter
from app.api.v1.endpoints import learning, onboarding, memory

api_router = APIRouter()
api_router.include_router(
    learning.router, prefix="/learning", tags=["Learning Intelligence"]
)
api_router.include_router(
    onboarding.router, prefix="/onboarding", tags=["Identity & Onboarding Intelligence"]
)
api_router.include_router(
    memory.router, prefix="/memory", tags=["Memory Intelligence"]
)
