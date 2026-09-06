from fastapi import APIRouter
from app.api.v1.endpoints import learning

api_router = APIRouter()
api_router.include_router(
    learning.router, prefix="/learning", tags=["Learning Intelligence"]
)
