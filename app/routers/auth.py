"""
Auth router — token info endpoint.
"""

from fastapi import APIRouter, Depends
from app.auth import require_api_key
from app.schemas import TokenResponse

router = APIRouter()


@router.get("/me", response_model=TokenResponse)
async def auth_me(api_key: str = Depends(require_api_key)):
    """Verify your API key and confirm it's valid."""
    return TokenResponse(api_key=api_key, message="API key is valid.")
