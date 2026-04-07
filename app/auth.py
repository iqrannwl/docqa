"""
Authentication — simple API-key based auth via request header.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import get_settings

settings = get_settings()

api_key_scheme = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def require_api_key(api_key: str = Security(api_key_scheme)) -> str:
    """Dependency — raises 401 if the key is missing or invalid."""
    if not api_key or api_key not in settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass it in the X-API-Key header.",
        )
    return api_key
