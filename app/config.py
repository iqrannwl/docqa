"""
Configuration — loads settings from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str = "AIzaSyD0f3xfDIds1_i1Fp8ogV2mpqS4Aw6rcZs"
    embedding_model: str = "gemini-embedding-001"  # 768-dim, free tier
    llm_model: str = "gemini-2.0-flash"                  # or gemini-2.0-flash-lite
    llm_temperature: float = 0.2
    max_tokens: int = 1024

    # Vector store
    faiss_index_path: str = "data/faiss_index"
    metadata_db_path: str = "data/metadata.db"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_chunks: int = 5

    # Auth
    api_key_header: str = "X-API-Key"
    api_keys: str = "dev-secret-key"
    # Upload limits
    max_upload_size_mb: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
