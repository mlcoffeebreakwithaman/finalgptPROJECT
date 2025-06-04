# app/config.py

from pydantic_settings import BaseSettings  # ✅ CORRECT for Pydantic v2
from pathlib import Path

class Settings(BaseSettings):
    # LLM Settings
    GEMINI_API_KEY: str ="AIzaSyBVe2VHlv9NcogJzZAm8b_zkNPkwDsn-n8"  
    EMBEDDING_MODEL_NAME: str = "text-embedding-004"

    # Paths
    DATA_PATH: Path = Path("data")
    INDEX_PATH: Path = DATA_PATH / "faiss_index.bin"
    CHUNKS_FILE_PATH: Path = DATA_PATH / "processed_chunks.json"
    PROGRESS_FILE_PATH: Path = DATA_PATH / "progress.json"

    # Prompt Parameters
    MAX_RELEVANT_CHUNKS: int = 3
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.7
    TOP_P: float = 1.0

    class Config:
        env_file = ".env"

# ✅ Global config instance used across app
settings = Settings()
