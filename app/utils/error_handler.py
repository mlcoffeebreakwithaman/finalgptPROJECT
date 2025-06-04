# app/utils/error_handler.py

from fastapi import HTTPException
from app.utils.logging_utils import logger

class AIError(Exception):
    """Base class for StudyGPT AI-related errors."""
    pass

class RetrievalError(AIError):
    """Raised when context retrieval from VectorStore fails."""
    pass

class LLMError(AIError):
    """Raised when the Gemini API fails."""
    pass

def handle_exception(exc: Exception, context: str = "Unknown"):
    """
    Logs the exception and raises a standardized HTTP error.
    Can be used in API routes or agents.
    """
    logger.error(f"[{context}] {type(exc).__name__}: {str(exc)}")
    raise HTTPException(status_code=500, detail=f"Internal error in {context}: {str(exc)}")
