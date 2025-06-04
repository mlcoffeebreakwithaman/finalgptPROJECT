"""
API Models - StudyGPT_AI
Updated for Pydantic v2 compatibility
"""

from typing import List, Dict, Literal, Optional, Annotated
from pydantic import BaseModel, Field
from pydantic.types import conint

# --------------------------
# REQUEST MODELS
# --------------------------

class AskRequest(BaseModel):
    """Request model for /api/ask endpoint"""
    topic: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Learning topic to explain",
        examples=["photosynthesis"]
    )

class QuizRequest(BaseModel):
    """Request model for /api/quiz endpoint"""
    topic: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Topic for quiz generation",
        examples=["mitosis"]
    )
    difficulty: Literal['easy', 'medium', 'hard'] = Field(
        default='medium',
        description="Difficulty level for quiz questions"
    )

class SubmitAnswerRequest(BaseModel):
    """Request model for /api/submit-answer endpoint"""
    topic: str = Field(..., description="Question topic")
    user_answer: str = Field(..., description="Student's answer")
    correct_answer: str = Field(..., description="Expected correct answer")

# --------------------------
# RESPONSE MODELS 
# --------------------------

from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field

class QuizQuestion(BaseModel):
    """Individual quiz question structure"""
    question: str = Field(..., description="The question text")
    options: List[str] = Field(
        ...,
        min_length=3,  # ✅ Correct for Pydantic v2 (replaces min_items)
        max_length=5,  # ✅ Correct for Pydantic v2 (replaces max_items)
        description="Answer choices"
    )
    correct_index: int = Field(
        ...,
        ge=0,
        lt=4,
        description="Index of correct answer (0-based)"
    )
    explanation: str = Field(..., description="Why this answer is correct")

class QuizResponse(BaseModel):
    """Response model for /api/quiz endpoint"""
    questions: List[QuizQuestion] = Field(
        ...,
        min_length=1,
        description="Generated quiz questions"
    )

class AskResponse(BaseModel):
    """Response model for /api/ask endpoint"""
    explanation: str = Field(..., description="Generated explanation text")

class SubmitAnswerResponse(BaseModel):
    """Response model for /api/submit-answer endpoint"""
    correct: bool = Field(..., description="Whether answer was correct")
    feedback: str = Field(..., description="Explanation of correctness")
    progress: Dict[str, float] = Field(
        default_factory=dict,
        description="Updated progress metrics"
    )

class RecommendationResponse(BaseModel):
    """Response model for /api/recommend endpoint"""
    recommendations: List[str] = Field(
        ...,
        description="Suggested study topics",
        examples=[["cell division", "dna replication"]]
    )

class ProgressSummaryResponse(BaseModel):
    """Response model for /api/progress endpoint"""
    topics: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Progress by topic with attempt/correct counts",
        examples=[{
            "mitosis": {"attempts": 5, "correct": 3},
            "meiosis": {"attempts": 2, "correct": 1}
        }]
    )

# --------------------------
# ERROR MODELS
# --------------------------

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    details: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional error context"
    )