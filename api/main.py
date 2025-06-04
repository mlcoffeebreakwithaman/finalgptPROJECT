"""
StudyGPT API Layer - Phase 5
FastAPI endpoints exposing MultiTaskAgent (Phase 4) functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agents.multitask_agent import MultiTaskAgent
from app.config import Settings
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

# --- Initialization (Matches Phase 4 setup) ---
app = FastAPI(title="StudyGPT API", version="1.0")
settings = Settings()
agent = MultiTaskAgent()  # Initializes all Phase 2-4 components

# CORS Configuration (For Flutter frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models (Phase 1 contract) ---
class AskRequest(BaseModel):
    topic: str

class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"

class AnswerSubmission(BaseModel):
    user_answer: str
    correct_answer: str
    topic: str

# --- API Endpoints ---
@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """Endpoint for topic explanations (Phase 4: explain())"""
    try:
        return {"explanation": agent.explain(request.topic)}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    """Endpoint for quiz generation (Phase 4: quiz())"""
    try:
        return agent.quiz(request.topic)
    except Exception as e:
        raise HTTPException(500, detail=f"Quiz generation failed: {str(e)}")

@app.post("/api/submit-answer")
async def submit_answer(answer: AnswerSubmission):
    """Endpoint for answer evaluation (Phase 4: submit_answer())"""
    try:
        return agent.submit_answer(
            answer.user_answer,
            answer.correct_answer,
            answer.topic
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/api/recommend")
async def get_recommendation():
    """Endpoint for study recommendations (Phase 4: recommend())"""
    try:
        return {"recommendation": agent.recommend()}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/api/progress")
async def get_progress():
    """Endpoint for progress tracking (Phase 4: _load_progress())"""
    try:
        return agent._load_progress()
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Integration verification for all phases"""
    return {
        "status": "ready",
        "components": {
            "llm": "connected" if agent.llm else "offline",
            "vector_store": "loaded" if agent.vector_store else "empty",
            "prompt_engine": "active"
        }
    }