"""
MultiTaskAgent - Phase 4
Core AI logic integrating VectorStore (Phase 2), GeminiClient (Phase 3), and PromptEngine (Phase 3).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from app.llm.gemini_client import GeminiClient
from app.llm.prompt_engine import PromptEngine
from app.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)

class MultiTaskAgent:
    def __init__(self):
        """
        Initializes all Phase 2 and Phase 3 dependencies.
        Maintains consistent config with previous phases.
        """
        self.llm = GeminiClient(max_retries=3)  # Phase 3
        self.prompts = PromptEngine()           # Phase 3
        self.vector_store = VectorStore()       # Phase 2
        self.progress_file = Path("data/progress.json")
        self._init_progress_file()

    def _init_progress_file(self) -> None:
        """Ensures progress file exists (matches Phase 2's file handling)"""
        self.progress_file.parent.mkdir(exist_ok=True)
        if not self.progress_file.exists():
            self.progress_file.write_text("{}")

    # --- Core Methods ---
    def explain(self, topic: str) -> str:
        """
        Generates educational explanation using:
        - VectorStore (Phase 2) for context
        - PromptEngine (Phase 3) for templating
        - GeminiClient (Phase 3) for generation
        """
        try:
            chunks = self.vector_store.retrieve(topic)  # Phase 2 integration
            prompt = self.prompts.render(               # Phase 3 integration
                "tutor_prompt.txt",
                {"topic": topic, "context": chunks}
            )
            return self.llm.generate(prompt)            # Phase 3 integration
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
            return "Could not generate explanation. Please try another topic."

    def quiz(self, topic: str) -> Dict:
        """
        Generates quiz JSON using same Phase 2/3 components.
        Returns format matching Phase 1 requirements:
        {
            "questions": [
                {
                    "question": "...",
                    "options": ["...", "..."],
                    "correct_index": 0,
                    "explanation": "..."
                }
            ]
        }
        """
        try:
            chunks = self.vector_store.retrieve(topic)
            prompt = self.prompts.render(
                "quiz_prompt.txt",
                {"topic": topic, "context": chunks}
            )
            return json.loads(self.llm.generate_structured(prompt))
        except json.JSONDecodeError:
            logger.error("Quiz JSON parsing failed")
            return {"error": "Failed to generate valid quiz format"}
        except Exception as e:
            logger.error(f"Quiz generation failed: {str(e)}")
            return {"error": "Quiz generation failed. Please try again."}

    def submit_answer(self, user_answer: str, correct_answer: str, topic: str) -> Dict:
        """
        Processes quiz answers and updates progress.
        Matches Phase 1's required return format:
        {
            "correct": bool,
            "feedback": str,
            "progress": dict
        }
        """
        is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
        feedback = (
            "Correct! Well done." if is_correct 
            else f"Incorrect. The right answer is: {correct_answer}"
        )
        
        progress = self._update_progress(
            topic,
            {
                "correct": is_correct,
                "user_answer": user_answer,
                "correct_answer": correct_answer
            }
        )
        
        return {
            "correct": is_correct,
            "feedback": feedback,
            "progress": progress
        }

    def recommend(self) -> str:
        """
        Generates study recommendations using:
        - Progress data from Phase 2's JSON format
        - Phase 3's prompt templating
        """
        try:
            progress = self._load_progress()
            prompt = self.prompts.render(
                "recommend_prompt.txt",
                {"progress": progress}
            )
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            return "Could not generate recommendations. Please try again later."

    # --- Progress Tracking ---
    def _update_progress(self, topic: str, result: Dict) -> Dict:
        """
        Updates progress JSON (matches Phase 2's file handling).
        Returns updated progress data.
        """
        progress = self._load_progress()
        
        # Initialize topic if needed
        if topic not in progress:
            progress[topic] = {
                "attempts": 0,
                "correct": 0,
                "history": []
            }
        
        # Update stats
        progress[topic]["attempts"] += 1
        if result["correct"]:
            progress[topic]["correct"] += 1
        progress[topic]["history"].append(result)
        
        # Limit history size (matches Phase 1 requirements)
        if len(progress[topic]["history"]) > 10:
            progress[topic]["history"] = progress[topic]["history"][-10:]
        
        # Save (Phase 2-style file ops)
        try:
            self.progress_file.write_text(json.dumps(progress, indent=2))
            return progress
        except Exception as e:
            logger.error(f"Progress save failed: {str(e)}")
            return {}

    def _load_progress(self) -> Dict:
        """Loads progress data with Phase 2's error handling"""
        try:
            return json.loads(self.progress_file.read_text())
        except Exception as e:
            logger.error(f"Progress load failed: {str(e)}")
            return {}