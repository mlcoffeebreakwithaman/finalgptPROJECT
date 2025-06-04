"""
Gemini API Client - Updated for google-generativeai v0.3.0+ with proper typing
"""

import os
import time
import logging
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GeminiClient:
    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 30,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None
    ):
        """
        Initialize client with proper SDK usage
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model = self._configure_client()

    def _configure_client(self):
        """Configure with proper SDK initialization"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing from .env")
        
        # Initialize the client with proper configuration
        config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )
        
        # Initialize the model with explicit typing
        return genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=config
        )

    def generate(self, prompt: str) -> str:
        """
        Generate content with proper response handling
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                # Proper response handling
                if not response.text:
                    raise ValueError("Empty response from Gemini")
                
                return response.text
                
            except Exception as e:
                last_error = e
                wait_time = min(2 ** attempt, 10)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(wait_time)
                
        raise RuntimeError(
            f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}"
        )

    def generate_structured(self, prompt: str) -> str:
        """
        Generate structured JSON output
        """
        structured_prompt = f"""{prompt}
        
        IMPORTANT: Respond with ONLY valid JSON formatted exactly as specified.
        Do not include any markdown formatting or additional explanations."""
        
        return self.generate(structured_prompt)