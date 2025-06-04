"""
StudyGPT Vector Store for Semantic Search
Handles FAISS index and text chunk retrieval.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from app.config import Settings
load_dotenv()  # Loads .env file

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize with paths from settings or defaults"""
        self.settings = settings or Settings()
        self.index = None
        self.chunks = []
        self._load_data()

    def _load_data(self) -> None:
        """Load FAISS index and processed chunks"""
        try:
            # Load FAISS index
            index_path = Path(self.settings.INDEX_PATH)
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # Load text chunks
            chunks_path = Path(self.settings.CHUNKS_FILE_PATH)
            if chunks_path.exists():
                with open(chunks_path, 'r') as f:
                    self.chunks = json.load(f)
            
            logger.info(f"Loaded {len(self.chunks)} chunks from knowledge base")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            self.index = None
            self.chunks = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for query (matches ingestion method)"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'], dtype='float32').reshape(1, -1)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query
        Returns:
            List of chunks with format:
            [{
                "id": str,
                "text": str,
                "metadata": dict
            }]
        """
        if not self.index or not self.chunks:
            logger.warning("Knowledge base not loaded")
            return []

        try:
            # Embed query
            query_embedding = self._get_embedding(query)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Return matched chunks
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    results.append(self.chunks[idx])
            
            return results
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            return []

    def health_check(self) -> Dict:
        """Check if vector store is operational"""
        return {
            "index_loaded": self.index is not None,
            "chunks_loaded": len(self.chunks) > 0
        }