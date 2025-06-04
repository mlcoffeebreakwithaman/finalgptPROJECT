#!/usr/bin/env python3
"""
StudyGPT Knowledge Base Ingestion Script
Processes textbook PDF into FAISS index and chunked JSON for retrieval.
"""

import os
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()  # Loads .env file

# Configuration
CHUNK_SIZE = 1000  # characters
OVERLAP = 200       # characters between chunks
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> None:
    """Load environment variables and configure Gemini"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise

def chunk_text(text: str) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks with metadata"""
    chunks = []
    i = 0
    chunk_id = 0
    
    while i < len(text):
        chunk = text[i:i + CHUNK_SIZE]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk,
            "metadata": {
                "start_char": i,
                "end_char": i + len(chunk),
                "source": "textbook"
            }
        })
        chunk_id += 1
        i += (CHUNK_SIZE - OVERLAP)
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Gemini API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        raise

def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    """Create and populate FAISS index"""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    np_embeddings = np.array(embeddings).astype('float32')
    index.add(np_embeddings)
    return index

def run_ingestion(pdf_path: Path) -> None:
    """Main ingestion workflow"""
    logger.info("Starting knowledge base ingestion...")
    
    # 1. Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        raise ValueError("No text extracted from PDF")
    
    # 2. Chunk text
    chunks = chunk_text(raw_text)
    
    # 3. Generate embeddings
    embeddings = []
    valid_chunks = []
    
    for chunk in chunks:
        try:
            embedding = get_embedding(chunk["text"])
            embeddings.append(embedding)
            valid_chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Skipping chunk {chunk['id']}: {str(e)}")
    
    # 4. Build and save index
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(DATA_DIR / "faiss_index.bin"))
    
    # 5. Save processed chunks
    with open(DATA_DIR / "processed_chunks.json", 'w') as f:
        json.dump(valid_chunks, f, indent=2)
    
    logger.info(f"Ingestion complete. Index: {len(valid_chunks)} chunks")

if __name__ == "__main__":
    try:
        load_config()
        pdf_path = Path("data/textbook.pdf")  # Update with your PDF path
        if not pdf_path.exists():
            raise FileNotFoundError(f"Textbook PDF not found at {pdf_path}")
        
        run_ingestion(pdf_path)
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise