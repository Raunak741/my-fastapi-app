## app.py

import os
import requests
import io
import json
import logging
import asyncio
import numpy as np
import re
import time
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
from google.api_core import exceptions as google_exceptions
from google.cloud import vision
import faiss # <-- NEW IMPORT FOR ON-DISK INDEXING

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")

if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")

genai.configure(api_key=GEMINI_API_KEY)

# --- On-Disk Cache Management ---
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# We will store the text chunks separately in memory, as they are smaller.
text_chunk_cache: Dict[str, List[str]] = {}


# --- FastAPI App Initialization ---
app = FastAPI()

# --- Authentication ---
security = HTTPBearer()

def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Pydantic Models ---
class ApiRequest(BaseModel):
    documents: str
    questions: List[str]

class ApiResponse(BaseModel):
    answers: List[str]

# --- Core Logic Functions ---
def get_pdf_text_from_url_sync(url: str) -> str:
    """Downloads and extracts all text from a PDF using PyMuPDF."""
    try:
        response = requests.get(url, timeout=120) # Longer timeout for large files
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            raise ValueError("Could not extract text from the PDF.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def get_text_chunks_advanced(text: str) -> List[str]:
    """Splits text into semantically meaningful chunks."""
    chunks = text.split('\n\n')
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 1500: # Slightly larger chunk size
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            final_chunks.extend(sentences)
        else:
            final_chunks.append(chunk)
    return [c.strip() for c in final_chunks if c.strip()]

async def get_single_answer(question: str, text_chunks: List[str], faiss_index_path: str) -> str:
    """Processes one question using the on-disk FAISS index."""
    generative_model = genai.GenerativeModel('gemini-1.5-pro')

    for attempt in range(3):
        try:
            # Load the FAISS index from disk for this search
            index = faiss.read_index(faiss_index_path)
            
            question_embedding = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=question, task_type="RETRIEVAL_QUERY")['embedding']
            )
            
            # FAISS expects a 2D array for searching
            query_vector = np.array([question_embedding], dtype=np.float32)
            
            # Perform the search
            distances, top_indices = index.search(query_vector, 10) # Retrieve top 10 results
            
            # Flatten the indices array
            relevant_indices = top_indices[0]
            
            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in relevant_indices])
            
            final_prompt = f"""
                You are an expert AI research analyst. Your task is to synthesize a single, clear, and accurate answer to the "Question" based *only* on the provided "Sources".

                **Sources:**
                ---
                {relevant_context}
                ---

                **Question:** {question}

                **Instructions:**
                - Synthesize the information to formulate a single, comprehensive, and well-written final answer.
                - If the sources do not contain enough information, state: "Based on the provided information, a definitive answer could not be found."
                - **Your output must ONLY be the final answer sentence.**

                **Final Answer:**
            """
            final_response = await asyncio.to_thread(lambda: generative_model.generate_content(final_prompt))
            return final_response.text.strip()

        except google_exceptions.ResourceExhausted as e:
            wait_time = (attempt + 1) * 10
            logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
            if attempt < 2:
                await asyncio.sleep(wait_time)
            else:
                return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}", exc_info=True)
            return f"Error processing question: {e}"
    return "Error: Failed to get an answer after all attempts."

# --- API Endpoint with On-Disk FAISS Caching ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    cache_key = document_url.split('?')[0].split('/')[-1].replace('.pdf', '') # Use filename as key
    faiss_index_path = os.path.join(CACHE_DIR, f"{cache_key}.index")
    
    logging.info(f"Processing request for document key: {cache_key}")

    if not os.path.exists(faiss_index_path):
        logging.info(f"Index for '{cache_key}' not found on disk. Processing and creating index now (this will be slow)...")
        try:
            pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
            text_chunks = get_text_chunks_advanced(pdf_text)
            
            # Store text chunks in the in-memory cache
            text_chunk_cache[cache_key] = text_chunks
            
            chunk_embeddings = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=text_chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
            )
            
            # --- Build and Save FAISS Index ---
            embeddings_np = np.array(chunk_embeddings, dtype=np.float32)
            d = embeddings_np.shape[1] # Dimension of vectors
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_np)
            faiss.write_index(index, faiss_index_path)
            
            logging.info(f"Successfully created and saved FAISS index for: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and create index for document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    else:
        logging.info(f"Found FAISS index for '{cache_key}' on disk.")
        # Ensure text chunks are loaded if they aren't in memory
        if cache_key not in text_chunk_cache:
             logging.warning(f"Index found but chunks not in memory for {cache_key}. This should not happen frequently.")
             # As a fallback, we'd need to re-process to get chunks, but this indicates a server restart.
             # For the hackathon, we assume the server stays up long enough.
             pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
             text_chunk_cache[cache_key] = get_text_chunks_advanced(pdf_text)

    
    # Retrieve text chunks from the in-memory cache
    current_text_chunks = text_chunk_cache[cache_key]
    
    tasks = [get_single_answer(q, current_text_chunks, faiss_index_path) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}
