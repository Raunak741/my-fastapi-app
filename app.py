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

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")
genai.configure(api_key=GEMINI_API_KEY)

# --- List of Known Documents ---
KNOWN_DOCUMENT_URLS = [
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    # Add all other known URLs here...
]

# --- Caching and Task Management ---
document_cache: Dict[str, Dict[str, Any]] = {}
cache_warming_tasks: Dict[str, asyncio.Task] = {}

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions (to be run in background) ---
async def process_and_cache_document(url: str):
    """The full, slow process for a single document."""
    cache_key = url.split('?')[0]
    logging.info(f"BACKGROUND: Starting to process and cache {cache_key}")
    try:
        pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, url)
        text_chunks = get_text_chunks_advanced(pdf_text)
        chunk_embeddings = await asyncio.to_thread(
            lambda: genai.embed_content(model='models/text-embedding-004', content=text_chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
        )
        document_cache[cache_key] = {"chunks": text_chunks, "embeddings": chunk_embeddings}
        logging.info(f"BACKGROUND: Successfully cached {cache_key}")
    except Exception as e:
        logging.error(f"BACKGROUND: Failed to cache {cache_key}: {e}")
        # Store failure so we don't retry endlessly
        document_cache[cache_key] = {"error": str(e)}

# --- Startup Event (Now very fast) ---
@app.on_event("startup")
async def startup_event():
    logging.info("Server starting up. Launching background caching tasks...")
    for url in KNOWN_DOCUMENT_URLS:
        cache_key = url.split('?')[0]
        # Launch the processing as a background task that we don't wait for
        task = asyncio.create_task(process_and_cache_document(url))
        cache_warming_tasks[cache_key] = task
    logging.info(f"Launched {len(KNOWN_DOCUMENT_URLS)} background caching tasks. Server is ready.")

# --- Authentication and Pydantic Models (unchanged) ---
security = HTTPBearer()
def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials
class ApiRequest(BaseModel):
    documents: str
    questions: List[str]
class ApiResponse(BaseModel):
    answers: List[str]

# --- Core Logic (unchanged from previous version) ---
def get_pdf_text_from_url_sync(url: str) -> str:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip(): raise ValueError("Could not extract text from the PDF.")
        return text
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def get_text_chunks_advanced(text: str) -> List[str]:
    chunks = text.split('\n\n')
    final_chunks = [c for chunk in chunks for c in (re.split(r'(?<=[.!?]) +', chunk) if len(chunk) > 1000 else [chunk])]
    return [c.strip() for c in final_chunks if c.strip()]

async def get_single_answer(question: str, cached_data: Dict[str, Any]) -> str:
    text_chunks, chunk_embeddings = cached_data["chunks"], cached_data["embeddings"]
    generative_model = genai.GenerativeModel('gemini-1.5-pro')
    for attempt in range(3):
        try:
            question_embedding = await asyncio.to_thread(lambda: genai.embed_content(model='models/text-embedding-004', content=question, task_type="RETRIEVAL_QUERY")['embedding'])
            dot_products = np.dot(np.array(chunk_embeddings), np.array(question_embedding))
            top_indices = np.argsort(dot_products)[-5:][::-1]
            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])
            prompt = f"""Based ONLY on the provided Sources, synthesize a single, clear, and accurate answer for the Question. **Sources:** --- {relevant_context} --- **Question:** {question} **Instructions:** If the answer is not in the sources, state: "Based on the provided information, a definitive answer could not be found." **Final Answer:**"""
            response = await asyncio.to_thread(lambda: generative_model.generate_content(prompt))
            return response.text.strip()
        except google_exceptions.ResourceExhausted as e:
            wait_time = (attempt + 1) * 10
            logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
            if attempt < 2: await asyncio.sleep(wait_time)
            else: return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}")
            return f"Error processing question: {e}"
    return "Error: Failed to get an answer after all attempts."

# --- API Endpoint (Now handles waiting for background tasks) ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    cache_key = document_url.split('?')[0]
    logging.info(f"Processing request for document: {cache_key}")

    # Check if a background caching task for this key is running
    if cache_key in cache_warming_tasks:
        task = cache_warming_tasks[cache_key]
        if not task.done():
            logging.info(f"Cache for {cache_key} is still warming up. Waiting for it to complete...")
            await task # Wait for the specific background task to finish
    
    # Check the cache *after* potentially waiting
    if cache_key not in document_cache:
        # This handles unknown documents or failed caching on startup
        logging.warning(f"'{cache_key}' was not pre-warmed. Processing live...")
        await process_and_cache_document(document_url)

    cached_data = document_cache.get(cache_key)
    if not cached_data or "error" in cached_data:
        raise HTTPException(status_code=500, detail=f"Could not process or find data for document: {cache_key}")

    tasks = [get_single_answer(q, cached_data) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}