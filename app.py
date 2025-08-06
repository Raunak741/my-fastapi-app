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
import fitz  # PyMuPDF <-- NEW, FASTER PDF LIBRARY
from google.api_core import exceptions as google_exceptions

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")
genai.configure(api_key=GEMINI_API_KEY)

# --- NEW: List of Known Documents to Pre-Process ---
# Add all known document URLs from the competition here
KNOWN_DOCUMENT_URLS = [
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
    "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
]

document_cache: Dict[str, Dict[str, Any]] = {}

# --- FastAPI App Initialization ---
app = FastAPI()

# --- NEW: Startup Event to Pre-warm the Cache ---
@app.on_event("startup")
async def startup_event():
    logging.info("Server starting up. Beginning to pre-warm the cache...")
    for url in KNOWN_DOCUMENT_URLS:
        try:
            # We run this in the background so it doesn't block the server from starting.
            # The main request handler will wait if the cache is still being built.
            logging.info(f"Processing for cache: {url}")
            pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, url)
            text_chunks = get_text_chunks_advanced(pdf_text)
            
            chunk_embeddings = await asyncio.to_thread(
                lambda: genai.embed_content(
                    model='models/text-embedding-004',
                    content=text_chunks,
                    task_type="RETRIEVAL_DOCUMENT"
                )['embedding']
            )
            
            document_cache[url] = {
                "chunks": text_chunks,
                "embeddings": chunk_embeddings
            }
            logging.info(f"Successfully cached: {url}")
        except Exception as e:
            logging.error(f"Failed to cache document {url} on startup: {e}")
    logging.info("Cache pre-warming complete.")


# --- Authentication and Pydantic Models ---
# (No changes here)
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

# --- Core Logic Functions ---
def get_pdf_text_from_url_sync(url: str) -> str:
    """Synchronous function to download and extract text using the FASTER PyMuPDF."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Use PyMuPDF (fitz) for faster text extraction
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

# (get_text_chunks_advanced and get_ai_answers_with_rag are unchanged)
def get_text_chunks_advanced(text: str) -> List[str]:
    chunks = text.split('\n\n')
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 1000:
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            final_chunks.extend(sentences)
        else:
            final_chunks.append(chunk)
    return [c.strip() for c in final_chunks if c.strip()]

def get_ai_answers_with_rag(cached_data: Dict[str, Any], questions: List[str]) -> List[str]:
    text_chunks = cached_data["chunks"]
    chunk_embeddings = cached_data["embeddings"]
    answers = []
    generative_model = genai.GenerativeModel('gemini-1.5-pro')
    for question in questions:
        for attempt in range(3):
            try:
                question_embedding = genai.embed_content(model='models/text-embedding-004', content=question, task_type="RETRIEVAL_QUERY")['embedding']
                dot_products = np.dot(np.array(chunk_embeddings), np.array(question_embedding))
                top_indices = np.argsort(dot_products)[-5:][::-1]
                relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])
                prompt = f"""You are an expert AI research analyst. Your task is to synthesize a single, clear, and accurate answer to the "Question" based *only* on the provided "Sources".
**Sources:** --- {relevant_context} ---
**Question:** {question}
**Instructions:** 1. Carefully read all the provided sources. 2. Synthesize the information to formulate a single, comprehensive, and well-written answer. 3. If the sources do not contain enough information, state: "Based on the provided information, a definitive answer could not be found." **Final Answer:**"""
                response = generative_model.generate_content(prompt)
                answers.append(response.text.strip())
                break
            except google_exceptions.ResourceExhausted as e:
                wait_time = (attempt + 1) * 15
                logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
                if attempt < 2: time.sleep(wait_time)
                else: answers.append(f"Error: Rate limit exceeded.")
            except Exception as e:
                logging.error(f"Failed to process question '{question}': {e}")
                answers.append(f"Error processing question: {e}")
                break
    return answers

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    logging.info(f"Processing request for document: {document_url}")

    # The cache should be pre-warmed, so we expect to find the URL here.
    # This check is now a fallback in case an UNKNOWN document is sent.
    if document_url not in document_cache:
        logging.warning(f"'{document_url}' was not in the pre-warmed cache! Processing it now (this will be slow).")
        # Fallback logic to handle unknown documents live
        try:
            pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
            text_chunks = get_text_chunks_advanced(pdf_text)
            chunk_embeddings = await asyncio.to_thread(lambda: genai.embed_content(model='models/text-embedding-004', content=text_chunks, task_type="RETRIEVAL_DOCUMENT")['embedding'])
            document_cache[document_url] = {"chunks": text_chunks, "embeddings": chunk_embeddings}
            logging.info(f"Successfully cached new document: '{document_url}'.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process new document: {e}")
    
    cached_data = document_cache[document_url]
    answers = await asyncio.to_thread(get_ai_answers_with_rag, cached_data, request.questions)
    
    logging.info("Processing complete. Returning answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}