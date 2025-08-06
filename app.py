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

# --- Caching Dictionary ---
document_cache: Dict[str, Dict[str, Any]] = {}

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
    """Downloads and extracts text using the fast PyMuPDF library."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        # --- STRATEGIC CHANGE: Process only the first 100 pages for large docs ---
        doc = fitz.open(stream=response.content, filetype="pdf")
        
        page_limit = 100
        if doc.page_count > page_limit:
            logging.warning(f"Document has {doc.page_count} pages. Processing only the first {page_limit} to save memory.")
            pages_to_process = range(page_limit)
        else:
            pages_to_process = range(doc.page_count)
            
        text = "".join(doc[i].get_text() for i in pages_to_process)
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
        if len(chunk) > 1000:
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            final_chunks.extend(sentences)
        else:
            final_chunks.append(chunk)
    return [c.strip() for c in final_chunks if c.strip()]

async def get_single_answer(question: str, cached_data: Dict[str, Any]) -> str:
    """Processes one question with improved retrieval and Chain-of-Thought prompting."""
    text_chunks = cached_data["chunks"]
    chunk_embeddings = cached_data["embeddings"]
    generative_model = genai.GenerativeModel('gemini-1.5-pro')

    for attempt in range(3):
        try:
            question_embedding = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=question, task_type="RETRIEVAL_QUERY")['embedding']
            )
            dot_products = np.dot(np.array(chunk_embeddings), np.array(question_embedding))
            top_indices = np.argsort(dot_products)[-7:][::-1]
            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])
            
            prompt = f"""
                You are an expert AI research analyst. Follow these steps precisely to answer the question based *only* on the provided sources.

                **Step 1: Fact Extraction**
                Carefully read the following sources and identify all facts, figures, and clauses relevant to the question. List them out as bullet points.

                **Step 2: Reasoning**
                Based on the extracted facts, explain your step-by-step reasoning for how you will arrive at the final answer. Consider any conflicting clauses or conditions.

                **Step 3: Final Answer**
                Synthesize your reasoning into a single, clear, and comprehensive final answer. This final answer should be the only thing you output after completing the first two steps internally.

                **Sources:**
                ---
                {relevant_context}
                ---

                **Question:** {question}

                **Final Answer:**
            """
            response = await asyncio.to_thread(lambda: generative_model.generate_content(prompt))
            return response.text.strip()
        except google_exceptions.ResourceExhausted as e:
            wait_time = (attempt + 1) * 10
            logging.warning(f"Rate limit hit on '{question}'. Waiting {wait_time}s...")
            if attempt < 2:
                await asyncio.sleep(wait_time)
            else:
                return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}")
            return f"Error processing question: {e}"
    return "Error: Failed to get an answer after all attempts."

# --- API Endpoint with On-Demand Caching ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    cache_key = document_url.split('?')[0]
    logging.info(f"Processing request for document: {cache_key}")

    if cache_key not in document_cache:
        logging.info(f"'{cache_key}' not in cache. Processing and caching now...")
        try:
            pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
            text_chunks = get_text_chunks_advanced(pdf_text)
            chunk_embeddings = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=text_chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
            )
            document_cache[cache_key] = {"chunks": text_chunks, "embeddings": chunk_embeddings}
            logging.info(f"Successfully cached: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and cache document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    else:
        logging.info(f"Found '{cache_key}' in cache. Using cached data.")
    
    cached_data = document_cache[cache_key]
    
    answers = []
    for question in request.questions:
        answer = await get_single_answer(question, cached_data)
        answers.append(answer)
        await asyncio.sleep(1)

    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}
