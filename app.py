## app.py

import os
import requests
import io
import json
import logging
import asyncio
import numpy as np
import re
import time # <-- NEW IMPORT
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from google.api_core import exceptions as google_exceptions # <-- NEW IMPORT for specific exception

# --- Configuration and Initialization ---
# (No changes here)
logging.basicConfig(level=logging.INFO)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")
genai.configure(api_key=GEMINI_API_KEY)

# --- FastAPI App and Authentication ---
# (No changes here)
app = FastAPI()
security = HTTPBearer()
def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Pydantic Models ---
# (No changes here)
class ApiRequest(BaseModel):
    documents: str
    questions: List[str]
class ApiResponse(BaseModel):
    answers: List[str]

# --- Core Logic Functions ---

def get_pdf_text_from_url_sync(url: str) -> str:
    # (No changes here)
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        pdf_stream = io.BytesIO(response.content)
        reader = PdfReader(pdf_stream)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def get_text_chunks_advanced(text: str) -> List[str]:
    # (No changes here)
    chunks = text.split('\n\n')
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 1000:
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            final_chunks.extend(sentences)
        else:
            final_chunks.append(chunk)
    return [c.strip() for c in final_chunks if c.strip()]


def get_ai_answers_with_rag(full_document_text: str, questions: List[str]) -> List[str]:
    """
    Generates answers using an ADVANCED RAG approach with rate-limit handling.
    """
    text_chunks = get_text_chunks_advanced(full_document_text)
    if not text_chunks:
        return ["Error: Document has no text content."] * len(questions)

    try:
        # This can also be rate-limited, so we add a retry here too
        chunk_embeddings = genai.embed_content(
            model='models/text-embedding-004',
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )['embedding']
    except google_exceptions.ResourceExhausted as e:
        logging.warning("Rate limit hit while embedding chunks. Waiting 20 seconds...")
        time.sleep(20)
        chunk_embeddings = genai.embed_content(
            model='models/text-embedding-004',
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )['embedding']
    except Exception as e:
        logging.error(f"Failed to create embeddings for document chunks: {e}")
        return [f"Error creating document embeddings: {e}"] * len(questions)

    answers = []
    generative_model = genai.GenerativeModel('gemini-1.5-pro') # Using Pro for higher quality

    for question in questions:
        for attempt in range(3): # Retry up to 3 times for each question
            try:
                question_embedding = genai.embed_content(
                    model='models/text-embedding-004',
                    content=question,
                    task_type="RETRIEVAL_QUERY"
                )['embedding']

                dot_products = np.dot(np.array(chunk_embeddings), np.array(question_embedding))
                top_indices = np.argsort(dot_products)[-5:][::-1]
                relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])

                prompt = f"""
                    You are an expert AI research analyst. Your task is to synthesize a single, clear, and accurate answer to the "Question" based *only* on the provided "Sources".
                    **Sources:** --- {relevant_context} ---
                    **Question:** {question}
                    **Instructions:**
                    1. Carefully read all the provided sources.
                    2. Synthesize the information to formulate a single, comprehensive, and well-written answer.
                    3. If the sources do not contain enough information, state: "Based on the provided information, a definitive answer could not be found."
                    **Final Answer:**
                """
                
                response = generative_model.generate_content(prompt)
                answers.append(response.text.strip())
                break # If successful, break the retry loop

            except google_exceptions.ResourceExhausted as e:
                wait_time = (attempt + 1) * 15 # Wait 15s, then 30s
                logging.warning(f"Rate limit hit on question '{question}'. Waiting {wait_time}s... Attempt {attempt + 1}/3")
                if attempt < 2:
                    time.sleep(wait_time)
                else:
                    answers.append(f"Error: Rate limit exceeded after multiple retries.")
            
            except Exception as e:
                logging.error(f"Failed to process question '{question}': {e}")
                answers.append(f"Error processing question: {e}")
                break # Don't retry on other errors

    return answers

# --- API Endpoint ---
# (No changes here)
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    logging.info(f"Processing request for document: {request.documents}")
    try:
        pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, request.documents)
        answers = await asyncio.to_thread(get_ai_answers_with_rag, pdf_text, request.questions)
        logging.info("Processing complete. Returning answers.")
        return ApiResponse(answers=answers)
    except Exception as e:
        logging.error(f"An unexpected error occurred in process_request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}