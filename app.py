## main.py

import os
import requests
import io
import json
import logging
import asyncio
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")

if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")

genai.configure(api_key=GEMINI_API_KEY)

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

# --- Core Logic Functions (Now defined as blocking functions) ---
def get_pdf_text_from_url_sync(url: str) -> str:
    """Synchronous function to download and extract text from a PDF URL."""
    try:
        response = requests.get(url, timeout=60) # Increased timeout for download
        response.raise_for_status()
        pdf_stream = io.BytesIO(response.content)
        reader = PdfReader(pdf_stream)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def get_ai_answers_in_batch_sync(full_document_text: str, questions: List[str]) -> List[str]:
    """Synchronous function to generate answers using the Gemini model."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""
        You are a specialized AI adjudicator. Your only function is to answer a list of questions based *exclusively* on the provided Document Text. You must be extremely precise.
        Your answer MUST include the specific quantitative details (numbers, percentages, times) you found.
        Your entire output MUST be a single, valid JSON array of strings, where each string is the answer to the corresponding question. The number of answers must exactly match the number of questions.

        **Document Text:**
        ---
        {full_document_text}
        ---
        **Questions:**
        {formatted_questions}
        **Your Final JSON Array Output:**
    """
    try:
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        answers = json.loads(cleaned_text)
        if isinstance(answers, list) and len(answers) == len(questions):
            return answers
        else:
            logging.error("AI response format is invalid.")
            return ["Error: AI response format is invalid."] * len(questions)
    except Exception as e:
        logging.error(f"Failed to get AI response: {e}")
        return [f"Error: An unexpected error occurred with the AI model: {e}"] * len(questions)

# --- API Endpoint (Asynchronous) ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    """
    Main endpoint that handles slow, blocking I/O operations in a non-blocking way.
    """
    logging.info(f"Processing request for document: {request.documents}")
    try:
        # Run the blocking PDF function in a separate thread
        pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, request.documents)
        
        # Run the blocking AI function in a separate thread
        answers = await asyncio.to_thread(get_ai_answers_in_batch_sync, pdf_text, request.questions)
        
        logging.info("Processing complete. Returning answers.")
        return ApiResponse(answers=answers)
    except HTTPException as e:
        # Re-raise HTTP exceptions to let FastAPI handle them
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred in process_request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}