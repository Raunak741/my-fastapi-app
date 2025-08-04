## main.py

import os
import requests
import io
import json
import logging
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
    raise ValueError("GEMINI_API_KEY or TEAM_TOKEN not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Authentication ---
security = HTTPBearer()

def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Pydantic Models for Request and Response ---
class ApiRequest(BaseModel):
    documents: str
    questions: List[str]

# THIS IS THE REQUIRED RESPONSE FORMAT
class ApiResponse(BaseModel):
    answers: List[str]

# --- Core Logic Functions ---
def get_pdf_text_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        response = requests.get(url, timeout=30) # 30-second timeout for the download
        response.raise_for_status()
        
        pdf_stream = io.BytesIO(response.content)
        reader = PdfReader(pdf_stream)
        
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def get_ai_answers_in_batch(full_document_text: str, questions: List[str]) -> List[str]:
    """Generates answers for a batch of questions from the Gemini model."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    prompt = f"""
        You are an AI adjudicator. Based ONLY on the Document Text provided, answer the questions.
        Your entire output must be a single, valid JSON array of strings, where each string is the direct answer to the corresponding question.
        The number of answers must exactly match the number of questions. Do not include any other text, reasoning, or markdown.

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
            logging.error("AI response format is invalid (mismatched answer count or not a list).")
            # Return error messages for each question if format is wrong
            return [f"Error: AI response format is invalid."] * len(questions)
            
    except Exception as e:
        logging.error(f"Failed to get AI response: {e}")
        # Return error messages for each question on failure
        return [f"Error: Failed to generate AI response. Details: {e}"] * len(questions)

# --- API Endpoint (Synchronous) ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    """
    Main endpoint to process documents and questions synchronously.
    """
    logging.info("Starting synchronous processing for request...")
    
    # Step 1: Get PDF text. This can be slow.
    pdf_text = get_pdf_text_from_url(request.documents)
    
    # Step 2: Get answers from AI. This can also be slow.
    answers = get_ai_answers_in_batch(pdf_text, request.questions)
    
    logging.info("Processing complete. Returning answers.")
    
    # Step 3: Return the answers in the required format.
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}