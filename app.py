## main.py

import os
import requests
import io
import time
import json
import logging
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
import google.api_core.exceptions

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Environment Variable Validation ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN") # Best practice: load token from env

if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")

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

# --- Pydantic Models ---
class ApiRequest(BaseModel):
    documents: str
    questions: List[str]

class ApiResponse(BaseModel):
    # This response is returned immediately
    message: str
    task_id: str # A way to track the background job

# --- Core Logic Functions (to be run in background) ---

def process_document_and_get_answers(document_url: str, questions: List[str], task_id: str):
    """
    This entire workflow runs in the background.
    It handles PDF download, text extraction, and AI processing.
    """
    logging.info(f"Task {task_id}: Starting background processing.")
    try:
        # Step 1: Download and Extract PDF Text
        pdf_text = get_pdf_text_from_url(document_url)

        # Step 2: Generate Answers using AI
        answers = get_ai_answers_in_batch(pdf_text, questions)

        # In a real-world app, you would save these answers to a database
        # or send them to a webhook using the task_id.
        logging.info(f"Task {task_id}: Successfully generated answers: {answers}")

    except HTTPException as e:
        # Log HTTP exceptions that occur within the background task
        logging.error(f"Task {task_id}: Failed with HTTPException. Status: {e.status_code}, Detail: {e.detail}")
    except Exception as e:
        # Log any other unexpected errors
        logging.error(f"Task {task_id}: An unexpected error occurred: {e}")

def get_pdf_text_from_url(url: str) -> str:
    """Downloads and extracts text from a PDF URL."""
    try:
        response = requests.get(url, timeout=30) # Add a timeout
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

def get_ai_answers_in_batch(full_document_text: str, questions: List[str]) -> List[str]:
    """Generates answers using the Gemini model with retry logic."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    # Using a clear, structured prompt improves model performance
    prompt = f"""
        You are an AI adjudicator. Based ONLY on the Document Text provided, answer the questions.
        Your entire output must be a single, valid JSON array of strings, where each string is the direct answer to the corresponding question.
        The number of answers must exactly match the number of questions.

        **Document Text:**
        ---
        {full_document_text}
        ---

        **Questions:**
        {formatted_questions}

        **Your Final JSON Array Output:**
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            answers = json.loads(cleaned_text)
            
            if isinstance(answers, list) and len(answers) == len(questions):
                return answers
            else:
                raise ValueError("AI response format is invalid (mismatched answer count or not a list).")
        
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"An unexpected error occurred with the AI model: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the AI model: {e}")

# --- API Endpoint ---

@app.post("/hackrx/run", response_model=ApiResponse, status_code=202)
async def process_request(
    request: ApiRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(check_token)
):
    """
    Accepts the request, responds immediately, and processes the document
    and questions in the background to avoid timeouts.
    """
    task_id = os.urandom(8).hex() # Generate a simple, unique ID for the task
    
    background_tasks.add_task(
        process_document_and_get_answers,
        document_url=request.documents,
        questions=request.questions,
        task_id=task_id
    )
    
    return ApiResponse(
        message="Request accepted. Processing is happening in the background.",
        task_id=task_id
    )

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}