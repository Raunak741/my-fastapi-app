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

# Securely load both keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")

# Validate that the keys were loaded successfully
if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")

genai.configure(api_key=GEMINI_API_KEY)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Authentication ---
security = HTTPBearer()

def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates the bearer token against the one loaded from the environment."""
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Pydantic Models for Request and Response ---
class ApiRequest(BaseModel):
    documents: str
    questions: List[str]

class ApiResponse(BaseModel):
    answers: List[str]

# --- Core Logic Functions ---
def get_pdf_text_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        response = requests.get(url, timeout=30)
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
    """Generates answers using the Gemini model with your improved, precise prompt."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    prompt = f"""
        You are a specialized AI adjudicator. Your only function is to answer a list of questions based *exclusively* on the provided Document Text. You must be extremely precise.

        **Document Text:**
        ---
        {full_document_text}
        ---

        **Questions:**
        {formatted_questions}

        **Instructions:**

        1.  **REASONING PHASE (Internal Thought Process for EACH question):**
            -   For each question, identify the key terms.
            -   Scan the ENTIRE Document Text to find the most relevant clauses.
            -   **Crucially, extract all specific numbers, percentages (e.g., 1%), time periods (e.g., 30 days, 24 months), and monetary limits.**
            -   Analyze the hierarchy of clauses. An explicit exclusion overrides a general benefit.

        2.  **FINAL ANSWER PHASE:**
            -   For each question, formulate a single, direct sentence.
            -   **Your answer MUST include the specific quantitative details (numbers, percentages, times) you found.**
            -   Briefly include the most important condition for the answer, if applicable.
            -   Your entire output MUST be a single, valid JSON array of strings, where each string is the answer to the corresponding question. The number of answers must exactly match the number of questions.

        **EXAMPLE OF YOUR TASK:**
        *Questions:*
        1. Are charges for a cervical collar covered?
        2. What is the room rent limit for Plan A?

        *Your Final Answer (Your only output):*
        [
          "No, a cervical collar is explicitly listed as a non-payable item in Annexure I.",
          "For Plan A, the daily room rent is capped at 1% of the Sum Insured."
        ]

        Now, perform this task for the provided questions and document. Your output must only be the final JSON array.
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

# --- API Endpoint (Synchronous) ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    """
    Main endpoint to process documents and questions synchronously and return final answers.
    """
    logging.info(f"Processing request for document: {request.documents}")
    
    pdf_text = get_pdf_text_from_url(request.documents)
    answers = get_ai_answers_in_batch(pdf_text, request.questions)
    
    logging.info("Processing complete. Returning answers.")
    
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}