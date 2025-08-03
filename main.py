import os
import requests
import io
import time
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
import google.api_core.exceptions

# --- Configuration and Initialization ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Authentication ---
TEAM_TOKEN = "feb5618c6bc6478c2e5c71edb6cd21445da1188ca4836db58d25a976f4441259"
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

class ApiResponse(BaseModel):
    answers: List[str]

# --- Core Logic Functions ---
def get_pdf_text_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        pdf_stream = io.BytesIO(response.content)
        reader = PdfReader(pdf_stream)
        
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
            
        return text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")
    except Exception as e:
        # Catches errors from PdfReader or other unexpected issues
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def get_ai_answers_in_batch(full_document_text: str, questions: List[str]) -> List[str]:
    """Generates answers for a batch of questions from the Gemini model."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create a numbered list of questions for the prompt
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    prompt = f"""
        You are a specialized AI adjudicator. Your only function is to answer a list of questions based *exclusively* on the provided Document Text. You must follow these instructions with absolute precision.

        **Document Text:**
        ---
        {full_document_text}
        ---

        **Questions:**
        {formatted_questions}

        **Instructions:**

        1.  **REASONING PHASE (Internal Thought Process for EACH question):**
            -   For each question, identify all key facts (e.g., condition, age, policy duration).
            -   Scan the ENTIRE Document Text to find every relevant clause for that question.
            -   Analyze the hierarchy of clauses. An explicit exclusion overrides a general benefit. The longest waiting period applies.
            -   Perform a logical deduction for each question based on the most specific rule.

        2.  **FINAL ANSWER PHASE:**
            -   Based on your reasoning, formulate a single, direct, and concise sentence that answers each question.
            -   Your entire output MUST be a single, valid JSON array of strings, where each string is the answer to the corresponding question in the list.
            -   The number of answers in the JSON array must exactly match the number of questions.
            -   Do not include your reasoning, any introductory phrases, or any text other than the final JSON array.

        **EXAMPLE OF YOUR TASK:**
        *Questions:*
        1. Are charges for a cervical collar covered?
        2. Is room rent covered up to Rs. 5000?

        *Your Final Answer (Your only output):*
        [
          "No, a cervical collar is explicitly listed as a non-payable item in Annexure I.",
          "Yes, the policy covers room rent up to a limit of Rs. 5000 per day."
        ]

        Now, perform this task for the provided questions and document. Your output must only be the final JSON array.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # The response should be a JSON array string, so we parse it.
            # First, clean up any potential markdown code fences.
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            answers = json.loads(cleaned_text)
            if isinstance(answers, list) and len(answers) == len(questions):
                return answers
            else:
                # If the response is not a list or the count doesn't match, raise an error to retry.
                raise ValueError("AI response format is invalid.")
        except (json.JSONDecodeError, ValueError) as e:
            # Handle cases where the AI doesn't return valid JSON
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return [f"Error: Failed to parse AI response after multiple retries. Details: {e}"] * len(questions)
        except google.api_core.exceptions.GoogleAPICallError as e:
            # Handle specific API call errors
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return [f"Error: The AI model returned an API error: {e}"] * len(questions)
        except Exception as e:
            # Handle other potential exceptions
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return [f"Error: An unexpected error occurred after several retries: {e}"] * len(questions)

@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    """Main endpoint to process documents and questions."""
    pdf_text = get_pdf_text_from_url(request.documents)
    
    # Send all questions in a single batch request to the AI
    answers = get_ai_answers_in_batch(pdf_text, request.questions)
    
    return ApiResponse(answers=answers)

# To run this app, save it as main.py and run: uvicorn main:app --reload
