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

# --- In-Memory Cache ---
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
    """Downloads and extracts the full text from a PDF, with no page limits."""
    try:
        response = requests.get(url, timeout=300) # Generous 5-minute timeout for huge files
        response.raise_for_status()
        pdf_content = response.content
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # --- NO PAGE LIMIT: Process the entire document for maximum accuracy ---
        text = "".join(page.get_text() for page in doc)
        doc.close()
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def get_text_chunks_advanced(text: str) -> List[str]:
    """Splits text into semantically meaningful chunks."""
    chunks = text.split('\n\n')
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 1500:
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            final_chunks.extend(sentences)
        else:
            final_chunks.append(chunk)
    return [c.strip() for c in final_chunks if c.strip()]

async def get_single_answer(question: str, cached_data: Dict[str, Any]) -> str:
    """Processes one question using Multi-Query RAG for maximum accuracy."""
    text_chunks = cached_data["chunks"]
    chunk_embeddings = cached_data["embeddings"]
    generative_model = genai.GenerativeModel('gemini-1.5-pro')
    
    for attempt in range(2): # Retry once on failure
        try:
            # --- NEW: Multi-Query Generation for better retrieval ---
            query_gen_prompt = f"""You are an expert at rephrasing questions for a retrieval system.
            Given the following question, generate 3 additional, different phrasings of it that cover different angles or keywords.
            Your output MUST be a valid JSON array of strings.
            
            Original Question: "{question}"
            
            JSON Array of Rephrased Questions:
            """
            query_gen_model = genai.GenerativeModel('gemini-1.5-flash')
            response = await asyncio.to_thread(lambda: query_gen_model.generate_content(query_gen_prompt))
            
            try:
                # Robust JSON parsing
                cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                rephrased_questions = json.loads(cleaned_text)
                all_queries = [question] + rephrased_questions
            except json.JSONDecodeError:
                logging.warning(f"Could not parse rephrased questions for '{question}'. Falling back to original question only.")
                all_queries = [question]

            # --- Embed all queries ---
            query_embeddings = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=all_queries, task_type="RETRIEVAL_QUERY")['embedding']
            )

            # --- Retrieve and combine results for all queries ---
            all_top_indices = set()
            for embedding in query_embeddings:
                dot_products = np.dot(np.array(chunk_embeddings), np.array(embedding))
                # Retrieve top 3 chunks for each query to get a diverse set of results
                top_indices = np.argsort(dot_products)[-3:][::-1]
                all_top_indices.update(top_indices)

            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in all_top_indices])
            
            # --- Final, High-Accuracy Prompt ---
            final_prompt = f"""
                You are a meticulous AI research analyst. Your task is to provide a single, definitive answer to the "Question" by strictly following these instructions, using *only* the provided "Sources".

                **Instructions:**
                1.  **Analyze Sources:** Carefully read all provided sources to understand the context.
                2.  **Extract Key Details:** Identify and extract all specific quantitative details (e.g., numbers, percentages, time periods like "30 days") and the most critical conditions or clauses related to the question.
                3.  **Formulate Final Answer:** Synthesize the extracted details into a single, comprehensive, and well-written sentence that directly answers the question. Your answer MUST include the specific quantitative details and the most important condition you found.
                4.  **Handle Missing Information:** If the sources do not contain enough information to answer the question, you must state: "Based on the provided information, a definitive answer could not be found."
                5.  **Output Format:** Your entire output must be ONLY the single, final answer sentence. Do not include your thought process, any introductory phrases, or any text other than the final answer.

                **Sources:**
                ---
                {relevant_context}
                ---

                **Question:** {question}

                **Final Answer:**
            """
            final_response = await asyncio.to_thread(lambda: generative_model.generate_content(final_prompt))
            return final_response.text.strip()

        except google_exceptions.ResourceExhausted as e:
            wait_time = 15
            logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
            if attempt < 1: await asyncio.sleep(wait_time)
            else: return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}")
            return f"Error processing question: {e}"
    return "Error: Failed to get an answer after all attempts."

# --- API Endpoint with Final Caching & Parallel Logic ---
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
    
    # --- Process all questions in PARALLEL for maximum speed ---
    tasks = [get_single_answer(q, cached_data) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}
