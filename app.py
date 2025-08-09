# --- FINAL HIGH-ACCURACY VERSION ---
# Key Improvements:
# 1. Intelligent Final Prompt: The prompt now distinguishes between yes/no and informational questions.
#    - It no longer incorrectly prepends "Yes." or "No." to "What is..." type questions.
#    - It is more forceful in instructing the LLM to synthesize facts rather than comment on the source text.
# 2. Wider Retrieval Scope: The RAG system now retrieves the top 7 chunks (up from 5) for each query.
#    - This significantly increases the likelihood of finding obscure but critical information (like the AYUSH clause).
# 3. Enhanced Robustness: Improved retry logic and error logging.

import os
import requests
import io
import json
import logging
import asyncio
import numpy as np
import re
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Set
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
    """Downloads and extracts the full text from a PDF."""
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        pdf_content = response.content
        
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF from URL {url}: {e}")

def get_text_chunks_advanced(text: str) -> List[str]:
    """
    Splits text into semantically meaningful chunks. This version uses a more robust
    strategy by first splitting by paragraphs and then subdividing large paragraphs by sentences.
    """
    initial_chunks = text.split('\n\n')
    final_chunks = []
    for chunk in initial_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # If a chunk is too large, split it by sentences. A more robust regex is used.
        if len(chunk) > 1500:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            final_chunks.extend([s.strip() for s in sentences if s.strip()])
        else:
            final_chunks.append(chunk)
    return [c for c in final_chunks if c] # Filter out any potential empty strings

def add_contextual_window(indices: Set[int], max_index: int) -> List[int]:
    """
    ACCURACY IMPROVEMENT: Contextual Windowing.
    For each retrieved index, also include the preceding and succeeding chunks
    to provide the LLM with more context for better interpretation.
    """
    expanded_indices = set(indices)
    for i in indices:
        if i > 0:
            expanded_indices.add(i - 1)
        if i < max_index:
            expanded_indices.add(i + 1)
    return sorted(list(expanded_indices))

async def get_single_answer(question: str, cached_data: Dict[str, Any]) -> str:
    """
    Processes one question using Multi-Query RAG, Contextual Windowing, 
    and an advanced, intelligent prompt that adapts to the question type.
    """
    text_chunks = cached_data["chunks"]
    chunk_embeddings = cached_data["embeddings"]
    generative_model = genai.GenerativeModel('gemini-1.5-pro')
    
    for attempt in range(3): # Retry up to 2 times on failure
        try:
            # Step 1: Generate multiple query variations for better semantic search
            query_gen_prompt = f"""You are an expert in semantic search. Given the user's question, generate 3 additional, rephrased versions of the question to improve the quality of information retrieval. The rephrased questions should be diverse and cover different angles of the original question.

Original Question: "{question}"

Output your response ONLY as a valid JSON array of strings. Example: ["rephrased question 1", "rephrased question 2", "rephrased question 3"]"""
            
            query_gen_model = genai.GenerativeModel('gemini-1.5-flash')
            response = await asyncio.to_thread(lambda: query_gen_model.generate_content(query_gen_prompt))
            
            try:
                cleaned_text = re.search(r'\[.*\]', response.text, re.DOTALL).group(0)
                rephrased_questions = json.loads(cleaned_text)
                all_queries = [question] + rephrased_questions
            except (json.JSONDecodeError, AttributeError):
                logging.warning(f"Failed to parse JSON for query variations. Using original question only for: '{question}'")
                all_queries = [question]

            # Step 2: Embed all queries
            query_embeddings_result = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=all_queries, task_type="RETRIEVAL_QUERY")
            )
            query_embeddings = query_embeddings_result['embedding']

            # Step 3: Retrieve relevant chunks for each query
            all_top_indices = set()
            for embedding in query_embeddings:
                dot_products = np.dot(np.array(chunk_embeddings), np.array(embedding))
                # ACCURACY IMPROVEMENT: Retrieve more context (top 7) to find obscure clauses
                top_indices = np.argsort(dot_products)[-7:][::-1]
                all_top_indices.update(top_indices)

            # Step 4: Apply Contextual Windowing for richer context
            expanded_indices = add_contextual_window(all_top_indices, len(text_chunks) - 1)
            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in expanded_indices])
            
            # Step 5: Generate the final answer using an advanced, multi-step prompt
            final_prompt = f"""
                You are a world-class AI analyst for legal and policy documents. Your task is to provide a single, definitive answer to the user's question based *only* on the provided "Sources". You must follow these instructions precisely.

                ### Instructions ###
                1.  **Analyze the Question Type:** First, determine if the question is a binary question that can be answered with "Yes" or "No" (e.g., "Does the policy cover X?") OR if it is an informational question seeking a specific detail (e.g., "What is the waiting period for X?", "How is X defined?").

                2.  **Scrutinize Sources:** Carefully read all provided sources to find the most relevant facts, figures, and clauses. Your primary goal is to extract the specific information that directly answers the question.

                3.  **Formulate the Final Answer:**
                    * **For Binary (Yes/No) Questions:**
                        * Begin your answer with "Yes," if the sources confirm the condition.
                        * Begin your answer with "No," if the sources deny the condition.
                        * After the "Yes," or "No,", write a single, comprehensive paragraph synthesizing all supporting details, conditions, and numbers from the sources to justify your decision.
                    * **For Informational Questions (What, How, etc.):**
                        * DO NOT begin with "Yes" or "No".
                        * Directly answer the question by starting with the information requested (e.g., "The waiting period for X is...", "X is defined as...").
                        * Synthesize all relevant details into a single, comprehensive paragraph.
                
                4.  **Handle Missing Information:** If, and only if, the sources contain absolutely no information to answer the question, you must state: "Based on the provided information, a definitive answer could not be found." Use this as a last resort.

                5.  **Final Output Rules:**
                    * Your entire output must be ONLY the single, final answer paragraph.
                    * Do not include your thought process or any introductory phrases not specified above.
                    * Base your answer strictly on the provided "Sources".

                ### Sources ###
                ---
                {relevant_context}
                ---

                ### Question ###
                {question}

                ### Final Answer ###
            """
            final_response = await asyncio.to_thread(lambda: generative_model.generate_content(final_prompt))
            return final_response.text.strip()

        except google_exceptions.ResourceExhausted as e:
            wait_time = (attempt + 1) * 10 # Exponential backoff
            logging.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}...")
            if attempt < 2:
                await asyncio.sleep(wait_time)
            else:
                return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"An unexpected error occurred processing question '{question}': {e}", exc_info=True)
            return f"Error processing question: An unexpected error occurred."
    
    return "Error: Failed to get an answer after all attempts."

# --- API Endpoint with Final Caching & Parallel Logic ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    cache_key = document_url.split('?')[0] # Use URL without query params as cache key
    logging.info(f"Processing request for document: {cache_key}")

    if cache_key not in document_cache:
        logging.info(f"'{cache_key}' not in cache. Processing and caching now...")
        try:
            pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
            text_chunks = get_text_chunks_advanced(pdf_text)
            
            # Embed chunks in batches to avoid overwhelming the API
            batch_size = 100
            chunk_embeddings = []
            for i in range(0, len(text_chunks), batch_size):
                batch_chunks = text_chunks[i:i+batch_size]
                embeddings_result = await asyncio.to_thread(
                    lambda: genai.embed_content(
                        model='models/text-embedding-004', 
                        content=batch_chunks, 
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                chunk_embeddings.extend(embeddings_result['embedding'])
                await asyncio.sleep(1) # Small delay to respect rate limits
            
            document_cache[cache_key] = {"chunks": text_chunks, "embeddings": chunk_embeddings}
            logging.info(f"Successfully cached {len(text_chunks)} chunks for: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and cache document: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    else:
        logging.info(f"Found '{cache_key}' in cache. Using cached data.")
    
    cached_data = document_cache[cache_key]
    
    # Process all questions in PARALLEL for maximum speed
    tasks = [get_single_answer(q, cached_data) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}
