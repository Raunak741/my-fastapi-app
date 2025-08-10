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
    """Downloads and extracts the full text from a PDF, with no page limits."""
    try:
        # Generous 5-minute timeout for huge files
        response = requests.get(url, timeout=300) 
        response.raise_for_status()
        pdf_content = response.content
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = "".join(page.get_text("text", sort=True) for page in doc) # Added sort=True for better reading order
        doc.close()
        
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def get_text_chunks_recursive(text: str, max_chunk_size: int = 2000, overlap: int = 250) -> List[str]:
    """
    Splits text into overlapping chunks to preserve semantic context.
    This is more robust than simple paragraph or sentence splitting.
    """
    if not text:
        return []
    
    # First, split by larger separators to respect document structure
    chunks = []
    initial_splits = text.split('\n\n\n') # Split by triple newlines first

    for split in initial_splits:
        if len(split) <= max_chunk_size:
            if split.strip():
                chunks.append(split.strip())
        else:
            # If the chunk is still too large, use recursive splitting
            sentences = re.split(r'(?<=[.!?])\s+', split)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk.strip())

    # Create overlapping chunks for better context retrieval
    overlapping_chunks = []
    for i in range(len(chunks)):
        start_index = max(0, i) # Each chunk starts normally
        end_index = i + 1
        
        # Combine the current chunk with a bit of the next one for overlap
        combined_chunk = " ".join(chunks[start_index:end_index])
        
        # Add look-ahead overlap
        if i + 1 < len(chunks):
            next_chunk_preview = chunks[i+1]
            overlap_text = ' '.join(next_chunk_preview.split()[:overlap // 10]) # Approx word count
            combined_chunk += " " + overlap_text

        overlapping_chunks.append(combined_chunk.strip())

    return [c for c in overlapping_chunks if c]


async def get_single_answer(question: str, cached_data: Dict[str, Any]) -> str:
    """Processes one question using Multi-Query RAG, Contextual Window Retrieval, and Chain-of-Thought reasoning."""
    text_chunks = cached_data["chunks"]
    chunk_embeddings = cached_data["embeddings"]
    generative_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    for attempt in range(2): # Retry once on failure
        try:
            # 1. Multi-Query Generation for better retrieval
            query_gen_prompt = f"""You are an expert at rephrasing questions for a vector database search.
Given the user's question, generate 3 additional, diverse, and specific phrasings that focus on different aspects of the original question.
For example, if the question is "What is the warranty period?", you might generate ["How long does the product warranty last?", "What are the terms of the warranty coverage?", "Are there any exclusions to the warranty?"].
Output ONLY a valid JSON array of strings in a single line.
Original Question: "{question}"
"""
            query_gen_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = await asyncio.to_thread(lambda: query_gen_model.generate_content(query_gen_prompt))
            
            all_queries = [question]
            try:
                cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                rephrased_questions = json.loads(cleaned_text)
                all_queries.extend(rephrased_questions)
            except (json.JSONDecodeError, AttributeError):
                logging.warning(f"Could not parse rephrased questions for: '{question}'. Using original query only.")

            # 2. Embed all queries
            query_embeddings_result = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=all_queries, task_type="RETRIEVAL_QUERY")
            )
            query_embeddings = query_embeddings_result['embedding']

            # 3. Retrieve, Expand Context, and Combine Results
            all_top_indices: Set[int] = set()
            for embedding in query_embeddings:
                dot_products = np.dot(np.array(chunk_embeddings), np.array(embedding))
                # ACCURACY TWEAK 1: Retrieve more initial candidates
                top_indices_for_query = np.argsort(dot_products)[-7:][::-1] # Top 7 for each query
                
                # ACCURACY TWEAK 2: Contextual Window Expansion
                # For each top index, also grab its neighbors for better context.
                for i in top_indices_for_query:
                    all_top_indices.add(i)
                    if i > 0:
                        all_top_indices.add(i - 1) # Add previous chunk
                    if i < len(text_chunks) - 1:
                        all_top_indices.add(i + 1) # Add next chunk
            
            # De-duplicate and sort indices to maintain document order
            sorted_indices = sorted(list(all_top_indices))
            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in sorted_indices])
            
            # 4. ACCURACY TWEAK 3: Chain-of-Thought & Self-Correction Prompt
            final_prompt = f"""
You are a meticulous AI research analyst. Your task is to provide a single, definitive answer to the "Question" using ONLY the provided "Sources". You must follow a strict reasoning process.

**Reasoning Process:**
1.  **Fact Extraction:** First, go through all the provided Sources and extract every single fact, number, condition, and exception that is directly relevant to the user's question. List them out internally.
2.  **Sufficiency Check:** Review the facts you extracted. Is there enough information to form a complete, unambiguous answer? Are there any contradictions?
3.  **Answer Synthesis:** Based ONLY on the extracted facts, construct a single, comprehensive sentence that directly answers the question.
    * Your answer **MUST** integrate all specific quantitative details (e.g., numbers like "36 months", percentages like "5%"), critical conditions, and **especially any exceptions or exclusions** (e.g., "...this limit does not apply if..."). This is the most important instruction.
    * If the sources contain conflicting information, you must state: "The provided information contains conflicting details and a definitive answer cannot be given."
    * If the sources do not contain enough information to provide a specific and complete answer, you **MUST** state: "Based on the provided information, a definitive answer could not befound."
4.  **Final Output:** Your entire output must be ONLY the single, final answer sentence you synthesized in step 3. Do not include your internal reasoning, any introductory phrases, or any text other than the final synthesized answer.

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
            wait_time = 20 # Increased wait time for heavier models
            logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
            if attempt < 1:
                await asyncio.sleep(wait_time)
            else:
                return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}", exc_info=True)
            return f"Error processing question: An unexpected error occurred."
            
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
            text_chunks = get_text_chunks_recursive(pdf_text) # Using the new chunking function
            
            if not text_chunks:
                raise HTTPException(status_code=500, detail="Failed to extract any text chunks from the document.")

            # Embed chunks in batches to avoid API limits for very large documents
            all_embeddings = []
            batch_size = 100 # Gemini API can handle up to 100 contents per request
            for i in range(0, len(text_chunks), batch_size):
                batch_chunks = text_chunks[i:i+batch_size]
                batch_embeddings = await asyncio.to_thread(
                    lambda: genai.embed_content(
                        model='models/text-embedding-004', 
                        content=batch_chunks, 
                        task_type="RETRIEVAL_DOCUMENT"
                    )['embedding']
                )
                all_embeddings.extend(batch_embeddings)
                await asyncio.sleep(1) # Small sleep to respect rate limits

            document_cache[cache_key] = {"chunks": text_chunks, "embeddings": all_embeddings}
            logging.info(f"Successfully cached: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and cache document: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
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