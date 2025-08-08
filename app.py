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
from google.cloud import vision
import faiss # <-- NEW IMPORT FOR ON-DISK INDEXING

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = os.getenv("TEAM_TOKEN")

if not GEMINI_API_KEY or not TEAM_TOKEN:
    raise ValueError("Required environment variables GEMINI_API_KEY or TEAM_TOKEN are not set.")

genai.configure(api_key=GEMINI_API_KEY)

# --- On-Disk Cache Management ---
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# We will store the text chunks separately in memory, as they are smaller.
text_chunk_cache: Dict[str, List[str]] = {}


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
    """Downloads and extracts all text from a PDF using PyMuPDF."""
    try:
        response = requests.get(url, timeout=180) # Longer timeout for very large files
        response.raise_for_status()
        pdf_content = response.content
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()

        # Smart Detection for OCR
        if doc.page_count > 1 and len(text.strip()) < (100 * doc.page_count):
            logging.warning(f"Low text detected. Attempting OCR with Google Cloud Vision for {url}")
            client = vision.ImageAnnotatorClient()
            gcs_feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            gcs_input_config = vision.InputConfig(content=pdf_content, mime_type='application/pdf')
            gcs_request = vision.AnnotateFileRequest(input_config=gcs_input_config, features=[gcs_feature])
            gcs_response = client.batch_annotate_files(requests=[gcs_request])
            
            full_text = ""
            for image_response in gcs_response.responses[0].responses:
                if image_response.full_text_annotation:
                    full_text += image_response.full_text_annotation.text + "\n"
            text = full_text

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

async def get_single_answer(question: str, text_chunks: List[str], faiss_index_path: str) -> str:
    """Processes one question using the on-disk FAISS index and Multi-Query RAG."""
    generative_model = genai.GenerativeModel('gemini-1.5-pro')

    for attempt in range(3):
        try:
            index = faiss.read_index(faiss_index_path)
            
            # Multi-Query Generation
            query_gen_prompt = f"""You are an expert at rephrasing questions for a retrieval system. Given the following question, generate 3 additional, different phrasings of it. Your output MUST be a valid JSON array of strings. Original Question: "{question}" JSON Array of Rephrased Questions:"""
            query_gen_model = genai.GenerativeModel('gemini-1.5-flash')
            response = await asyncio.to_thread(lambda: query_gen_model.generate_content(query_gen_prompt))
            
            try:
                cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                rephrased_questions = json.loads(cleaned_text)
                all_queries = [question] + rephrased_questions
            except json.JSONDecodeError:
                logging.warning(f"Could not parse rephrased questions. Falling back to original question.")
                all_queries = [question]

            query_embeddings = await asyncio.to_thread(lambda: genai.embed_content(model='models/text-embedding-004', content=all_queries, task_type="RETRIEVAL_QUERY")['embedding'])

            all_top_indices = set()
            for embedding in query_embeddings:
                query_vector = np.array([embedding], dtype=np.float32)
                distances, top_indices = index.search(query_vector, 3) # Top 3 for each query
                all_top_indices.update(top_indices[0])

            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in all_top_indices])
            
            final_prompt = f"""You are an expert AI research analyst. Your task is to synthesize a single, clear, and accurate answer to the "Question" based *only* on the provided "Sources".
**Sources:** --- {relevant_context} ---
**Question:** {question}
**Instructions:** - Synthesize the information to formulate a single, comprehensive, and well-written final answer. - If the sources do not contain enough information, state: "Based on the provided information, a definitive answer could not be found." - **Your output must ONLY be the final answer sentence.**
**Final Answer:**"""
            final_response = await asyncio.to_thread(lambda: generative_model.generate_content(final_prompt))
            return final_response.text.strip()

        except google_exceptions.ResourceExhausted as e:
            wait_time = (attempt + 1) * 10
            logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
            if attempt < 2: await asyncio.sleep(wait_time)
            else: return "Error: Rate limit exceeded after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}", exc_info=True)
            return f"Error processing question: {e}"
    return "Error: Failed to get an answer after all attempts."

# --- API Endpoint with On-Disk FAISS Caching ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    cache_key = document_url.split('?')[0].split('/')[-1].replace('.pdf', '')
    faiss_index_path = os.path.join(CACHE_DIR, f"{cache_key}.index")
    
    logging.info(f"Processing request for document key: {cache_key}")

    if not os.path.exists(faiss_index_path):
        logging.info(f"Index for '{cache_key}' not found on disk. Processing and creating index now (this will be slow)...")
        try:
            pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
            text_chunks = get_text_chunks_advanced(pdf_text)
            text_chunk_cache[cache_key] = text_chunks
            
            chunk_embeddings = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=text_chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
            )
            
            embeddings_np = np.array(chunk_embeddings, dtype=np.float32)
            d = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_np)
            faiss.write_index(index, faiss_index_path)
            
            logging.info(f"Successfully created and saved FAISS index for: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and create index for document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    else:
        logging.info(f"Found FAISS index for '{cache_key}' on disk.")
        if cache_key not in text_chunk_cache:
             pdf_text = await asyncio.to_thread(get_pdf_text_from_url_sync, document_url)
             text_chunk_cache[cache_key] = get_text_chunks_advanced(pdf_text)

    current_text_chunks = text_chunk_cache[cache_key]
    
    # Process questions sequentially to respect API rate limits
    answers = []
    for question in request.questions:
        answer = await get_single_answer(question, current_text_chunks, faiss_index_path)
        answers.append(answer)

    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}
