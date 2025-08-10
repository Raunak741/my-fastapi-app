# app.py

# --- IMPORTANT: System and Library Requirements ---
# You must install Google's Tesseract OCR engine on your system first.
# For Debian/Ubuntu: sudo apt-get install tesseract-ocr
# For other OSes, check the official Tesseract documentation.

# Then, install the required Python libraries:
# pip install "fastapi[all]" python-dotenv google-generativeai numpy "PyMuPDF<1.24.0>" pytesseract pdf2image python-docx python-magic aiohttp

import os
import io
import json
import logging
import asyncio
import numpy as np
import re
import fitz  # PyMuPDF
import magic
import aiohttp
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- OCR Dependencies ---
import pytesseract
from pdf2image import convert_from_bytes

# --- Document Processing for .docx ---
from docx import Document

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

async def get_document_content_and_type(url: str) -> Tuple[bytes, str]:
    """
    UPGRADE 1: Fetches content and validates file type using headers.
    This acts as a gatekeeper to reject invalid files early.
    """
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB limit
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=300) as response:
                response.raise_for_status()

                # Check file size
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise ValueError(f"File size {int(content_length) / 1024**2:.2f} MB exceeds limit of {MAX_FILE_SIZE / 1024**2} MB.")

                content = await response.read()
                
                # Check content type using python-magic for higher accuracy
                mime_type = magic.from_buffer(content, mime=True)
                logging.info(f"Detected MIME type: {mime_type} for URL: {url}")

                if 'pdf' in mime_type:
                    return content, 'pdf'
                elif 'vnd.openxmlformats-officedocument.wordprocessingml.document' in mime_type:
                    return content, 'docx'
                else:
                    raise ValueError(f"Unsupported file type: {mime_type}")

    except aiohttp.ClientError as e:
        raise RuntimeError(f"Network error fetching document: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to get or validate document: {e}")


def extract_text_from_pdf_with_ocr(pdf_content: bytes) -> str:
    """
    UPGRADE 2: Hybrid text extraction with OCR for scanned PDFs.
    This is the key to handling image-based documents.
    """
    full_text = []
    min_text_length_per_page = 20  # Threshold to trigger OCR

    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        for page_num, page in enumerate(doc):
            # First, try normal text extraction
            text = page.get_text().strip()
            
            # If text is minimal or absent, assume it's a scanned page and use OCR
            if len(text) < min_text_length_per_page:
                logging.warning(f"Page {page_num + 1} has little to no text. Attempting OCR.")
                try:
                    images = convert_from_bytes(pdf_content, first_page=page_num + 1, last_page=page_num + 1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        full_text.append(ocr_text)
                        logging.info(f"Successfully extracted text from page {page_num + 1} using OCR.")
                except Exception as ocr_error:
                    logging.error(f"OCR failed for page {page_num + 1}: {ocr_error}")
                    full_text.append(text) # Append the little text we found earlier
            else:
                full_text.append(text)
        
        doc.close()
        final_text = "\n\n".join(full_text)
        if not final_text.strip():
            raise ValueError("Could not extract any meaningful text from the PDF, even after OCR attempt.")
        return final_text

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF with OCR: {e}")


def extract_text_from_docx(docx_content: bytes) -> str:
    """
    UPGRADE 3: Added support for DOCX files.
    """
    try:
        doc = Document(io.BytesIO(docx_content))
        return "\n\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        raise RuntimeError(f"Failed to process DOCX file: {e}")


def get_text_chunks_recursive(text: str, max_chunk_size: int = 2000, overlap: int = 250) -> List[str]:
    """Splits text into overlapping chunks to preserve semantic context."""
    if not text: return []
    chunks = []
    initial_splits = text.split('\n\n\n')
    for split in initial_splits:
        if len(split) <= max_chunk_size:
            if split.strip(): chunks.append(split.strip())
        else:
            sentences = re.split(r'(?<=[.!?])\s+', split)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk: chunks.append(current_chunk.strip())
    
    final_chunks = [c for c in chunks if c]
    return final_chunks


async def get_single_answer(question: str, cached_data: Dict[str, Any]) -> str:
    # This function remains largely the same, as the improvements are in the upstream document processing.
    # The prompt and retrieval logic are already quite advanced.
    text_chunks = cached_data["chunks"]
    chunk_embeddings = cached_data["embeddings"]
    generative_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    for attempt in range(2): 
        try:
            query_gen_prompt = f"""You are an expert at rephrasing questions for vector search. Generate 3 diverse phrasings for the question. Output ONLY a valid JSON array of strings. Question: "{question}" """
            query_gen_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = await asyncio.to_thread(lambda: query_gen_model.generate_content(query_gen_prompt))
            
            all_queries = [question]
            try:
                cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                all_queries.extend(json.loads(cleaned_text))
            except (json.JSONDecodeError, AttributeError):
                logging.warning(f"Could not parse rephrased questions for: '{question}'.")

            query_embeddings = await asyncio.to_thread(lambda: genai.embed_content(model='models/text-embedding-004', content=all_queries, task_type="RETRIEVAL_QUERY")['embedding'])

            all_top_indices = set()
            for embedding in query_embeddings:
                dot_products = np.dot(np.array(chunk_embeddings), np.array(embedding))
                top_indices_for_query = np.argsort(dot_products)[-7:][::-1]
                for i in top_indices_for_query:
                    all_top_indices.add(i)
                    if i > 0: all_top_indices.add(i - 1)
                    if i < len(text_chunks) - 1: all_top_indices.add(i + 1)
            
            sorted_indices = sorted(list(all_top_indices))
            relevant_context = "\n\n---\n\n".join([text_chunks[i] for i in sorted_indices])
            
            final_prompt = f"""
You are a meticulous AI research analyst. Your task is to provide a single, definitive answer to the "Question" using ONLY the provided "Sources". Follow a strict reasoning process.
**Reasoning Process:**
1. **Fact Extraction:** Internally, extract every single fact, number, condition, and exception relevant to the question from the Sources.
2. **Sufficiency Check:** Do you have enough information to form a complete, unambiguous answer?
3. **Answer Synthesis:** Based ONLY on the extracted facts, construct a single, comprehensive sentence. Your answer MUST integrate all specific details (numbers, percentages), conditions, and especially any exceptions. If the information is missing or contradictory, you MUST state: "Based on the provided information, a definitive answer could not be found."
4. **Final Output:** Your entire output must be ONLY the single, final answer sentence. Do not include your reasoning or any introductory phrases.

**Sources:**
---
{relevant_context}
---
**Question:** {question}
**Final Answer:**
"""
            final_response = await asyncio.to_thread(lambda: generative_model.generate_content(final_prompt))
            return final_response.text.strip()

        except google_exceptions.ResourceExhausted:
            wait_time = 20
            logging.warning(f"Rate limit hit. Waiting {wait_time}s...")
            if attempt < 1: await asyncio.sleep(wait_time)
            else: return "Error: Rate limit exceeded after multiple retries."
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
            # Main document processing router
            content, file_type = await get_document_content_and_type(document_url)
            
            extracted_text = ""
            if file_type == 'pdf':
                extracted_text = await asyncio.to_thread(extract_text_from_pdf_with_ocr, content)
            elif file_type == 'docx':
                extracted_text = await asyncio.to_thread(extract_text_from_docx, content)
            
            if not extracted_text:
                raise HTTPException(status_code=422, detail="Could not extract any processable text from the document.")

            text_chunks = get_text_chunks_recursive(extracted_text)
            
            if not text_chunks:
                raise HTTPException(status_code=500, detail="Failed to extract any text chunks from the document.")

            all_embeddings = []
            batch_size = 100
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i+batch_size]
                batch_embeddings = await asyncio.to_thread(lambda: genai.embed_content(model='models/text-embedding-004', content=batch, task_type="RETRIEVAL_DOCUMENT")['embedding'])
                all_embeddings.extend(batch_embeddings)
                await asyncio.sleep(1)

            document_cache[cache_key] = {"chunks": text_chunks, "embeddings": all_embeddings}
            logging.info(f"Successfully cached: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and cache document: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        logging.info(f"Found '{cache_key}' in cache. Using cached data.")
    
    cached_data = document_cache[cache_key]
    
    tasks = [get_single_answer(q, cached_data) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}