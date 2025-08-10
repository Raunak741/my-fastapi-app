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
import docx # python-docx
import extract_msg # for .msg and .eml files
from google.api_core import exceptions as google_exceptions
import faiss

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
METADATA_CACHE_DIR = "metadata_cache"
os.makedirs(METADATA_CACHE_DIR, exist_ok=True)


# --- FastAPI App Initialization ---
app = FastAPI()

# --- Authentication ---
security = HTTPBearer()

def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Pydantic Models (Reverted to simple string list) ---
class ApiResponse(BaseModel):
    answers: List[str]

class ApiRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Core Logic Functions ---

def get_document_content(url: str) -> (bytes, str):
    """Downloads content from a URL and determines its file type."""
    try:
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        
        if 'pdf' in content_type:
            file_type = 'pdf'
        elif 'vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            file_type = 'docx'
        elif 'message/rfc822' in content_type:
            file_type = 'eml'
        else:
            if url.lower().endswith('.pdf'): file_type = 'pdf'
            elif url.lower().endswith('.docx'): file_type = 'docx'
            elif url.lower().endswith('.eml') or url.lower().endswith('.msg'): file_type = 'eml'
            else: raise ValueError(f"Unsupported file type: {content_type}")
            
        return response.content, file_type
    except Exception as e:
        raise RuntimeError(f"Failed to download or identify document: {e}")

def chunk_document_with_metadata(content: bytes, file_type: str) -> List[Dict[str, Any]]:
    """Extracts text and creates chunks with page/clause metadata."""
    chunks_with_metadata = []
    
    if file_type == 'pdf':
        doc = fitz.open(stream=content, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    chunks_with_metadata.append({
                        "text": para.strip(),
                        "source": f"Page {page_num + 1}"
                    })
        doc.close()

    elif file_type == 'docx':
        doc = docx.Document(io.BytesIO(content))
        current_heading = "General"
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                current_heading = para.text.strip()
            if para.text.strip():
                chunks_with_metadata.append({
                    "text": para.text.strip(),
                    "source": f"Section: {current_heading}"
                })

    elif file_type == 'eml':
        msg = extract_msg.Message(io.BytesIO(content))
        chunks_with_metadata.append({"text": f"From: {msg.sender}\nTo: {msg.to}\nSubject: {msg.subject}", "source": "Email Header"})
        chunks_with_metadata.append({"text": msg.body, "source": "Email Body"})

    return chunks_with_metadata


async def get_single_answer(question: str, text_chunks_with_metadata: List[Dict[str, Any]], faiss_index_path: str) -> str:
    """Processes one question and returns a single answer string."""
    generative_model = genai.GenerativeModel('gemini-1.5-pro')

    for attempt in range(2):
        try:
            index = faiss.read_index(faiss_index_path)
            
            question_embedding = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=question, task_type="RETRIEVAL_QUERY")['embedding']
            )
            
            query_vector = np.array([question_embedding], dtype=np.float32)
            distances, top_indices = index.search(query_vector, 10)
            
            relevant_chunks = [text_chunks_with_metadata[i] for i in top_indices[0]]
            context_str = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # --- FINAL, MOST ACCURATE PROMPT for simple string output ---
            final_prompt = f"""
                You are a meticulous AI legal and compliance analyst. Your task is to provide a single, clear, and accurate answer to the "Question" based *only* on the provided "Sources".

                **Sources:**
                ---
                {context_str}
                ---

                **Question:** {question}

                **Instructions:**
                1.  **Synthesize:** Formulate a comprehensive final answer by synthesizing all relevant information from the sources.
                2.  **Be Specific:** Your answer MUST include all critical details, conditions, quantitative values (e.g., "36 months", "5%"), and especially any exceptions or exclusions (e.g., "...this does not apply if...").
                3.  **Handle Missing Info:** If the answer cannot be found in the sources, you must state: "Based on the provided information, a definitive answer could not be found."
                4.  **Output Format:** Your entire output must be ONLY the single, final answer sentence. Do not include your thought process, any introductory phrases, or any text other than the final answer.

                **Final Answer:**
            """
            final_response = await asyncio.to_thread(lambda: generative_model.generate_content(final_prompt))
            return final_response.text.strip()

        except google_exceptions.ResourceExhausted as e:
            logging.warning(f"Retrying question '{question}' due to: {e}")
            if attempt < 1: await asyncio.sleep(15)
            else: return "Error: Failed after multiple retries."
        except Exception as e:
            logging.error(f"Failed to process question '{question}': {e}", exc_info=True)
            return f"Error processing question: {e}"
    return "Error: Failed after all attempts."


# --- API Endpoint with On-Disk FAISS Caching ---
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(request: ApiRequest, token: str = Depends(check_token)):
    document_url = request.documents
    cache_key = document_url.split('?')[0].split('/')[-1]
    faiss_index_path = os.path.join(CACHE_DIR, f"{cache_key}.index")
    metadata_path = os.path.join(METADATA_CACHE_DIR, f"{cache_key}.json")
    
    logging.info(f"Processing request for document key: {cache_key}")

    if not os.path.exists(faiss_index_path):
        logging.info(f"Index for '{cache_key}' not found. Processing and creating index now (this will be slow)...")
        try:
            content, file_type = await asyncio.to_thread(get_document_content, document_url)
            chunks_with_metadata = chunk_document_with_metadata(content, file_type)
            
            with open(metadata_path, 'w') as f:
                json.dump(chunks_with_metadata, f)
            
            text_chunks = [chunk['text'] for chunk in chunks_with_metadata]
            
            chunk_embeddings = await asyncio.to_thread(
                lambda: genai.embed_content(model='models/text-embedding-004', content=text_chunks, task_type="RETRIEVAL_DOCUMENT")['embedding']
            )
            
            embeddings_np = np.array(chunk_embeddings, dtype=np.float32)
            d = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_np)
            faiss.write_index(index, faiss_index_path)
            
            logging.info(f"Successfully created FAISS index and metadata for: {cache_key}")
        except Exception as e:
            logging.error(f"Failed to process and create index for document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    
    with open(metadata_path, 'r') as f:
        current_chunks_with_metadata = json.load(f)
    
    tasks = [get_single_answer(q, current_chunks_with_metadata, faiss_index_path) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info("Processing complete. Returning all answers.")
    return ApiResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Adjudicator API is running."}
