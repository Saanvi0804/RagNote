# backend/app/main.py
"""
RagNote backend main app (FastAPI).
Uses a local Ollama HTTP model for generation and a FAISS vectorstore for retrieval.
This version adds HYBRID mode:
- Uses PDF embeddings first (RAG)
- If RAG is insufficient, model uses its general knowledge to fill gaps
- Robust streaming assembly from Ollama with retries and long timeouts
"""

import os
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Any, Dict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_reader import extract_text_from_pdf
from vectorstore import create_embeddings_and_store, load_vectorstore

# ------------------------
# Useful file path (uploaded file)
# ------------------------
# Developer note: user uploaded a file earlier; expose its local path/URI here so tooling can transform it.
LAST_UPLOADED_FILE = r"C:\Users\Saanvi Shetty\Desktop\ragnote\backend\app\uploads\Agile_planning&_estimation-3[1].pdf"
# As a file:// URI (useful if you need to pass a URL to other tools)
try:
    LAST_UPLOADED_FILE_URL = Path(LAST_UPLOADED_FILE).as_uri()
except Exception:
    LAST_UPLOADED_FILE_URL = LAST_UPLOADED_FILE

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ragnote")

# ------------------------
# Paths
# ------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STORE_PATH = BASE_DIR / "faiss_store"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Env / Ollama config
# ------------------------
load_dotenv(str(BASE_DIR / ".env"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# ------------------------
# FastAPI
# ------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    chat_history: Optional[List] = []


# ------------------------
# Upload endpoint
# ------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Save uploaded PDF, extract text, create embeddings and save vectorstore to disk.
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info("Saved file: %s", file_path)
        text = extract_text_from_pdf(str(file_path))
        logger.info("Extracted chars: %d", len(text))

        db = create_embeddings_and_store(text, str(STORE_PATH))
        logger.info("Index created/updated at: %s", STORE_PATH)
        return {"detail": "uploaded and indexed", "filename": file.filename, "file_path": str(file_path)}
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Error during upload/indexing")
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})


# ------------------------
# Robust Ollama streaming + assemble with retries
# ------------------------
def query_ollama_stream_assemble(prompt: str,
                                max_tokens: int = 400,
                                temperature: float = 0.0,
                                timeout_seconds: int = 300,
                                max_retries: int = 2) -> str:
    """
    Call Ollama /api/generate and assemble streaming chunks into a single string.
    Retries on transient errors. Uses long timeout by default.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Note: Ollama may stream JSON-lines by default.
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 2):
        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout_seconds) as resp:
                resp.raise_for_status()
                pieces: List[str] = []
                # Iterate over streamed lines
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue
                    # Attempt to parse JSON per-line
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # ignore non-json lines
                        logger.debug("Non-JSON line from Ollama stream (ignored): %s", line[:200])
                        continue
                    # Common keys: "response", "text", "output"
                    piece = obj.get("response") or obj.get("text") or obj.get("output") or ""
                    if piece:
                        pieces.append(piece)
                    # If stream indicates finished
                    if obj.get("done") is True:
                        break
                full_text = "".join(pieces).strip()
                return full_text
        except requests.exceptions.RequestException as e:
            if attempt <= max_retries:
                logger.warning("Ollama request failed (attempt %d/%d): %s — retrying in %ss",
                               attempt, max_retries + 1, str(e), backoff)
                time.sleep(backoff)
                backoff *= 2.0
                continue
            logger.exception("Ollama request failed, no retries left")
            raise RuntimeError(f"Ollama request failed: {e}")


# ------------------------
# Utility: load metadata previews (faiss_store/metadata.txt)
# ------------------------
def load_metadata_lines(store_path: str = None) -> List[str]:
    """
    Read faiss_store/metadata.txt and return list of preview strings indexed by chunk idx.
    Defensive: handles missing file or malformed lines.
    """
    if store_path is None:
        store_path = str(STORE_PATH)
    meta_path = os.path.join(store_path, "metadata.txt")
    lines: List[str] = []
    try:
        with open(meta_path, "r", encoding="utf8") as mf:
            for ln in mf:
                raw = ln.rstrip("\n\r")
                if not raw:
                    lines.append("")
                    continue
                parts = raw.split("\t", 1)
                if len(parts) == 2:
                    lines.append(parts[1])
                else:
                    lines.append(parts[0])
    except FileNotFoundError:
        logger.warning("metadata.txt not found at %s", meta_path)
    except Exception:
        logger.exception("Failed reading metadata.txt")
    return lines


# ------------------------
# Normalize retriever output (robust)
# ------------------------
def _normalize_retriever_output(raw: Any) -> List[Dict[str, Any]]:
    """
    Accept many shapes and normalize to a list of dicts with 'page_content' and 'metadata'.
    """
    out: List[Dict[str, Any]] = []
    if raw is None:
        return out

    if isinstance(raw, tuple) and len(raw) == 2:
        raw = raw[0]

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, tuple) and len(item) >= 1:
                doc_candidate = item[0]
            else:
                doc_candidate = item

            if hasattr(doc_candidate, "page_content"):
                out.append({
                    "page_content": getattr(doc_candidate, "page_content", "") or str(doc_candidate),
                    "metadata": getattr(doc_candidate, "metadata", {}) or {}
                })
            elif isinstance(doc_candidate, dict):
                pc = doc_candidate.get("page_content") or doc_candidate.get("content") or doc_candidate.get("text") or str(doc_candidate)
                meta = doc_candidate.get("metadata") or doc_candidate.get("meta") or {}
                out.append({"page_content": pc, "metadata": meta})
            else:
                out.append({"page_content": str(doc_candidate), "metadata": {}})
        return out

    if isinstance(raw, dict):
        pc = raw.get("page_content") or raw.get("content") or raw.get("text") or str(raw)
        meta = raw.get("metadata") or raw.get("meta") or {}
        return [{"page_content": pc, "metadata": meta}]

    return [{"page_content": str(raw), "metadata": {}}]


# ------------------------
# Ask endpoint (hybrid RAG + general knowledge)
# ------------------------
@app.post("/ask")
async def ask(req: AskRequest):
    """
    Retrieve top-k chunks, build prompt and ask Ollama to generate an answer.
    Returns assembled answer and readable source previews.
    """
    question = req.question.strip()

    # Load vectorstore
    try:
        db = load_vectorstore(str(STORE_PATH))
    except Exception as e:
        logger.exception("Failed to load vectorstore")
        return {"error": "Failed to load vectorstore: " + str(e)}

    if db is None:
        return {"error": "No documents indexed. Upload a PDF first."}

    # Obtain retriever robustly
    retriever = None
    try:
        retriever = db.as_retriever(search_kwargs={"k": 4})
    except Exception:
        try:
            retriever = db.as_retriever()
        except Exception:
            retriever = None

    # Call retriever in multiple ways
    raw_docs = None
    try:
        if callable(retriever):
            raw_docs = retriever(question)
        elif hasattr(retriever, "get_relevant_documents"):
            raw_docs = retriever.get_relevant_documents(question)
        elif hasattr(db, "get_relevant_documents"):
            raw_docs = db.get_relevant_documents(question)
        elif hasattr(db, "similarity_search"):
            raw_docs = db.similarity_search(question, k=4)
        else:
            if callable(db):
                raw_docs = db(question)
            else:
                raise RuntimeError("No suitable retriever method found on the vectorstore.")
    except Exception as e:
        logger.exception("Retriever invocation failed: %s", e)
        return {"error": "Retriever error: " + str(e)}

    docs = _normalize_retriever_output(raw_docs)

    if not docs:
        # Continue with an empty excerpt block so the model can still answer from general knowledge
        logger.info("Retriever returned no docs for query; falling back to general knowledge.")

    # Build excerpts block (truncate)
    excerpts = []
    source_indices: List[int] = []
    for i, d in enumerate(docs):
        preview = (d.get("page_content") or "").strip()
        if len(preview) > 2000:
            preview = preview[:2000] + " ... (truncated)"
        excerpts.append(f"[{i}] {preview}")
        meta = d.get("metadata", {}) or {}
        si = None
        if isinstance(meta, dict):
            si = meta.get("source_idx") or meta.get("source_index") or meta.get("idx") or meta.get("id")
        try:
            if si is not None:
                source_indices.append(int(si))
        except Exception:
            pass

    context_block = "\n\n".join(excerpts) if excerpts else "(No matching excerpts found)"

    # Hybrid prompt: prefer PDF, but allow general knowledge when needed
    prompt = f"""
You are a helpful, factual assistant. You have two sources of information:
1) PDF excerpts (may be incomplete) supplied below.
2) Your own general knowledge.

Task:
- First check the PDF excerpts to see if they contain the answer.
- If the PDF contains the answer, answer using those excerpts (cite indices).
- If the PDF is incomplete or doesn't contain the exact answer, use your general knowledge to answer correctly, and indicate that you supplemented with general knowledge.
- Do NOT respond with "I don't know" unless the concept truly cannot be answered.

User question:
{question}

PDF excerpts:
----------------
{context_block}
----------------

Answer concisely and clearly. If you used general knowledge, you may say "Using general knowledge:" before that part.
"""

    logger.debug("Prompt length: %d", len(prompt))

    try:
        generated = query_ollama_stream_assemble(prompt, max_tokens=600, timeout_seconds=300, max_retries=2)
    except Exception as e:
        logger.exception("Ollama generation failed")
        return {"error": f"Ollama generation failed: {e}"}

    # Map source_indices to readable previews
    meta_lines = load_metadata_lines(str(STORE_PATH))
    readable_sources = []
    for idx in source_indices:
        preview = meta_lines[idx] if (0 <= idx < len(meta_lines)) else ""
        readable_sources.append({"source_idx": idx, "preview": preview})

    return {"answer": generated, "source_documents": readable_sources, "uploaded_file": LAST_UPLOADED_FILE_URL}
