import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from pinecone import Pinecone

# -------------------------
# ENV
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
# IMPORTANT: must match Pinecone index field_mapping text field (you got "chunk_text" error)
TEXT_FIELD = os.getenv("TEXT_FIELD", "chunk_text")

# API key for Dify -> External Knowledge API
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

app = FastAPI()


# -------------------------
# helpers
# -------------------------
def _check_auth(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    if not EXTERNAL_API_KEY:
        return

    token = None
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        else:
            token = authorization.strip()

    if not token and x_api_key:
        token = x_api_key.strip()

    if token != EXTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _pdf_to_pages_text(file_path: str) -> List[str]:
    reader = PdfReader(file_path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        txt = " ".join(txt.split())
        pages.append(txt)
    return pages


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        i = max(0, end - overlap)
    return chunks


def _batched(lst: List[Any], batch_size: int) -> List[List[Any]]:
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


# -------------------------
#
