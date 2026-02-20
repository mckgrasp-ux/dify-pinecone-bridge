import os
import re
import hashlib
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from pinecone import Pinecone


# -----------------------------
# Config / env
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_HOST = os.getenv("PINECONE_HOST", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()  # optional (tylko informacyjnie)
TEXT_FIELD = os.getenv("TEXT_FIELD", "text").strip()  # w Pinecone UI masz field map "text"

if not PINECONE_API_KEY:
    raise RuntimeError("Missing env var: PINECONE_API_KEY")
if not PINECONE_HOST:
    raise RuntimeError("Missing env var: PINECONE_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


# -----------------------------
# Helpers
# -----------------------------
def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io_bytes := bytes(pdf_bytes))  # keep a local ref
    # pypdf accepts file-like, but also bytes in some contexts; safest: use memoryview via BytesIO
    from io import BytesIO
    bio = BytesIO(io_bytes)
    reader = PdfReader(bio)

    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _clean_text(txt)
        if txt:
            parts.append(f"[page {i+1}]\n{txt}")
    return "\n\n".join(parts).strip()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    """
    Prosty chunking po znakach z overlapem.
    max_chars ~ bezpiecznie dla większości embedderów; masz limit 507 tokens w e5-large,
    więc trzymamy się raczej krótszych kawałków.
    """
    text = _clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def stable_id(namespace: str, filename: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha1()
    h.update(namespace.encode("utf-8"))
    h.update(b"|")
    h.update(filename.encode("utf-8"))
    h.update(b"|")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"|")
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="dify-pinecone-bridge", version="1.0.0")


@app.get("/")
def root():
    return {"ok": True}


@app.get("/health")
def health():
    return {
        "ok": True,
        "pinecone_host": PINECONE_HOST,
        "index_name": PINECONE_INDEX_NAME or None,
        "text_field": TEXT_FIELD,
    }


@app.post("/ingest")
async def ingest(
    namespace: str = Query(..., description="Pinecone namespace, np. nordkalk"),
    file: UploadFile = File(...),
    chunk_chars: int = Query(1800, ge=500, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=800),
):
    """
    Upload PDF -> extract text -> chunk -> upsert_records() do Pinecone
    Wymaga indexu z Integrated embedding (field map = TEXT_FIELD, domyślnie 'text').
    """
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="namespace is required")

    filename = file.filename or "file"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Basic type check (nie blokujemy na 100%, ale pomagamy)
    if not (filename.lower().endswith(".pdf") or (file.content_type or "").lower() == "application/pdf"):
        # możesz to zmienić na warning, jeśli chcesz inne pliki
        raise HTTPException(status_code=400, detail="Only PDF is supported for now (.pdf)")

    # Extract
    try:
        text = extract_text_from_pdf(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parse error: {e}")

    if not text:
        raise HTTPException(status_code=400, detail="No extractable text in PDF")

    # Chunk
    chunks = chunk_text(text, max_chars=chunk_chars, overlap=chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced")

    # Build records for Integrated Embedding
    # Pinecone expects: {"id": "...", "<TEXT_FIELD>": "...", "metadata": {...}}
    records: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        rid = stable_id(namespace, filename, i, ch)
        records.append(
            {
                "id": rid,
                TEXT_FIELD: ch,
                "metadata": {
                    "source": filename,
                    "chunk": i,
                    "chunks_total": len(chunks),
                    "namespace": namespace,
                },
            }
        )

    # Upsert to Pinecone using integrated embedding
    try:
        # upsert_records działa w pakiecie "pinecone" (nowe SDK)
        index.upsert_records(namespace=namespace, records=records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone upsert error: {e}")

    return JSONResponse(
        {
            "ok": True,
            "namespace": namespace,
            "file": filename,
            "chunks": len(chunks),
            "note": "Upserted via integrated embedding",
        }
    )
