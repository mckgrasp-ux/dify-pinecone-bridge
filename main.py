import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document as DocxDocument
from openai import OpenAI
from pinecone import Pinecone


# =========================
# ENV VARIABLES
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # np. dify-knowledge
PINECONE_HOST = os.getenv("PINECONE_HOST")              # Twój host z Pinecone

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


app = FastAPI(title="Dify External Knowledge Bridge (Pinecone)")


# =========================
# HELPERS
# =========================

def get_openai():
    if not OPENAI_API_KEY:
        raise Exception("Missing OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_index():
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not PINECONE_HOST:
        raise Exception("Missing Pinecone configuration")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks


def extract_text(file: UploadFile, data: bytes) -> str:
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        from io import BytesIO
        reader = PdfReader(BytesIO(data))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])

    if filename.endswith(".docx"):
        from io import BytesIO
        doc = DocxDocument(BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])

    # txt / md / fallback
    return data.decode("utf-8", errors="ignore")


# =========================
# MODELS
# =========================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    namespace: Optional[str] = None


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), namespace: Optional[str] = None):

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys")

    data = await file.read()
    text = extract_text(file, data)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted")

    client = get_openai()

    embeddings_response = client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks
    )

    vectors = []
    for chunk, emb in zip(chunks, embeddings_response.data):
        vectors.append(
            (
                str(uuid.uuid4()),
                emb.embedding,
                {
                    "content": chunk,
                    "source": file.filename
                }
            )
        )

    index = get_index()
    index.upsert(vectors=vectors, namespace=namespace or "")

    return {
        "uploaded": file.filename,
        "chunks": len(chunks),
        "namespace": namespace or ""
    }


@app.post("/search")
def search(req: SearchRequest):

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys")

    client = get_openai()

    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=req.query
    ).data[0].embedding

    index = get_index()

    results = index.query(
        vector=query_embedding,
        top_k=req.top_k,
        include_metadata=True,
        namespace=req.namespace or ""
    )

    documents = []

    for match in (results.matches or []):
        metadata = match.metadata or {}
        documents.append({
            "content": metadata.get("content", ""),
            "score": float(match.score or 0.0),
            "source": metadata.get("source")
        })

    return {"documents": documents}
