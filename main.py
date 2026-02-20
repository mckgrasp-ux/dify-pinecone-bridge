import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document as DocxDocument
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

# MUSI się zgadzać z Field map w Pinecone (ustawiłeś chunk_text)
TEXT_FIELD = os.getenv("TEXT_FIELD", "chunk_text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

app = FastAPI(title="Dify External Knowledge Bridge (Pinecone Integrated Embedding)")


def get_index():
    if not (PINECONE_API_KEY and PINECONE_HOST and PINECONE_INDEX_NAME):
        raise Exception("Missing Pinecone env vars")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        from io import BytesIO
        reader = PdfReader(BytesIO(data))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])

    if name.endswith(".docx"):
        from io import BytesIO
        doc = DocxDocument(BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])

    return data.decode("utf-8", errors="ignore")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    namespace: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), namespace: Optional[str] = None):
    try:
        index = get_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = await file.read()
    text = extract_text(file.filename or "file", data)

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted (empty or scanned PDF?)")

    records = []
    for ch in chunks:
        records.append({
            "_id": str(uuid.uuid4()),
            TEXT_FIELD: ch,              # Pinecone zrobi embedding automatycznie
            "source": file.filename or "file",
        })

    ns = namespace or NAMESPACE
    index.upsert_records(ns, records)

    return {"uploaded": file.filename, "chunks": len(chunks), "namespace": ns}


@app.post("/search")
def search(req: SearchRequest):
    try:
        index = get_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    ns = req.namespace or NAMESPACE

    res = index.search(
        namespace=ns,
        query={
            "inputs": {"text": req.query},
            "top_k": req.top_k
        },
        fields=["source", TEXT_FIELD]
    )

    hits = res.get("result", {}).get("hits", []) if isinstance(res, dict) else []
    documents = []
    for h in hits:
        fields = h.get("fields", {})
        documents.append({
            "content": fields.get(TEXT_FIELD, ""),
            "score": float(h.get("score", 0.0)),
            "source": fields.get("source")
        })

    return {"documents": documents}
