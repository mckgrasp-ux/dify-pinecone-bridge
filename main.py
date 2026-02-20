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
TEXT_FIELD = os.getenv("TEXT_FIELD", "text")

# API key for Dify -> External Knowledge API
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")

if not PINECONE_API_KEY or not PINECONE_HOST:
    # index name is optional if you target by host, but we keep it for clarity
    pass

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

app = FastAPI()


# -------------------------
# helpers
# -------------------------
def _check_auth(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    """
    Accept:
      - Authorization: Bearer <token>
      - X-API-Key: <token>
    If EXTERNAL_API_KEY is empty -> no auth enforced (not recommended).
    """
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
        # lekkie czyszczenie
        txt = " ".join(txt.split())
        pages.append(txt)
    return pages


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Prosty chunker po znakach (wystarczy na start).
    """
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


# -------------------------
# health
# -------------------------
@app.get("/")
def root():
    return {"ok": True}


# -------------------------
# ingest: multipart file -> pinecone upsert_records
# -------------------------
@app.post("/ingest")
async def ingest(namespace: str = "default", file: UploadFile = File(...)):
    # zapis tymczasowy
    tmp_name = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    content = await file.read()
    with open(tmp_name, "wb") as f:
        f.write(content)

    try:
        pages = _pdf_to_pages_text(tmp_name)
        pages_with_text = 0
        records: List[Dict[str, Any]] = []

        for page_idx, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            pages_with_text += 1

            chunks = _chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks, start=1):
                rec_id = uuid.uuid4().hex
                # Records API: _id + fields (zgodne z field_map w indexie)
                rec = {
                    "_id": rec_id,
                    TEXT_FIELD: chunk,
                    "filename": file.filename,
                    "page": page_idx,
                    "chunk": chunk_idx,
                    "source": "upload",
                }
                records.append(rec)

        if not records:
            return JSONResponse(
                {
                    "ok": True,
                    "filename": file.filename,
                    "namespace": namespace,
                    "pages_with_text": 0,
                    "chunks": 0,
                    "upserted": 0,
                }
            )

        # Pinecone integrated embedding: upsert_records(namespace, records)
        index.upsert_records(namespace, records)

        return JSONResponse(
            {
                "ok": True,
                "filename": file.filename,
                "namespace": namespace,
                "pages_with_text": pages_with_text,
                "chunks": len(records),
                "upserted": len(records),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass


# -------------------------
# retrieval: Dify External Knowledge API
# POST /retrieval
# -------------------------
@app.post("/retrieval")
async def retrieval(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Dify wysyła m.in.:
    {
      "query": "...",
      "knowledge_id": "nordkalk",
      "retrieval_setting": { "top_k": 4, "score_threshold": 0.5 }
    }

    Oczekuje:
    { "records": [ { "content": "...", "score": 0.8, "title": "...", "metadata": {...} }, ... ] }
    """
    _check_auth(authorization, x_api_key)

    query = (payload.get("query") or "").strip()
    knowledge_id = (payload.get("knowledge_id") or "").strip()
    rs = payload.get("retrieval_setting") or {}

    if not query:
        return {"records": []}

    namespace = knowledge_id or "default"
    top_k = int(rs.get("top_k") or 4)
    score_threshold = rs.get("score_threshold", None)
    score_threshold = float(score_threshold) if score_threshold is not None else None

    try:
        # Integrated embedding search by text:
        res = index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": query},
            },
        )

        hits = (((res or {}).get("result") or {}).get("hits") or [])
        out: List[Dict[str, Any]] = []

        for h in hits:
            score = float(h.get("_score") or 0.0)
            fields = h.get("fields") or {}

            if score_threshold is not None and score < score_threshold:
                continue

            content = fields.get(TEXT_FIELD) or fields.get("text") or ""
            title = fields.get("filename") or "document"

            metadata = dict(fields)
            metadata["id"] = h.get("_id")
            metadata["namespace"] = namespace

            out.append(
                {
                    "content": content,
                    "score": score,
                    "title": title,
                    "metadata": metadata,
                }
            )

        return {"records": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
