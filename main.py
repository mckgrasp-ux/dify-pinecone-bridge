import os
import uuid
import time
import requests
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from pinecone import Pinecone

# -------------------------
# ENV
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")
TEXT_FIELD = os.getenv("TEXT_FIELD", "chunk_text") 
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "") 

# Klucz do inteligencji czytającej tabele w PDF
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

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

# -------------------------
# LlamaParse - Bezpośrednie wywołanie API (Bez konfliktów bibliotek!)
# -------------------------
def _pdf_to_pages_text(file_path: str) -> List[str]:
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError("Brak LLAMA_CLOUD_API_KEY w Renderze!")
    
    base_url = "https://api.cloud.llamaindex.ai/api/parsing"
    headers = {
        "Authorization": f"Bearer {LLAMA_CLOUD_API_KEY}",
        "Accept": "application/json"
    }
    
    # 1. Wysyłamy plik na serwery AI
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/pdf")}
        data = {"language": "pl"} # Wymuszamy język polski
        
        resp = requests.post(f"{base_url}/upload", headers=headers, files=files, data=data)
        if resp.status_code != 200:
            raise Exception(f"Błąd wysyłania do LlamaParse: {resp.text}")
            
        job_id = resp.json()["id"]

    # 2. Czekamy cierpliwie aż AI skończy czytać (odpytujemy co 3 sekundy)
    while True:
        time.sleep(3)
        status_resp = requests.get(f"{base_url}/job/{job_id}", headers=headers)
        status_resp.raise_for_status()
        status = status_resp.json()["status"]
        
        if status == "SUCCESS":
            break
        elif status in ["ERROR", "FAILED", "CANCELED"]:
            raise Exception(f"LlamaParse napotkało błąd podczas analizy. Status: {status}")

    # 3. Pobieramy gotowy, zrekonstruowany tekst (Markdown z tabelami)
    res_resp = requests.get(f"{base_url}/job/{job_id}/result/markdown", headers=headers)
    res_resp.raise_for_status()
    markdown_text = res_resp.json()["markdown"]

    # Zwracamy całość - LlamaParse samo zadbało o odpowiedni format
    return [markdown_text]

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

def _batched(lst: List[Dict[str, Any]], batch_size: int = 90):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

# -------------------------
# health
# -------------------------
@app.get("/")
def root():
    return {"ok": True}

# -------------------------
# ingest
# -------------------------
@app.post("/ingest")
async def ingest(namespace: str = "default", file: UploadFile = File(...)):
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
                records.append(
                    {
                        "_id": uuid.uuid4().hex,
                        TEXT_FIELD: chunk,                 
                        "filename": file.filename,
                        "page": page_idx,
                        "chunk": chunk_idx,
                        "source": "upload",
                    }
                )

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

        upserted = 0
        for batch in _batched(records, batch_size=90):
            index.upsert_records(namespace, batch)
            upserted += len(batch)

        return JSONResponse(
            {
                "ok": True,
                "filename": file.filename,
                "namespace": namespace,
                "pages_with_text": pages_with_text,
                "chunks": len(records),
                "upserted": upserted,
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
# retrieval
# -------------------------
@app.post("/retrieval")
async def retrieval(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
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
        res = index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": query},
            },
            return_fields=[TEXT_FIELD, "text", "filename"]
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
