import os
import re
import uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

from pypdf import PdfReader

from pinecone import Pinecone


# -----------------------------
# Config (ENV)
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_HOST = os.getenv("PINECONE_HOST", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()

# Pinecone Inference embedding model (z UI Pinecone: np. multilingual-e5-large)
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large").strip()

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))      # znaki
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200")) # znaki

# Batch sizes
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))


def _require_env():
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not PINECONE_HOST:
        missing.append("PINECONE_HOST")
    if not PINECONE_INDEX_NAME:
        missing.append("PINECONE_INDEX_NAME")
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing environment variables: {', '.join(missing)}",
        )


def _clean_text(t: str) -> str:
    t = t.replace("\u0000", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_text(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Zwraca listę: [{page: int, text: str}, ...]"""
    reader = PdfReader(io_bytes := _BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _clean_text(txt)
        if txt:
            pages.append({"page": i + 1, "text": txt})
    return pages


class _BytesIO:
    """Minimalny BytesIO bez importu io (żeby było ultra-prosto na Render)."""
    def __init__(self, b: bytes):
        self._b = b
        self._i = 0

    def read(self, n: Optional[int] = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._b) - self._i
        out = self._b[self._i:self._i + n]
        self._i += len(out)
        return out

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._i = pos
        elif whence == 1:
            self._i += pos
        elif whence == 2:
            self._i = len(self._b) + pos
        self._i = max(0, min(self._i, len(self._b)))
        return self._i

    def tell(self) -> int:
        return self._i


def _get_clients():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    return pc, index


app = FastAPI(title="dify-pinecone-bridge", version="1.0.0")


@app.get("/")
def root():
    return {"ok": True}


@app.get("/health")
def health():
    _require_env()
    return {
        "ok": True,
        "index": PINECONE_INDEX_NAME,
        "host": PINECONE_HOST,
        "embed_model": PINECONE_EMBED_MODEL,
    }


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    namespace: str = Query("default"),
    source: str = Query("upload"),
):
    """
    Upload PDF -> extract text -> embed via Pinecone Inference -> upsert vectors.
    """
    _require_env()

    filename = file.filename or "file.pdf"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) Extract per page
    try:
        pages = _extract_pdf_text(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")

    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    # 2) Chunk
    chunks: List[Dict[str, Any]] = []
    for p in pages:
        page_no = p["page"]
        for j, ch in enumerate(_chunk_text(p["text"], CHUNK_SIZE, CHUNK_OVERLAP), start=1):
            chunks.append(
                {
                    "id": f"{uuid.uuid4()}",
                    "text": ch,
                    "metadata": {
                        "source": source,
                        "filename": filename,
                        "page": page_no,
                        "chunk": j,
                    },
                }
            )

    # 3) Embed + upsert
    pc, index = _get_clients()

    total_upserted = 0
    try:
        # batch embed
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            texts = [b["text"] for b in batch]

            emb = pc.inference.embed(
                model=PINECONE_EMBED_MODEL,
                inputs=texts,
                parameters={"input_type": "passage"},
            )

            vectors = []
            for item, vec in zip(batch, emb.data):
                vectors.append(
                    {
                        "id": item["id"],
                        "values": vec["values"],
                        "metadata": {**item["metadata"], "text": item["text"]},
                    }
                )

            # upsert in smaller batches if needed
            for k in range(0, len(vectors), UPSERT_BATCH):
                part = vectors[k:k + UPSERT_BATCH]
                index.upsert(vectors=part, namespace=namespace)
                total_upserted += len(part)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

    return JSONResponse(
        {
            "ok": True,
            "filename": filename,
            "namespace": namespace,
            "pages_with_text": len(pages),
            "chunks": len(chunks),
            "upserted": total_upserted,
        }
    )
