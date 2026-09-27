"""API REST (FastAPI) — socle backend pour un futur frontend React et pour les recruteurs.

Lance :  uvicorn api:app --host 0.0.0.0 --port 8000
Docs auto : http://localhost:8000/docs

Endpoints :
- GET  /health  -> état + nb de chunks indexés
- POST /ask     -> {question, k?, provider?} -> {answer, sources, provider}
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import SETTINGS
from src.rag_chain import answer

app = FastAPI(title="RAG Chatbot API", version="1.0.0")


class AskRequest(BaseModel):
    """Corps de la requête /ask."""

    question: str = Field(..., min_length=3, description="Question en français")
    k: int = Field(default=SETTINGS.top_k, ge=1, le=8, description="Nombre de passages")
    provider: str = Field(default=SETTINGS.llm_provider, description="ollama ou extractive")


class Source(BaseModel):
    """Un passage cité."""

    file: str
    page_or_chunk: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    """Réponse de /ask."""

    answer: str
    provider: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    """Vérifie que l'index Chroma existe et est lisible."""
    from src.embed_index import load_index

    try:
        store = load_index()
        count = store._collection.count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Index indisponible : {exc}")
    return {"status": "ok", "chunks": count, "collection": SETTINGS.collection_name}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Pose une question au RAG."""
    try:
        out = answer(req.question, k=req.k, provider=req.provider)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    sources = [
        Source(
            file=Path(d.metadata.get("source", "?")).name,
            page_or_chunk=str(d.metadata.get("page", d.metadata.get("chunk_id", "?"))),
            score=round(float(s), 3),
            excerpt=d.page_content[:300],
        )
        for d, s in out["sources"]
    ]
    return AskResponse(answer=out["answer"], provider=out["provider"], sources=sources)
