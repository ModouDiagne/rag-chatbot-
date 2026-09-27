"""Configuration centrale du chatbot RAG.

Tous les chemins sont résolus par rapport à la racine du projet pour que
le code fonctionne aussi bien en local que sur Hugging Face Spaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Renvoie la racine du projet (dossier parent de `src/`)."""
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Réglages immuables, surchargeables par variables d'environnement."""

    project_root: Path = field(default_factory=_project_root)

    # --- Documents ---
    docs_dir: Path = field(default_factory=lambda: _project_root() / "data" / "documents")

    # --- Découpage (stratégie récursive, voir src/chunk.py) ---
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

    # --- Embeddings ---
    # Modèle léger + multilingue (FR OK) + rapide sur CPU.
    # Piste d'amélioration : intfloat/multilingual-e5-large (1024 dim, plus lourd).
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # --- Base vectorielle (Chroma, locale et persistante) ---
    chroma_dir: Path = field(default_factory=lambda: _project_root() / "chroma_db")
    collection_name: str = os.getenv("RAG_COLLECTION", "rag_docs")

    # --- Recherche ---
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    # --- LLM ---
    # Défaut : Ollama en local (gratuit, privé). Nécessite `ollama serve` + `ollama pull mistral`.
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "ollama")  # ollama | extractive
    # Modèle léger par défaut (tient sur 8 Go de RAM). Montée en gamme : export RAG_OLLAMA_MODEL="mistral".
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "llama3.2:1b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.1"))


SETTINGS = Settings()
