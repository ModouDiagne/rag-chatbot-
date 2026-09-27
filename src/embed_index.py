"""Embeddings + indexation (étapes 3-4) avec Chroma (locale, persistante).

Modèle par défaut : paraphrase-multilingual-MiniLM-L12-v2 — 384 dimensions,
multilingue (FR OK), rapide sur CPU. Surchargeable via RAG_EMBEDDING_MODEL.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

# Le repli langchain-community émet un avertissement de dépréciation inoffensif : on le masque.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    # Paquet dédié (recommandé si installé).
    from langchain_chroma import LangChainChroma as ChromaStore
except ImportError:  # repli : version fournie avec langchain-community
    from langchain_community.vectorstores import Chroma as ChromaStore

from .chunk import chunk_documents
from .config import SETTINGS
from .ingest import load_all_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_embeddings(model_name: str = SETTINGS.embedding_model) -> HuggingFaceEmbeddings:
    """Construit les embeddings sentence-transformers (CPU, normalisés)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(
    chunks: list[Document],
    persist_dir: Path | str = SETTINGS.chroma_dir,
    collection_name: str = SETTINGS.collection_name,
    model_name: str = SETTINGS.embedding_model,
) -> Any:
    """Encode les chunks et les persiste dans Chroma (recrée la collection)."""
    embeddings = make_embeddings(model_name)
    store = ChromaStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )
    logger.info("Indexé %d chunks -> %s (collection=%s)", len(chunks), persist_dir, collection_name)
    return store


# Mémoire des index déjà ouverts : évite de recharger le modèle (~7 s) à chaque question.
_STORE_CACHE: dict[tuple[str, str, str], Any] = {}


def load_index(
    persist_dir: Path | str = SETTINGS.chroma_dir,
    collection_name: str = SETTINGS.collection_name,
    model_name: str = SETTINGS.embedding_model,
) -> Any:
    """Rouvre une collection Chroma persistante existante (mise en cache)."""
    key = (str(persist_dir), collection_name, model_name)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = ChromaStore(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            embedding_function=make_embeddings(model_name),
        )
    return _STORE_CACHE[key]


def main() -> None:
    """Point d'entrée CLI : `python -m src.embed_index`."""
    parser = argparse.ArgumentParser(description="Construit l'index vectoriel Chroma.")
    parser.add_argument("--docs-dir", default=str(SETTINGS.docs_dir))
    parser.add_argument("--persist-dir", default=str(SETTINGS.chroma_dir))
    args = parser.parse_args()

    docs = load_all_documents(args.docs_dir)
    if not docs:
        raise SystemExit("Aucun document. Ajoutez des PDF/MD/TXT dans data/documents d'abord.")
    chunks = chunk_documents(docs)
    build_index(chunks, persist_dir=args.persist_dir)


if __name__ == "__main__":
    main()
