"""Découpage en chunks (étape 2) — stratégie récursive (défaut recommandé).

Pourquoi : un document de 50 pages ne tient pas dans la fenêtre de contexte
d'un LLM, et on veut citer *le passage exact*. Le découpage récursif essaie
d'abord les paragraphes, puis les phrases, puis les mots, ce qui préserve
la cohérence sémantique.
"""

from __future__ import annotations

import argparse
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import SETTINGS
from .ingest import load_all_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_splitter(
    chunk_size: int = SETTINGS.chunk_size,
    chunk_overlap: int = SETTINGS.chunk_overlap,
) -> RecursiveCharacterTextSplitter:
    """Construit le découpeur récursif (paragraphes -> phrases -> mots)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(
    docs: list[Document],
    chunk_size: int = SETTINGS.chunk_size,
    chunk_overlap: int = SETTINGS.chunk_overlap,
) -> list[Document]:
    """Découpe les Documents en chunks en gardant source + index du chunk."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    logger.info("%d docs -> %d chunks (taille=%d, chevauchement=%d)", len(docs), len(chunks), chunk_size, chunk_overlap)
    return chunks


def main() -> None:
    """Point d'entrée CLI : `python -m src.chunk`."""
    parser = argparse.ArgumentParser(description="Découpe les documents pour le RAG.")
    parser.add_argument("--docs-dir", default=str(SETTINGS.docs_dir))
    parser.add_argument("--chunk-size", type=int, default=SETTINGS.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=SETTINGS.chunk_overlap)
    args = parser.parse_args()

    docs = load_all_documents(args.docs_dir)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    if chunks:
        logger.info("Exemple chunk [0] : %.300s...", chunks[0].page_content)
        logger.info("Métadonnées : %s", chunks[0].metadata)


if __name__ == "__main__":
    main()
