"""Chargement des documents (étape 1 du pipeline RAG).

Formats supportés : .pdf, .txt, .md, .html/.htm
Un Document par page PDF (pour citer la page exacte),
un Document par fichier pour les autres formats.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from .clean import clean_documents
from .config import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Chargeur associé à chaque extension de fichier.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def load_file(file_path: Path) -> list[Document]:
    """Charge un seul fichier selon son extension."""
    ext = file_path.suffix.lower()
    loader_cls = LOADERS.get(ext)
    if loader_cls is None:
        logger.warning("Format non supporté, ignoré : %s", file_path.name)
        return []
    try:
        return loader_cls(str(file_path)).load()
    except Exception as exc:  # on continue avec les autres fichiers
        logger.error("Échec sur %s : %s", file_path.name, exc)
        return []


def load_all_documents(directory: Path | str = SETTINGS.docs_dir) -> list[Document]:
    """Charge + nettoie tous les fichiers supportés d'un dossier (récursif)."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dossier de documents introuvable : {directory}")

    all_docs: list[Document] = []
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in LOADERS)
    logger.info("Chargement de %d fichier(s) depuis %s", len(files), directory)
    for fp in files:
        logger.info("  %s", fp.name)
        all_docs.extend(load_file(fp))

    logger.info("%d Document(s) brut(s) chargé(s), nettoyage...", len(all_docs))
    cleaned = clean_documents(all_docs)
    logger.info("%d Document(s) après nettoyage", len(cleaned))
    return cleaned


def print_stats(docs: list[Document]) -> None:
    """Affiche des statistiques utiles pour régler le découpage."""
    if not docs:
        logger.warning("Aucun document chargé.")
        return
    total_chars = sum(len(d.page_content) for d in docs)
    word_counts = [len(d.page_content.split()) for d in docs]
    logger.info(
        "Stats : docs=%d | total_car=%d | moy_car=%d | moy_mots=%d | min_mots=%d | max_mots=%d",
        len(docs),
        total_chars,
        total_chars // len(docs),
        sum(word_counts) // len(word_counts),
        min(word_counts),
        max(word_counts),
    )
    if docs:
        logger.info("Premier doc source=%s", docs[0].metadata.get("source"))
        logger.info("Aperçu : %.200s...", docs[0].page_content)


def main() -> None:
    """Point d'entrée CLI : `python -m src.ingest`."""
    parser = argparse.ArgumentParser(description="Charge et nettoie les documents RAG.")
    parser.add_argument("--docs-dir", default=str(SETTINGS.docs_dir))
    args = parser.parse_args()
    docs = load_all_documents(args.docs_dir)
    print_stats(docs)


if __name__ == "__main__":
    main()
