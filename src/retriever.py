"""Recherche sémantique (étape 5) — similarité cosinus sur Chroma."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from langchain_core.documents import Document

from .config import SETTINGS
from .embed_index import load_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def retrieve(query: str, k: int = SETTINGS.top_k) -> list[tuple[Document, float]]:
    """Renvoie les top-k chunks les plus similaires, avec leurs scores."""
    store = load_index()
    results = store.similarity_search_with_score(query, k=k)
    return results


def format_sources(results: list[tuple[Document, float]]) -> str:
    """Formate les chunks retrouvés pour affichage (fichier + page + score)."""
    lines = []
    for i, (doc, score) in enumerate(results, 1):
        src = Path(doc.metadata.get("source", "?")).name
        page = doc.metadata.get("page", doc.metadata.get("chunk_id", "?"))
        lines.append(f"[{i}] {src} (page/chunk={page}, score={score:.3f})\n    {doc.page_content[:220]}...")
    return "\n".join(lines)


def main() -> None:
    """Point d'entrée CLI : `python -m src.retriever "question"`."""
    parser = argparse.ArgumentParser(description="Interroge l'index vectoriel.")
    parser.add_argument("query", help="Question à rechercher")
    parser.add_argument("-k", type=int, default=SETTINGS.top_k)
    args = parser.parse_args()

    try:
        results = retrieve(args.query, k=args.k)
    except Exception as exc:
        raise SystemExit(f"Index introuvable ou vide. Indexez d'abord : python -m src.embed_index ({exc})")
    print(format_sources(results) or "Aucun résultat.")


if __name__ == "__main__":
    main()
