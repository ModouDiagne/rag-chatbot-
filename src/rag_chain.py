"""Chaîne RAG complète (étape 6) : recherche -> prompt -> LLM -> réponse + sources.

Deux fournisseurs :
- `ollama` (défaut) : Mistral en local via le serveur Ollama. Nécessite
  `ollama serve` + `ollama pull mistral`. Gratuit, privé, hors-ligne.
- `extractive` : sans LLM — renvoie les meilleurs passages tels quels.
  Utile pour la CI, les démos sans serveur, et le débogage de la recherche.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from langchain_core.documents import Document

from .config import SETTINGS
from .retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Gabarit de prompt : force le LLM à rester ancré aux documents fournis.
PROMPT_TEMPLATE = """Tu es un assistant qui répond UNIQUEMENT à partir des documents fournis.
Si l'information n'est pas dans les documents, dis-le clairement.

Documents :
{context}

Question : {question}

Réponse (en français, concise, avec citations [1], [2]) :"""


def build_context(results: list[tuple[Document, float]]) -> str:
    """Numérote les chunks retrouvés [1], [2]... pour permettre les citations."""
    parts = []
    for i, (doc, _score) in enumerate(results, 1):
        src = Path(doc.metadata.get("source", "?")).name
        parts.append(f"[{i}] (source : {src})\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_extractive(question: str, k: int = SETTINGS.top_k) -> dict:
    """Répond sans LLM en renvoyant les passages les plus pertinents."""
    results = retrieve(question, k=k)
    context = build_context(results)
    answer = (
        "Réponse extractive (sans LLM — passages les plus pertinents) :\n\n" + context
        if context
        else "Je ne trouve pas l'information dans les documents fournis."
    )
    return {"answer": answer, "sources": results, "provider": "extractive"}


def answer_ollama(question: str, k: int = SETTINGS.top_k) -> dict:
    """Répond avec un modèle Ollama local, ancré sur les chunks retrouvés."""
    import ollama  # client python (pip install ollama)

    results = retrieve(question, k=k)
    if not results:
        return {"answer": "Je ne trouve pas l'information dans les documents fournis.", "sources": [], "provider": "ollama"}
    prompt = PROMPT_TEMPLATE.format(context=build_context(results), question=question)
    resp = ollama.chat(
        model=SETTINGS.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        # num_predict borne le temps de génération sur CPU (sinon file d'attente).
        options={"temperature": SETTINGS.temperature, "num_predict": 300},
    )
    return {"answer": resp["message"]["content"], "sources": results, "provider": f"ollama/{SETTINGS.ollama_model}"}


def answer(question: str, k: int = SETTINGS.top_k, provider: str = SETTINGS.llm_provider) -> dict:
    """Route vers le fournisseur configuré, avec repli gracieux vers l'extractif."""
    if provider == "ollama":
        try:
            return answer_ollama(question, k=k)
        except Exception as exc:
            logger.warning("Ollama indisponible (%s), repli en mode extractif.", exc)
            return answer_extractive(question, k=k)
    return answer_extractive(question, k=k)


def main() -> None:
    """Point d'entrée CLI : `python -m src.rag_chain "question"`."""
    parser = argparse.ArgumentParser(description="Pose une question au RAG.")
    parser.add_argument("question", help="Question en français")
    parser.add_argument("-k", type=int, default=SETTINGS.top_k)
    parser.add_argument("--provider", default=SETTINGS.llm_provider, choices=["ollama", "extractive"])
    args = parser.parse_args()

    out = answer(args.question, k=args.k, provider=args.provider)
    print(f"\n--- Réponse ({out['provider']}) ---\n{out['answer']}\n")
    print("--- Sources ---")
    for i, (doc, score) in enumerate(out["sources"], 1):
        print(f"[{i}] {Path(doc.metadata.get('source', '?')).name} (score={score:.3f})")


if __name__ == "__main__":
    main()
