"""Évaluation (étape 7b) : taux de succès de la recherche + test de fumée.

Méthode : un petit jeu de questions/réponses écrit à la main sur les
documents d'exemple. Pour chaque question, on vérifie que le mot-clé
attendu apparaît dans les top-k chunks retrouvés. Rapide, déterministe,
sans LLM requis.
"""

from __future__ import annotations

import argparse
import logging
import time

from .config import SETTINGS
from .rag_chain import answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# (question, mot-clé attendu) — le mot-clé doit apparaître dans un chunk retrouvé.
TEST_SET = [
    ("Quel est le délai pour demander un remboursement ?", "30 jours"),
    ("Qui paie les frais de retour ?", "charge du client"),
    ("Sous combien de jours le remboursement est-il effectué ?", "14 jours"),
    ("Comment réinitialiser mon mot de passe ?", "Mot de passe oublié"),
    ("Quel est l'email du support ?", "support@exemple.com"),
]


def evaluate(k: int = SETTINGS.top_k, provider: str = "extractive") -> dict:
    """Exécute le jeu de test et renvoie précision + latences."""
    from .retriever import retrieve

    hits, latencies = 0, []
    details = []
    for question, keyword in TEST_SET:
        t0 = time.perf_counter()
        results = retrieve(question, k=k)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        found = any(keyword.lower() in doc.page_content.lower() for doc, _ in results)
        hits += int(found)
        details.append((question, keyword, found, round(dt, 2)))
        logger.info("%s | mot-clé='%s' | trouvé=%s | %.2fs", question[:50], keyword, found, dt)

    # Test de fumée : la chaîne complète tourne sans planter.
    smoke = answer(TEST_SET[0][0], k=k, provider=provider)["answer"][:100]
    report = {
        "accuracy": hits / len(TEST_SET),
        "hits": f"{hits}/{len(TEST_SET)}",
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "details": details,
        "smoke_answer_preview": smoke,
    }
    return report


def main() -> None:
    """Point d'entrée CLI : `python -m src.evaluate`."""
    parser = argparse.ArgumentParser(description="Évalue la qualité de la recherche.")
    parser.add_argument("-k", type=int, default=SETTINGS.top_k)
    parser.add_argument("--provider", default="extractive", choices=["ollama", "extractive"])
    args = parser.parse_args()

    try:
        report = evaluate(k=args.k, provider=args.provider)
    except Exception as exc:
        raise SystemExit(f"Évaluation impossible — avez-vous indexé ? python -m src.embed_index ({exc})")
    print(f"\nSuccès : {report['hits']} | Précision : {report['accuracy']:.0%} | Latence moy. : {report['avg_latency_s']}s")
    print(f"Aperçu réponse : {report['smoke_answer_preview']}...")


if __name__ == "__main__":
    main()
