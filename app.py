"""Démo Gradio — prête pour Hugging Face Spaces.

En local :  conda run -n tf python app.py
Sur HF Spaces (SDK Gradio) : fichier app = app.py.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.config import SETTINGS
from src.rag_chain import answer

# Questions d'exemple cliquables dans l'interface.
EXAMPLES = [
    "Quel est le délai pour demander un remboursement ?",
    "Comment réinitialiser mon mot de passe ?",
    "Quels sont les délais de livraison ?",
]


def respond(message: str, history: list, k: int) -> tuple[str, list]:
    """Répond à la question puis ajoute les sources citées."""
    out = answer(message, k=k)
    reply = out["answer"]
    if out["sources"]:
        cites = "\n".join(
            f"- [{i}] {Path(d.metadata.get('source', '?')).name} (score={s:.3f})"
            for i, (d, s) in enumerate(out["sources"], 1)
        )
        reply += f"\n\n**Sources :**\n{cites}"
    return reply, history


def build_demo() -> gr.Blocks:
    """Construit l'interface de démonstration."""
    with gr.Blocks(title="RAG Chatbot — vos documents") as demo:
        gr.Markdown(
            "# 🤖 RAG Chatbot — posez vos questions sur vos documents\n"
            "Recherche sémantique (Chroma) + LLM local (Ollama/Mistral). "
            "Chaque réponse cite ses sources."
        )
        with gr.Row():
            k = gr.Slider(1, 8, value=SETTINGS.top_k, step=1, label="Top-K passages")
        # Avec additional_inputs, chaque exemple doit couvrir tous les inputs : [question, k].
        exemples = [[q, SETTINGS.top_k] for q in EXAMPLES]
        gr.ChatInterface(
            fn=lambda msg, hist, k=k: respond(msg, hist, k)[0],
            examples=exemples,
            additional_inputs=[k],
        )
        gr.Markdown(
            "_Docs d'exemple : remboursement, support, livraison. "
            "Ajoutez vos PDF/MD dans `data/documents/`, puis relancez l'indexation._"
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
