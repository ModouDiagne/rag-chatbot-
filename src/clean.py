"""Nettoyage du texte avant indexation.

Les PDF et pages web contiennent des en-têtes, pieds de page, numéros de
page et caractères de contrôle. Ces artefacts polluent les embeddings, on
normalise donc le texte *avant* le découpage et l'indexation.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalise une chaîne de texte."""
    # Réduit 3+ sauts de ligne en un simple saut de paragraphe.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Supprime les blancs en début/fin de chaque ligne.
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Supprime les numéros de page isolés ("42" seul sur une ligne).
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Normalise les espaces/tabulations répétés.
    text = re.sub(r"[ \t]+", " ", text)
    # Supprime les caractères de contrôle non imprimables (garde \n et \t).
    text = "".join(c for c in text if c.isprintable() or c in "\n\t ")
    return text.strip()


def clean_documents(documents: list[Document], min_chars: int = 20) -> list[Document]:
    """Nettoie une liste de Documents en place et écarte les quasi-vides."""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return [doc for doc in documents if len(doc.page_content) > min_chars]
