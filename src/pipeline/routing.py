"""Model routing cheap-first com fallback.

Reaproveita o notebook 05. Voce vai preencher 1 TODO aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium).

    Estrategia heuristica simples. Em producao, evoluiria para classifier treinado.
    """
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    # SEU CODIGO AQUI — TODO 6
    # Implemente heuristica simples para classificar a query como "simple" ou "complex".
    # Sugestao de regras:
    #   - len(query) < 60 e query termina em "?" → simple
    #   - contem palavras como "explique", "compare", "analise", "projete" → complex
    #   - default → simple
    # Retorne RouteDecision(model=cheap_model OU premium_model, complexity=..., reason="por que")
    # Dica: notebook 05, Etapa 5 — Model Routing.
    COMPLEX_MARKERS = (
        "explique", "compare", "analise", "diferença", "quando devo", "por que",
        "como funciona", "quais são", "descreva", "detalhe", "explica",
        "o que significa", "me explica", "quais disciplinas", "como é estruturado",
    )

    if len(query) < 60 and query.endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="pergunta curta (<60) terminando em '?'",
        )

    lowered = query.lower()
    for marker in COMPLEX_MARKERS:
        if marker in lowered:
            return RouteDecision(
                model=premium_model,
                complexity="complex",
                reason=f"contém marcador de complexidade: '{marker}'",
            )

    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="default (nenhuma regra de complexidade disparou)",
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()
