"""Streamlit UI — entrada principal do app. Pronta para deploy 1-click no Streamlit Cloud.

Voce nao precisa editar quase nada aqui — ja faz integracao com:
- src.pipeline.rag (TODOs 1-3)
- src.pipeline.cache (TODO 5)
- src.pipeline.routing (TODO 6)
- src.pipeline.tools (TODO 4, opcional)
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Adiciona o root do projeto no path para imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402
from openai import RateLimitError  # noqa: E402
from src.observability.trace import trace, log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402
from src.pipeline.tools import PPC_SECTIONS, lookup_section  # noqa: E402


def is_quota_error(error: Exception) -> bool:
    """Detecta erros de quota/rate limit para evitar crash vermelho no Streamlit."""
    error_name = type(error).__name__
    error_text = str(error)
    status_code = getattr(error, "status_code", None)

    return (
        status_code == 429
        or "RateLimit" in error_name
        or "rate limit" in error_text.lower()
        or "429" in error_text
        or "RESOURCE_EXHAUSTED" in error_text
        or "quota" in error_text.lower()
        or "exceeded" in error_text.lower()
    )


# ---------------------------------------------------------------- Streamlit UI
st.set_page_config(
    page_title="PPC Engenharia de Software — IFSP São Carlos",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 PPC Engenharia de Software — IFSP São Carlos")
st.caption(
    "Tire dúvidas sobre o curso: currículo, disciplinas, TCC, estágio, perfil do egresso "
    "e mais. Respostas baseadas no PPC 2023 (atualizado nov/2025)."
)


# Inicializacao lazy de pipeline + caches
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# Sidebar — metricas e debug
with st.sidebar:
    st.header("Metricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])

    if st.button("Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a pagina.")
    st.divider()
    st.header("Tool-use")

    section_keys = list(PPC_SECTIONS.keys())

    def format_section(key: str) -> str:
        data = PPC_SECTIONS[key]
        title = data.get("titulo") or data.get("title") or key
        return f"{key} — {title}"

    selected_section = st.selectbox(
        "Consultar seção estruturada do PPC",
        options=section_keys,
        format_func=format_section,
    )

    if st.button("Executar lookup_section"):
        tool_result = lookup_section(selected_section)
        st.success("Tool lookup_section executada")
        st.json(tool_result)


# Main — chat interface
query = st.text_input(
    "Sua pergunta:",
    placeholder="Ex: Quais disciplinas tem no 3º semestre? O que é necessário para fazer o TCC? Qual o perfil do egresso?",
)

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        # 1. Exact cache
        cached = exact_cache.get(query)
        if cached:
            st.success("Cache hit (exact)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        # 2. Semantic cache
        try:
            cached = semantic_cache.get(query)
        except NotImplementedError:
            cached = None
            st.warning("Semantic cache nao implementado (TODO 5). Caindo no LLM real.")
        except Exception as e:
            cached = None
            if is_quota_error(e):
                st.warning(
                    "Cache semântico indisponível no momento por limite de quota da API. "
                    "Continuando com RAG sem cache semântico."
                )
            else:
                raise

        if cached:
            st.success("Cache hit (semantic)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        # 3. Pipeline RAG + Routing
        try:
            decision = classify_complexity(query)
            st.info(f"Routing: {decision.complexity} -> {decision.model}")
            log_event("route_decision", trace_id=trace_id, **decision.__dict__)
        except NotImplementedError:
            st.warning("Routing nao implementado (TODO 6). Usando modelo default.")
        except Exception as e:
            if is_quota_error(e):
                st.warning(
                    "Routing indisponível temporariamente por limite de quota. "
                    "Usando modelo default."
                )
            else:
                raise

        try:
            result = pipeline.answer(query)
        except NotImplementedError as e:
            st.error(f"Pipeline nao implementado: {e}")
            st.info("Implemente TODOs 1-3 em `src/pipeline/rag.py` para destravar.")
            st.stop()
        except Exception as e:
            if not is_quota_error(e):
                raise

            result = {
                "answer": (
                    "A cota da API Gemini foi atingida no momento. "
                    "O app continua funcionando, mas novas respostas podem depender da liberação da quota. "
                    "Tente novamente em alguns minutos ou use uma pergunta que já esteja em cache."
                ),
                "sources": [],
                "rate_limited": True,
            }

        # 4. Renderiza + cacheia
        st.write(result["answer"])

        if result.get("rate_limited"):
            st.warning(
                "A resposta acima foi gerada em modo fallback porque a cota da API foi atingida."
            )

        if result.get("sources"):
            with st.expander("Fontes citadas"):
                for source, page in result["sources"]:
                    st.write(f"- `{source}:p{page}`")

        exact_cache.put(query, result["answer"])

        try:
            semantic_cache.put(query, result["answer"])
        except Exception as e:
            if is_quota_error(e):
                st.warning(
                    "A resposta foi salva no cache exato, mas o cache semântico não foi atualizado "
                    "porque a cota da API foi atingida."
                )
            else:
                raise

        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))


st.divider()
st.caption(
    "Corpus: PPC Bacharelado em Engenharia de Software — IFSP Campus São Carlos "
    "(Atualização Nov/2025) | RAG + Gemini Embeddings + Groq/Llama + Chroma"
)