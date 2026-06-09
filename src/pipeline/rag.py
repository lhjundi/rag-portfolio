"""RAG pipeline — chunk, embed, index, retrieve, generate.

Usa Gemini para embeddings e, se disponivel, Groq para gerar respostas.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _is_quota_error(error: Exception) -> bool:
    """Detecta erros de quota/rate limit mesmo quando a exception muda no deploy."""
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


def _make_chat_client() -> tuple[OpenAI, str]:
    """Cliente de geração: prefere Groq; se não houver, cai para Gemini/OpenAI."""
    if "GROQ_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=GROQ_BASE_URL,
        )
        model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        return client, model

    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=GEMINI_BASE_URL,
        )
        model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        return client, model

    if "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        return client, model

    raise RuntimeError("Configure GROQ_API_KEY, GEMINI_API_KEY ou OPENAI_API_KEY.")


def _make_embedding_function(embed_model: str) -> OpenAIEmbeddingFunction:
    """Função de embedding: usa Gemini se disponível; senão OpenAI."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAIEmbeddingFunction(
            api_key=os.environ["GEMINI_API_KEY"],
            api_base=GEMINI_BASE_URL,
            model_name=embed_model,
        )

    if "OPENAI_API_KEY" in os.environ:
        return OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=embed_model,
        )

    raise RuntimeError(
        "Configure GEMINI_API_KEY ou OPENAI_API_KEY para embeddings. "
        "A GROQ_API_KEY sozinha não gera embeddings para o Chroma."
    )


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, default_llm_model = _make_chat_client()
        self.llm_model = llm_model or default_llm_model

        # Mantem Gemini como embedding default para ser compativel com o indice versionado.
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")
        self.embed_fn = _make_embedding_function(self.embed_model)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma."""
        docs: list[dict] = []

        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            source = pdf_path.name
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue

                docs.append(
                    {
                        "text": text,
                        "source": source,
                        "page": page_number,
                    }
                )

        chunks: list[dict] = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        for doc in docs:
            source = doc["source"]
            page = doc["page"]

            for i, piece in enumerate(splitter.split_text(doc["text"])):
                chunks.append(
                    {
                        "id": f"{source}-{page}-{i}",
                        "text": piece,
                        "source": source,
                        "page": page,
                    }
                )

        batch_size = 80
        throttle_seconds = 60
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for index in range(total_batches):
            start = index * batch_size
            batch = chunks[start : start + batch_size]

            ids = [c["id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = [{"source": c["source"], "page": c["page"]} for c in batch]

            for attempt in range(6):
                try:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    break
                except Exception as e:
                    if not _is_quota_error(e):
                        raise

                    if attempt == 5:
                        raise

                    time.sleep(65)

            if index < total_batches - 1:
                time.sleep(throttle_seconds)

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        results = self.collection.query(query_texts=[query], n_results=k)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        hits: list[dict] = []

        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}

            hits.append(
                {
                    "text": text,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "distance": distance,
                }
            )

        return hits

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate."""
        try:
            hits = self.retrieve(question, k=k)
        except Exception as e:
            if not _is_quota_error(e):
                raise

            return {
                "answer": (
                    "A cota da API de embeddings foi atingida durante a busca semântica. "
                    "Tente novamente em alguns minutos."
                ),
                "sources": [],
                "rate_limited": True,
            }

        context = "\n".join(
            f"[{h['source']}:p{h['page']}]\n{h['text']}\n---" for h in hits
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0,
            )

            answer = response.choices[0].message.content or ""

            return {
                "answer": answer,
                "sources": [(h["source"], h["page"]) for h in hits],
                "rate_limited": False,
            }

        except Exception as e:
            if not _is_quota_error(e):
                raise

            fallback_parts = []
            for h in hits[:3]:
                excerpt = " ".join((h["text"] or "").split())
                excerpt = excerpt[:450] + ("..." if len(excerpt) > 450 else "")
                fallback_parts.append(f"- [{h['source']}:p{h['page']}] {excerpt}")

            fallback_answer = (
                "A busca no PPC funcionou, mas a cota do modelo de geração foi atingida. "
                "Abaixo estão os trechos mais relevantes recuperados do corpus:\n\n"
                + "\n\n".join(fallback_parts)
            )

            return {
                "answer": fallback_answer,
                "sources": [(h["source"], h["page"]) for h in hits],
                "rate_limited": True,
            }


PROMPT_TEMPLATE = """Você é um assistente especializado no PPC (Projeto Pedagógico de Curso)
do Bacharelado em Engenharia de Software do IFSP São Carlos.

Responda APENAS com base no contexto abaixo. Se a informação não estiver no contexto,
diga "Não encontrei essa informação no PPC."

Sempre cite a seção ou página usando o formato [p.{{página}}].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)

    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()

    return pipeline