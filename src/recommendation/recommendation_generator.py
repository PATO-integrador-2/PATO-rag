"""
Recommendation generator — orchestrator of the Recommendation module.

Pipeline:
    1. Classify the question → topic, source, entity, dimension, focus
       (LLM: qwen2.5:1.5b)
    2. Retrieve relevant context from the RAG retriever
       (knowledge.document_chunks, hybrid search)
    3. Generate concrete project proposals with qwen2.5:3b
       (grounded in the retrieved context)
    4. Return the answer + sources + token usage

No simulation, no invented numbers — only grounded proposals.

Public API:
    gen = RecommendationGenerator()
    result = gen.answer("¿Qué proyectos para reducir el hacinamiento en Antioquia?")
    print(result["answer"])
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_retriever import RagRetriever
from src.recommendation.recommendation_intent_classifier import (
    RecommendationIntentClassifier,
)
from src.recommendation.recommendation_answer_builder import (
    RecommendationAnswerBuilder,
)
from src.recommendation.recommendation_config import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """Orchestrates the Recommendation pipeline."""

    def __init__(
        self,
        retriever: Optional[RagRetriever] = None,
        classifier: Optional[RecommendationIntentClassifier] = None,
        answer_builder: Optional[RecommendationAnswerBuilder] = None,
    ):
        self.retriever = retriever or RagRetriever()
        self.classifier = classifier or RecommendationIntentClassifier()
        self.answer_builder = answer_builder or RecommendationAnswerBuilder()

    def close(self):
        if self.retriever:
            self.retriever.close()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def answer(self, question: str, verbose: bool = False) -> dict:
        """
        Full Recommendation pipeline.

        Returns:
            {
                "question": str,
                "topic": str,
                "source": "dane"|"cepal"|"both"|"general",
                "entity": str,
                "dimension": str,
                "focus": str,
                "classifier_method": "llm"|"fallback",
                "answer": str,
                "sources": list[dict],
                "context_chunks": int,
                "status": "ok"|"error",
                "tokens": {"input": int, "output": int},
                "latency_ms": int,
            }
        """
        start = time.time()

        # ------------------------------------------------------
        # 1. Classify intent (includes its own tokens)
        # ------------------------------------------------------
        try:
            classified = self.classifier.classify(question)
        except Exception as exc:
            logger.exception("Error clasificando: %s", exc)
            classified = {
                "topic": question,
                "source": "general",
                "entity": "general",
                "dimension": "general",
                "focus": question,
                "method": "fallback",
                "tokens": {"input": 0, "output": 0},
            }

        classifier_tokens = classified.get("tokens", {"input": 0, "output": 0})

        if verbose:
            print(f"[Recommendation] topic={classified['topic']!r}")
            print(f"[Recommendation] source={classified.get('source', 'general')!r} "
                  f"entity={classified['entity']!r} "
                  f"dimension={classified['dimension']!r}")
            print(f"[Recommendation] focus={classified['focus']!r}")

        # ------------------------------------------------------
        # 2. Retrieve context (enriched with source hints)
        # ------------------------------------------------------
        retrieval_query = self._build_retrieval_query(classified)

        if verbose:
            print(f"[Recommendation] retrieval_query={retrieval_query!r}")

        try:
            chunks = self.retriever.search(
                retrieval_query, top_k=RETRIEVAL_TOP_K,
            )
        except Exception as exc:
            logger.exception("Error en retrieval: %s", exc)
            chunks = []

        if verbose:
            print(f"[Recommendation] {len(chunks)} chunks recuperados")

        # ------------------------------------------------------
        # 3. Build answer (returns tuple: answer, tokens)
        # ------------------------------------------------------
        try:
            answer, answer_tokens = self.answer_builder.build(
                question,
                chunks,
                source=classified.get("source", "general"),
                dimension=classified.get("dimension", "general"),
            )
        except Exception as exc:
            logger.exception("Error generando respuesta: %s", exc)
            answer = (
                "No pude generar recomendaciones en este momento. "
                "Intenta nuevamente."
            )
            answer_tokens = {"input": 0, "output": 0}

        # ------------------------------------------------------
        # 4. Extract sources for citations
        # ------------------------------------------------------
        sources = self._extract_sources(chunks)

        # ------------------------------------------------------
        # 5. Sum tokens (classifier + answer)
        # ------------------------------------------------------
        total_tokens = {
            "input": classifier_tokens.get("input", 0) + answer_tokens.get("input", 0),
            "output": classifier_tokens.get("output", 0) + answer_tokens.get("output", 0),
        }

        return {
            "question": question,
            "topic": classified["topic"],
            "source": classified.get("source", "general"),
            "entity": classified["entity"],
            "dimension": classified["dimension"],
            "focus": classified["focus"],
            "classifier_method": classified.get("method", "unknown"),
            "answer": answer,
            "sources": sources,
            "context_chunks": len(chunks),
            "status": "ok",
            "tokens": total_tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _build_retrieval_query(classified: dict) -> str:
        """
        Build the retrieval query from topic + focus + entity.
        Keep it focused — do NOT add generic keywords.
        """
        topic = classified.get("topic", "") or ""
        focus = classified.get("focus", "") or ""
        entity = classified.get("entity", "") or ""

        parts = [topic, focus]
        if entity and entity != "general":
            parts.append(entity)
        return " ".join(p for p in parts if p).strip()

    @staticmethod
    def _extract_sources(chunks: list[dict]) -> list[dict]:
        """
        Deduplicate chunks by (document_id, chunk_index) and return
        a compact source list with metadata.
        """
        seen = set()
        sources = []
        for c in chunks:
            key = (str(c.get("document_id")), c.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            preview = (c.get("content") or "").strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            sources.append({
                "document_id": str(c.get("document_id")),
                "document_title": c.get("document_title"),
                "document_url": c.get("document_url"),
                "organization": c.get("organization"),
                "page_number": c.get("page_number"),
                "chunk_index": c.get("chunk_index"),
                "preview": preview,
                "similarity": c.get("similarity"),
                "rrf_score": c.get("rrf_score"),
            })
        return sources


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = RecommendationGenerator()
    try:
        tests = [
            "¿Qué proyectos puedo implementar para reducir el hacinamiento en Antioquia?",
            "Dame ideas de políticas para combatir la pobreza multidimensional",
        ]
        for q in tests:
            print(f"\n{'=' * 72}")
            print(f"❓ {q}")
            print('=' * 72)
            r = gen.answer(q, verbose=True)
            print(f"\n  Source: {r['source']} | Entity: {r['entity']} "
                    f"| Dimension: {r['dimension']}")
            print(f"  Status: {r['status']} | Chunks: {r['context_chunks']} "
                    f"| Latency: {r['latency_ms']} ms")
            print(f"  Tokens: in={r['tokens']['input']} out={r['tokens']['output']}")
            print(f"\n💬 RESPUESTA:\n{r['answer'][:900]}")
            if r["sources"]:
                print(f"\n📚 Fuentes ({len(r['sources'])}):")
                for s in r["sources"][:3]:
                    print(f"  - {s['document_title']} p.{s['page_number']}")
    finally:
        gen.close()