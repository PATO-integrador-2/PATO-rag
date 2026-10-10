"""
RAG generator — orchestrator of the RAG module.

Pipeline:
    1. Embed the query
    2. Hybrid retrieval (semantic + lexical with RRF)
    3. Build context from top-K chunks
    4. Generate answer with qwen2.5:3b
    5. Return the answer + sources + tokens
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
from src.rag.rag_answer_builder import RagAnswerBuilder
from src.rag.rag_config import TOP_K_FINAL

logger = logging.getLogger(__name__)


class RagGenerator:
    """Orchestrates the RAG pipeline."""

    def __init__(
        self,
        retriever: Optional[RagRetriever] = None,
        answer_builder: Optional[RagAnswerBuilder] = None,
    ):
        self.retriever = retriever or RagRetriever()
        self.answer_builder = answer_builder or RagAnswerBuilder()

    def close(self):
        if self.retriever:
            self.retriever.close()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def answer(self, question: str, verbose: bool = False) -> dict:
        start = time.time()

        # 1. Retrieve
        try:
            chunks = self.retriever.search(question, top_k=TOP_K_FINAL)
        except Exception as exc:
            logger.exception("Error en retrieval: %s", exc)
            return {
                "question": question,
                "answer": (
                    "Ocurrió un error consultando los documentos disponibles. "
                    "Intenta nuevamente."
                ),
                "sources": [],
                "context_chunks": 0,
                "status": "error",
                "error": str(exc),
                "tokens": {"input": 0, "output": 0},
                "latency_ms": int((time.time() - start) * 1000),
            }

        if verbose:
            print(f"[RagGenerator] {len(chunks)} chunks recuperados")
            for i, c in enumerate(chunks[:5], 1):
                lex = "✅" if c.get("lexical_match") else "❌"
                print(f"  [{i}] lex={lex} score={c.get('score', 0):.3f} "
                      f"| {c.get('title', '?')[:50]} p.{c.get('page_number')}")

        # 2. Build answer
        try:
            answer, answer_tokens = self.answer_builder.build(question, chunks)
        except Exception as exc:
            logger.exception("Error generando respuesta: %s", exc)
            answer = "No pude generar una respuesta con los documentos disponibles."
            answer_tokens = {"input": 0, "output": 0}

        # 3. Sources
        sources = self._extract_sources(chunks)

        status = "ok" if chunks else "no_context"

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_chunks": len(chunks),
            "status": status,
            "tokens": answer_tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _extract_sources(chunks: list[dict]) -> list[dict]:
        seen = set()
        sources = []
        for c in chunks:
            key = (str(c.get("document_id")), c.get("page_number"), c.get("id"))
            if key in seen:
                continue
            seen.add(key)
            preview = (c.get("content") or "").strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            sources.append({
                "document_id": str(c.get("document_id")),
                "document_title": c.get("title"),
                "document_url": c.get("url"),
                "organization": c.get("organization"),
                "page_number": c.get("page_number"),
                "preview": preview,
                "score": c.get("score"),
                "lexical_match": c.get("lexical_match"),
                "rrf": c.get("rrf"),
            })
        return sources


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = RagGenerator()
    try:
        tests = [
            "¿Qué es el hacinamiento crítico?",
            "¿Cuáles son las dimensiones del IPM?",
            "¿Cómo se mide el analfabetismo?",
        ]
        for q in tests:
            print(f"\n{'=' * 70}")
            print(f"❓ {q}")
            r = gen.answer(q, verbose=True)
            print(f"  Status: {r['status']} | Chunks: {r['context_chunks']} | {r['latency_ms']} ms")
            print(f"  Tokens: in={r['tokens']['input']} out={r['tokens']['output']}")
            print(f"\n💬 {r['answer'][:500]}")
            if r["sources"]:
                print(f"\n📚 Fuentes ({len(r['sources'])}):")
                for s in r["sources"][:3]:
                    lex = "✅" if s.get("lexical_match") else "❌"
                    print(f"  {lex} {s['document_title'][:50]} p.{s['page_number']}")
    finally:
        gen.close()