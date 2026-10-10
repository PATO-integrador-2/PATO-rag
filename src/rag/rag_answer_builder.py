"""
RAG answer builder.

Takes the top-K retrieved chunks (with the new hybrid format) and
asks qwen2.5:3b to produce a grounded Spanish answer.

Handles mojibake and returns (answer, tokens) tuple.
"""
import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ollama
from dotenv import load_dotenv

from src.rag.rag_config import (
    ANSWER_LLM_MODEL, ANSWER_MAX_TOKENS, ANSWER_TEMPERATURE, CONTEXT_MAX_CHARS,
)
from src.rag.rag_prompts import RAG_ANSWER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)


def _fix_mojibake(text: str) -> str:
    """Fix common UTF-8-as-Latin-1 mojibake."""
    if not text:
        return text
    try:
        if "Ã" in text or "Â" in text:
            return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def _clean_text(text: str) -> str:
    text = _fix_mojibake(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class RagAnswerBuilder:
    """Builds a Spanish answer from retrieved chunks."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or ANSWER_LLM_MODEL

    def build(self, question: str, chunks: list[dict]) -> tuple[str, dict]:
        """
        Produce the final answer.

        Returns:
            (answer_text, tokens_dict)
        """
        if not chunks:
            return (
                "No encontré información relevante en los documentos disponibles "
                "para responder esa pregunta.",
                {"input": 0, "output": 0},
            )

        context = self._build_context(chunks)
        if not context:
            return (
                "No encontré información relevante en los documentos disponibles "
                "para responder esa pregunta.",
                {"input": 0, "output": 0},
            )

        prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": ANSWER_TEMPERATURE,
                    "top_p": 0.9,
                    "num_predict": ANSWER_MAX_TOKENS,
                },
            )
            answer = response.get("response", "").strip()
            tokens = {
                "input": response.get("prompt_eval_count", 0),
                "output": response.get("eval_count", 0),
            }
            if not answer:
                return self._fallback_summary(chunks), tokens
            return answer, tokens
        except Exception as exc:
            logger.error("Error generando respuesta RAG: %s", exc)
            return self._fallback_summary(chunks), {"input": 0, "output": 0}

    # ----------------------------------------------------------
    # Context building
    # ----------------------------------------------------------
    @staticmethod
    def _build_context(chunks: list[dict], max_chars: int = CONTEXT_MAX_CHARS) -> str:
        """Concatenate chunks with document metadata headers."""
        parts = []
        total = 0

        for i, c in enumerate(chunks, start=1):
            content = _clean_text(c.get("content", ""))
            if not content:
                continue

            header_bits = []
            if c.get("title"):
                header_bits.append(c["title"])
            if c.get("organization"):
                header_bits.append(c["organization"])
            if c.get("page_number") is not None:
                header_bits.append(f"p. {c['page_number']}")
            header = " — ".join(header_bits) if header_bits else f"fragmento {i}"

            block = f"[Fragmento {i} · {header}]\n{content}"

            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(block[:remaining] + "\n...[truncado]")
                break
            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------
    @staticmethod
    def _fallback_summary(chunks: list[dict]) -> str:
        top = _clean_text(chunks[0].get("content", "")).strip()
        if len(top) > 500:
            top = top[:500] + "..."
        return (
            "No pude generar una respuesta elaborada, pero encontré este "
            "fragmento relevante en los documentos:\n\n" + top
        )


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = RagAnswerBuilder()

    sample_mojibake = "Se realiza un filtro para las personas mayores de 15 aÃ±os."
    print("Before:", sample_mojibake)
    print("After: ", _clean_text(sample_mojibake))
    print()

    fake_chunks = [
        {
            "content": (
                "Un hogar se considera en hacinamiento crítico cuando el número "
                "de personas por cuarto es mayor o igual a tres en zonas urbanas, "
                "y mayor a tres en zonas rurales."
            ),
            "title": "bol-PMultidimensional-2025.pdf",
            "organization": "DANE",
            "page_number": 5,
        },
    ]
    answer, tokens = builder.build("¿Qué es el hacinamiento crítico?", fake_chunks)
    print(answer)
    print(f"\nTokens: {tokens}")