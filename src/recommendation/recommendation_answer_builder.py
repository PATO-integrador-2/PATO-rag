"""
Answer builder for the Recommendation module.

Takes the user's question and context from the RAG retriever, and uses
qwen2.5:3b to draft concrete project proposals in a compact markdown
table.

Sanitization:
    The LLM sometimes generates more rows than requested, or duplicates
    the table. `_sanitize_table` truncates to at most `max_rows` data
    rows and keeps only the first table.

Returns a tuple (answer_text, tokens_dict) where tokens_dict has
"input" and "output" keys with the real token counts from Ollama.
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

from ollama import Client
from dotenv import load_dotenv

from src.recommendation.recommendation_config import (
    ANSWER_LLM_MODEL, ANSWER_MAX_TOKENS, ANSWER_TEMPERATURE, CONTEXT_MAX_CHARS,
)
from src.recommendation.recommendation_prompts import RECOMMENDATION_ANSWER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
# ============================================================
# Text sanitization
# ============================================================
def _fix_mojibake(text: str) -> str:
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


# ============================================================
# Table sanitizer
# ============================================================
def _sanitize_table(answer: str, max_rows: int = 3) -> str:
    """
    Truncate a markdown table to at most `max_rows` data rows.
    Keeps only the first table, drops extra rows and trailing prose.
    """
    lines = answer.split("\n")
    out: list[str] = []
    data_rows = 0
    in_table = False
    table_finished = False
    separator_seen = False

    for line in lines:
        stripped = line.strip()

        if table_finished:
            break

        if not in_table and stripped.startswith("|") and "proyecto" in stripped.lower():
            in_table = True
            out.append(line)
            continue

        if not in_table:
            continue

        if (not separator_seen
                and stripped.startswith("|")
                and set(stripped.replace(" ", "")) <= set("|-:")):
            separator_seen = True
            out.append(line)
            continue

        if stripped.startswith("|"):
            if data_rows < max_rows:
                out.append(line)
                data_rows += 1
            else:
                table_finished = True
                break
            continue

        if stripped == "" or not stripped.startswith("|"):
            if data_rows > 0:
                table_finished = True
                break
            continue

    return "\n".join(out).strip()


def _user_asked_more(question: str) -> bool:
    """Detect if the user explicitly asked for more than 3 projects."""
    q = question.lower()
    patterns = [
        r"\b5\s+proyectos?\b",
        r"\bm[aá]s\s+proyectos?\b",
        r"\bm[aá]s\s+intervenciones?\b",
        r"\bm[aá]s\s+ideas?\b",
        r"\bampl[ií]a\b",
        r"\bampliar\b",
    ]
    return any(re.search(p, q) for p in patterns)


# ============================================================
# Answer builder
# ============================================================
class RecommendationAnswerBuilder:
    """Builds the Spanish answer with concrete project proposals."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or ANSWER_LLM_MODEL

    def build(
        self,
        question: str,
        chunks: list[dict],
        source: str = "general",
        dimension: str = "general",
    ) -> tuple[str, dict]:
        """
        Produce the final answer.

        Returns:
            (answer_text, tokens_dict)
        """
        context = self._build_context(chunks)
        if not context:
            context = "(Sin contexto adicional de los documentos.)"

        source_hint = self._source_hint(source)
        dim_hint = (
            f"Dimensión del IPM más relevante: {dimension}."
            if dimension and dimension != "general" else ""
        )

        prompt = RECOMMENDATION_ANSWER_PROMPT.format(
            context=context,
            question=question,
        )

        extra_hints = "\n".join(filter(None, [source_hint, dim_hint]))
        if extra_hints:
            prompt = prompt + f"\n\nPISTAS ADICIONALES:\n{extra_hints}"

        max_rows = 5 if _user_asked_more(question) else 3

        try:
            client = Client(host=OLLAMA_HOST)
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": ANSWER_TEMPERATURE,
                    "top_p": 0.9,
                    "num_predict": 350,
                    "stop": ["\n\n\n", "==="],
                },
            )
            answer = response.get("response", "").strip()
            tokens = {
                "input": response.get("prompt_eval_count", 0),
                "output": response.get("eval_count", 0),
            }
            if not answer:
                return self._fallback(chunks), tokens
            sanitized = _sanitize_table(answer, max_rows=max_rows)
            return sanitized, tokens
        except Exception as exc:
            logger.error("Error generando recomendaciones: %s", exc)
            return self._fallback(chunks), {"input": 0, "output": 0}

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _source_hint(source: str) -> str:
        if source == "dane":
            return (
                "La pregunta se refiere a Colombia. Usa las 5 dimensiones del "
                "IPM-Colombia (Educación, Niñez y Juventud, Salud, Trabajo, "
                "Vivienda)."
            )
        if source == "cepal":
            return (
                "La pregunta se refiere a Latinoamérica (IPM-AL de la CEPAL). "
                "Usa las 4 dimensiones del IPM-AL (Vivienda, Salud, Educación, "
                "Empleo y pensiones)."
            )
        if source == "both":
            return (
                "La pregunta compara Colombia con otros países latinoamericanos. "
                "Considera ambas perspectivas (DANE y CEPAL)."
            )
        return (
            "La pregunta no especifica ubicación. Considera intervenciones "
            "aplicables tanto a Colombia como a Latinoamérica."
        )

    @staticmethod
    def _build_context(chunks: list[dict], max_chars: int = CONTEXT_MAX_CHARS) -> str:
        if not chunks:
            return ""

        parts = []
        total = 0
        MAX_PER_CHUNK = 500

        for i, c in enumerate(chunks, start=1):
            content = _clean_text(c.get("content", ""))
            if not content:
                continue
            if len(content) > MAX_PER_CHUNK:
                content = content[:MAX_PER_CHUNK] + "..."

            header_bits = []
            if c.get("document_title"):
                header_bits.append(c["document_title"])
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

    @staticmethod
    def _fallback(chunks: list[dict]) -> str:
        return (
            "No pude generar recomendaciones en este momento. "
            "Intenta reformular la pregunta o intenta nuevamente en unos segundos."
        )


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the sanitizer standalone
    sample = """| Proyecto | Descripción | Dimensión IPM |
|---|---|---|
| A | Foo | Vivienda |
| B | Bar | Trabajo |
| C | Baz | Salud |
| D | Qux | Educación |

Trailing prose.
| Proyecto | Ignored |
|---|---|"""

    print("=== Test del sanitizador ===")
    print(_sanitize_table(sample, max_rows=3))

    builder = RecommendationAnswerBuilder()
    fake_chunks = [
        {
            "content": (
                "El hacinamiento crítico se define como la privación en el hogar "
                "cuando hay tres o más personas por cuarto en zonas urbanas."
            ),
            "document_title": "Algoritmo_IPM.pdf",
            "page_number": 15,
        },
    ]
    answer, tokens = builder.build(
        "¿Qué proyectos para reducir el hacinamiento?",
        fake_chunks,
        source="dane",
        dimension="Vivienda",
    )
    print("\n=== Test del builder ===")
    print(answer)
    print(f"\nTokens: {tokens}")