"""
Central router: receives a question and sends it to the right module.

This is the single entry point for the generative AI. All external
consumers (API, CLI) call `RagRouter.ask()` and never the modules directly.

The RAG module (definition) is no longer implemented here. Backend now
handles it separately. For now, this router returns a controlled placeholder
when the intent is 'definition'.
"""
import time
from typing import Optional
import sys
from pathlib import Path
from src.retrieval.sql_intent_classifier import classify_intent
from src.retrieval.sql_generator import SqlGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    


class RagRouter:
    """
    High-level router that decides what to do with each question.

    Intents:
        - data           → SqlGenerator (implemented here)
        - simulation     → placeholder (future module)
        - recommendation → placeholder (future module)
        - definition     → placeholder (backend handles RAG separately)
    """

    def __init__(self, sql_generator: Optional[SqlGenerator] = None):
        self.sql_generator = sql_generator or SqlGenerator()

    def close(self):
        self.sql_generator.close()

    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Classify the question and route it to the right module.

        Returns a unified dict:
            {
                "question": str,
                "intent": "definition" | "data" | "simulation" | "recommendation",
                "answer": str,
                "sources": list,          # reserved for external RAG
                "sql": str | dict | None, # only for 'data'
                "latency_ms": int,
                "raw": dict,              # raw result from the module
            }
        """
        start = time.time()

        intent = classify_intent(question)
        if verbose:
            print(f"[Router] Intención detectada: {intent}")

        if intent == "data":
            raw = self.sql_generator.answer(question, verbose=verbose)
            return {
                "question": question,
                "intent": intent,
                "answer": raw["answer"],
                "sources": [],
                "sql": raw.get("sql"),
                "latency_ms": int((time.time() - start) * 1000),
                "raw": raw,
            }

        if intent == "definition":
            # RAG now lives outside this module. Return a controlled placeholder.
            return {
                "question": question,
                "intent": intent,
                "answer": (
                    "La búsqueda de definiciones se maneja en otro servicio. "
                    "Por favor consulta el endpoint de documentos."
                ),
                "sources": [],
                "sql": None,
                "latency_ms": int((time.time() - start) * 1000),
                "raw": {"handled_externally": True},
            }

        # simulation / recommendation: not yet implemented
        if intent == "simulation":
            answer = (
                "El módulo de simulación aún no está disponible. "
                "Pronto podré ayudarte con escenarios hipotéticos."
            )
        else:  # recommendation
            answer = (
                "El módulo de generación de propuestas aún no está disponible. "
                "Pronto podré sugerirte proyectos e intervenciones."
            )

        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "sources": [],
            "sql": None,
            "latency_ms": int((time.time() - start) * 1000),
            "raw": {},
        }