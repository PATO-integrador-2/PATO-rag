"""
Central router / orchestrator.

The single entry point for the entire generative AI. All external
consumers (FastAPI, CLI, tests) call `Orchestrator.ask()` and never
the modules directly.

Flow:
    1. Classify the question → data | simulation | rag | recommendation.
    2. Dispatch to the corresponding module.
    3. Return a unified response.

Modules:
    - data           → SqlGenerator
    - simulation     → SimulationGenerator
    - rag            → RagGenerator (placeholder for now)
    - recommendation → RecommendationGenerator (placeholder for now)
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

from src.orchestrator.intent_classifier import OrchestratorIntentClassifier
from src.orchestrator.response_builder import ResponseBuilder


logger = logging.getLogger(__name__)


# ============================================================
# Lazy module loaders
# ============================================================
# We avoid importing the heavy modules at the top so the orchestrator
# can be imported even if some modules are not yet implemented.

def _load_sql_generator():
    from src.retrieval.sql_generator import SqlGenerator
    return SqlGenerator


def _load_simulation_generator():
    from src.simulation.simulation_generator import SimulationGenerator
    return SimulationGenerator


def _load_rag_generator():
    from src.rag.rag_generator import RagGenerator
    return RagGenerator


def _load_recommendation_generator():
    from src.recommendation.recommendation_generator import RecommendationGenerator
    return RecommendationGenerator


# ============================================================
# Orchestrator
# ============================================================
class Orchestrator:
    """
    Central router that decides which module handles a question.

    Example:
        orch = Orchestrator()
        result = orch.ask("¿Cuál fue el IPM de Huila en 2018?")
        print(result["answer"])
        orch.close()
    """

    def __init__(
        self,
        classifier: Optional[OrchestratorIntentClassifier] = None,
        response_builder: Optional[ResponseBuilder] = None,
        use_llm_classifier: bool = True,
    ):
        self.classifier = classifier or OrchestratorIntentClassifier(
            use_llm=use_llm_classifier
        )
        self.response_builder = response_builder or ResponseBuilder()

        # Lazy-loaded module instances
        self._sql_gen = None
        self._sim_gen = None
        self._rag_gen = None
        self._rec_gen = None

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------
    def close(self):
        for gen in (self._sql_gen, self._sim_gen, self._rag_gen, self._rec_gen):
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception as exc:
                    logger.warning("Error cerrando módulo: %s", exc)

    # ----------------------------------------------------------
    # Lazy getters
    # ----------------------------------------------------------
    def _get_sql(self):
        if self._sql_gen is None:
            cls = _load_sql_generator()
            self._sql_gen = cls()
        return self._sql_gen

    def _get_simulation(self):
        if self._sim_gen is None:
            cls = _load_simulation_generator()
            self._sim_gen = cls()
        return self._sim_gen

    def _get_rag(self):
        if self._rag_gen is None:
            cls = _load_rag_generator()
            if cls is None:
                return None
            self._rag_gen = cls()
        return self._rag_gen

    def _get_recommendation(self):
        if self._rec_gen is None:
            cls = _load_recommendation_generator()
            if cls is None:
                return None
            self._rec_gen = cls()
        return self._rec_gen

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Classify the question, dispatch it, and return a unified response.

        Response shape:
            {
                "question": str,
                "intent": "data" | "simulation" | "rag" | "recommendation",
                "intent_reason": str,
                "intent_method": "llm" | "keyword",
                "answer": str,
                "sources": list,          # RAG only
                "sql": str | dict | None, # data only
                "simulation": dict | None,# simulation only
                "status": "ok" | "no_data" | "not_implemented" | "error",
                "latency_ms": int,
                "raw": dict,              # raw module response
            }
        """
        start = time.time()

        # 1. Classify
        classification = self.classifier.classify(question)
        intent = classification["intent"]

        if verbose:
            print(f"[Orchestrator] Intent: {intent} ({classification['method']})")
            print(f"[Orchestrator] Reason: {classification.get('reason', '')}")

        # 2. Dispatch
        try:
            if intent == "data":
                raw = self._get_sql().answer(question, verbose=verbose)
            elif intent == "simulation":
                raw = self._get_simulation().answer(question, verbose=verbose)
            elif intent == "rag":
                rag = self._get_rag()
                if rag is None:
                    return self.response_builder.not_implemented(
                        question, intent, classification, start,
                        message=(
                            "El módulo de consultas conceptuales (RAG) aún no "
                            "está conectado en este entorno. Intenta con una "
                            "pregunta de datos o de simulación."
                        ),
                    )
                raw = rag.answer(question, verbose=verbose)
            elif intent == "recommendation":
                rec = self._get_recommendation()
                if rec is None:
                    return self.response_builder.not_implemented(
                        question, intent, classification, start,
                        message=(
                            "El módulo de recomendaciones aún no está disponible. "
                            "Pronto podré sugerirte proyectos e intervenciones."
                        ),
                    )
                raw = rec.answer(question, verbose=verbose)
            else:
                return self.response_builder.error(
                    question, intent, classification, start,
                    error=f"Intent desconocido: {intent}",
                )
        except Exception as exc:
            logger.exception("Error despachando intent %s: %s", intent, exc)
            return self.response_builder.error(
                question, intent, classification, start,
                error=f"Error ejecutando el módulo: {exc}",
            )

        # 3. Normalize into unified response
        return self.response_builder.build(
            question=question,
            intent=intent,
            classification=classification,
            raw=raw,
            start=start,
        )


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator()
    try:
        tests = [
            "¿Cuál fue el IPM de Huila en 2018?",
            "¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?",
            "¿Qué es el hacinamiento crítico?",
            "¿Qué proyectos puedo implementar para reducir el IPM en Antioquia?",
        ]
        for q in tests:
            print(f"\n{'=' * 70}")
            print(f"❓ {q}")
            r = orch.ask(q, verbose=False)
            print(f"  Intent: {r['intent']} ({r['intent_method']})")
            print(f"  Status: {r['status']}")
            print(f"  Latency: {r['latency_ms']} ms")
            print(f"  Answer: {r['answer'][:250]}")
    finally:
        orch.close()