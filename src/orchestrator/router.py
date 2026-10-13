"""
Central router / orchestrator with memory support.

The single entry point for the entire generative AI.
Modules:
    - data           → SqlGenerator
    - simulation     → SimulationGenerator
    - rag            → RagGenerator
    - recommendation → RecommendationGenerator
    - general        → GeneralGenerator (guarded fallback)
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
from src.general.general_generator import GeneralGenerator

load_dotenv = None
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


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


class Orchestrator:
    """Central router that decides which module handles a question."""

    def __init__(
        self,
        classifier: Optional[OrchestratorIntentClassifier] = None,
        response_builder: Optional[ResponseBuilder] = None,
        general_generator: Optional[GeneralGenerator] = None,
        use_llm_classifier: bool = True,
    ):
        self.classifier = classifier or OrchestratorIntentClassifier(
            use_llm=use_llm_classifier
        )
        self.response_builder = response_builder or ResponseBuilder()
        self.general_generator = general_generator or GeneralGenerator()

        self._sql_gen = None
        self._sim_gen = None
        self._rag_gen = None
        self._rec_gen = None

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
            self._sql_gen = _load_sql_generator()()
        return self._sql_gen

    def _get_simulation(self):
        if self._sim_gen is None:
            self._sim_gen = _load_simulation_generator()()
        return self._sim_gen

    def _get_rag(self):
        if self._rag_gen is None:
            self._rag_gen = _load_rag_generator()()
        return self._rag_gen

    def _get_recommendation(self):
        if self._rec_gen is None:
            self._rec_gen = _load_recommendation_generator()()
        return self._rec_gen

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def ask(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        verbose: bool = False,
    ) -> dict:
        start = time.time()

        # 1. Classify (with optional history for reformulation)
        classification = self.classifier.classify(question, history=history)
        intent = classification["intent"]
        effective_question = classification.get("question", question)

        if verbose:
            print(f"[Orchestrator] Intent: {intent} ({classification['method']})")
            print(f"[Orchestrator] Effective question: {effective_question}")
            if classification.get("reformulated"):
                print(f"[Orchestrator] Original: {question}")

        # 2. Dispatch
        try:
            if intent == "data":
                raw = self._get_sql().answer(effective_question, verbose=verbose)
            elif intent == "simulation":
                raw = self._get_simulation().answer(effective_question, verbose=verbose)
            elif intent == "rag":
                raw = self._get_rag().answer(effective_question, verbose=verbose)
            elif intent == "recommendation":
                raw = self._get_recommendation().answer(effective_question, verbose=verbose)
            elif intent == "general":
                raw = self.general_generator.answer(
                    effective_question, history=history, verbose=verbose,
                )
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

        # 3. Normalize
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
            "¿Qué pasaría si el hacinamiento baja 10%?",
            "¿Qué es el hacinamiento crítico?",
            "¿Qué proyectos para reducir el hacinamiento?",
            "Cuéntame la historia del IPM",
        ]
        for q in tests:
            print(f"\n{'=' * 70}")
            print(f"❓ {q}")
            r = orch.ask(q)
            print(f"  Intent: {r['intent']}")
            print(f"  Status: {r['status']}")
            print(f"  Latency: {r['latency_ms']} ms")
            print(f"  Answer: {r['answer'][:200]}")
    finally:
        orch.close()