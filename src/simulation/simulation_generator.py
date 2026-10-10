"""
Simulation generator — orchestrator.

Pipeline:
    1. Classify the question → intent + params (LLM, ~1.5s)
    2. Validate the scenario (Python, ~0s)
    3. Load households from clean_data_db (~0.3s)
    4. Run microsimulation (Python, ~0.2s)
    5. Format the answer (Python, ~0s)

Total latency on CPU: ~2-3s.
"""
import os
import sys
import time
import logging
from pathlib import Path
from tracemalloc import start
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.data_db import DataDatabase
from src.simulation.simulation_intent_classifier import SimulationIntentClassifier
from src.simulation.scenario_validator import ScenarioValidator
from src.simulation.microsimulator import Microsimulator
from src.simulation.simulation_answer_builder import SimulationAnswerBuilder

logger = logging.getLogger(__name__)


class SimulationGenerator:
    """Orchestrates the simulation pipeline."""

    def __init__(
        self,
        db: Optional[DataDatabase] = None,
        classifier: Optional[SimulationIntentClassifier] = None,
    ):
        self.db = db or DataDatabase()
        self.classifier = classifier or SimulationIntentClassifier()
        self.validator = ScenarioValidator()
        self.simulator = Microsimulator(db=self.db)
        self.answer_builder = SimulationAnswerBuilder()

    def close(self):
        if self.db:
            self.db.close()

    # ============================================================
    # Public API
    # ============================================================
    def answer(self, question: str, verbose: bool = False) -> dict:
        start = time.time()

        # 1. Classify
        classified = self.classifier.classify(question)
        intent = classified["intent"]
        params = classified["params"]

        if verbose:
            print(f"[SimulationGenerator] Intent: {intent} | params: {params}")

        if intent == "no_data":
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "answer": (
                    "No pude interpretar esa pregunta como un escenario de "
                    "simulación sobre pobreza multidimensional. Intenta "
                    "reformularla, por ejemplo: '¿Qué pasaría si el hacinamiento "
                    "bajara 10% en Huila?'"
                ),
                "status": "no_data",
                "latency_ms": int((time.time() - start) * 1000),
            }

        # 2. Validate
        ok, reason = self.validator.validate(intent, params)
        if not ok:
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "answer": f"No puedo simular ese escenario: {reason}",
                "status": "validation_error",
                "latency_ms": int((time.time() - start) * 1000),
            }

        # 3. Load households
        try:
            households = self.simulator.load_households(
                entity_type=params.get("entity_type", "national"),
                entity_name=params.get("entity_name"),
                year=params.get("year"),
            )
        except Exception as exc:
            logger.exception("Error cargando hogares: %s", exc)
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "answer": "No pude cargar los datos para la simulación.",
                "status": "error",
                "error": str(exc),
                "latency_ms": int((time.time() - start) * 1000),
            }

        if households["matrix"].shape[0] == 0:
            return {
            "question": question,
            "intent": intent,
            "params": params,
            "answer": (
                "No encontré hogares en la base para esa ubicación o año. "
                "Verifica el departamento o el año."
            ),
            "status": "no_data",
            "latency_ms": int((time.time() - start) * 1000),
        }

        if verbose:
            print(f"[SimulationGenerator] {len(households)} hogares cargados")

        # 4. Simulate
        try:
            result = self._run_simulation(intent, params, households)
        except Exception as exc:
            logger.exception("Error en simulación: %s", exc)
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "answer": "No pude completar la simulación.",
                "status": "error",
                "error": str(exc),
                "latency_ms": int((time.time() - start) * 1000),
            }

        if verbose:
            print(f"[SimulationGenerator] Resultado: {result}")

        # 5. Format
        answer = self.answer_builder.build(intent, params, result)

        return {
            "question": question,
            "intent": intent,
            "params": params,
            "answer": answer,
            "result": result,
            "status": "ok",
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ============================================================
    # Simulation dispatcher
    # ============================================================
    def _run_simulation(
        self, intent: str, params: dict, households: list[dict]
    ) -> dict:
        if intent in ("reduce_one_deprivation", "increase_one_deprivation"):
            return self.simulator.simulate(households, [{
                "deprivation": params["deprivation"],
                "delta_pct": params["delta_pct"],
            }])

        if intent == "eliminate_one_deprivation":
            return self.simulator.simulate(households, [{
                "deprivation": params["deprivation"],
                "delta_pct": -100,
            }])

        if intent in ("reduce_multiple", "increase_multiple"):
            return self.simulator.simulate(households, params["changes"])

        if intent in ("dimension_impact", "worsen_all_in_dimension"):
            from src.simulation.ipm_weights import DANE_DIMENSIONS
            dim = params["dimension"]
            cols = DANE_DIMENSIONS.get(dim, [])
            changes = [
                {"deprivation": c, "delta_pct": params["delta_pct"]}
                for c in cols
            ]
            return self.simulator.simulate(households, changes)

        if intent in ("compare_improvements", "compare_worsenings"):
            delta = params["delta_pct"]
            r_a = self.simulator.simulate(households, [{
                "deprivation": params["deprivation_a"],
                "delta_pct": delta,
            }])
            r_b = self.simulator.simulate(households, [{
                "deprivation": params["deprivation_b"],
                "delta_pct": delta,
            }])
            # Return the "better" one as main result, with both for the formatter
            better = r_a if r_a["delta_ipm_pp"] < r_b["delta_ipm_pp"] else r_b
            out = dict(better)
            out["result_a"] = r_a
            out["result_b"] = r_b
            return out

        if intent in ("target_ipm", "target_ipm_increase"):
            return self._solve_target(intent, params, households)

        raise ValueError(f"Intent no manejado: {intent}")

    def _solve_target(
        self, intent: str, params: dict, households: dict
    ) -> dict:
        target = params["target_ipm"]
        dep = params["deprivation"]
        sign = -1 if intent == "target_ipm" else 1

        baseline = self.simulator.compute_aggregate(
            self.simulator.compute_ipm_scores(households["matrix"])
        )["ipm"]

        # Already met
        if (sign < 0 and baseline <= target) or (sign > 0 and baseline >= target):
            r = self.simulator.simulate(households, [{
                "deprivation": dep, "delta_pct": 0,
            }])
            r["delta_required_pct"] = 0.0
            r["target_reached"] = True
            return r

        low, high = 0.0, 100.0
        best = None
        best_delta = None

        for _ in range(12):
            mid = (low + high) / 2
            delta = sign * mid
            r = self.simulator.simulate(households, [{
                "deprivation": dep, "delta_pct": delta,
            }])
            reached = (r["ipm_after"] <= target) if sign < 0 else (r["ipm_after"] >= target)
            if reached:
                best, best_delta = r, delta
                high = mid
            else:
                low = mid

        if best is None:
            delta = sign * 100
            r = self.simulator.simulate(households, [{
                "deprivation": dep, "delta_pct": delta,
            }])
            r["delta_required_pct"] = delta
            r["target_reached"] = False
            r["reason_unreachable"] = (
                f"Incluso con un cambio del 100% en {dep}, el IPM no alcanza "
                f"la meta de {target}%."
            )
            return r

        best["delta_required_pct"] = best_delta
        best["target_reached"] = True
        return best


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = SimulationGenerator()
    try:
        q = "¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?"
        print(f"\n❓ {q}")
        result = gen.answer(q, verbose=True)
        print(f"\n💬 {result['answer']}")
        print(f"⏱️  {result['latency_ms']} ms")
    finally:
        gen.close()