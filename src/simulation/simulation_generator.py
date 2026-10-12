"""
Simulation generator — orchestrator.

Pipeline:
    1. Classify the question → intent + params (LLM, ~1.5s)
    2. Validate the scenario (Python, ~0s)
    3. Load households (Colombia) or aggregate contributions (CEPAL)
    4. Run microsimulation (NumPy) or CEPAL linear simulation
    5. Format the answer (Python, ~0s)

Dispatch:
    - If entity_type == "country" and entity_name != "colombia"
      → CEPAL simulator (aggregate contributions).
    - Otherwise (department or national)
      → Microsimulator (household-level, DANE).

Total latency on CPU: ~2-6s (DANE) or ~1-2s (CEPAL).
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

from src.storage.data_db import DataDatabase
from src.simulation.simulation_intent_classifier import SimulationIntentClassifier
from src.simulation.scenario_validator import ScenarioValidator
from src.simulation.microsimulator import Microsimulator
from src.simulation.cepal_simulator import CepalSimulator
from src.simulation.simulation_answer_builder import SimulationAnswerBuilder
from src.simulation.ipm_weights import (
    DANE_DEPRIVATION_LABELS,
    CEPAL_DEPRIVATION_LABELS,
)

logger = logging.getLogger(__name__)


# ============================================================
# Mapping DANE ↔ CEPAL deprivations
# ============================================================
# Some DANE privations don't have a CEPAL equivalent, and vice versa.
# This mapping is best-effort: it maps the closest CEPAL privation.
DANE_TO_CEPAL_MAP = {
    "dep_critical_overcrowding":          "Overcrowded",
    "dep_illiteracy":                     "Illiteracy",
    "dep_low_educational_achievement":    "Educational achievement",
    "dep_school_nonattendance":           "Non-attendance or lateness",
    "dep_school_lag":                     "Non-attendance or lateness",
    "dep_informal_employment":            "Quality of employment",
    "dep_long_term_unemployment":         "Labor insertion",
    "dep_no_health_insurance":            "Health insurance",
    "dep_no_improved_water_source":       "Water",
    "dep_inadequate_excreta_disposal":    "Sanitization",
    "dep_inadequate_floor_material":      "Housing conditions",
    "dep_inadequate_wall_material":       "Housing conditions",
    # No direct equivalent:
    # dep_child_labor, dep_early_childhood_care_barriers,
    # dep_health_access_barriers → CEPAL doesn't have them separately.
}

# Países NO soportados (existen en CEPAL pero sin datos útiles)
UNSUPPORTED_CEPAL = {
    "argentina", "brasil", "brazil", "peru", "bolivia", "venezuela",
    "cuba", "nicaragua", "uruguay",
}


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
        self.cepal_simulator = CepalSimulator(db=self.db)
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
        classifier_tokens = classified.get("tokens", {"input": 0, "output": 0})

        if verbose:
            print(f"[SimulationGenerator] Intent: {intent} | params: {params}")

        if intent == "no_data":
            return self._unsupported_response(question, params, start, classifier_tokens)

        # 2. Dispatch by entity_type
        entity_type = params.get("entity_type", "national")
        entity_name = (params.get("entity_name") or "").lower().strip()

        # Check if it's an unsupported CEPAL country (Argentina, Peru, etc.)
        if entity_name in UNSUPPORTED_CEPAL:
            return self._no_data_response(
                question, intent, params, start, classifier_tokens,
                f"No hay datos de simulación disponibles para {entity_name.title()}. "
                f"Países soportados: Chile, México, Costa Rica, "
                f"República Dominicana, Ecuador, El Salvador, Guatemala, "
                f"Honduras, Panamá, Paraguay."
            )

        # CEPAL path (any country != Colombia)
        if entity_type == "country" and entity_name and entity_name != "colombia":
            return self._answer_cepal(
                question, intent, params, start, verbose, classifier_tokens
            )

        # DANE path (default: Colombia department or national)
        return self._answer_dane(
            question, intent, params, start, verbose, classifier_tokens
        )

    # ============================================================
    # DANE (Colombia) — microsimulation
    # ============================================================
    def _answer_dane(self, question, intent, params, start, verbose, classifier_tokens):
        # Validate
        ok, reason = self.validator.validate(intent, params)
        if not ok:
            return self._error_response(
                question, intent, params, start, classifier_tokens,
                f"No puedo simular ese escenario: {reason}"
            )

        # Load households
        try:
            households = self.simulator.load_households(
                entity_type=params.get("entity_type", "national"),
                entity_name=params.get("entity_name"),
                year=params.get("year"),
            )
        except Exception as exc:
            logger.exception("Error cargando hogares: %s", exc)
            return self._error_response(
                question, intent, params, start, classifier_tokens,
                "No pude cargar los datos para la simulación."
            )

        if households["matrix"].shape[0] == 0:
            return self._no_data_response(
                question, intent, params, start, classifier_tokens,
                "No encontré hogares en la base para esa ubicación o año."
            )

        if verbose:
            print(f"[SimulationGenerator] {households['matrix'].shape[0]} hogares cargados")

        # Simulate
        try:
            result = self._run_simulation(intent, params, households)
        except Exception as exc:
            logger.exception("Error en simulación: %s", exc)
            return self._error_response(
                question, intent, params, start, classifier_tokens,
                "No pude completar la simulación."
            )

        if verbose:
            print(f"[SimulationGenerator] Resultado: {result}")

        # Format
        answer = self.answer_builder.build(intent, params, result)

        return {
            "question": question,
            "intent": intent,
            "params": params,
            "answer": answer,
            "result": result,
            "status": "ok",
            "source": "DANE",
            "tokens": classifier_tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ============================================================
    # CEPAL — aggregate contribution simulation
    # ============================================================
    def _answer_cepal(self, question, intent, params, start, verbose, classifier_tokens):
        country = params["entity_name"]
        country_title = country.title()

        # Convert intent to changes
        changes = self._build_cepal_changes(intent, params)
        if not changes:
            return self._no_data_response(
                question, intent, params, start, classifier_tokens,
                f"No pude mapear la privación a un indicador de la CEPAL "
                f"para {country_title}. Es posible que la privación no tenga "
                f"un equivalente directo en el índice regional. "
                f"Privaciones CEPAL disponibles: hacinamiento, acceso a Internet, "
                f"calidad del empleo, inserción laboral, pensiones, analfabetismo, "
                f"acceso a agua, saneamiento, logro educativo, "
                f"condiciones de la vivienda, seguro de salud, inasistencia o rezago."
            )

        if verbose:
            print(f"[SimulationGenerator] CEPAL simulation for {country_title}: {changes}")

        # Run CEPAL simulation
        result = self.cepal_simulator.simulate(
            country=country,
            changes=changes,
            year=params.get("year"),
        )

        if not result["available"]:
            return self._no_data_response(
                question, intent, params, start, classifier_tokens,
                f"No tengo datos de simulación para {country_title}. "
                f"{result.get('reason', '')}"
            )

        # If no actual change occurred, tell the user
        if abs(result["delta_ipm_pp"]) < 0.0001:
            return self._no_data_response(
                question, intent, params, start, classifier_tokens,
                f"La privación indicada no está en los datos de {country_title} "
                f"o el cambio es insignificante."
            )

        # Format
        answer = self._format_cepal_answer(country_title, intent, params, result, changes)

        return {
            "question": question,
            "intent": intent,
            "params": params,
            "answer": answer,
            "result": result,
            "status": "ok",
            "source": "CEPAL",
            "tokens": classifier_tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ----------------------------------------------------------
    # CEPAL helpers
    # ----------------------------------------------------------
    def _build_cepal_changes(self, intent, params):
        """Convert the intent into changes format for the CEPAL simulator."""
        if intent in ("reduce_one_deprivation", "increase_one_deprivation"):
            dep = params.get("deprivation")
            if not dep:
                return []
            label = self._label_for_cepal(dep)
            if not label:
                return []
            return [{"deprivation": label, "delta_pct": params.get("delta_pct", -10)}]

        if intent == "eliminate_one_deprivation":
            dep = params.get("deprivation")
            if not dep:
                return []
            label = self._label_for_cepal(dep)
            if not label:
                return []
            return [{"deprivation": label, "delta_pct": -100}]

        if intent in ("reduce_multiple", "increase_multiple"):
            changes = []
            for c in params.get("changes", []):
                label = self._label_for_cepal(c["deprivation"])
                if label:
                    changes.append({
                        "deprivation": label,
                        "delta_pct": c["delta_pct"],
                    })
            return changes

        return []

    def _label_for_cepal(self, column_or_label):
        """
        Map a DANE column key or Spanish/English label to the CEPAL
        deprivation name.

        Handles four cases:
            1. Already an English CEPAL name → return as-is.
            2. DANE column key (dep_*) → mapped via DANE_TO_CEPAL_MAP.
            3. Spanish label → mapped via CEPAL_DEPRIVATION_LABELS.
            4. Unknown → return None.
        """
        if not column_or_label:
            return None

        val = str(column_or_label).strip()

        # 1. Already an English CEPAL name?
        for eng in CEPAL_DEPRIVATION_LABELS.keys():
            if val.lower() == eng.lower():
                return eng

        # 2. DANE column → CEPAL English name
        if val in DANE_TO_CEPAL_MAP:
            return DANE_TO_CEPAL_MAP[val]

        # 3. Spanish label → CEPAL English name
        for eng, esp in CEPAL_DEPRIVATION_LABELS.items():
            if val.lower() == esp.lower():
                return eng
        # Partial match on Spanish
        for eng, esp in CEPAL_DEPRIVATION_LABELS.items():
            if val.lower() in esp.lower() or esp.lower() in val.lower():
                return eng

        # 4. Unknown
        return None

    def _format_cepal_answer(self, country_title, intent, params, result, changes):
        """Format the CEPAL answer in Spanish."""
        method_note = ""
        if result["method"] == "Incidence (H) — aproximación":
            method_note = (
                " Nota: para este país la base de la CEPAL solo tiene la "
                "incidencia, no el IPM ajustado; el valor mostrado es la incidencia."
            )
        elif result["method"] == "Calculated (H × A)":
            method_note = " (IPM calculado como incidencia × intensidad)"

        delta = result["delta_ipm_pp"]
        direction = "reducción" if delta < 0 else "aumento"
        article = "una" if delta < 0 else "un"

        # Describe the changes
        change_parts = []
        for c in changes:
            dep = c["deprivation"]
            esp = CEPAL_DEPRIVATION_LABELS.get(dep, dep)
            delta_pct = c["delta_pct"]
            if delta_pct < 0:
                change_parts.append(f"reduce {abs(delta_pct):.0f}%")
            else:
                change_parts.append(f"aumenta {abs(delta_pct):.0f}%")

        if len(change_parts) == 1:
            verb_and_pct = change_parts[0]
            changes_txt = f"{esp} {verb_and_pct}"
        elif len(change_parts) == 2:
            changes_txt = " y ".join([f"{esp} {p}" for p in change_parts])
        else:
            changes_txt = " y ".join(
                [f"{esp} {p}" for p in change_parts]
            ) if len(change_parts) <= 2 else ", ".join(
                [f"{esp} {p}" for p in change_parts]
            )

        return (    
            f"Según datos de la CEPAL, si {changes_txt} en {country_title} "
            f"({result['year']}), el IPM pasaría de {result['ipm_before']:.2f}% "
            f"a {result['ipm_after']:.2f}%, {article} {direction} de "
            f"{abs(delta):.2f} puntos porcentuales.{method_note}"
        )   

    # ============================================================
    # Shared response helpers
    # ============================================================
    def _unsupported_response(self, question, params, start, tokens):
        return {
            "question": question,
            "intent": "no_data",
            "params": params,
            "answer": (
                "No pude interpretar esa pregunta como un escenario de "
                "simulación sobre pobreza multidimensional. Intenta "
                "reformularla, por ejemplo: '¿Qué pasaría si el hacinamiento "
                "bajara 10% en Huila?'"  # ← asegúrate de que haya espacio aquí
            ),
            "status": "no_data",
            "tokens": tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    def _no_data_response(self, question, intent, params, start, tokens, message):
        return {
            "question": question,
            "intent": intent,
            "params": params,
            "answer": message,
            "status": "no_data",
            "tokens": tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    def _error_response(self, question, intent, params, start, tokens, message):
        return {
            "question": question,
            "intent": intent,
            "params": params,
            "answer": message,
            "status": "error",
            "tokens": tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ============================================================
    # Simulation dispatcher (DANE only)
    # ============================================================
    def _run_simulation(self, intent, params, households):
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
            better = r_a if r_a["delta_ipm_pp"] < r_b["delta_ipm_pp"] else r_b
            out = dict(better)
            out["result_a"] = r_a
            out["result_b"] = r_b
            return out

        if intent in ("target_ipm", "target_ipm_increase"):
            return self._solve_target(intent, params, households)

        raise ValueError(f"Intent no manejado: {intent}")

    def _solve_target(self, intent, params, households):
        target = params["target_ipm"]
        dep = params["deprivation"]
        sign = -1 if intent == "target_ipm" else 1

        baseline = self.simulator.compute_aggregate(
            self.simulator.compute_ipm_scores(households["matrix"])
        )["ipm"]

        if (sign < 0 and baseline <= target) or (sign > 0 and baseline >= target):
            r = self.simulator.simulate(households, [{"deprivation": dep, "delta_pct": 0}])
            r["delta_required_pct"] = 0.0
            r["target_reached"] = True
            return r

        low, high = 0.0, 100.0
        best = None
        best_delta = None

        for _ in range(12):
            mid = (low + high) / 2
            delta = sign * mid
            r = self.simulator.simulate(households, [{"deprivation": dep, "delta_pct": delta}])
            reached = (r["ipm_after"] <= target) if sign < 0 else (r["ipm_after"] >= target)
            if reached:
                best, best_delta = r, delta
                high = mid
            else:
                low = mid

        if best is None:
            delta = sign * 100
            r = self.simulator.simulate(households, [{"deprivation": dep, "delta_pct": delta}])
            r["delta_required_pct"] = delta
            r["target_reached"] = False
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
        tests = [
            "¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?",
            "¿Qué pasaría si el hacinamiento baja 10% en México?",
            "Si la contribución de Internet sube 20% en Chile",
            "Si el trabajo informal sube 15% en Antioquia",
            "Si el hacinamiento baja 20% en Argentina",
            "Si el trabajo infantil baja 20% en México",  # ← sin equivalente CEPAL
        ]
        for q in tests:
            print(f"\n{'=' * 70}")
            print(f"❓ {q}")
            r = gen.answer(q, verbose=True)
            print(f"  Intent: {r.get('intent')}")
            print(f"  Status: {r.get('status')}")
            print(f"  Source: {r.get('source', 'n/a')}")
            print(f"  Answer: {r.get('answer', '')[:300]}")
    finally:
        gen.close()