"""
Simulation intent classifier.

Classifies a Spanish question into one of 12 simulation intents.
Uses qwen2.5:1.5b (fast, reliable for short JSON).

On any failure, falls back to `no_data`.
"""
import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Optional
from src.simulation.simulation_intent_postprocessor import postprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ollama
from dotenv import load_dotenv

from src.core.text_normalizer import normalize
from src.simulation.simulation_intent_promts import SIMULATION_INTENT_PROMPT
from src.simulation.ipm_weights import (
    label_to_column,
    DANE_DEPRIVATION_LABELS,
    CEPAL_DEPRIVATION_LABELS,
)

load_dotenv()
logger = logging.getLogger(__name__)


INTENT_LLM_MODEL = os.getenv(
    "OLLAMA_INTENT_MODEL",
    os.getenv("OLLAMA_SQL_MODEL", "qwen2.5:7b"),
)
INTENT_MAX_TOKENS = int(os.getenv("INTENT_MAX_TOKENS", "220"))
INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", "0.0"))


# ============================================================
# Valid intents and their required params
# ============================================================
VALID_INTENTS = {
    # Improvement
    "reduce_one_deprivation":    {"required": ["deprivation"],          "optional": ["delta_pct", "entity_type", "entity_name", "year"]},
    "eliminate_one_deprivation": {"required": ["deprivation"],          "optional": ["entity_type", "entity_name", "year"]},
    "reduce_multiple":           {"required": ["changes"],              "optional": ["entity_type", "entity_name", "year"]},
    "target_ipm":                {"required": ["deprivation", "target_ipm"], "optional": ["entity_type", "entity_name", "year"]},
    "compare_improvements":      {"required": ["deprivation_a", "deprivation_b", "delta_pct"], "optional": ["entity_type", "entity_name"]},
    "dimension_impact":          {"required": ["dimension", "delta_pct"], "optional": ["entity_type", "entity_name", "year"]},
    # Worsening
    "increase_one_deprivation":  {"required": ["deprivation"],          "optional": ["delta_pct", "entity_type", "entity_name", "year"]},
    "worsen_all_in_dimension":   {"required": ["dimension", "delta_pct"], "optional": ["entity_type", "entity_name"]},
    "increase_multiple":         {"required": ["changes"],              "optional": ["entity_type", "entity_name", "year"]},
    "target_ipm_increase":       {"required": ["deprivation", "target_ipm"], "optional": ["entity_type", "entity_name", "year"]},
    "compare_worsenings":        {"required": ["deprivation_a", "deprivation_b", "delta_pct"], "optional": ["entity_type", "entity_name"]},
    # Fallback
    "no_data":                   {"required": [], "optional": []},
}

VALID_ENTITY_TYPES = {"national", "department", "region"}


# ============================================================
# Classifier
# ============================================================
class SimulationIntentClassifier:
    """Classifies a Spanish question into a simulation intent."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or INTENT_LLM_MODEL

    def classify(self, question: str) -> dict:
        if not question or not question.strip():
            return {"intent": "no_data", "params": {}, "raw": "", "tokens": {"input": 0, "output": 0}}

        prompt = SIMULATION_INTENT_PROMPT.format(question=question)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": INTENT_TEMPERATURE,
                    "top_p": 0.9,
                    "num_predict": INTENT_MAX_TOKENS,
                    "stop": ["\n\n", "###"],
                },  
            )
            raw = response.get("response", "").strip()
            tokens = {
                "input": response.get("prompt_eval_count", 0),
                "output": response.get("eval_count", 0),
            }
        except Exception as exc:
            logger.error("Error llamando a Ollama (%s): %s", self.model, exc)
            return {"intent": "no_data", "params": {}, "raw": "", "tokens": {"input": 0, "output": 0}}

        parsed = self._parse_response(raw)
        parsed["raw"] = raw
        parsed["tokens"] = tokens
        return parsed

    # ----------------------------------------------------------
    # Parsing
    # ----------------------------------------------------------
    def _parse_response(self, raw: str) -> dict:
        cleaned = self._clean_json(raw)
        if not cleaned:
            logger.warning("Respuesta no parseable: %r", raw[:200])
            return {"intent": "no_data", "params": {}}

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            repaired = self._repair_json(cleaned)
            try:
                data = json.loads(repaired)
                logger.info("JSON repaired successfully.")
            except json.JSONDecodeError:
                logger.warning("JSON inválido del clasificador: %s — %r", exc, cleaned[:200])
                return {"intent": "no_data", "params": {}}

        intent = (data.get("intent") or "").strip().lower().replace(" ", "_")
        if intent not in VALID_INTENTS:
            logger.warning("Intención desconocida: %r", intent)
            return {"intent": "no_data", "params": {}}

        params = data.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        sanitized = self._sanitize_params(intent, params)
        if sanitized is None:
            return {"intent": "no_data", "params": {}}

        return {"intent": intent, "params": sanitized}

    @staticmethod
    def _clean_json(raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else ""

    # ----------------------------------------------------------
    # Param sanitization
    # ----------------------------------------------------------
    def _sanitize_params(self, intent: str, params: dict) -> Optional[dict]:
        spec = VALID_INTENTS[intent]
        allowed = set(spec["required"]) | set(spec["optional"])
        clean: dict = {}

        for key, value in params.items():
            if key not in allowed or value is None:
                continue

            if key == "delta_pct":
                try:
                    clean["delta_pct"] = float(value)
                except (TypeError, ValueError):
                    continue

            elif key == "target_ipm":
                try:
                    clean["target_ipm"] = float(value)
                except (TypeError, ValueError):
                    continue

            elif key == "year":
                try:
                    clean["year"] = int(value)
                except (TypeError, ValueError):
                    continue

            elif key in ("deprivation", "deprivation_a", "deprivation_b"):
                col = label_to_column("DANE", str(value)) or label_to_column("CEPAL", str(value))
                if col:
                    clean[key] = col

            elif key == "dimension":
                dim = self._match_dimension(str(value))
                if dim:
                    clean[key] = dim

            elif key == "entity_type":
                et = str(value).strip().lower()
                if et in VALID_ENTITY_TYPES:
                    clean["entity_type"] = et

            elif key == "entity_name":
                clean["entity_name"] = normalize(str(value)).strip()

            elif key == "changes":
                changes = self._sanitize_changes(value)
                if changes:
                    clean["changes"] = changes

            else:
                clean[key] = value

        # Default entity_type
        if "entity_type" not in clean:
            clean["entity_type"] = "national"

        # Validation of required params
        for req in spec["required"]:
            if req not in clean:
                logger.warning(
                    "Falta '%s' para intent '%s'. Cayendo a no_data.", req, intent
                )
                return None

        # delta_pct sign sanity checks
        if intent in ("reduce_one_deprivation", "reduce_multiple", "dimension_impact",
                      "compare_improvements"):
            # These are improvements: force negative delta
            if "delta_pct" in clean and clean["delta_pct"] > 0:
                clean["delta_pct"] = -abs(clean["delta_pct"])
        if intent in ("increase_one_deprivation", "increase_multiple",
                      "worsen_all_in_dimension", "compare_worsenings"):
            if "delta_pct" in clean and clean["delta_pct"] < 0:
                clean["delta_pct"] = abs(clean["delta_pct"])

        # Default delta_pct if missing
        if intent in ("reduce_one_deprivation", "increase_one_deprivation"):
            clean.setdefault("delta_pct", -10 if intent.startswith("reduce") else 10)

        return clean

    def _sanitize_changes(self, raw) -> list[dict]:
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            col = label_to_column("DANE", str(item.get("deprivation", ""))) or \
                  label_to_column("CEPAL", str(item.get("deprivation", "")))
            if not col:
                continue
            try:
                delta = float(item.get("delta_pct"))
            except (TypeError, ValueError):
                continue
            out.append({"deprivation": col, "delta_pct": delta})
        return out

    @staticmethod
    def _match_dimension(name: str) -> Optional[str]:
        """
        Match a Spanish dimension name to a canonical key in DANE_DIMENSIONS.
        Accepts variations with/without accents: 'Educacion', 'Educación', etc.
        Returns the exact key used in DANE_DIMENSIONS (e.g. 'Educación').
        """
        from src.simulation.ipm_weights import DANE_DIMENSIONS

        key = normalize(name).strip()
        if not key:
            return None

        # normalize() strips accents: "educación" → "educacion"
        # We compare against the normalized canonical names.
        aliases = {
            "educacion": "Educación",
            "ninez y juventud": "Niñez y Juventud",
            "ninez": "Niñez y Juventud",
            "juventud": "Niñez y Juventud",
            "salud": "Salud",
            "trabajo": "Trabajo",
            "vivienda": "Vivienda",
        }
        if key in aliases:
            return aliases[key]

        # Substring match as a fallback
        for alias, canonical in aliases.items():
            if alias in key or key in alias:
                return canonical

        return None


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf = SimulationIntentClassifier()
    tests = [
        "¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?",
        "Si se elimina el analfabetismo en Antioquia",
        "Si el trabajo informal baja 20% y la deserción escolar 15%",
        "¿Cuánto hay que reducir el trabajo informal para que el IPM llegue a 20%?",
        "¿Qué reduce más: hacinamiento o analfabetismo, ambos al 10%?",
        "Si la dimensión Educación mejora 20%",
        "¿Qué pasa si el trabajo informal sube 15% en Antioquia?",
        "Si empeora toda la dimensión Vivienda 20%",
        "Si el analfabetismo sube 10% y el desempleo 5%",
        "¿Cuánto debe subir el trabajo informal para que el IPM llegue a 35% en La Guajira?",
        "¿Qué aumenta más: hacinamiento o trabajo informal al 10%?",
        "¿Cuál es el IPM de Huila?",  # no_data
    ]
    for q in tests:
        r = clf.classify(q)
        print(f"\n❓ {q}")
        print(f"   → {json.dumps({'intent': r['intent'], 'params': r['params']}, ensure_ascii=False)}")