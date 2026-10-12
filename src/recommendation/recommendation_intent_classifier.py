"""
Intent classifier for the Recommendation module.

Extracts: topic, source (DANE/CEPAL/BOTH/general), entity, dimension,
focus. Uses qwen2.5:1.5b for speed.
"""
import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ollama import Client
from dotenv import load_dotenv

from src.core.text_normalizer import normalize
from src.recommendation.recommendation_config import (
    INTENT_LLM_MODEL, INTENT_MAX_TOKENS, INTENT_TEMPERATURE,
)
from src.recommendation.recommendation_prompts import RECOMMENDATION_INTENT_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
# ============================================================
# Valid values
# ============================================================
VALID_SOURCES = {"dane", "cepal", "both", "general"}

# DANE dimensions (5)
DANE_DIMENSIONS = {
    "educacion": "Educación",
    "ninez y juventud": "Niñez y Juventud",
    "ninez": "Niñez y Juventud",
    "juventud": "Niñez y Juventud",
    "salud": "Salud",
    "trabajo": "Trabajo",
    "vivienda": "Vivienda",
}

# CEPAL dimensions (4)
CEPAL_DIMENSIONS = {
    "vivienda": "Vivienda",
    "salud": "Salud",
    "educacion": "Educación",
    "empleo y pensiones": "Empleo y pensiones",
    "empleo": "Empleo y pensiones",
    "pensiones": "Empleo y pensiones",
    "insercion laboral": "Empleo y pensiones",
    "calidad del empleo": "Empleo y pensiones",
}

# Common dimensions (apply to both)
COMMON_DIMENSIONS = {
    "educacion": "Educación",
    "salud": "Salud",
    "vivienda": "Vivienda",
    "general": "general",
}

# Countries (CEPAL)
CEPAL_COUNTRIES = {
    "chile", "colombia", "costa rica", "republica dominicana",
    "dominican republic", "ecuador", "el salvador", "guatemala",
    "honduras", "mexico", "panama", "paraguay",
    "brasil", "brazil", "argentina", "peru", "bolivia", "venezuela",
}

# Departments (DANE)
DANE_DEPARTMENTS = {
    "antioquia", "huila", "bogota", "choco", "narino", "valle del cauca",
    "atlantico", "bolivar", "santander", "cordoba", "sucre", "boyaca",
    "tolima", "cesar", "magdalena", "la guajira", "guajira", "cauca",
    "caldas", "risaralda", "quindio", "caqueta", "putumayo", "meta",
    "casanare", "vichada", "guainia", "guaviare", "vaupes", "amazonas",
    "arauca", "san andres", "providencia", "cundinamarca",
    "norte de santander",
}

# Regions/latam keywords
LATAM_KEYWORDS = {
    "latinoamerica", "america latina", "cepal", "region", "regional",
    "ipm-al", "al", "america",
}


class RecommendationIntentClassifier:
    """Extracts structured info from the user's recommendation request."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or INTENT_LLM_MODEL

    def classify(self, question: str) -> dict:
        """
        Return:
            {
                "topic": str,
                "source": "dane"|"cepal"|"both"|"general",
                "entity": str,
                "dimension": str,
                "focus": str,
                "raw": str,
                "method": "llm"|"fallback",
            }
        """
        if not question or not question.strip():
            return self._fallback(question, reason="empty_question")

        prompt = RECOMMENDATION_INTENT_PROMPT.format(question=question)

        try:
            client = Client(host=OLLAMA_HOST)
            response = client.generate(
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
        except Exception as exc:
            logger.warning("Error en LLM de intención: %s", exc)
            return self._fallback(question, reason=f"llm_error: {exc}")

        parsed = self._parse_response(raw)
        if parsed is None:
            return self._fallback(question, reason="invalid_json")

        parsed["raw"] = raw
        parsed["method"] = "llm"
        return parsed

    # ----------------------------------------------------------
    # Parsing
    # ----------------------------------------------------------
    @staticmethod
    def _clean_json(raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else ""

    def _parse_response(self, raw: str) -> Optional[dict]:
        cleaned = self._clean_json(raw)
        if not cleaned:
            return None
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

        topic = (data.get("topic") or "pobreza multidimensional").strip()
        source = self._normalize_source(data.get("source") or "general")
        entity = (data.get("entity") or "general").strip()
        dimension = self._match_dimension(
            data.get("dimension") or "general", source
        )
        focus = (data.get("focus") or topic).strip()

        # Auto-correct source if the entity clearly implies one
        if source == "general":
            inferred = self._infer_source_from_entity(entity)
            if inferred != "general":
                source = inferred

        return {
            "topic": topic,
            "source": source,
            "entity": entity,
            "dimension": dimension,
            "focus": focus,
        }

    @staticmethod
    def _normalize_source(source: str) -> str:
        key = (source or "general").strip().lower()
        if key in VALID_SOURCES:
            return key
        # Aliases
        if key in {"colombia", "dane"}:
            return "dane"
        if key in {"latam", "latinoamerica", "cepal", "region"}:
            return "cepal"
        if key in {"ambos", "comparar", "comparison"}:
            return "both"
        return "general"

    def _match_dimension(self, name: str, source: str) -> str:
        key = normalize(name).strip()
        if not key or key == "general":
            return "general"

        # Source-specific first
        if source == "dane":
            if key in DANE_DIMENSIONS:
                return DANE_DIMENSIONS[key]
        elif source == "cepal":
            if key in CEPAL_DIMENSIONS:
                return CEPAL_DIMENSIONS[key]
        elif source == "both":
            if key in COMMON_DIMENSIONS:
                return COMMON_DIMENSIONS[key]

        # Try common
        if key in COMMON_DIMENSIONS:
            return COMMON_DIMENSIONS[key]
        # Try DANE
        if key in DANE_DIMENSIONS:
            return DANE_DIMENSIONS[key]
        # Try CEPAL
        if key in CEPAL_DIMENSIONS:
            return CEPAL_DIMENSIONS[key]
        return "general"

    def _infer_source_from_entity(self, entity: str) -> str:
        key = normalize(entity).strip()
        if not key or key == "general":
            return "general"

        # Check for CEPAL countries
        for c in CEPAL_COUNTRIES:
            if c in key:
                return "cepal"

        # Check for DANE departments
        for d in DANE_DEPARTMENTS:
            if d in key:
                return "dane"

        # Colombia explicitly (could be DANE)
        if "colombia" in key:
            return "dane"

        # Latam keywords
        for kw in LATAM_KEYWORDS:
            if kw in key:
                return "cepal"

        return "general"

    # ----------------------------------------------------------
    # Fallback (keyword-based)
    # ----------------------------------------------------------
    def _fallback(self, question: str, reason: str = "") -> dict:
        q_norm = normalize(question or "")

        # Source detection
        source = "general"
        is_dane = any(d in q_norm for d in DANE_DEPARTMENTS) or \
                  "colombia" in q_norm
        is_cepal = any(c in q_norm for c in CEPAL_COUNTRIES) or \
                   any(kw in q_norm for kw in LATAM_KEYWORDS)
        if is_dane and is_cepal:
            source = "both"
        elif is_dane:
            source = "dane"
        elif is_cepal:
            source = "cepal"

        # Dimension detection
        dimension = "general"
        dim_kwargs = {
            "educacion": "Educación",
            "escuela": "Educación",
            "analfabet": "Educación",
            "logro educativo": "Educación",
            "salud": "Salud",
            "aseguramiento": "Salud",
            "trabajo": "Trabajo",
            "empleo": "Trabajo",
            "desempleo": "Trabajo",
            "informal": "Trabajo",
            "pension": "Empleo y pensiones",
            "vivienda": "Vivienda",
            "hacinamiento": "Vivienda",
            "agua": "Vivienda",
            "excreta": "Vivienda",
            "piso": "Vivienda",
            "pared": "Vivienda",
            "ninez": "Niñez y Juventud",
            "nino": "Niñez y Juventud",
            "infancia": "Niñez y Juventud",
            "juventud": "Niñez y Juventud",
            "trabajo infantil": "Niñez y Juventud",
        }
        for k, v in dim_kwargs.items():
            if k in q_norm:
                dimension = v
                break

        # Entity detection (pick the first match)
        entity = "general"
        for c in sorted(CEPAL_COUNTRIES):
            if c in q_norm:
                entity = c.title()
                break
        if entity == "general":
            for d in sorted(DANE_DEPARTMENTS):
                if d in q_norm:
                    entity = d.title()
                    break
        if entity == "general" and "colombia" in q_norm:
            entity = "Colombia"

        return {
            "topic": question.strip() or "pobreza multidimensional",
            "source": source,
            "entity": entity,
            "dimension": dimension,
            "focus": question.strip() or "reducir pobreza multidimensional",
            "raw": "",
            "method": "fallback",
            "fallback_reason": reason,
        }


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf = RecommendationIntentClassifier()
    tests = [
        # DANE examples
        "¿Qué proyectos puedo implementar para reducir el hacinamiento en Antioquia?",
        "¿Cómo puedo bajar el desempleo de larga duración en Colombia?",
        "¿Qué puedo hacer para reducir el analfabetismo?",
        "Propón proyectos para la primera infancia en zonas rurales",
        # CEPAL examples
        "¿Qué proyectos para reducir la pobreza multidimensional en México?",
        "Sugiere políticas para mejorar el acceso a pensiones en Chile",
        "¿Cómo mejorar la calidad del empleo en la región?",
        "¿Qué intervenciones reducen la pobreza en Ecuador?",
        # Both
        "Compara las políticas de vivienda entre Colombia y México",
        # General
        "Dame ideas de políticas para combatir la pobreza multidimensional",
    ]
    for q in tests:
        r = clf.classify(q)
        print(f"\n❓ {q}")
        print(f"   → source={r['source']!r} entity={r['entity']!r} "
            f"dimension={r['dimension']!r} ({r['method']})")