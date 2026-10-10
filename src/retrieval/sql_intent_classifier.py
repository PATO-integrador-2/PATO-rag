"""
SQL intent classifier.

Uses an LLM to classify a Spanish question into a structured intent.
The LLM only produces a small JSON — no SQL. This makes classification
fast and robust.

Fallback: if the LLM returns invalid JSON, an unknown intent, or an
intent with missing required params, we return
`{"intent": "no_data", "params": {}}`.

Special cases:
    - `dimension_top_in_domain` defaults its `domain` to "nacional" if
      the LLM did not extract one. This is because the underlying view
      (vw_ipm_contribution_dimension) does not have department-level
      granularity — it only has domains (Nacional, Caribe, etc.).
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

import ollama
from dotenv import load_dotenv

from src.core.text_normalizer import normalize
from src.retrieval.sql_intent_prompts import SQL_INTENT_CLASSIFICATION_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================
INTENT_LLM_MODEL = os.getenv(
    "OLLAMA_INTENT_MODEL",
    os.getenv("OLLAMA_SQL_MODEL", "qwen2.5:1.5b"),
)
INTENT_MAX_TOKENS = int(os.getenv("INTENT_MAX_TOKENS", "150"))
INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", "0.0"))


# ============================================================
# Intent schema
# ============================================================
VALID_INTENTS = {
    "ipm_department_year":        {"required": ["department"], "optional": ["year"]},
    "ipm_national_latest":        {"required": [],             "optional": []},
    "ipm_national_year":          {"required": ["year"],       "optional": []},
    "dimension_top_in_domain":    {"required": ["domain"],     "optional": ["year"]},
    "deprivation_top":            {"required": [],             "optional": ["year"]},
    "sex_comparison_national":    {"required": [],             "optional": ["year"]},
    "household_count_department": {"required": ["department"], "optional": ["year"]},
    "ipm_country_year":           {"required": ["country"],    "optional": ["year"]},
    "country_top_poverty":        {"required": [],             "optional": ["year"]},
    "ipm_country_age_group":      {"required": ["country"],    "optional": ["year"]},
    "country_vs_latam":           {"required": ["country"],    "optional": []},
    "no_data":                    {"required": [],             "optional": []},
}

# Canonical department names (accent-free, as stored in DB)
VALID_DEPARTMENTS = {
    "amazonas", "antioquia", "arauca", "archipielago de san andres",
    "atlantico", "bogota", "bolivar", "boyaca", "caldas", "caqueta",
    "casanare", "cauca", "cesar", "choco", "cordoba", "cundinamarca",
    "guainia", "guaviare", "huila", "la guajira", "magdalena", "meta",
    "narino", "norte de santander", "putumayo", "quindio", "risaralda",
    "santander", "sucre", "tolima", "valle del cauca", "vaupes", "vichada",
}

# Canonical country names (as stored in DB)
VALID_COUNTRIES = {
    "chile", "colombia", "costa rica", "dominican republic",
    "ecuador", "el salvador", "guatemala", "honduras", "mexico",
    "panama", "paraguay",
}

# Canonical domains in vw_ipm_incidence / vw_ipm_contribution_dimension
VALID_DOMAINS = {
    "nacional", "caribe", "central", "oriental", "pacifica",
    "orinoquia - amazonia", "bogota",
}

# Known typos / variants the LLM might produce
INTENT_ALIASES = {
    "ipm_nacional_year": "ipm_national_year",
    "ipm_nacional_latest": "ipm_national_latest",
    "country_ms_latam": "country_vs_latam",
    "pais_top_poverty": "country_top_poverty",
    "dimension_top": "dimension_top_in_domain",
    "deprivation_most_common": "deprivation_top",
    "sex_comparison": "sex_comparison_national",
    "household_count": "household_count_department",
    "household_count_department_year": "household_count_department",
}

# Intent-specific default params (applied when the LLM omits them)
DEFAULT_PARAMS = {
    "dimension_top_in_domain": {"domain": "nacional"},
}


# ============================================================
# Classifier
# ============================================================
class SqlIntentClassifier:
    """Classifies a Spanish question into a structured SQL intent."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or INTENT_LLM_MODEL

    def classify(self, question: str) -> dict:
        """
        Return:
            {
                "intent": str,
                "params": dict,
                "raw": str,
                "tokens": {"input": int, "output": int},
            }
        """
        if not question or not question.strip():
            return {"intent": "no_data", "params": {}, "raw": "", "tokens": {"input": 0, "output": 0}}

        prompt = SQL_INTENT_CLASSIFICATION_PROMPT.format(question=question)

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
    # Parsing / normalization
    # ----------------------------------------------------------
    def _parse_response(self, raw: str) -> dict:
        """Parse the LLM JSON output, sanitize it, validate it."""
        cleaned = self._clean_json(raw)
        if not cleaned:
            logger.warning("Respuesta del clasificador no parseable: %r", raw[:200])
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



        intent = self._normalize_intent(data.get("intent") or "")
        params = data.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if intent not in VALID_INTENTS:
            logger.warning("Intención desconocida: %r", intent)
            return {"intent": "no_data", "params": {}}

        # `_sanitize_params` returns None if required params are missing.
        sanitized = self._sanitize_params(intent, params)
        if sanitized is None:
            # Fall back to no_data instead of an error status.
            return {"intent": "no_data", "params": {}}

        return {"intent": intent, "params": sanitized}

    @staticmethod
    def _normalize_intent(intent: str) -> str:
        """Map known typos/variants to canonical intent names."""
        key = intent.strip().lower().replace(" ", "_")
        if key in VALID_INTENTS:
            return key
        return INTENT_ALIASES.get(key, key)

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences and any leading/trailing prose."""
        text = raw.strip()
        # Remove ```json ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        # Extract the first {...} block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else ""

    # ----------------------------------------------------------
    # Param sanitization
    # ----------------------------------------------------------
    def _sanitize_params(self, intent: str, params: dict) -> Optional[dict]:
        """
        Normalize names, cast years, drop unknown keys.

        Returns:
            - dict with clean params if required params are present.
            - dict with defaults applied + still-valid params.
            - None if a required param is missing after sanitization
              (caller should switch intent to `no_data`).
        """
        spec = VALID_INTENTS[intent]
        allowed_keys = set(spec["required"]) | set(spec["optional"])
        clean: dict = {}

        for key, value in params.items():
            if key not in allowed_keys:
                continue
            if value is None:
                continue

            if key == "year":
                try:
                    clean["year"] = int(value)
                except (TypeError, ValueError):
                    continue

            elif key == "department":
                name = self._match_department(normalize(str(value)).strip())
                if name:
                    clean["department"] = name

            elif key == "country":
                name = self._match_country(normalize(str(value)).strip())
                if name:
                    clean["country"] = name

            elif key == "domain":
                name = self._match_domain(normalize(str(value)).strip())
                if name:
                    clean["domain"] = name

            else:
                clean[key] = value

        # Apply intent-specific defaults for missing params
        defaults = DEFAULT_PARAMS.get(intent, {})
        for key, default_value in defaults.items():
            if key not in clean:
                clean[key] = default_value

        # If a required param is still missing → signal failure
        for req in spec["required"]:
            if req not in clean:
                logger.warning(
                    "Falta parámetro requerido '%s' para intent '%s'. Cayendo a no_data.",
                    req, intent,
                )
                return None

        return clean

    # ----------------------------------------------------------
    # Matching helpers
    # ----------------------------------------------------------
    @staticmethod
    def _match_department(name: str) -> Optional[str]:
        """Find the closest department by prefix/substring."""
        if not name:
            return None
        if name in VALID_DEPARTMENTS:
            return name
        for dep in VALID_DEPARTMENTS:
            if dep in name or name in dep:
                return dep
        return None

    @staticmethod
    def _match_country(name: str) -> Optional[str]:
        """Find the closest country by prefix/substring."""
        if not name:
            return None
        aliases = {
            "republica dominicana": "dominican republic",
            "mexico": "mexico",
            "panama": "panama",
        }
        name = aliases.get(name, name)
        if name in VALID_COUNTRIES:
            return name
        for country in VALID_COUNTRIES:
            if country in name or name in country:
                return country
        return None

    @staticmethod
    def _match_domain(name: str) -> Optional[str]:
        """Find the closest domain."""
        if not name:
            return None
        if name in VALID_DOMAINS:
            return name
        aliases = {
            "bogota d.c.": "bogota",
            "bogota dc": "bogota",
            "region caribe": "caribe",
            "region central": "central",
            "region oriental": "oriental",
            "region pacifica": "pacifica",
            "region pacifico": "pacifica",
            "orinoquia": "orinoquia - amazonia",
            "amazonia": "orinoquia - amazonia",
        }
        if name in aliases:
            return aliases[name]
        for dom in VALID_DOMAINS:
            if dom in name or name in dom:
                return dom
        return None

    @staticmethod
    def _repair_json(raw: str) -> str:
        """
        Attempt to repair common JSON syntax errors produced by small LLMs.

        Examples:
            '{"year": 2018"}'      → '{"year": 2018}'
            "{'key': 'value'}"     → '{"key": "value"}'
            '{"a": 1,}'            → '{"a": 1}'
        """
        # Remove trailing commas before closing brace
        raw = re.sub(r",\s*}", "}", raw)
        # Fix unmatched quotes after numbers/booleans: `2018"` → `2018`
        raw = re.sub(r'(\d+)"(\s*[},])', r"\1\2", raw)
        raw = re.sub(r'(true|false|null)"(\s*[},])', r"\1\2", raw)
        # Replace single quotes with double quotes for keys/values
        # (best-effort; only if all quotes are single)
        if raw.count("'") >= 2 and '"' not in raw:
            raw = raw.replace("'", '"')
        return raw

# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf = SqlIntentClassifier()
    test_questions = [
        "¿Cuál fue el IPM de Huila en 2018?",
        "¿Cuál es el IPM nacional más reciente?",
        "¿Qué dimensión contribuye más al IPM en Antioquia?",
        "¿Cuál es la privación más común en Colombia?",
        "Diferencia de IPM entre hogares con jefe hombre y mujer",
        "¿Cuántos hogares hay en Antioquia en 2022?",
        "¿Cuál es el IPM de Argentina en 2020?",
        "¿Qué país tiene mayor pobreza multidimensional?",
        "IPM por grupo de edad en México",
        "Compara el IPM de Colombia con el promedio de Latinoamérica",
    ]
    for q in test_questions:
        result = clf.classify(q)
        print(f"\n? {q}")
        print(f"   → {json.dumps({'intent': result['intent'], 'params': result['params']}, ensure_ascii=False)}")