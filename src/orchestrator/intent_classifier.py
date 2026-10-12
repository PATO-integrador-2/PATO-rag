"""
Top-level intent classifier.

Classifies a Spanish question into one of 4 categories:
    data | simulation | rag | recommendation

Uses qwen2.5:1.5b for speed and reliability on short JSON outputs.
Falls back to a keyword-based classifier if the LLM fails.
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
from src.orchestrator.intent_prompts import ORCHESTRATOR_INTENT_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
INTENT_LLM_MODEL = os.getenv(
    "OLLAMA_ORCHESTRATOR_INTENT_MODEL",
    os.getenv("OLLAMA_INTENT_MODEL", "qwen2.5:1.5b"),
)
INTENT_MAX_TOKENS = int(os.getenv("ORCHESTRATOR_INTENT_MAX_TOKENS", "120"))
INTENT_TEMPERATURE = float(os.getenv("ORCHESTRATOR_INTENT_TEMPERATURE", "0.0"))


VALID_INTENTS = {"data", "simulation", "rag", "recommendation"}

INTENT_ALIASES = {
    "sql": "data",
    "query": "data",
    "consulta": "data",
    "consulta_datos": "data",
    "sim": "simulation",
    "simulacion": "simulation",
    "simulación": "simulation",
    "whatif": "simulation",
    "what_if": "simulation",
    "definition": "rag",
    "definicion": "rag",
    "definición": "rag",
    "docs": "rag",
    "documentos": "rag",
    "recomendacion": "recommendation",
    "recomendación": "recommendation",
    "recommend": "recommendation",
    "proposal": "recommendation",
    "propuesta": "recommendation",
    "projects": "recommendation",
}


# ============================================================
# Keyword-based fallback
# ============================================================
_SIMULATION_CUES = [
    r"\bque pasaria si\b",
    r"\bsi .* (bajara|subiera|aumentara|redujera|mejorara|empeorara|creciera|cayera)\b",
    r"\bcual seria el impacto\b",
    r"\bcuanto (bajaria|subiria|aumentaria|reduciria)\b",
    r"\bcuanto (hay que|debe|deberia) (reducir|aumentar|subir|bajar)\b",
    r"\bque reduce mas\b",
    r"\bque aumenta mas\b",
    r"\bsimula\b",
    r"\bsimular\b",
    r"\bescenario\b",
]

_RECOMMENDATION_CUES = [
    r"\bque proyectos\b",
    r"\bque se puede implementar\b",
    r"\bcomo puedo reducir\b",
    r"\bcomo puedo mejorar\b",
    r"\bcomo reducir\b",
    r"\bcomo mejorar\b",
    r"\bsugiere\b",
    r"\bsugiereme\b",
    r"\bpropone\b",
    r"\bpropon\b",
    r"\bpropuesta de proyecto\b",
    r"\bintervencion(es)?\b",
    r"\bdame ideas\b",
    r"\bque puedo hacer\b",
]

_RAG_CUES = [
    r"\bque es\b",
    r"\bque significa\b",
    r"\bcomo se mide\b",
    r"\bcomo se calcula\b",
    r"\bcuales son los criterios\b",
    r"\bdefinicion de\b",
    r"\bexplicame\b",
    r"\bexplica el concepto\b",
    r"\bmetodologia\b",
    r"\bcuales son las dimensiones\b",
]

_DATA_CUES = [
    r"\bcual (fue|es|era) el\b",
    r"\bcuanto (fue|es|era|son)\b",
    r"\bcuantos\b",
    r"\bcual es el (ipm|promedio|valor|total)\b",
    r"\bdame el dato\b",
    r"\branking\b",
    r"\bel (departamento|pais|país) con (mayor|menor)\b",
    r"\bdiferencia (de|entre)\b",
    r"\bipm de\b",
    r"\bipm en\b",
    r"\bipm por\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _keyword_fallback(question: str) -> str:
    q_norm = normalize(question)
    if _matches_any(q_norm, _SIMULATION_CUES):
        return "simulation"
    if _matches_any(q_norm, _RECOMMENDATION_CUES):
        return "recommendation"
    if _matches_any(q_norm, _RAG_CUES):
        return "rag"
    if _matches_any(q_norm, _DATA_CUES):
        return "data"
    return "rag"


# ============================================================
# Classifier
# ============================================================
class OrchestratorIntentClassifier:
    """Classifies a question into one of the 4 top-level intents."""

    def __init__(self, model: Optional[str] = None, use_llm: bool = True):
        self.model = model or INTENT_LLM_MODEL
        self.use_llm = use_llm

    def classify(self, question: str) -> dict:
        """
        Return:
            {
                "intent": str,
                "reason": str,
                "method": "llm"|"keyword",
                "raw": str,
                "tokens": {"input": int, "output": int},
            }
        """
        if not question or not question.strip():
            return {
                "intent": "rag",
                "reason": "empty question",
                "method": "keyword",
                "raw": "",
                "tokens": {"input": 0, "output": 0},
            }

        if self.use_llm:
            try:
                llm_result = self._classify_with_llm(question)
                if llm_result:
                    return llm_result
            except Exception as exc:
                logger.warning("LLM classifier failed: %s. Falling back to keywords.", exc)

        intent = _keyword_fallback(question)
        return {
            "intent": intent,
            "reason": "keyword_fallback",
            "method": "keyword",
            "raw": "",
            "tokens": {"input": 0, "output": 0},
        }

    # ----------------------------------------------------------
    # LLM path
    # ----------------------------------------------------------
    def _classify_with_llm(self, question: str) -> Optional[dict]:
        prompt = ORCHESTRATOR_INTENT_PROMPT.format(question=question)

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
        if not raw:
            return None

        cleaned = self._clean_json(raw)
        if not cleaned:
            return None

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        intent = self._normalize_intent(data.get("intent") or "")
        if intent not in VALID_INTENTS:
            return None

        return {
            "intent": intent,
            "reason": (data.get("reason") or "").strip()[:120],
            "method": "llm",
            "raw": raw,
            "tokens": {
                "input": response.get("prompt_eval_count", 0),
                "output": response.get("eval_count", 0),
            },
        }

    @staticmethod
    def _clean_json(raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else ""

    @staticmethod
    def _normalize_intent(intent: str) -> str:
        key = intent.strip().lower().replace(" ", "_")
        if key in VALID_INTENTS:
            return key
        return INTENT_ALIASES.get(key, key)


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)

    clf = OrchestratorIntentClassifier()
    tests = [
        ("¿Cuál fue el IPM de Huila en 2018?", "data"),
        ("¿Cuántos hogares hay en Antioquia en 2022?", "data"),
        ("¿Qué país tiene mayor pobreza multidimensional?", "data"),
        ("Diferencia de IPM entre hogares con jefe hombre y mujer", "data"),
        ("¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?", "simulation"),
        ("Si se elimina el analfabetismo en Antioquia", "simulation"),
        ("¿Cuánto hay que reducir el trabajo informal para que el IPM llegue a 20%?", "simulation"),
        ("¿Qué reduce más el IPM: hacinamiento o analfabetismo?", "simulation"),
        ("¿Qué es el hacinamiento crítico?", "rag"),
        ("¿Cómo se mide el analfabetismo?", "rag"),
        ("Explícame la metodología de Alkire-Foster", "rag"),
        ("¿Cuáles son las dimensiones del IPM?", "rag"),
        ("¿Qué proyectos puedo implementar para reducir el IPM en Antioquia?", "recommendation"),
        ("Sugiere intervenciones para el hacinamiento crítico", "recommendation"),
        ("¿Cómo puedo mejorar la educación en zonas rurales?", "recommendation"),
    ]

    correct = 0
    for q, expected in tests:
        t0 = time.time()
        r = clf.classify(q)
        dt = (time.time() - t0) * 1000
        got = r["intent"]
        ok = "OK " if got == expected else "BAD"
        if got == expected:
            correct += 1
        print(f"  [{ok}] {got:<15} | esperado: {expected:<15} | {dt:6.0f}ms | {q[:50]}")

    print(f"\n  Clasificación: {correct}/{len(tests)}")