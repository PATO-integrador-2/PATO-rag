"""
Unified response builder.

Normalizes the outputs of the different modules into a single shape
so the API layer has one contract to serve.

Token handling:
    Each module returns its own "tokens" dict with the real Ollama
    counts (prompt_eval_count / eval_count). This builder sums:
        1. The orchestrator's classifier tokens.
        2. The module's own tokens (if the module exposes them).
    Modules that don't use LLM for the answer (SQL, Simulation via
    templates) contribute 0 output tokens, but the internal classifier
    tokens may still be included.
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Builds a unified response dict from a module's raw output."""

    def build(self, question, intent, classification, raw, start):
        # --- Tokens: orchestrator classifier ---
        cls_tokens = classification.get("tokens", {"input": 0, "output": 0})
        if not isinstance(cls_tokens, dict):
            cls_tokens = {"input": 0, "output": 0}

        # --- Tokens: module (may be missing or nested) ---
        module_tokens = raw.get("tokens", {"input": 0, "output": 0})
        if not isinstance(module_tokens, dict):
            module_tokens = {"input": 0, "output": 0}

        try:
            input_tokens = int(cls_tokens.get("input", 0)) + int(module_tokens.get("input", 0))
        except (TypeError, ValueError):
            input_tokens = 0
        try:
            output_tokens = int(cls_tokens.get("output", 0)) + int(module_tokens.get("output", 0))
        except (TypeError, ValueError):
            output_tokens = 0

        total_tokens = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        base = {
            "question": question,
            "intent": intent,
            "intent_reason": classification.get("reason", ""),
            "intent_method": classification.get("method", "llm"),
            "answer": raw.get("answer") or "No pude generar una respuesta.",
            "sources": [],
            "sql": None,
            "simulation": None,
            "status": raw.get("status") or "ok",
            "tokens": total_tokens,
            "latency_ms": int((time.time() - start) * 1000),
            "raw": raw,
        }

        # Module-specific fields
        if intent == "data":
            base["sql"] = raw.get("sql")
        elif intent == "simulation":
            base["simulation"] = raw.get("result")
        elif intent == "rag":
            base["sources"] = raw.get("sources", []) or []
        elif intent == "recommendation":
            base["sources"] = raw.get("sources", []) or []

        return base

    def not_implemented(
        self,
        question: str,
        intent: str,
        classification: dict,
        start: float,
        message: str,
    ) -> dict:
        return {
            "question": question,
            "intent": intent,
            "intent_reason": classification.get("reason", ""),
            "intent_method": classification.get("method", "llm"),
            "answer": message,
            "sources": [],
            "sql": None,
            "simulation": None,
            "status": "not_implemented",
            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "latency_ms": int((time.time() - start) * 1000),
            "raw": {},
        }

    def error(
        self,
        question: str,
        intent: str,
        classification: dict,
        start: float,
        error: str,
    ) -> dict:
        return {
            "question": question,
            "intent": intent,
            "intent_reason": classification.get("reason", ""),
            "intent_method": classification.get("method", "llm"),
            "answer": "Ocurrió un error procesando tu pregunta.",
            "sources": [],
            "sql": None,
            "simulation": None,
            "status": "error",
            "error": error,
            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "latency_ms": int((time.time() - start) * 1000),
            "raw": {},
        }