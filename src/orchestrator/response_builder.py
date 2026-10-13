"""
Unified response builder.
"""
import time
import logging

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Builds a unified response dict from a module's raw output."""

    def build(self, question, intent, classification, raw, start):
        cls_tokens = classification.get("tokens", {"input": 0, "output": 0})
        if not isinstance(cls_tokens, dict):
            cls_tokens = {"input": 0, "output": 0}

        module_tokens = raw.get("tokens", {"input": 0, "output": 0})
        if not isinstance(module_tokens, dict):
            module_tokens = {"input": 0, "output": 0}

        # Normalize: some modules use "input"/"output", others "input_tokens"/"output_tokens"
        def _get(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        continue
            return 0

        input_tokens = _get(cls_tokens, "input", "input_tokens")
        output_tokens = _get(cls_tokens, "output", "output_tokens")
        module_input = _get(module_tokens, "input", "input_tokens")
        module_output = _get(module_tokens, "output", "output_tokens")

        total = {
            "input_tokens": input_tokens + module_input,
            "output_tokens": output_tokens + module_output,
        }
        total["total_tokens"] = total["input_tokens"] + total["output_tokens"]

        base = {
            "question": question,
            "intent": intent,
            "intent_reason": classification.get("reason", ""),
            "intent_method": classification.get("method", "llm"),
            "reformulated": bool(classification.get("reformulated", False)),
            "effective_question": classification.get("question", question),
            "answer": raw.get("answer") or "No pude generar una respuesta.",
            "sources": [],
            "sql": None,
            "simulation": None,
            "status": raw.get("status") or "ok",
            "tokens": total,
            "latency_ms": int((time.time() - start) * 1000),
            "raw": raw,
        }

        if intent == "data":
            base["sql"] = raw.get("sql")
        elif intent == "simulation":
            base["simulation"] = raw.get("result")
        elif intent in ("rag", "recommendation"):
            base["sources"] = raw.get("sources", []) or []

        return base

    def not_implemented(self, question, intent, classification, start, message):
        return {
            "question": question,
            "intent": intent,
            "intent_reason": classification.get("reason", ""),
            "intent_method": classification.get("method", "llm"),
            "reformulated": False,
            "effective_question": question,
            "answer": message,
            "sources": [],
            "sql": None,
            "simulation": None,
            "status": "not_implemented",
            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "latency_ms": int((time.time() - start) * 1000),
            "raw": {},
        }

    def error(self, question, intent, classification, start, error):
        return {
            "question": question,
            "intent": intent,
            "intent_reason": classification.get("reason", ""),
            "intent_method": classification.get("method", "llm"),
            "reformulated": False,
            "effective_question": question,
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