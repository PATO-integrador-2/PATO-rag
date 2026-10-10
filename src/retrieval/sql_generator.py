"""
Text-to-SQL module — refactored to LLM-as-classifier architecture.

Pipeline:
    1. classify(question)       → {intent, params}    (LLM, ~2-5s)
    2. build_sql(intent, params) → parameterized SQL   (Python, 0s)
    3. execute(sql, sql_params)  → rows                 (DB, <1s)
    4. format(intent, rows)      → Spanish answer       (Python, 0s)

Total latency on CPU: ~3-6s per question (vs 30-500s before).

The LLM never writes SQL, never sees columns, never multiplies by 100.
All of that is deterministic Python.
"""
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.data_db import DataDatabase
from src.retrieval.sql_intent_classifier import SqlIntentClassifier
from src.retrieval.sql_builder import SqlBuilder
from src.retrieval.answer_builder import AnswerBuilder

logger = logging.getLogger(__name__)


SQL_TIMEOUT_SECONDS = int(os.getenv("SQL_TIMEOUT_SECONDS", "15"))


class SqlGenerator:
    """
    Orchestrates the intent → SQL → execute → format pipeline.

    Kept name-compatible with the old SqlGenerator so existing callers
    (router, tests) don't break.
    """

    def __init__(
        self,
        db: Optional[DataDatabase] = None,
        classifier: Optional[SqlIntentClassifier] = None,
    ):
        self.db = db or DataDatabase()
        self.classifier = classifier or SqlIntentClassifier()
        self.builder = SqlBuilder()
        self.answer_builder = AnswerBuilder()

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
            print(f"[SqlGenerator] Intención: {intent} | params: {params}")

        # 2. Handle no_data
        if intent == "no_data":
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "sql": None,
                "answer": (
                    "No tengo información para responder esa pregunta con los "
                    "datos disponibles de pobreza multidimensional. Verifica el "
                    "país, departamento o año indicado."
                ),
                "rows_count": 0,
                "status": "no_data",
                "latency_ms": int((time.time() - start) * 1000),
            }

        # 3. Build SQL
        built = self.builder.build(intent, params)
        if not built["ok"]:
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "sql": None,
                "answer": "No pude construir la consulta para esa pregunta.",
                "rows_count": 0,
                "status": "error",
                "error": built.get("error"),
                "latency_ms": int((time.time() - start) * 1000),
            }

        # 4. Execute (single or both)
        if built["source"] == "BOTH":
            return self._answer_both(question, intent, params, built, start, verbose)

        sql, sql_params = built["queries"]["main"]
        if verbose:
            print(f"[SqlGenerator] SQL: {sql}")
            print(f"[SqlGenerator] params: {sql_params}")

        try:
            rows = self.db.execute_select(
                sql, params=sql_params, timeout_seconds=SQL_TIMEOUT_SECONDS
            )
        except Exception as exc:
            logger.exception("Error ejecutando SQL: %s", exc)
            return {
                "question": question,
                "intent": intent,
                "params": params,
                "sql": sql,
                "answer": "No pude ejecutar la consulta solicitada.",
                "rows_count": 0,
                "status": "error",
                "error": str(exc),
                "latency_ms": int((time.time() - start) * 1000),
            }

        # 5. Format (pure Python)
        answer = self.answer_builder.build(intent, params, rows, built["source"])

        return {
            "question": question,
            "intent": intent,
            "params": params,
            "sql": sql,
            "sql_params": sql_params,
            "answer": answer,
            "rows_count": len(rows),
            "status": "ok" if rows else "no_data",
            "latency_ms": int((time.time() - start) * 1000),
        }

    # ============================================================
    # BOTH pipeline
    # ============================================================
    def _answer_both(
        self,
        question: str,
        intent: str,
        params: dict,
        built: dict,
        start: float,
        verbose: bool,
    ) -> dict:
        sql_dane, params_dane = built["queries"]["dane"]
        sql_cepal, params_cepal = built["queries"]["cepal"]

        if verbose:
            print(f"[SqlGenerator] SQL DANE: {sql_dane}")
            print(f"[SqlGenerator] SQL CEPAL: {sql_cepal}")

        rows_dane: list[dict] = []
        rows_cepal: list[dict] = []
        error_dane = None
        error_cepal = None

        try:
            rows_dane = self.db.execute_select(
                sql_dane, params=params_dane, timeout_seconds=SQL_TIMEOUT_SECONDS
            )
        except Exception as exc:
            error_dane = str(exc)
            logger.exception("Error DANE: %s", exc)

        try:
            rows_cepal = self.db.execute_select(
                sql_cepal, params=params_cepal, timeout_seconds=SQL_TIMEOUT_SECONDS
            )
        except Exception as exc:
            error_cepal = str(exc)
            logger.exception("Error CEPAL: %s", exc)

        answer = self.answer_builder.build(
            intent, params,
            {"dane": rows_dane, "cepal": rows_cepal},
            "BOTH",
        )

        return {
            "question": question,
            "intent": intent,
            "params": params,
            "sql": {"dane": sql_dane, "cepal": sql_cepal},
            "sql_params": {"dane": params_dane, "cepal": params_cepal},
            "answer": answer,
            "rows_count": len(rows_dane) + len(rows_cepal),
            "status": "ok" if (rows_dane or rows_cepal) else "no_data",
            "errors": {"dane": error_dane, "cepal": error_cepal},
            "latency_ms": int((time.time() - start) * 1000),
        }


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = SqlGenerator()
    try:
        q = "¿Cuál fue el IPM de Huila en 2018?"
        print(f"\n❓ {q}")
        result = gen.answer(q, verbose=True)
        print(f"\n💬 {result['answer']}")
        print(f"⏱️ {result['latency_ms']} ms")
    finally:
        gen.close()