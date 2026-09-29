"""
Módulo Text-to-SQL para responder preguntas sobre datos específicos
consultando la base clean_data_db.

Flujo:
    1. route_group(question)          → decide si va por DANE, CEPAL o ambos.
    2. build_schema_prompt(group)     → construye el texto del esquema.
    3. generate_sql(question, group)  → LLM genera SQL.
    4. validate_sql(sql, group)       → valida contra whitelist y regex.
    5. execute_sql(sql)               → ejecuta con DataDatabase.
    6. format_answer(...)             → LLM formatea el resultado.

Uso:
    from src.retrieval.sql_generator import SqlGenerator
    gen = SqlGenerator()
    result = gen.answer("¿Cuál fue el IPM de Huila en 2018?")
    print(result["answer"])
"""
import os
import re
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
from decimal import Decimal
from datetime import date, datetime

import ollama
from dotenv import load_dotenv

from src.storage.data_db import DataDatabase, DATA_GROUPS
from src.core.text_normalizer import normalize
from src.retrieval.sql_prompts import (
    SQL_GENERATION_PROMPT,
    ANSWER_FORMATTING_PROMPT,
    BOTH_GROUPS_PROMPT,
)

load_dotenv()

LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "100"))
SQL_TIMEOUT_SECONDS = int(os.getenv("SQL_TIMEOUT_SECONDS", "15"))


# ============================================================
# Enrutador por palabras clave
# ============================================================
def route_group(question: str) -> str:
    """
    Devuelve 'dane', 'cepal' o 'both' según las palabras clave de la pregunta.
    """
    q_norm = normalize(question)

    dane_score = sum(
        1 for kw in DATA_GROUPS["dane"]["keywords"]
        if normalize(kw) in q_norm
    )
    cepal_score = sum(
        1 for kw in DATA_GROUPS["cepal"]["keywords"]
        if normalize(kw) in q_norm
    )

    # Si ambos scores son positivos y similares → both
    if dane_score > 0 and cepal_score > 0 and abs(dane_score - cepal_score) <= 1:
        return "both"
    if dane_score > cepal_score:
        return "dane"
    if cepal_score > dane_score:
        return "cepal"
    # Sin match → both (fallback)
    return "both"


# ============================================================
# Construcción del prompt de esquema
# ============================================================
def _schema_to_text(schema: dict) -> str:
    """
    Convierte un dict de esquema en texto legible para el LLM.
    """
    lines = [f"# Grupo: {schema['label']}"]

    lines.append("\n## Vistas disponibles:")
    for view_name, cols in schema["views"].items():
        col_str = ", ".join(f"{c['column_name']} {c['data_type']}" for c in cols)
        lines.append(f"  {view_name}({col_str})")

    if schema["ref_tables"]:
        lines.append("\n## Tablas de referencia (catálogos):")
        for ref_name, cols in schema["ref_tables"].items():
            col_str = ", ".join(f"{c['column_name']} {c['data_type']}" for c in cols)
            lines.append(f"  {ref_name}({col_str})")

    return "\n".join(lines)


# ============================================================
# Validación del SQL
# ============================================================
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "COPY", "VACUUM", "ANALYZE",
    "EXECUTE", "CALL", "DO", "MERGE",
]


def validate_sql(sql: str, allowed_views: list[str]) -> tuple[bool, str]:
    """
    Valida que el SQL sea seguro.
    Devuelve (es_valido, motivo).
    """
    if not sql or not sql.strip():
        return False, "SQL vacío"

    upper = sql.upper()

    # 1. Solo SELECT
    if not upper.lstrip().startswith("SELECT") and not upper.lstrip().startswith("WITH"):
        return False, "Solo se permite SELECT o WITH"

    # 2. Sin keywords prohibidas
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"Keyword prohibida: {kw}"

    # 3. Sin múltiples statements
    cleaned = sql.rstrip().rstrip(";")
    if ";" in cleaned:
        return False, "Múltiples statements no permitidos"

    # 4. Sin comentarios
    if "--" in sql or "/*" in sql:
        return False, "Comentarios no permitidos"

    # 5. Referencias a tablas deben estar en la whitelist
    # Extraemos todos los identificadores que parezcan nombres de tabla (con o sin vw_)
    table_refs = re.findall(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    )
    allowed_lower = {v.lower() for v in allowed_views}
    for ref in table_refs:
        if ref.lower() not in allowed_lower:
            return False, f"Tabla/vista no permitida: {ref}"

    return True, "OK"


# ============================================================
# Serialización para el prompt
# ============================================================
def _serialize_result(rows: list[dict]) -> str:
    """
    Convierte el resultado de la BD a JSON serializable.
    Maneja Decimal, date, datetime.
    """
    if not rows:
        return "[]"

    def default(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return str(obj)

    # Limitar filas para no saturar el prompt
    limited = rows[:50]
    serialized = json.dumps(limited, default=default, ensure_ascii=False, indent=2)

    if len(rows) > 50:
        serialized += f"\n\n... y {len(rows) - 50} filas más (total: {len(rows)})"

    return serialized


# ============================================================
# Clase principal
# ============================================================
class SqlGenerator:
    """Orquesta el flujo de Text-to-SQL."""

    def __init__(self, db: Optional[DataDatabase] = None):
        self.db = db or DataDatabase()
        self._schema_cache: dict = {}

    def close(self):
        if self.db:
            self.db.close()

    # --------------------------------------------------------
    # Caché de esquemas
    # --------------------------------------------------------
    def _get_schema(self, group: str) -> dict:
        if group not in self._schema_cache:
            self._schema_cache[group] = self.db.get_group_schema(group)
        return self._schema_cache[group]

    # --------------------------------------------------------
    # Generación de SQL con el LLM
    # --------------------------------------------------------
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.0,       # máxima determinación
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
        )
        return response.get("response", "").strip()

    def _clean_sql_output(self, raw: str) -> str:
        """
        Limpia el output del LLM: quita ```sql, ```, espacios, ; final.
        """
        sql = raw.strip()
        # Quitar bloques markdown si los hay
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)
        # Quitar ; final
        sql = sql.rstrip().rstrip(";").rstrip()
        return sql

    def generate_sql(self, question: str, group: str) -> str:
        """
        Genera una consulta SQL para el grupo indicado.
        """
        if group == "both":
            return self._generate_sql_both(question)

        schema = self._get_schema(group)
        schema_text = _schema_to_text(schema)
        prompt = SQL_GENERATION_PROMPT.format(schema=schema_text, question=question)
        raw = self._call_llm(prompt, max_tokens=400)
        return self._clean_sql_output(raw)

    def _generate_sql_both(self, question: str) -> str:
        """
        Genera dos consultas SQL (DANE + CEPAL) separadas por '---SPLIT---'.
        """
        schema_dane = _schema_to_text(self._get_schema("dane"))
        schema_cepal = _schema_to_text(self._get_schema("cepal"))
        prompt = BOTH_GROUPS_PROMPT.format(
            schema_dane=schema_dane,
            schema_cepal=schema_cepal,
            question=question,
        )
        raw = self._call_llm(prompt, max_tokens=800)
        return raw  # No limpiamos aquí; el split se hace después

    # --------------------------------------------------------
    # Ejecución
    # --------------------------------------------------------
    def _execute(self, sql: str, group: str) -> tuple[bool, list[dict] | str]:
        """
        Valida y ejecuta. Devuelve (ok, rows_o_error).
        """
        allowed = self.db.get_views_by_group(group)
        valid, reason = validate_sql(sql, allowed)
        if not valid:
            return False, f"SQL inválido: {reason}"

        try:
            rows = self.db.execute_select(sql, timeout_seconds=SQL_TIMEOUT_SECONDS)
            return True, rows
        except Exception as e:
            return False, str(e)

    # --------------------------------------------------------
    # Formateo de respuesta con el LLM
    # --------------------------------------------------------
    def _format_answer(
        self,
        question: str,
        sql: str,
        rows: list[dict],
    ) -> str:
        result_json = _serialize_result(rows)
        prompt = ANSWER_FORMATTING_PROMPT.format(
            question=question,
            sql=sql,
            result=result_json,
        )
        return self._call_llm(prompt, max_tokens=500)

    # --------------------------------------------------------
    # API pública
    # --------------------------------------------------------
    def answer(self, question: str, verbose: bool = False) -> dict:
        """
        Flujo completo: pregunta → grupo → SQL → ejecución → respuesta.
        """
        start = time.time()

        # 1. Enrutar
        group = route_group(question)
        if verbose:
            print(f"[SqlGenerator] Grupo detectado: {group}")

        # 2. Caso "both": dos SQL separadas
        if group == "both":
            return self._answer_both(question, verbose, start)

        # 3. Generar SQL
        sql = self.generate_sql(question, group)
        if verbose:
            print(f"[SqlGenerator] SQL generado:\n{sql}\n")

        if "NO_SE_PUEDE_RESPONDER" in sql:
            return {
                "question": question,
                "group": group,
                "sql": None,
                "answer": "No puedo responder esa pregunta con los datos disponibles en la base.",
                "rows_count": 0,
                "latency_ms": int((time.time() - start) * 1000),
            }

        # 4. Ejecutar
        ok, result = self._execute(sql, group)
        if not ok:
            return {
                "question": question,
                "group": group,
                "sql": sql,
                "answer": f"Error al consultar los datos: {result}",
                "rows_count": 0,
                "latency_ms": int((time.time() - start) * 1000),
            }

        rows = result

        # 5. Formatear
        answer_text = self._format_answer(question, sql, rows)

        return {
            "question": question,
            "group": group,
            "sql": sql,
            "answer": answer_text,
            "rows_count": len(rows),
            "latency_ms": int((time.time() - start) * 1000),
        }

    def _answer_both(self, question: str, verbose: bool, start: float) -> dict:
        """
        Maneja el caso 'both': dos SQL, dos resultados, una respuesta unificada.
        """
        raw = self.generate_sql(question, "both")
        parts = raw.split("---SPLIT---")

        if len(parts) != 2:
            return {
                "question": question,
                "group": "both",
                "sql": raw,
                "answer": "No pude separar las consultas para DANE y CEPAL.",
                "rows_count": 0,
                "latency_ms": int((time.time() - start) * 1000),
            }

        sql_dane = self._clean_sql_output(parts[0])
        sql_cepal = self._clean_sql_output(parts[1])

        if verbose:
            print(f"[SqlGenerator] SQL DANE:\n{sql_dane}\n")
            print(f"[SqlGenerator] SQL CEPAL:\n{sql_cepal}\n")

        # Ejecutar cada uno
        ok_dane, rows_dane = self._execute(sql_dane, "dane")
        ok_cepal, rows_cepal = self._execute(sql_cepal, "cepal")

        rows_dane = rows_dane if ok_dane else []
        rows_cepal = rows_cepal if ok_cepal else []

        # Formatear respuesta combinada
        combined_json = (
            "=== DATOS DANE (Colombia) ===\n"
            + (_serialize_result(rows_dane) if ok_dane else f"ERROR: {rows_dane}")
            + "\n\n=== DATOS CEPAL (Latinoamérica) ===\n"
            + (_serialize_result(rows_cepal) if ok_cepal else f"ERROR: {rows_cepal}")
        )

        prompt = ANSWER_FORMATTING_PROMPT.format(
            question=question,
            sql=f"DANE:\n{sql_dane}\n\nCEPAL:\n{sql_cepal}",
            result=combined_json,
        )
        answer_text = self._call_llm(prompt, max_tokens=600)

        return {
            "question": question,
            "group": "both",
            "sql": {"dane": sql_dane, "cepal": sql_cepal},
            "answer": answer_text,
            "rows_count": len(rows_dane) + len(rows_cepal),
            "latency_ms": int((time.time() - start) * 1000),
        }


# ============================================================
# Prueba directa
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PRUEBA DEL SQL GENERATOR")
    print("=" * 60)

    gen = SqlGenerator()
    try:
        # Prueba simple
        q = "¿Cuál fue el IPM de Huila en 2018?"
        print(f"\nPregunta: {q}")
        result = gen.answer(q, verbose=True)
        print(f"\nRespuesta: {result['answer']}")
        print(f"\nTiempo de respuesta: {result['latency_ms']} ms")
    finally:
        gen.close()