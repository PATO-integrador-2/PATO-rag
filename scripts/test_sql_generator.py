"""
SQL module test — new architecture (LLM-as-classifier).
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.sql_generator import SqlGenerator
from src.retrieval.sql_intent_classifier import SqlIntentClassifier
from src.storage.data_db import DataDatabase


TEST_QUESTIONS = [
    ("¿Cuál fue el IPM de Huila en 2018?", "ipm_department_year"),
    ("¿Cuál es el IPM nacional más reciente?", "ipm_national_latest"),
    ("¿Qué dimensión contribuye más al IPM en Antioquia?", "dimension_top_in_domain"),
    ("¿Cuál es la privación más común en Colombia?", "deprivation_top"),
    ("Diferencia de IPM entre hogares con jefe hombre y mujer", "sex_comparison_national"),
    ("¿Cuántos hogares hay en Antioquia en 2022?", "household_count_department"),
    ("¿Cuál es el IPM de Argentina en 2020?", "no_data"),
    ("¿Qué país tiene mayor pobreza multidimensional?", "country_top_poverty"),
    ("IPM por grupo de edad en México", "ipm_country_age_group"),
    ("Compara el IPM de Colombia con el promedio de Latinoamérica", "country_vs_latam"),
]


def main() -> None:
    print("=" * 70)
    print("  TEST SQL GENERATOR — LLM-as-classifier architecture")
    print("=" * 70)

    # ---- Classifier-only test (fast, no DB) ----
    print("\n=== FASE 1: Clasificador de intención ===")
    clf = SqlIntentClassifier()
    correct = 0
    for q, expected in TEST_QUESTIONS:
        result = clf.classify(q)
        got = result["intent"]
        ok = "✅" if got == expected else "❌"
        if got == expected:
            correct += 1
        print(f"  {ok} [{got:<28}] esperado: {expected:<28} | {q[:50]}")
        if got != expected:
            print(f"      raw: {result.get('raw', '')[:120]}")
    print(f"\n  Clasificación: {correct}/{len(TEST_QUESTIONS)}")

    # ---- End-to-end ----
    print("\n=== FASE 2: Pipeline completo ===")
    gen = SqlGenerator()
    results = []
    try:
        for i, (q, expected_intent) in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'-' * 70}")
            print(f"  [{i}/{len(TEST_QUESTIONS)}] {q}")
            t0 = time.time()
            try:
                result = gen.answer(q, verbose=False)
                elapsed = int((time.time() - t0) * 1000)
                print(f"  Intent: {result['intent']}  Params: {result['params']}")
                print(f"  Filas: {result['rows_count']} | Estado: {result['status']}")
                print(f"  Respuesta:\n    {result['answer'][:400]}")
                print(f"  ⏱️  {elapsed} ms")
                results.append({
                    "q": q,
                    "intent": result["intent"],
                    "ok": result["status"] in {"ok", "no_data"},
                    "status": result["status"],
                    "rows": result["rows_count"],
                    "ms": elapsed,
                })
            except Exception as exc:
                print(f"  ❌ EXCEPCIÓN: {exc}")
                results.append({
                    "q": q, "intent": "error",
                    "ok": False, "status": "error", "rows": 0, "ms": 0,
                })
    finally:
        gen.close()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    ok_count = sum(1 for r in results if r["ok"])
    total_ms = sum(r["ms"] for r in results)
    avg_ms = total_ms // max(len(results), 1)
    print(f"\n  Preguntas exitosas: {ok_count}/{len(results)}")
    print(f"  Tiempo total: {total_ms / 1000:.1f} s")
    print(f"  Promedio por pregunta: {avg_ms} ms\n")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(
            f"  {status} [{r['intent']:<28}] [{r['status']:<8}] "
            f"({r['rows']} filas, {r['ms']} ms) — {r['q'][:60]}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()