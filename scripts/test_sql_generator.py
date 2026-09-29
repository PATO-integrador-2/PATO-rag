"""
Script de prueba del módulo Text-to-SQL.

Ejecuta una batería de preguntas de los 3 grupos (dane, cepal, both)
y muestra el resultado de cada una.

Ejecutar: python scripts/test_sql_generator.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.sql_generator import SqlGenerator, route_group


# ============================================================
# Batería de preguntas
# ============================================================
TEST_QUESTIONS = [
    # --- DANE (Colombia) ---
    "¿Cuál fue el IPM de Huila en 2018?",
    "¿Cuál es el IPM nacional más reciente?",
    "¿Qué dimensión contribuye más al IPM en Antioquia?",
    "¿Cuál es la privación más común en Colombia?",
    "Diferencia de IPM entre hogares con jefe hombre y mujer",

    # --- CEPAL (Latinoamérica) ---
    "¿Cuál es el IPM de Argentina en 2020?",
    "¿Qué país tiene mayor pobreza multidimensional?",
    "IPM por grupo de edad en México",

    # --- BOTH (ambiguo o cruce) ---
    "Compara el IPM de Colombia con el promedio de Latinoamérica",
    "¿Cómo está Colombia en pobreza multidimensional comparado con la región?",
]


def main():
    print("=" * 70)
    print("  PRUEBA DEL MÓDULO TEXT-TO-SQL")
    print("=" * 70)

    gen = SqlGenerator()

    results = []

    try:
        for i, q in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'=' * 70}")
            print(f"  PREGUNTA {i}/{len(TEST_QUESTIONS)}")
            print(f"{'=' * 70}")
            print(f"\n❓ {q}")
            print(f"   [Enrutador predice: {route_group(q)}]")

            t0 = time.time()
            try:
                result = gen.answer(q, verbose=True)
                elapsed = int((time.time() - t0) * 1000)

                print(f"\n💬 RESPUESTA:\n{result['answer']}")
                print(f"\n📊 Grupo: {result['group']}")
                print(f"📈 Filas devueltas: {result['rows_count']}")
                print(f"⏱️  Latencia: {elapsed} ms")

                results.append({
                    "question": q,
                    "group": result["group"],
                    "ok": True,
                    "rows": result["rows_count"],
                    "ms": elapsed,
                })

            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                results.append({
                    "question": q,
                    "group": route_group(q),
                    "ok": False,
                    "error": str(e),
                })

        # ============================================================
        # Resumen final
        # ============================================================
        print("\n\n" + "=" * 70)
        print("  RESUMEN")
        print("=" * 70)

        ok_count = sum(1 for r in results if r["ok"])
        print(f"\nPreguntas exitosas: {ok_count}/{len(results)}")

        print("\nDetalle:")
        for r in results:
            status = "✅" if r["ok"] else "❌"
            if r["ok"]:
                print(f"  {status} [{r['group']:<6}] ({r['rows']} filas, {r['ms']} ms) — {r['question'][:60]}")
            else:
                print(f"  {status} [{r['group']:<6}] ERROR: {r.get('error', 'desconocido')[:80]}")

    finally:
        gen.close()

    print("\n" + "=" * 70)
    print("  FIN DE LA PRUEBA")
    print("=" * 70)


if __name__ == "__main__":
    main()