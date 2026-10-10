"""
Massive test of the Orchestrator across all 4 modules.

Measures:
    - Intent classification correctness
    - End-to-end latency per question
    - Module-specific performance
    - Model swapping impact (first vs subsequent questions per module)

Run: python scripts/test_orchestrator_massive.py
"""
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.router import Orchestrator


# ============================================================
# Test battery — 30 questions
# ============================================================
TEST_CASES = [
    # ---- data (8) ----
    ("¿Cuál fue el IPM de Huila en 2018?", "data"),
    ("¿Cuál es el IPM nacional más reciente?", "data"),
    ("¿Cuál es el IPM de Antioquia en 2020?", "data"),
    ("¿Qué dimensión contribuye más al IPM?", "data"),
    ("¿Cuál es la privación más común en Colombia?", "data"),
    ("Diferencia de IPM entre hogares con jefe hombre y mujer", "data"),
    ("¿Cuántos hogares hay en Antioquia en 2022?", "data"),
    ("¿Cuál es el IPM de México en 2020?", "data"),

    # ---- simulation (8) ----
    ("¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?", "simulation"),
    ("Si se elimina el analfabetismo en Antioquia", "simulation"),
    ("¿Cuánto hay que reducir el trabajo informal para que el IPM llegue a 20%?", "simulation"),
    ("¿Qué reduce más el IPM: hacinamiento o analfabetismo?", "simulation"),
    ("Si la dimensión Educación mejora 20% en Colombia", "simulation"),
    ("¿Qué pasa si el trabajo informal sube 15% en Antioquia?", "simulation"),
    ("Si empeora toda la dimensión Vivienda 20% en Colombia", "simulation"),
    ("¿Qué aumenta más el IPM: hacinamiento o trabajo informal?", "simulation"),

    # ---- rag (7) ----
    ("¿Qué es el hacinamiento crítico?", "rag"),
    ("¿Cuáles son las dimensiones del IPM?", "rag"),
    ("¿Cómo se mide el analfabetismo?", "rag"),
    ("¿Qué es pobreza multidimensional?", "rag"),
    ("¿Qué es el IPM?", "rag"),
    ("¿Cómo se calcula el IPM?", "rag"),
    ("¿Cuál es la metodología del IPM?", "rag"),

    # ---- recommendation (7) ----
    ("¿Qué proyectos puedo implementar para reducir el hacinamiento en Antioquia?", "recommendation"),
    ("Sugiere intervenciones para mejorar la educación rural", "recommendation"),
    ("¿Cómo puedo bajar el desempleo de larga duración en Colombia?", "recommendation"),
    ("Dame ideas de políticas para combatir la pobreza multidimensional", "recommendation"),
    ("Propón proyectos para mejorar el acceso a agua potable en La Guajira", "recommendation"),
    ("¿Qué puedo hacer para reducir el analfabetismo?", "recommendation"),
    ("Sugiere proyectos para la primera infancia en zonas rurales", "recommendation"),
]


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 74)
    print("  TEST MASIVO DEL ORQUESTADOR — 30 preguntas")
    print("=" * 74)

    orch = Orchestrator()
    results = []
    by_intent = defaultdict(lambda: {"count": 0, "total_ms": 0, "ok": 0, "wrong_intent": 0})

    try:
        for i, (q, expected) in enumerate(TEST_CASES, 1):
            print(f"\n{'-' * 74}")
            print(f"  [{i:02d}/{len(TEST_CASES)}] {q}")
            t0 = time.time()
            try:
                r = orch.ask(q, verbose=False)
                elapsed = int((time.time() - t0) * 1000)
                got = r.get("intent", "?")
                correct = "✅" if got == expected else "❌"

                # Track
                stats = by_intent[got]
                stats["count"] += 1
                stats["total_ms"] += elapsed
                if r.get("status") == "ok":
                    stats["ok"] += 1
                if got != expected:
                    stats["wrong_intent"] += 1

                print(f"  Intent: {got} {correct} (esperado: {expected}) | "
                      f"Status: {r.get('status')} | {elapsed} ms")
                print(f"  Answer: {r.get('answer', '')[:200]}")

                results.append({
                    "q": q,
                    "expected": expected,
                    "got": got,
                    "status": r.get("status"),
                    "ms": elapsed,
                    "correct_intent": got == expected,
                })
            except Exception as exc:
                print(f"  ❌ EXCEPCIÓN: {exc}")
                results.append({
                    "q": q, "expected": expected, "got": "error",
                    "status": "error", "ms": 0, "correct_intent": False,
                })

        # ============================================================
        # Summary
        # ============================================================
        print("\n\n" + "=" * 74)
        print("  RESUMEN GENERAL")
        print("=" * 74)

        total = len(results)
        correct = sum(1 for r in results if r["correct_intent"])
        ok = sum(1 for r in results if r["status"] == "ok")
        total_ms = sum(r["ms"] for r in results)
        avg_ms = total_ms // max(total, 1)

        print(f"\n  Total preguntas:       {total}")
        print(f"  Intent correcto:       {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Status ok:             {ok}/{total} ({100*ok/total:.1f}%)")
        print(f"  Tiempo total:          {total_ms/1000:.1f} s")
        print(f"  Promedio por pregunta: {avg_ms} ms")

        print("\n  DESGLOSE POR MÓDULO:")
        print(f"  {'Módulo':<18} {'Cant':<6} {'OK':<6} {'Promedio':<10} {'Errores intent'}")
        print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*10} {'-'*15}")
        for intent in ["data", "simulation", "rag", "recommendation"]:
            s = by_intent[intent]
            if s["count"] == 0:
                continue
            avg = s["total_ms"] // s["count"]
            print(f"  {intent:<18} {s['count']:<6} {s['ok']:<6} {avg:<10} {s['wrong_intent']}")

        # Per-question detail
        print("\n  DETALLE POR PREGUNTA:")
        for i, r in enumerate(results, 1):
            status = "✅" if r["correct_intent"] and r["status"] == "ok" else "⚠️"
            print(f"  {status} [{r['got']:<15}] {r['ms']:>6} ms | {r['q'][:55]}")

    finally:
        orch.close()

    print("\n" + "=" * 74)
    print("  FIN DEL TEST MASIVO")
    print("=" * 74)


if __name__ == "__main__":
    main()