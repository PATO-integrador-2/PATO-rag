"""Simulation module test battery."""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.simulation_generator import SimulationGenerator
from src.simulation.simulation_intent_classifier import SimulationIntentClassifier


TEST_CASES = [
    # Improvement (reduce)
    ("¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?", "reduce_one_deprivation"),
    ("Si se elimina el analfabetismo en Antioquia", "eliminate_one_deprivation"),
    ("Si el trabajo informal baja 20% y la deserción escolar 15% en Colombia", "reduce_multiple"),
    ("¿Cuánto hay que reducir el trabajo informal para que el IPM nacional llegue a 20%?", "target_ipm"),
    ("¿Qué reduce más el IPM nacional: hacinamiento o analfabetismo, ambos al 10%?", "compare_improvements"),
    ("Si la dimensión Educación mejora 20% en Colombia", "dimension_impact"),
    # Worsening (increase)
    ("¿Qué pasa si el trabajo informal sube 15% en Antioquia?", "increase_one_deprivation"),
    ("Si empeora toda la dimensión Vivienda 20% en Colombia", "worsen_all_in_dimension"),
    ("Si el analfabetismo sube 10% y el desempleo 5% en Colombia", "increase_multiple"),
    ("¿Cuánto debe subir el trabajo informal para que el IPM nacional llegue a 40%?", "target_ipm_increase"),
    ("¿Qué aumenta más el IPM: hacinamiento o trabajo informal al 10%?", "compare_worsenings"),
    # Fallbacks
    ("¿Cuál es el IPM de Huila?", "no_data"),
    ("¿Qué es el hacinamiento crítico?", "no_data"),
]


def main() -> None:
    print("=" * 70)
    print("  TEST SIMULATION MODULE")
    print("=" * 70)

    # Phase 1: classifier
    print("\n=== FASE 1: Clasificador ===")
    clf = SimulationIntentClassifier()
    correct = 0
    for q, expected in TEST_CASES:
        r = clf.classify(q)
        got = r["intent"]
        ok = "✅" if got == expected else "❌"
        if got == expected:
            correct += 1
        print(f"  {ok} [{got:<28}] esperado: {expected:<28} | {q[:55]}")
        if got != expected:
            print(f"      raw: {r.get('raw', '')[:150]}")
    print(f"\n  Clasificación: {correct}/{len(TEST_CASES)}")

    # Phase 2: end-to-end
    print("\n=== FASE 2: Pipeline completo ===")
    gen = SimulationGenerator()
    results = []
    try:
        for i, (q, _) in enumerate(TEST_CASES, 1):
            print(f"\n{'-' * 70}")
            print(f"  [{i}/{len(TEST_CASES)}] {q}")
            t0 = time.time()
            try:
                result = gen.answer(q, verbose=False)
                elapsed = int((time.time() - t0) * 1000)
                print(f"  Intent: {result['intent']} | Status: {result['status']}")
                print(f"  Respuesta:\n    {result['answer'][:500]}")
                print(f"  ⏱️  {elapsed} ms")
                results.append({
                    "q": q,
                    "intent": result["intent"],
                    "status": result["status"],
                    "ok": result["status"] in {"ok", "no_data"},
                    "ms": elapsed,
                })
            except Exception as exc:
                print(f"  ❌ EXCEPCIÓN: {exc}")
                results.append({
                    "q": q, "intent": "error", "status": "error",
                    "ok": False, "ms": 0,
                })
    finally:
        gen.close()

    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    ok_count = sum(1 for r in results if r["ok"])
    total_ms = sum(r["ms"] for r in results)
    avg_ms = total_ms // max(len(results), 1)
    print(f"\n  Exitosas: {ok_count}/{len(results)}")
    print(f"  Tiempo total: {total_ms / 1000:.1f} s")
    print(f"  Promedio: {avg_ms} ms\n")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} [{r['intent']:<28}] [{r['status']:<10}] "
            f"({r['ms']} ms) — {r['q'][:55]}")


if __name__ == "__main__":
    main()