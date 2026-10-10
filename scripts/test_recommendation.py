"""Recommendation module test battery."""
import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.recommendation_generator import RecommendationGenerator
from src.recommendation.recommendation_intent_classifier import (
    RecommendationIntentClassifier,
)


TEST_QUESTIONS = [
    "¿Qué proyectos puedo implementar para reducir el hacinamiento en Antioquia?",
    "Sugiere intervenciones para mejorar la educación rural",
    "¿Cómo puedo bajar el desempleo de larga duración en Colombia?",
    "Dame ideas de políticas para combatir la pobreza multidimensional",
    "Propón proyectos para mejorar el acceso a agua potable en La Guajira",
    "¿Qué puedo hacer para reducir el analfabetismo?",
    "Sugiere proyectos para la primera infancia en zonas rurales",
    "¿Qué intervenciones reducen las barreras de acceso a salud?",
]


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 70)
    print("  TEST RECOMMENDATION MODULE")
    print("=" * 70)

    # Phase 1: classifier
    print("\n=== FASE 1: Clasificador de intención ===")
    clf = RecommendationIntentClassifier()
    for q in TEST_QUESTIONS:
        r = clf.classify(q)
        print(f"\n  ❓ {q[:60]}")
        print(f"     topic={r['topic']!r} entity={r['entity']!r} "
              f"dimension={r['dimension']!r} ({r['method']})")

    # Phase 2: full pipeline
    print("\n=== FASE 2: Pipeline completo ===")
    gen = RecommendationGenerator()
    results = []
    try:
        for i, q in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'-' * 70}")
            print(f"  [{i}/{len(TEST_QUESTIONS)}] {q}")
            t0 = time.time()
            try:
                r = gen.answer(q, verbose=False)
                elapsed = int((time.time() - t0) * 1000)
                print(f"  Topic: {r['topic']} | Entity: {r['entity']} | Dim: {r['dimension']}")
                print(f"  Chunks: {r['context_chunks']} | Fuentes: {len(r['sources'])}")
                print(f"  Respuesta:\n    {r['answer'][:600]}")
                print(f"  ⏱️  {elapsed} ms")
                results.append({
                    "q": q, "ok": True, "ms": elapsed,
                    "topic": r["topic"], "dimension": r["dimension"],
                })
            except Exception as exc:
                print(f"  ❌ EXCEPCIÓN: {exc}")
                results.append({
                    "q": q, "ok": False, "ms": 0,
                    "topic": "", "dimension": "",
                })
    finally:
        gen.close()

    # Summary
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    ok_count = sum(1 for r in results if r["ok"])
    total_ms = sum(r["ms"] for r in results)
    avg_ms = total_ms // max(len(results), 1)
    print(f"\n  Exitosas: {ok_count}/{len(results)}")
    print(f"  Tiempo total: {total_ms/1000:.1f} s")
    print(f"  Promedio: {avg_ms} ms\n")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} [{r['dimension']:<20}] ({r['ms']:>6} ms) — {r['q'][:55]}")


if __name__ == "__main__":
    main()