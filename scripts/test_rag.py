"""RAG module test battery."""
import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_generator import RagGenerator
from src.rag.rag_retriever import RagRetriever


TEST_QUESTIONS = [
    "¿Qué es el hacinamiento crítico?",
    "¿Cuáles son las dimensiones del IPM?",
    "¿Cómo se mide el analfabetismo?",
    "¿Qué es pobreza multidimensional?",
    "¿Qué es el IPM?",
    "¿Cómo se calcula el IPM?",
    "¿Qué es el trabajo informal según el DANE?",
    "¿Cuál es la metodología del IPM?",
    "¿Qué indicadores componen la dimensión de educación?",
    "¿Qué es un hogar en pobreza multidimensional?",
]


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 70)
    print("  TEST RAG MODULE")
    print("=" * 70)

    # Ping first
    print("\n=== FASE 0: Conexión ===")
    ret = RagRetriever()
    try:
        if not ret.ping():
            print("❌ No se pudo conectar a la BD vectorial.")
            return
        print("✅ Conexión OK a embedding_db")
    finally:
        ret.close()

    # RAG tests
    print("\n=== FASE 1: Preguntas conceptuales ===")
    gen = RagGenerator()
    results = []
    try:
        for i, q in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'-' * 70}")
            print(f"  [{i}/{len(TEST_QUESTIONS)}] {q}")
            t0 = time.time()
            try:
                r = gen.answer(q)
                elapsed = int((time.time() - t0) * 1000)
                print(f"  Status: {r['status']} | Chunks: {r['context_chunks']} | {elapsed} ms")
                print(f"  Respuesta:\n    {r['answer'][:500]}")
                if r["sources"]:
                    print(f"  Fuentes: {len(r['sources'])}")
                    for s in r["sources"][:2]:
                        print(f"    - {s['document_title']} p.{s['page_number']}")
                results.append({
                    "q": q,
                    "status": r["status"],
                    "chunks": r["context_chunks"],
                    "ok": r["status"] in {"ok", "no_context"},
                    "ms": elapsed,
                })
            except Exception as exc:
                print(f"  ❌ EXCEPCIÓN: {exc}")
                results.append({
                    "q": q, "status": "error", "chunks": 0,
                    "ok": False, "ms": 0,
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
        print(f"  {status} [{r['status']:<10}] ({r['chunks']} chunks, {r['ms']} ms) — {r['q'][:55]}")


if __name__ == "__main__":
    main()