"""
Inspección rápida de las tablas del RAG en Postgres.

Muestra:
    - Fuentes (sources) registradas
    - Documentos (documents) y su cantidad de chunks
    - Estadísticas de chunks (totales, por dominio, por dimensión)
    - Chunks sin embedding (para detectar problemas)
    - Prueba de búsqueda vectorial con el primer chunk como consulta

Ejecutar: python -m src.inspect_db
"""
import os
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.database import Database


def inspect():
    """Inspecciona las tablas del RAG y muestra estadísticas."""
    db = Database()

    print("=" * 60)
    print("  INSPECCIÓN DE LA BASE DE DATOS — RAG IPM")
    print("=" * 60)

    with db.cursor() as cur:
        # ============================================================
        # SOURCES
        # ============================================================
        cur.execute("SELECT id, name, website_url FROM sources ORDER BY id")
        sources = cur.fetchall()
        print(f"\n📚 SOURCES ({len(sources)}):")
        for s in sources:
            print(f"  [{s['id']}] {s['name']}")

        # ============================================================
        # DOCUMENTS
        # ============================================================
        cur.execute("""
            SELECT d.id, d.title, d.publication_year, s.name AS source_name,
                   COUNT(dc.id) AS n_chunks
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            LEFT JOIN document_chunks dc ON dc.document_id = d.id
            GROUP BY d.id, d.title, d.publication_year, s.name
            ORDER BY d.created_at DESC
            LIMIT 10
        """)
        docs = cur.fetchall()
        print(f"\n📄 DOCUMENTS (últimos 10 de {len(docs)} mostrados):")
        if not docs:
            print("  (no hay documentos indexados)")
        for d in docs:
            title = d['title'][:60] + ("..." if len(d['title']) > 60 else "")
            year = d['publication_year'] if d['publication_year'] else "s/a"
            print(f"  [{d['source_name']}] {title} ({year}) → {d['n_chunks']} chunks")

        # Total de documentos
        cur.execute("SELECT COUNT(*) AS total FROM documents")
        total_docs = cur.fetchone()["total"]
        if total_docs > len(docs):
            print(f"  ... y {total_docs - len(docs)} documentos más")

        # ============================================================
        # CHUNKS: totales
        # ============================================================
        cur.execute("SELECT COUNT(*) AS total FROM document_chunks")
        total = cur.fetchone()["total"]
        print(f"\n📦 CHUNKS TOTALES: {total}")

        if total == 0:
            print("\n⚠️  No hay chunks en la base. Ejecuta primero:")
            print("    python -m src.indexer --limit 3")
            db.close()
            return

        # ============================================================
        # CHUNKS: por dominio
        # ============================================================
        cur.execute("""
            SELECT domain, COUNT(*) AS n
            FROM document_chunks
            GROUP BY domain
            ORDER BY n DESC
        """)
        print("\n  Por dominio:")
        for row in cur.fetchall():
            print(f"    {row['domain'] or '(sin dominio)'}: {row['n']}")

        # ============================================================
        # CHUNKS: por dimensión
        # ============================================================
        cur.execute("""
            SELECT COALESCE(dimension, 'no_clasificada') AS dimension,
                   COUNT(*) AS n
            FROM document_chunks
            GROUP BY dimension
            ORDER BY n DESC
        """)
        print("\n  Por dimensión:")
        for row in cur.fetchall():
            pct = 100 * row['n'] / total
            print(f"    {row['dimension']:<30} {row['n']:>5} ({pct:.1f}%)")

        # ============================================================
        # CHUNKS: por indicador (top 10)
        # ============================================================
        cur.execute("""
            SELECT indicator, COUNT(*) AS n
            FROM document_chunks
            WHERE indicator IS NOT NULL
            GROUP BY indicator
            ORDER BY n DESC
            LIMIT 10
        """)
        indicators = cur.fetchall()
        if indicators:
            print("\n  Top 10 indicadores:")
            for row in indicators:
                ind = row['indicator'][:50] + ("..." if len(row['indicator']) > 50 else "")
                print(f"    {ind:<55} {row['n']:>5}")

        # ============================================================
        # CHUNKS: sin embedding
        # ============================================================
        cur.execute("""
            SELECT COUNT(*) AS n
            FROM document_chunks
            WHERE embedding IS NULL
        """)
        nulls = cur.fetchone()["n"]
        if nulls > 0:
            print(f"\n  ⚠️  Chunks SIN embedding: {nulls} (deberían ser 0)")
        else:
            print(f"\n  ✅ Todos los chunks tienen embedding")

        # ============================================================
        # PRUEBA DE BÚSQUEDA VECTORIAL
        # ============================================================
        print("\n🔍 PRUEBA DE BÚSQUEDA VECTORIAL:")
        print("  (usando el primer chunk como consulta)")

        cur.execute("""
            SELECT embedding FROM document_chunks
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)
        sample = cur.fetchone()

        if sample:
            cur.execute("""
                SELECT
                    LEFT(content, 70) AS preview,
                    dimension,
                    indicator,
                    page_number,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 5
            """, (sample["embedding"], sample["embedding"]))

            print()
            for i, row in enumerate(cur.fetchall(), 1):
                dim = row['dimension'] or '(sin dim)'
                ind = row['indicator'] or '(sin ind)'
                page = row['page_number'] if row['page_number'] else '?'
                print(f"  {i}. [sim={row['similarity']:.4f}] pág {page} | {dim} / {ind}")
                print(f"     {row['preview']}...")

            print("\n  Interpretación:")
            print("  El primer resultado debe tener similitud ≈ 1.0 (es el mismo chunk).")
            print("  Los siguientes deben ser progresivamente menos similares.")

    db.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    inspect()