"""
Analiza los chunks no clasificados para detectar patrones.
Ejecutar: python -m src.analyze_unclassified
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.database import Database
from collections import Counter
import re


def analyze():
    db = Database()

    with db.cursor() as cur:
        # Cantidad total de no clasificados
        cur.execute("""
            SELECT COUNT(*) AS n
            FROM document_chunks
            WHERE dimension IS NULL OR dimension = 'no_clasificada'
        """)
        total_unclassified = cur.fetchone()["n"]
        print(f"Chunks no clasificados: {total_unclassified}")

        # Extraer muestra aleatoria
        cur.execute("""
            SELECT content, document_id
            FROM document_chunks
            WHERE dimension IS NULL OR dimension = 'no_clasificada'
            ORDER BY random()
            LIMIT 30
        """)
        samples = cur.fetchall()

        print("\n" + "=" * 70)
        print("  MUESTRA ALEATORIA DE 30 CHUNKS NO CLASIFICADOS")
        print("=" * 70)
        for i, s in enumerate(samples, 1):
            preview = s["content"][:200].replace("\n", " ")
            print(f"\n--- Muestra {i} ---")
            print(preview + "...")

        # Análisis de frecuencia de palabras
        print("\n" + "=" * 70)
        print("  PALABRAS MÁS FRECUENTES EN CHUNKS NO CLASIFICADOS")
        print("=" * 70)

        # Traer más chunks para análisis de palabras
        cur.execute("""
            SELECT content FROM document_chunks
            WHERE dimension IS NULL OR dimension = 'no_clasificada'
            LIMIT 2000
        """)
        all_unclassified = cur.fetchall()

        # Contar palabras (excluyendo stopwords básicas)
        stopwords = {
            "de", "la", "el", "en", "y", "a", "los", "del", "las", "un", "una",
            "por", "con", "para", "que", "es", "al", "se", "su", "lo", "como",
            "más", "o", "pero", "sus", "le", "ha", "me", "si", "sin", "sobre",
            "este", "ya", "entre", "cuando", "todo", "esta", "ser", "son", "dos",
            "también", "fue", "había", "era", "muy", "años", "hasta", "desde",
            "cada", "puede", "tiene", "donde", "todos", "nos", "ni", "parte",
            "tiene", "estos", "esos", "esas", "estas", "otro", "otra",
        }
        words = Counter()
        for chunk in all_unclassified:
            texto = chunk["content"].lower()
            tokens = re.findall(r"\b[a-záéíóúñ]{4,}\b", texto)
            for t in tokens:
                if t not in stopwords:
                    words[t] += 1

        print("\nTop 50 palabras (frecuencia):")
        for word, count in words.most_common(50):
            print(f"  {word:<30} {count}")

    db.close()


if __name__ == "__main__":
    analyze()