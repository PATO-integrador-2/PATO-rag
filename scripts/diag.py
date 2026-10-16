import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import find_dotenv, load_dotenv, dotenv_values

print("=== ANTES de load_dotenv ===")
print(f"cwd            = {os.getcwd()!r}")
print(f"find_dotenv()  = {find_dotenv()!r}")
print(f"POSTGRES_HOST  = {os.environ.get('POSTGRES_HOST')!r}")
print(f"POSTGRES_USER  = {os.environ.get('POSTGRES_USER')!r}")

vals = dotenv_values(find_dotenv())
print(f"dotenv_values() tiene {len(vals)} claves")
print(f"  POSTGRES_HOST = {vals.get('POSTGRES_HOST')!r}")
print(f"  POSTGRES_USER = {vals.get('POSTGRES_USER')!r}")

load_dotenv()

print("\n=== DESPUES de load_dotenv ===")
print(f"POSTGRES_HOST  = {os.environ.get('POSTGRES_HOST')!r}")
print(f"POSTGRES_USER  = {os.environ.get('POSTGRES_USER')!r}")


from src.rag.rag_retriever import RagRetriever
from src.rag.rag_config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER,
    MIN_TOKENS_FOR_VECTOR_FALLBACK, MIN_SCORE, TOP_K_SEMANTIC, TOP_K_LEXICAL,
)

print("=== Config que ve el proceso ===")
print(f"  host={POSTGRES_HOST}:{POSTGRES_PORT} db={POSTGRES_DB} user={POSTGRES_USER}")
print(f"  MIN_TOKENS_FOR_VECTOR_FALLBACK={MIN_TOKENS_FOR_VECTOR_FALLBACK}")
print(f"  MIN_SCORE={MIN_SCORE}")
print(f"  TOP_K_SEMANTIC={TOP_K_SEMANTIC}  TOP_K_LEXICAL={TOP_K_LEXICAL}")

r = RagRetriever()
print(f"\n=== Conexión ===")
print(f"  host={r.host}:{r.port} db={r.dbname} user={r.user}")
print(f"  ping={r.ping()}")

q = "¿Qué es el DANE?"
print(f"\n=== Query: {q!r} ===")
lex = r._build_lexical_query(q)
kw = r._extract_query_keywords(q)
print(f"  lex_query = {lex!r}")
print(f"  keywords  = {kw}")
print(f"  use_vector_fallback = {len(lex.split('&')) >= MIN_TOKENS_FOR_VECTOR_FALLBACK}")

# Prueba directa: ¿el tsquery matchea?
conn = r._connect()
with conn.cursor() as cur:
    cur.execute("""
        SELECT COUNT(*) AS n
        FROM knowledge.document_chunks dc
        WHERE dc.content_tsv @@ to_tsquery('spanish', %s)
    """, (lex,))
    print(f"\n  Filas que matchean tsquery('spanish', {lex!r}): {cur.fetchone()['n']}")

    cur.execute("""
        SELECT dc.id, dc.page_number,
               to_tsvector('spanish', dc.content) @@ to_tsquery('spanish', %s) AS matches
        FROM knowledge.document_chunks dc
        WHERE dc.content ILIKE '%%DANE%%'
        LIMIT 5
    """, (lex,))
    print(f"\n  Chunks con 'DANE' y su match con tsquery:")
    for row in cur.fetchall():
        print(f"    id={row['id']}  p.{row['page_number']}  matches={row['matches']}")

# Ahora la búsqueda completa
print(f"\n=== search() completo ===")
chunks = r.search(q, top_k=5)
print(f"  chunks={len(chunks)}")
for c in chunks[:3]:
    print(f"    {c.get('title')} p.{c.get('page_number')} score={c.get('score')} rrf={c.get('rrf')}")

r.close()