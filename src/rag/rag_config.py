"""
RAG configuration.

Uses HYBRID retrieval: semantic (vector cosine) + full-text (tsvector)
fused with RRF (Reciprocal Rank Fusion). Based on the reference
implementation provided by the backend team.

The hybrid approach guarantees that returned chunks contain actual
lexical matches (real words from the query), while still leveraging
the semantic similarity of embeddings.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Connection
# ============================================================
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "embedding_db")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


# ============================================================
# Embedding model
# ============================================================
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EXPECTED_EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
PREFIX_QUERY = "search_query: "


# ============================================================
# Retrieval parameters
# ============================================================
TOP_K_SEMANTIC = int(os.getenv("RAG_TOP_K_SEMANTIC", "100"))
TOP_K_LEXICAL = int(os.getenv("RAG_TOP_K_LEXICAL", "50"))
TOP_K_FINAL = int(os.getenv("RAG_TOP_K_FINAL", "10"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.50"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
MIN_TOKENS_FOR_VECTOR_FALLBACK = 4


# ============================================================
# Answer generation
# ============================================================
ANSWER_LLM_MODEL = os.getenv(
    "OLLAMA_RAG_ANSWER_MODEL",
    os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b"),
)
ANSWER_MAX_TOKENS = int(os.getenv("RAG_ANSWER_MAX_TOKENS", "400"))
ANSWER_TEMPERATURE = float(os.getenv("RAG_ANSWER_TEMPERATURE", "0.2"))
CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "4000"))


# ============================================================
# SQL: hybrid retrieval (semantic + full-text with RRF)
# ============================================================
# Reference: implementation by backend team.
# Params:
#   %(embedding)s           vector as string "[0.1,0.2,...]"
#   %(lex_query)s           tokens joined with " & "
#   %(limit)s               final limit
#   %(min_score)s           min score for vector fallback
#   %(organization)s        organization filter (or NULL)
#   %(use_vector_fallback)s boolean
#   %(top_semantic)s        top N semantic
#   %(top_lexical)s         top N lexical
SQL_HYBRID_SEARCH = """
WITH vec AS (
    SELECT dc.id,
           ROW_NUMBER() OVER (ORDER BY dc.embedding <=> %(embedding)s::vector) AS rnk
    FROM knowledge.document_chunks dc
    JOIN knowledge.documents d ON d.id = dc.document_id
    WHERE dc.embedding IS NOT NULL
      AND (%(organization)s::text IS NULL OR lower(d.organization) = lower(%(organization)s))
    ORDER BY dc.embedding <=> %(embedding)s::vector
    LIMIT %(top_semantic)s
),
lex AS (
    SELECT dc.id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(dc.content_tsv, to_tsquery('spanish', %(lex_query)s)) DESC
           ) AS rnk
    FROM knowledge.document_chunks dc
    JOIN knowledge.documents d ON d.id = dc.document_id
    WHERE dc.content_tsv @@ to_tsquery('spanish', %(lex_query)s)
      AND (%(organization)s::text IS NULL OR lower(d.organization) = lower(%(organization)s))
    ORDER BY ts_rank_cd(dc.content_tsv, to_tsquery('spanish', %(lex_query)s)) DESC
    LIMIT %(top_lexical)s
),
fused AS (
    SELECT id, SUM(1.0 / (60 + rnk)) AS rrf
    FROM (
        SELECT id, rnk FROM vec
        UNION ALL
        SELECT id, rnk FROM lex
    ) u
    GROUP BY id
)
SELECT
    dc.id,
    d.id AS document_id,
    d.title,
    d.organization,
    d.url,
    dc.content,
    dc.page_number,
    dc.section,
    1 - (dc.embedding <=> %(embedding)s::vector) AS score,
    (dc.id IN (SELECT id FROM lex)) AS lexical_match,
    f.rrf
FROM fused f
JOIN knowledge.document_chunks dc ON dc.id = f.id
JOIN knowledge.documents d ON d.id = dc.document_id
WHERE dc.id IN (SELECT id FROM lex)
   OR (
       %(use_vector_fallback)s::boolean
       AND NOT EXISTS (SELECT 1 FROM lex)
       AND 1 - (dc.embedding <=> %(embedding)s::vector) >= %(min_score)s
   )
ORDER BY f.rrf DESC
LIMIT %(limit)s
"""


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    print("RAG configuration (hybrid):")
    print(f"  Postgres:   {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"  Embed:      {EMBED_MODEL} ({EXPECTED_EMBED_DIM} dims)")
    print(f"  Retrieval:  top_{TOP_K_SEMANTIC} sem + top_{TOP_K_LEXICAL} lex → top_{TOP_K_FINAL}")
    print(f"  Min score:  {MIN_SCORE}")
    print(f"  RRF k:      {RRF_K}")
    print(f"  Answer:     {ANSWER_LLM_MODEL}")