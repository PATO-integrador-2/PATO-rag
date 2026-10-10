"""
RAG retriever.

Hybrid retrieval: semantic (vector cosine) + full-text (tsvector),
fused with RRF (Reciprocal Rank Fusion). Based on the reference
implementation by the backend team.

Post-filter:
    After the SQL query, chunks are re-scored using:
      1. Keyword matches (how many query keywords appear in content).
      2. Title relevance (chunks whose title mentions IPM-related
         topics get full score; others get a 0.6 penalty).
    This ensures the chunk that actually contains the answer rises
    to the top, even if its semantic similarity was slightly lower.

Uses psycopg (v3) for PostgreSQL connectivity.
"""
import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from psycopg.rows import dict_row

from src.rag.rag_config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    EMBED_MODEL, EXPECTED_EMBED_DIM, PREFIX_QUERY,
    TOP_K_SEMANTIC, TOP_K_LEXICAL, TOP_K_FINAL, MIN_SCORE,
    MIN_TOKENS_FOR_VECTOR_FALLBACK,
    SQL_HYBRID_SEARCH,
)

logger = logging.getLogger(__name__)


# ============================================================
# Spanish stopwords
# ============================================================
SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "de", "del", "a", "al", "en", "con", "por", "para", "sin",
    "sobre", "bajo", "entre", "hacia", "hasta", "desde", "según",
    "y", "o", "u", "e", "ni", "pero", "sino", "que",
    "se", "su", "sus", "me", "te", "nos", "os", "le", "les",
    "mi", "tu", "mis", "tus", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "aquel", "aquella",
    "es", "son", "fue", "era", "ser", "está", "estan", "están",
    "hay", "han", "ha", "he", "has", "hemos",
    "muy", "más", "menos", "ya", "no", "sí", "si",
    "como", "cuando", "donde", "cual", "cuales",
    "qué", "que", "quien", "quienes", "cuanto", "cuanta",
    "cuantos", "cuantas", "cómo",
    "dame", "dime", "diga", "explica", "explicame", "explícame",
    "muestra", "muestrame", "muéstrame", "indica", "indícame",
    "puedo", "puede", "podría", "podria", "quiero", "necesito",
    "hacer", "hace", "hacen", "tiene", "tienen", "hay",
}


class RagRetriever:
    """Hybrid retriever over knowledge.document_chunks."""

    # Chunks whose title mentions any of these get full score.
    # Others are penalized by TITLE_PENALTY.
    RELEVANT_TITLE_KEYWORDS = [
        "ipm", "multidimensional", "pobreza", "conpes",
        "calidad de vida", "privacion", "privación",
        "ecv", "pobreza monetaria",
    ]
    TITLE_PENALTY = 0.6

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.host = host or POSTGRES_HOST
        self.port = port or POSTGRES_PORT
        self.dbname = dbname or POSTGRES_DB
        self.user = user or POSTGRES_USER
        self.password = password or POSTGRES_PASSWORD
        self._conn = None

    # ----------------------------------------------------------
    # Connection
    # ----------------------------------------------------------
    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                row_factory=dict_row,
            )
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def ping(self) -> bool:
        try:
            with self._connect().cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                return cur.fetchone()["ok"] == 1
        except Exception as e:
            logger.error("RAG ping falló: %s", e)
            return False

    # ----------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------
    def _embed_query(self, query: str) -> list[float]:
        import ollama
        prefixed = PREFIX_QUERY + query.strip()
        response = ollama.embeddings(model=EMBED_MODEL, prompt=prefixed)
        vector = response.get("embedding", [])
        if len(vector) != EXPECTED_EMBED_DIM:
            raise RuntimeError(
                f"Dimensión inesperada: {len(vector)} "
                f"(esperada {EXPECTED_EMBED_DIM})"
            )
        return vector

    # ----------------------------------------------------------
    # Lexical query builder
    # ----------------------------------------------------------
    @staticmethod
    def _build_lexical_query(query: str) -> str:
        tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
        meaningful = [
            t for t in tokens
            if t not in SPANISH_STOPWORDS and len(t) >= 3
        ]
        if not meaningful:
            meaningful = [t for t in tokens if len(t) >= 2]
        if not meaningful:
            return ""
        return " & ".join(meaningful)

    # ----------------------------------------------------------
    # Keyword extraction
    # ----------------------------------------------------------
    @staticmethod
    def _extract_query_keywords(query: str, min_len: int = 4) -> set[str]:
        tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
        return {
            t for t in tokens
            if t not in SPANISH_STOPWORDS and len(t) >= min_len
        }

    # ----------------------------------------------------------
    # Title relevance
    # ----------------------------------------------------------
    @classmethod
    def _title_relevance(cls, chunk: dict) -> float:
        """
        Return 1.0 if the chunk title mentions an IPM topic, else
        TITLE_PENALTY (0.6).
        """
        title = (chunk.get("title") or "").lower()
        for kw in cls.RELEVANT_TITLE_KEYWORDS:
            if kw in title:
                return 1.0
        return cls.TITLE_PENALTY

    # ----------------------------------------------------------
    # Boost by keywords + title penalty
    # ----------------------------------------------------------
    @classmethod
    def _boost_by_keywords(
        cls,
        results: list[dict],
        query_keywords: set[str],
        weight: float = 0.15,
    ) -> list[dict]:
        """
        Boost chunks by keyword matches and penalize off-topic titles.

        Final score = (base + weight * matches) * title_multiplier
        where base is the RRF (or semantic score) as a float.
        """
        if not query_keywords:
            return results

        scored = []
        for r in results:
            content_lower = (r.get("content") or "").lower()
            matches = sum(1 for kw in query_keywords if kw in content_lower)

            # Coerce base to float (Postgres returns SUM as Decimal)
            raw_base = r.get("rrf") or r.get("score") or 0.0
            try:
                base = float(raw_base)
            except (TypeError, ValueError):
                base = 0.0

            title_mult = cls._title_relevance(r)
            final = (base + (weight * matches)) * title_mult

            r_copy = dict(r)
            try:
                r_copy["rrf"] = float(r_copy["rrf"]) if r_copy.get("rrf") is not None else 0.0
            except (TypeError, ValueError):
                r_copy["rrf"] = 0.0
            try:
                r_copy["score"] = float(r_copy["score"]) if r_copy.get("score") is not None else 0.0
            except (TypeError, ValueError):
                r_copy["score"] = 0.0

            r_copy["_keyword_matches"] = matches
            r_copy["_title_mult"] = title_mult
            r_copy["_final_score"] = final
            scored.append(r_copy)

        scored.sort(key=lambda c: c["_final_score"], reverse=True)
        return scored

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        organization: Optional[str] = None,
        min_score: float = MIN_SCORE,
    ) -> list[dict]:
        if not query or not query.strip():
            return []

        lex_query = self._build_lexical_query(query)
        if not lex_query:
            logger.warning("Empty lexical query for: %r", query[:100])
            return []
        use_vector_fallback = (
            len(lex_query.split("&")) >= MIN_TOKENS_FOR_VECTOR_FALLBACK
        )

        query_keywords = self._extract_query_keywords(query, min_len=4)

        embedding = self._embed_query(query)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(SQL_HYBRID_SEARCH, {
                    "embedding": embedding_str,
                    "lex_query": lex_query,
                    "limit": top_k,
                    "min_score": min_score,
                    "organization": organization,
                    "use_vector_fallback": use_vector_fallback,
                    "top_semantic": TOP_K_SEMANTIC,
                    "top_lexical": TOP_K_LEXICAL,
                })
                rows = cur.fetchall()
            results = [dict(r) for r in rows]
        except Exception as exc:
            logger.error("Error en hybrid search: %s", exc)
            return []

        if query_keywords and results:
            results = self._boost_by_keywords(results, query_keywords)

        return results[:top_k]


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = RagRetriever()
    try:
        if not retriever.ping():
            print("❌ No se pudo conectar a la BD vectorial.")
            raise SystemExit(1)
        print("✅ Conexión OK a embedding_db")

        print("\n=== TEST _build_lexical_query ===")
        for q in [
            "¿Qué es el hacinamiento crítico?",
            "¿Cuáles son las dimensiones del IPM?",
            "¿Cómo se mide el analfabetismo?",
            "IPM",
        ]:
            lq = retriever._build_lexical_query(q)
            kws = retriever._extract_query_keywords(q)
            print(f"  {q!r}")
            print(f"    lex_query → {lq!r}")
            print(f"    keywords  → {kws}")

        print("\n=== SIN FILTRO ===")
        for q in [
            "¿Qué es el hacinamiento crítico?",
            "¿Cuáles son las dimensiones del IPM?",
            "¿Cómo se mide el analfabetismo?",
            "IPM",
        ]:
            print(f"\n❓ {q}")
            results = retriever.search(q, top_k=5)
            for i, r in enumerate(results, 1):
                lex = "✅" if r.get("lexical_match") else "❌"
                org = r.get("organization") or "?"
                kw = r.get("_keyword_matches", 0)
                tm = r.get("_title_mult", 1.0)
                final = r.get("_final_score", 0)
                print(f"  [{i}] lex={lex} kw={kw} title×{tm:.1f} final={final:.4f}")
                print(f"      {r['title'][:60]} | {org} | p.{r['page_number']}")

    finally:
        retriever.close()