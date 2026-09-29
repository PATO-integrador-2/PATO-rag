"""
Módulo de conexión a PostgreSQL con soporte para pgvector.

Proporciona una clase Database con un pool de conexiones y
métodos helper para las operaciones del RAG (insertar documentos,
chunks, buscar por similitud vectorial, etc.).
"""
import os
import contextlib
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor, execute_values, Json
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Cargar variables de entorno al importar el módulo
load_dotenv()

#Cliente de PostgreSQL con pool de conexiones.
#
class Database:

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        minconn: int = 1,
        maxconn: int = 5,
    ):
        self.host = host or os.getenv("POSTGRES_HOST")
        self.port = int(port or os.getenv("POSTGRES_PORT"))
        self.dbname = dbname or os.getenv("POSTGRES_DB")
        self.user = user or os.getenv("POSTGRES_USER")
        self.password = password or os.getenv("POSTGRES_PASSWORD")

        try:
            self._pool = pg_pool.SimpleConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
            )
        except psycopg2.Error as e:
            raise ConnectionError(
                f"No se pudo crear el pool de conexiones a "
                f"{self.host}:{self.port}/{self.dbname}: {e}"
            ) from e

    
    # Gestión del pool: Context manager para obtener y devolver una conexión al pool.
    # Registra el tipo vector en cada conexión para que pgvector funcione.
    
    @contextlib.contextmanager
    def connection(self):
        conn = self._pool.getconn()
        try:
            # Registrar el adaptador de pgvector en esta conexión
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # Context manager para obtener un cursor.
    @contextlib.contextmanager
    def cursor(self, dict_cursor: bool = True):

        with self.connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
            finally:
                cur.close()
    # Cierra todas las conexiones del pool.
    def close(self):
        if self._pool:
            self._pool.closeall()

    
    # Health check
    # Verifica que la conexión funcione.
    
    def ping(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                result = cur.fetchone()
                return result["ok"] == 1
        except Exception:
            return False

    # Verifica que la extensión pgvector esté instalada.
    def check_pgvector(self) -> bool:
        with self.cursor() as cur:
            cur.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            )
            return cur.fetchone() is not None

    # Devuelve la lista de tablas en el esquema configurado.
    def list_tables(self) -> list[str]:
        schema = os.getenv("POSTGRES_SCHEMA", "public")
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
                """,
                (schema,),
            )
            return [row["table_name"] for row in cur.fetchall()]

    
    # Operaciones del RAG (se usarán en indexer.py y retriever.py)
    # Devuelve el id de una fuente (DANE, CEPAL, OPHI...). La crea si no existe.
    def get_or_create_source(self, name: str, website_url: Optional[str] = None) -> int:

        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (name, website_url)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (name, website_url),
            )
            return cur.fetchone()["id"]

    # Devuelve el UUID de un documento. Si ya existe (mismo source_id + title), lo reutiliza.
    
    def get_or_create_document(
        self,
        source_id: int,
        title: str,
        publication_year: Optional[int] = None,
        file_url: Optional[str] = None,
    ) -> str:
        with self.cursor() as cur:
            # Buscar primero
            cur.execute(
                """
                SELECT id FROM documents
                WHERE source_id = %s AND title = %s
                """,
                (source_id, title),
            )
            row = cur.fetchone()
            if row:
                return row["id"]

            # Crear nuevo
            cur.execute(
                """
                INSERT INTO documents
                    (source_id, title, publication_year, file_url)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (source_id, title, publication_year, file_url),
            )
            return cur.fetchone()["id"]

    # Borra los chunks de un documento (útil para reindexar).
    def delete_chunks_for_document(self, document_id: str) -> int:
        with self.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )
            return cur.rowcount

    # Inserta chunks en lote.
    def insert_chunks(self, chunks: list[dict]) -> int:
        
        if not chunks:
            return 0

        rows = [
            (
                c["document_id"],
                c["chunk_index"],
                c["content"],
                c["embedding"],
                c.get("page_number"),
                c.get("domain"),
                c.get("dimension"),
                c.get("indicator"),
                Json(c.get("metadata") or {}),
            )
            for c in chunks
        ]

        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, content, embedding,
                    page_number, domain, dimension, indicator, metadata)
                VALUES %s
                ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    page_number = EXCLUDED.page_number,
                    domain = EXCLUDED.domain,
                    dimension = EXCLUDED.dimension,
                    indicator = EXCLUDED.indicator,
                    metadata = EXCLUDED.metadata
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                page_size=100,
            )
            return cur.rowcount

    # Busca los chunks más similares al embedding dado, con filtros opcionales.
    
    def search_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        domain: Optional[str] = None,
        dimension: Optional[str] = None,
        indicator: Optional[str] = None,
    ) -> list[dict]:
        # Construir el WHERE dinámicamente según los filtros
        filters = []
        params: list[Any] = [query_embedding]

        if domain:
            filters.append("dc.domain = %s")
            params.append(domain)
        if dimension:
            filters.append("dc.dimension = %s")
            params.append(dimension)
        if indicator:
            filters.append("dc.indicator = %s")
            params.append(indicator)

        where_clause = "WHERE " + " AND ".join(filters) if filters else ""
        params.append(top_k)

        sql = f"""
            SELECT
                dc.content,
                dc.page_number,
                dc.domain,
                dc.dimension,
                dc.indicator,
                dc.metadata,
                d.title AS document_title,
                s.name AS source_name,
                1 - (dc.embedding <=> %s::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            JOIN sources s ON s.id = d.source_id
            {where_clause}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """

        # Reordenar params: el vector va dos veces (SELECT y ORDER BY)
        final_params = [query_embedding] + params[1:-1] + [query_embedding, params[-1]]

        with self.cursor() as cur:
            cur.execute(sql, final_params)
            return cur.fetchall()


    # Conversaciones (persistencia del chat) Guarda un mensaje (user o assistant) en el historial.
    
    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self.cursor(dict_cursor=False) as cur:
            cur.execute(
                """
                INSERT INTO conversations (session_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (session_id, role, content),
            )

    # Devuelve los últimos N mensajes de una sesión, en orden cronológico.
    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, created_at
                FROM conversations
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
            return list(reversed(rows))


# Prueba rápida al ejecutar el módulo directamente

    
    # Conversaciones (tablas ai_conversation y ai_conversation_message)

    def conversation_exists(self, conversation_id: str) -> bool:
        """Verifica si una conversación existe y no está eliminada."""
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM ai_conversation
                WHERE id = %s AND is_deleted = false
                """,
                (conversation_id,),
            )
            return cur.fetchone() is not None

    def create_conversation(
        self,
        user_id: str,
        session_id: Optional[str],
        name: str,
    ) -> str:
        """
        Crea una fila en ai_conversation.
        Devuelve el UUID de la conversación creada.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_conversation (user_id, session_id, name)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (user_id, session_id, name[:150]),  # límite de 150 chars
            )
            return str(cur.fetchone()["id"])

    def get_next_message_position(self, conversation_id: str) -> int:
        """
        Devuelve la siguiente posición disponible dentro de la conversación.
        Si no hay mensajes, devuelve 0.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(position), -1) + 1 AS next_pos
                FROM ai_conversation_message
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            )
            return cur.fetchone()["next_pos"]

    def save_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        position: int,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Guarda un mensaje en ai_conversation_message.
        Devuelve el UUID del mensaje insertado.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_conversation_message
                    (conversation_id, role, content, position, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    conversation_id,
                    role,
                    content,
                    position,
                    Json(metadata) if metadata else None,
                ),
            )
            return str(cur.fetchone()["id"])

    def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: int = 10,
    ) -> list[dict]:
        """
        Devuelve los últimos N mensajes de la conversación, en orden
        cronológico, para usar como contexto.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, position
                FROM ai_conversation_message
                WHERE conversation_id = %s AND is_deleted = false
                ORDER BY position DESC
                LIMIT %s
                """,
                (conversation_id, max_messages),
            )
            rows = cur.fetchall()
            # Revertir para orden cronológico
            return list(reversed(rows))

if __name__ == "__main__":
    print("Probando conexión a PostgreSQL...")
    try:
        db = Database()
    except ConnectionError as e:
        print(f"Error de conexión: {e}")
        raise SystemExit(1)

    if db.ping():
        print(f"Conexión OK a {db.host}:{db.port}/{db.dbname}")
    else:
        print("El ping falló")

    if db.check_pgvector():
        print("Extensión pgvector instalada")
    else:
        print("Extensión pgvector NO instalada")

    tables = db.list_tables()
    print(f"Tablas encontradas: {tables}")

    db.close()
    print("Pool cerrado.")