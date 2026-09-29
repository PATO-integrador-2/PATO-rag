"""
Conexión a la base de datos de usuarios (user_management_db).

Esta base contiene:
    - app_user: usuarios del sistema
    - user_session: sesiones
    - ai_conversation: conversaciones con el chatbot
    - ai_conversation_message: mensajes individuales
    - roles, permisos, etc.

Se mantiene separada de la base del RAG (embedding_db) por diseño:
son dominios distintos administrados por equipos distintos.
"""
import os
import contextlib
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()


class UserDatabase:
    """
    Cliente de PostgreSQL para la base de usuarios (conversaciones).
    """

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
        self.host = host or os.getenv("USERS_DB_HOST", "localhost")
        self.port = int(port or os.getenv("USERS_DB_PORT", "5432"))
        self.dbname = dbname or os.getenv("USERS_DB_NAME", "user_management_db")
        self.user = user or os.getenv("USERS_DB_USER", "postgres")
        self.password = password or os.getenv("USERS_DB_PASSWORD", "postgres")

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

    # ============================================================
    # Gestión del pool
    # ============================================================
    @contextlib.contextmanager
    def connection(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @contextlib.contextmanager
    def cursor(self, dict_cursor: bool = True):
        with self.connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
            finally:
                cur.close()

    def close(self):
        if self._pool:
            self._pool.closeall()

    def ping(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                return cur.fetchone()["ok"] == 1
        except Exception:
            return False

    # ============================================================
    # Conversaciones
    # ============================================================
    def conversation_exists(self, conversation_id: str) -> bool:
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
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_conversation (user_id, session_id, name)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (user_id, session_id, name[:150]),
            )
            return str(cur.fetchone()["id"])

    def get_next_message_position(self, conversation_id: str) -> int:
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
            return list(reversed(rows))

    def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, created_at
                FROM ai_conversation
                WHERE user_id = %s AND is_deleted = false
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return cur.fetchall()

    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> list[dict]:
        with self.cursor() as cur:
            sql = """
                SELECT id, role, content, position, metadata, created_at
                FROM ai_conversation_message
                WHERE conversation_id = %s AND is_deleted = false
                ORDER BY position ASC
            """
            params = [conversation_id]
            if limit:
                sql += " LIMIT %s"
                params.append(limit)
            cur.execute(sql, params)
            return cur.fetchall()


# ============================================================
# Prueba rápida
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: Conexión a user_management_db")
    print("=" * 60)

    try:
        db = UserDatabase()
    except ConnectionError as e:
        print(f"❌ {e}")
        raise SystemExit(1)

    if db.ping():
        print(f"✅ Conexión OK a {db.host}:{db.port}/{db.dbname} (user: {db.user})")
    else:
        print("❌ El ping falló")
        raise SystemExit(1)

    with db.cursor() as cur:
        # Verificar tablas requeridas
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN (
                  'ai_conversation',
                  'ai_conversation_message',
                  'app_user',
                  'user_session'
              )
            ORDER BY table_name
        """)
        tables = [r["table_name"] for r in cur.fetchall()]
        print(f"\n📋 Tablas encontradas: {tables}")

        required = ["ai_conversation", "ai_conversation_message", "app_user"]
        for t in required:
            if t in tables:
                print(f"  ✅ {t}")
            else:
                print(f"  ❌ FALTA: {t}")

        # Contar usuarios
        try:
            cur.execute("SELECT COUNT(*) AS n FROM app_user")
            n_users = cur.fetchone()["n"]
            print(f"\n👥 Usuarios en app_user: {n_users}")
            if n_users > 0:
                cur.execute("SELECT id, username, email FROM app_user LIMIT 3")
                for r in cur.fetchall():
                    print(f"  [{r['id']}] {r['username']} ({r['email']})")
        except Exception as e:
            print(f"⚠️  No se pudo leer app_user: {e}")

    db.close()
    print("\n" + "=" * 60)
    print("  TEST COMPLETADO")
    print("=" * 60)