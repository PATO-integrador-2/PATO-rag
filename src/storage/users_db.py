"""
Connection to the users database (user_management_db).

Contains tables:
    - ai_conversation
    - ai_conversation_message

Uses psycopg v3 with a connection pool.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


USERS_DB_HOST = os.getenv("USERS_DB_HOST")
USERS_DB_PORT = int(os.getenv("USERS_DB_PORT"))
USERS_DB_NAME = os.getenv("USERS_DB_NAME")
USERS_DB_USER = os.getenv("USERS_DB_USER")
USERS_DB_PASSWORD = os.getenv("USERS_DB_PASSWORD")
USERS_DB_SCHEMA = os.getenv("USERS_DB_SCHEMA", "public")


class UsersDatabase:
    """PostgreSQL client for user_management_db."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        minconn: int = 1,
        maxconn: int = 5,
    ):
        self.host = host or USERS_DB_HOST
        self.port = port or USERS_DB_PORT
        self.dbname = dbname or USERS_DB_NAME
        self.user = user or USERS_DB_USER
        self.password = password or USERS_DB_PASSWORD
        self.schema = schema or USERS_DB_SCHEMA

        conninfo = (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )

        try:
            self._pool = ConnectionPool(
                conninfo=conninfo,
                min_size=minconn,
                max_size=maxconn,
                kwargs={"row_factory": dict_row},
                open=True,
            )
        except psycopg.Error as e:
            raise ConnectionError(
                f"No se pudo crear el pool a {self.host}:{self.port}/{self.dbname}: {e}"
            ) from e

    def close(self):
        if getattr(self, "_pool", None):
            self._pool.close()

    def ping(self) -> bool:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    return cur.fetchone()["ok"] == 1
        except Exception:
            return False

    def execute_select(
        self,
        sql: str,
        params: Optional[dict] = None,
        timeout_seconds: int = 15,
    ) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {timeout_seconds * 1000}")
                cur.execute(sql, params or {})
                return cur.fetchall()

    def execute_write(
        self,
        sql: str,
        params: Optional[dict] = None,
        timeout_seconds: int = 15,
        returning: bool = False,
    ) -> Optional[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {timeout_seconds * 1000}")
                cur.execute(sql, params or {})
                if returning:
                    return cur.fetchone()
                return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Test UsersDatabase")
    try:
        db = UsersDatabase()
        if db.ping():
            print(" Conexión OK a user_management_db")
            rows = db.execute_select("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %(schema)s
                    AND table_name LIKE 'ai_%%'
                ORDER BY table_name
            """, {"schema": db.schema})
            for r in rows:
                print(f"  - {r['table_name']}")
        else:
            print("X Ping falló")
        db.close()
    except Exception as e:
        print(f"X Error: {e}")