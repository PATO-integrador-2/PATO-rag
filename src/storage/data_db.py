"""
Connection to the clean data database (clean_data_db) using psycopg v3.
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


DATA_GROUPS = {
    "dane": {
        "label": "DANE - Colombia",
        "views": [
            "vw_ipm_incidence",
            "vw_ipm_contribution_dimension",
            "vw_ipm_deprivation_proportion",
            "vw_ipm_deprivations_variable",
            "vw_ipm_incidence_household_head_sex",
            "vw_ipm_incidence_person_sex",
            "vw_ipm_household_departmental",
        ],
        "ref_tables": [
            "ref_domain", "ref_department", "ref_dimension",
            "ref_variable", "ref_region", "ref_sex",
        ],
        "keywords": [
            "colombia", "colombiano", "dane", "departamento", "municipio",
            "antioquia", "huila", "bogota", "caribe", "andina",
            "pacifico", "orinoquia", "amazonia", "nacional",
            "cundinamarca", "valle", "atlantico", "bolivar", "santander",
            "narino", "cauca", "choco", "caldas", "risaralda", "quindio",
            "tolima", "cesar", "magdalena", "guajira", "cordoba", "sucre",
            "boyaca", "casanare", "vichada", "guainia", "vaupes",
            "guaviare", "putumayo", "caqueta", "arauca",
            "san andres", "providencia",
        ],
    },
    "cepal": {
        "label": "CEPAL - Latinoamérica",
        "views": [
            "vw_latam_ipm_age_group",
            "vw_latam_ipm_deprivation_contribution",
            "vw_latam_ipm_national_population",
        ],
        "ref_tables": [
            "ref_country", "ref_age_group", "ref_geographic_area",
            "ref_measurement_type", "ref_deprivation",
        ],
        "keywords": [
            "latinoamerica", "cepal", "america latina",
            "region", "regional",
            "argentina", "brasil", "brazil", "mexico", "chile",
            "peru", "ecuador", "bolivia", "paraguay", "uruguay",
            "costa rica", "panama", "honduras", "guatemala",
            "el salvador", "nicaragua", "cuba", "republica dominicana",
            "haiti", "venezuela",
            "rural", "urbano", "grupo de edad", "grupos de edad",
            "pais", "paises",
        ],
    },
}


class DataDatabase:
    """PostgreSQL client for the clean data database (psycopg v3)."""

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
        self.host = host or os.getenv("DATA_DB_HOST", "localhost")
        self.port = int(port or os.getenv("DATA_DB_PORT", "5432"))
        self.dbname = dbname or os.getenv("DATA_DB_NAME", "clean_data_db")
        self.user = user or os.getenv("DATA_DB_USER", "postgres")
        self.password = password or os.getenv("DATA_DB_PASSWORD", "")

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
                f"No se pudo crear el pool: {e}"
            ) from e

        self._schema_cache: dict = {}

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

    # ---------- Introspection ----------
    def list_views(self) -> list[str]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.views
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                return [r["table_name"] for r in cur.fetchall()]

    def get_group_schema(self, group: str, force_refresh: bool = False) -> dict:
        if group not in DATA_GROUPS:
            raise ValueError(f"Grupo desconocido: {group}")
        if not force_refresh and group in self._schema_cache:
            return self._schema_cache[group]

        info = DATA_GROUPS[group]
        schema = {"label": info["label"], "views": {}, "ref_tables": {}}

        available_views = set(self.list_views())
        allowed_views = [v for v in info["views"] if v in available_views]

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if allowed_views:
                    cur.execute("""
                        SELECT table_name, column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = ANY(%s)
                        ORDER BY table_name, ordinal_position
                    """, (allowed_views,))
                    for row in cur.fetchall():
                        schema["views"].setdefault(row["table_name"], []).append({
                            "column_name": row["column_name"],
                            "data_type": row["data_type"],
                        })

                if info["ref_tables"]:
                    cur.execute("""
                        SELECT table_name, column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = ANY(%s)
                        ORDER BY table_name, ordinal_position
                    """, (info["ref_tables"],))
                    for row in cur.fetchall():
                        schema["ref_tables"].setdefault(row["table_name"], []).append({
                            "column_name": row["column_name"],
                            "data_type": row["data_type"],
                        })

        self._schema_cache[group] = schema
        return schema

    # ---------- Execution ----------
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


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Test DataDatabase (psycopg v3)")
    try:
        db = DataDatabase()
        if db.ping():
            print("✅ Conexión OK a clean_data_db")
            for g in DATA_GROUPS:
                views = db.get_group_schema(g)["views"]
                print(f"  {g}: {len(views)} vistas")
        else:
            print("❌ Ping falló")
        db.close()
    except Exception as e:
        print(f"❌ Error: {e}")