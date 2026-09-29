"""
Conexión a la base de datos limpios (clean_data_db).

Esta base contiene dos grupos independientes de datos:

    GRUPO DANE (Colombia):
        - ipm_incidence, ipm_contribution_dimension, ipm_deprivation_proportion,
        ipm_deprivations_variable, ipm_incidence_household_head_sex,
        ipm_incidence_person_sex, ipm_household_departmental
        - ref_domain, ref_department, ref_dimension, ref_variable,
        ref_region, ref_sex
        - Cobertura: departamentos y dominios geográficos de Colombia

    GRUPO CEPAL (Latinoamérica):
        - latam_ipm_age_group, latam_ipm_deprivation_contribution,
        latam_ipm_national_population
        - ref_country, ref_age_group, ref_geographic_area,
        ref_measurement_type, ref_deprivation
        - Cobertura: países de Latinoamérica

Los dos grupos son completamente independientes. NO se deben hacer JOINs
entre tablas de distintos grupos.

Se usa exclusivamente para responder preguntas de datos específicos
vía Text-to-SQL. Se mantiene separada de:
    - embedding_db (RAG de PDFs)
    - user_management_db (conversaciones)
"""
import os
import contextlib
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# Definición de grupos
# Cada grupo tiene:
#   - label: nombre legible
#   - views: vistas expuestas al LLM (las que el generador SQL puede usar)
#   - ref_tables: tablas de referencia (catálogos)
#   - keywords: palabras clave para el enrutador

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
            "ref_domain",
            "ref_department",
            "ref_dimension",
            "ref_variable",
            "ref_region",
            "ref_sex",
        ],
        "keywords": [
            "colombia", "colombiano", "dane", "departamento", "municipio",
            "antioquia", "huila", "bogota", "bogotá", "caribe", "andina",
            "pacifico", "pacífico", "orinoquia", "amazonia", "amazonía",
            "nacional", "cundinamarca", "valle", "atlantico", "atlántico",
            "bolivar", "bolívar", "santander", "narino", "nariño", "cauca",
            "choco", "chocó", "caldas", "risaralda", "quindio", "quindío",
            "tolima", "cesar", "magdalena", "guajira", "cordoba", "córdoba",
            "sucre", "bolivar", "boyaca", "boyacá", "casanare", "vichada",
            "guainia", "guainía", "vaupes", "vaupés", "guaviare", "putumayo",
            "caqueta", "caquetá", "arauca", "san andres", "san andrés",
            "providencia",
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
            "ref_country",
            "ref_age_group",
            "ref_geographic_area",
            "ref_measurement_type",
            "ref_deprivation",
        ],
        "keywords": [
            "latinoamerica", "latinoamérica", "cepal", "america latina",
            "américa latina", "region", "región", "regional",
            "argentina", "brasil", "brazil", "mexico", "méxico", "chile",
            "peru", "perú", "ecuador", "bolivia", "paraguay", "uruguay",
            "costa rica", "panama", "panamá", "honduras", "guatemala",
            "el salvador", "nicaragua", "cuba", "republica dominicana",
            "república dominicana", "haiti", "haití", "venezuela",
            "rural", "urbano", "grupo de edad", "grupos de edad",
            "pais", "país", "paises", "países",
        ],
    },
}

# Cliente de PostgreSQL para la base de datos limpios.
class DataDatabase:

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
        self.host = host or os.getenv("DATA_DB_HOST")
        self.port = int(port or os.getenv("DATA_DB_PORT"))
        self.dbname = dbname or os.getenv("DATA_DB_NAME")
        self.user = user or os.getenv("DATA_DB_USER")
        self.password = password or os.getenv("DATA_DB_PASSWORD")

        try:
            self._pool = pg_pool.SimpleConnectionPool(
                minconn=minconn, maxconn=maxconn,
                host=self.host, port=self.port, dbname=self.dbname,
                user=self.user, password=self.password,
            )
        except psycopg2.Error as e:
            raise ConnectionError(
                f"No se pudo crear el pool a {self.host}:{self.port}/{self.dbname}: {e}"
            ) from e

    # Pool y cursores
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

    # Introspección
    # Lista las vistas disponibles en el esquema public.
    def list_views(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            return [r["table_name"] for r in cur.fetchall()]

    #Lista las tablas base en el esquema public.
    def list_tables(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            return [r["table_name"] for r in cur.fetchall()]

    # Devuelve las columnas y tipos de una vista específica.
    def get_view_columns(self, view_name: str) -> list[dict]:
        with self.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                ORDER BY ordinal_position
            """, (view_name,))
            return cur.fetchall()

    # Devuelve las vistas del grupo indicado, verificando que existan realmente en la base de datos.
    def get_views_by_group(self, group: str) -> list[str]:
        
        if group not in DATA_GROUPS:
            raise ValueError(f"Grupo desconocido: {group}")
        existing = set(self.list_views())
        return [v for v in DATA_GROUPS[group]["views"] if v in existing]

    # Devuelve el esquema completo de un grupo:
    def get_group_schema(self, group: str) -> dict:

        if group not in DATA_GROUPS:
            raise ValueError(f"Grupo desconocido: {group}")

        info = DATA_GROUPS[group]
        schema = {
            "label": info["label"],
            "views": {},
            "ref_tables": {},
        }

        for view in self.get_views_by_group(group):
            schema["views"][view] = self.get_view_columns(view)

        # Las ref tables también pueden ser vistas o tablas base; se intentan ambas
        for ref in info["ref_tables"]:
            cols = self.get_view_columns(ref)
            if not cols:
                # Intentar como tabla base
                with self.cursor() as cur:
                    cur.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = %s
                        ORDER BY ordinal_position
                    """, (ref,))
                    cols = cur.fetchall()
            if cols:
                schema["ref_tables"][ref] = cols

        return schema

    
    # Ejecución de consultas (solo SELECT)
    # Ejecuta un SELECT y devuelve las filas.
    # Aplica timeout a nivel de transacción (compatible con PgBouncer).
    def execute_select(self, sql: str, timeout_seconds: int = 15) -> list[dict]:

        with self.cursor() as cur:
            try:
                cur.execute(f"SET LOCAL statement_timeout = {timeout_seconds * 1000}")
                cur.execute(sql)
                return cur.fetchall()
            except Exception as e:
                raise RuntimeError(f"Error ejecutando SELECT: {e}") from e

# Prueba directa

if __name__ == "__main__":
    print("  TEST: Conexión a clean_data_db")
    print("-" * 60)

    try:
        db = DataDatabase()
    except ConnectionError as e:
        print(f"Error de conexión: {e}")
        raise SystemExit(1)

    if not db.ping():
        print(" El ping falló")
        raise SystemExit(1)

    print(f" Conexión OK a {db.host}:{db.port}/{db.dbname}")
    print(f" Usuario: {db.user}")

    # Vistas por grupo
    for group in DATA_GROUPS:
        label = DATA_GROUPS[group]["label"]
        views = db.get_views_by_group(group)
        print(f"\n Grupo '{group}' ({label}) — {len(views)} vistas:")
        for v in views:
            print(f"  - {v}")

    # Esquema resumido de cada grupo
    print("\n" + "-" * 60)
    print("  ESQUEMA POR GRUPO")
    print("-" * 60)

    for group in DATA_GROUPS:
        print(f"\n--- Grupo: {group} ---")
        schema = db.get_group_schema(group)

        print(f"\n  Vistas ({len(schema['views'])}):")
        for view_name, cols in schema["views"].items():
            col_names = ", ".join(c["column_name"] for c in cols)
            print(f"    {view_name}: [{col_names}]")

        print(f"\n  Ref tables ({len(schema['ref_tables'])}):")
        for ref_name, cols in schema["ref_tables"].items():
            col_names = ", ".join(c["column_name"] for c in cols)
            print(f"    {ref_name}: [{col_names}]")

    # SELECT de prueba en cada grupo
    print("\n" + "-" * 60)
    print("  SELECT DE PRUEBA")
    print("-" * 60)

    print("\n DANE — vw_ipm_incidence LIMIT 3:")
    try:
        rows = db.execute_select("SELECT * FROM vw_ipm_incidence LIMIT 3")
        for r in rows:
            print(f"  {dict(r)}")
    except Exception as e:
        print(f" Error:  {e}")

    print("\nCEPAL — vw_latam_ipm_national_population LIMIT 3:")
    try:
        rows = db.execute_select("SELECT * FROM vw_latam_ipm_national_population LIMIT 3")
        for r in rows:
            print(f"  {dict(r)}")
    except Exception as e:
        print(f" Error: {e}")

    db.close()
    print("  TEST COMPLETADO")
