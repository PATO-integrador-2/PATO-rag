"""
Verifica dónde están las tablas de conversaciones y usuarios.
Ejecutar: python -m src.check_conversation_tables
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import psycopg2

CONN = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def check():
    print("=" * 60)
    print("  DIAGNÓSTICO DE TABLAS DE CONVERSACIÓN")
    print("=" * 60)

    conn = psycopg2.connect(**CONN)
    with conn.cursor() as cur:
        # 1. Base de datos actual
        cur.execute("SELECT current_database(), current_user")
        row = cur.fetchone()
        print(f"\n📦 Base de datos: {row[0]}")
        print(f"👤 Usuario: {row[1]}")

        # 2. search_path actual
        cur.execute("SHOW search_path")
        print(f"🔍 search_path: {cur.fetchone()[0]}")

        # 3. Esquemas disponibles
        cur.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT LIKE 'pg_%'
              AND schema_name != 'information_schema'
            ORDER BY schema_name
        """)
        schemas = [r[0] for r in cur.fetchall()]
        print(f"\n📂 Esquemas disponibles: {schemas}")

        # 4. Buscar las tablas en todos los esquemas
        print("\n🔎 Buscando tablas relacionadas con usuarios y conversaciones:")
        cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_name IN (
                'ai_conversation',
                'ai_conversation_message',
                'app_user',
                'user_session',
                'user_role',
                'role',
                'permission'
            )
            ORDER BY table_schema, table_name
        """)
        rows = cur.fetchall()
        if not rows:
            print("  ⚠️  NO se encontró NINGUNA tabla relacionada.")
            print("  → Pide al área de BD que ejecute el DDL completo.")
        else:
            current_schema = None
            for schema, table in rows:
                if schema != current_schema:
                    print(f"\n  Esquema '{schema}':")
                    current_schema = schema
                print(f"    - {table}")

        # 5. Verificar si alguna de las tablas está accesible desde search_path
        print("\n🎯 Test: ¿Puedo acceder a 'ai_conversation' sin calificar esquema?")
        try:
            cur.execute("SELECT COUNT(*) FROM ai_conversation")
            count = cur.fetchone()[0]
            print(f"  ✅ Sí, se puede. Hay {count} filas.")
        except Exception as e:
            print(f"  ❌ No: {e}")

        # 6. Contar filas en app_user para tener el UUID de prueba
        print("\n👥 Usuarios disponibles en app_user:")
        try:
            cur.execute("SELECT id, username, email FROM app_user LIMIT 5")
            for r in cur.fetchall():
                print(f"  [{r[0]}] {r[1]} ({r[2]})")
        except Exception as e:
            print(f"  ❌ No se puede leer app_user: {e}")

    conn.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    check()