"""
Verificación rápida del estado del proyecto RAG.
Chequea: archivos, dependencias, taxonomía, PDFs, Ollama y PostgreSQL.
Ejecutar: python -m src.verify_setup
"""
import os
import sys
from pathlib import Path

# ============================================================
# Asegurar que la raíz del proyecto esté en sys.path
# ============================================================
# Si este archivo está en <raiz>/src/verify_setup.py,
# entonces la raíz es el padre de src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================================

import importlib
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Utilidades de reporte
# ============================================================
OK = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "


def report(status: str, msg: str):
    print(f"{status} {msg}")


# ============================================================
# 1. Archivos y carpetas
# ============================================================
def check_files():
    print("\n=== 1. ARCHIVOS Y CARPETAS ===")
    required = [
        "config/taxonomy.yaml",
        "src/__init__.py",
        "src/text_normalizer.py",
        "src/taxonomy_loader.py",
        "src/pdf_extractor.py",
        "src/chunker.py",
        "src/db.py",
        ".env",
        "requirements.txt",
    ]
    for path in required:
        if Path(path).exists():
            report(OK, f"Existe: {path}")
        else:
            report(FAIL, f"FALTA: {path}")

    pdf_dir = os.getenv("PDFS_DIR", "data/pdfs")
    if Path(pdf_dir).exists():
        report(OK, f"Carpeta de PDFs: {pdf_dir}")
    else:
        report(FAIL, f"FALTA carpeta de PDFs: {pdf_dir}")


# ============================================================
# 2. Dependencias Python
# ============================================================
def check_dependencies():
    print("\n=== 2. DEPENDENCIAS PYTHON ===")
    deps = {
        "pymupdf": "PyMuPDF",
        "ollama": "ollama",
        "psycopg2": "psycopg2",
        "pgvector": "pgvector",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
        "tqdm": "tqdm",
    }
    for module_name, display_name in deps.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "?")
            report(OK, f"{display_name} ({module_name}) v{version}")
        except ImportError:
            report(FAIL, f"FALTA: {display_name} ({module_name})")


# ============================================================
# 3. Taxonomía
# ============================================================
def check_taxonomy():
    print("\n=== 3. TAXONOMÍA ===")
    try:
        from ingestion.taxonomy_loader import load_taxonomy, list_domains, list_dimensions
        tax = load_taxonomy()
        domains = list_domains(tax)
        report(OK, f"Dominios cargados: {domains}")
        for d in domains:
            dims = list_dimensions(tax, d)
            report(INFO, f"  {d}: {len(dims)} dimensiones")
    except Exception as e:
        report(FAIL, f"Error cargando taxonomía: {e}")


# ============================================================
# 4. Módulos del proyecto
# ============================================================
def check_modules():
    print("\n=== 4. MÓDULOS DEL PROYECTO ===")
    modules = [
        "src.text_normalizer",
        "src.taxonomy_loader",
        "src.pdf_extractor",
        "src.chunker",
        "src.db",
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            report(OK, f"Importable: {mod_name}")
        except Exception as e:
            report(FAIL, f"No importable {mod_name}: {e}")


# ============================================================
# 5. PDFs
# ============================================================
def check_pdfs():
    print("\n=== 5. PDFs ===")
    try:
        from ingestion.pdf_extractor import walk_pdfs
        pdf_dir = os.getenv("PDFS_DIR", "data/pdfs")
        pdfs = walk_pdfs(pdf_dir)
        report(OK, f"PDFs encontrados: {len(pdfs)}")
        if pdfs:
            report(INFO, f"  Primer PDF: {pdfs[0]}")
    except Exception as e:
        report(FAIL, f"Error listando PDFs: {e}")


# ============================================================
# 6. Ollama
# ============================================================
def check_ollama():
    print("\n=== 6. OLLAMA ===")
    try:
        import ollama
        models_response = ollama.list()
        if isinstance(models_response, dict):
            model_names = [m.get("name", "") for m in models_response.get("models", [])]
        else:
            model_names = [getattr(m, "model", str(m)) for m in models_response]

        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
        llm_model = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")

        for name, label in [(embed_model, "embed"), (llm_model, "llm")]:
            base = name.split(":")[0]
            found = any(base in m for m in model_names)
            if found:
                report(OK, f"Modelo {label} disponible: {name}")
            else:
                report(FAIL, f"Modelo {label} NO descargado: {name}")
                report(INFO, f"  Ejecuta: ollama pull {name}")

        # Prueba real de embedding
        try:
            test_embed = ollama.embeddings(model=embed_model, prompt="prueba")
            dim = len(test_embed.get("embedding", []))
            expected_dim = int(os.getenv("EMBEDDING_DIM", "768"))
            if dim == expected_dim:
                report(OK, f"Embedding de prueba OK (dim={dim})")
            else:
                report(WARN, f"Embedding dim={dim}, se esperaba {expected_dim}")
        except Exception as e:
            report(FAIL, f"Embedding de prueba falló: {e}")

    except ImportError:
        report(FAIL, "Librería ollama no instalada. Ejecuta: pip install ollama")
    except Exception as e:
        report(FAIL, f"Error conectando a Ollama: {e}")
        report(INFO, "  ¿Está Ollama corriendo? Verifica con: ollama list")


# ============================================================
# 7. PostgreSQL + pgvector
# ============================================================
def check_postgres():
    print("\n=== 7. POSTGRESQL + PGVECTOR ===")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "rag_ipm")

    report(INFO, f"Configuración: {host}:{port}/{dbname}")

    try:
        from storage.database import Database
    except ImportError as e:
        report(FAIL, f"No se pudo importar src.db: {e}")
        return

    try:
        db = Database()
    except ConnectionError as e:
        report(FAIL, f"No se pudo conectar: {e}")
        report(INFO, "  Verifica que Postgres esté corriendo y las credenciales del .env.")
        return

    # Ping
    if db.ping():
        report(OK, "Conexión OK (ping exitoso)")
    else:
        report(FAIL, "El ping falló")
        db.close()
        return

    # pgvector
    if db.check_pgvector():
        report(OK, "Extensión pgvector instalada")
    else:
        report(FAIL, "Extensión pgvector NO instalada")
        report(INFO, "  Ejecuta en Postgres: CREATE EXTENSION vector;")

    # Tablas
    tables = db.list_tables()
    report(INFO, f"Tablas encontradas: {tables}")
    expected = ["sources", "documents", "document_chunks"]
    for t in expected:
        if t in tables:
            report(OK, f"Tabla existe: {t}")
        else:
            report(FAIL, f"FALTA tabla: {t}")
            report(INFO, "  Pide al área de BD que ejecute el script de esquema.")

    # Conversaciones (opcional)
    if "conversations" in tables:
        report(OK, "Tabla existe: conversations")
    else:
        report(WARN, "Tabla conversations no existe (opcional, para persistencia del chat)")

    db.close()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  VERIFICACIÓN DEL SETUP — PROYECTO RAG IPM")
    print("=" * 60)

    check_files()
    check_dependencies()
    check_taxonomy()
    check_modules()
    check_pdfs()
    check_ollama()
    check_postgres()

    print("\n" + "=" * 60)
    print("  FIN DE LA VERIFICACIÓN")
    print("=" * 60)


if __name__ == "__main__":
    main()