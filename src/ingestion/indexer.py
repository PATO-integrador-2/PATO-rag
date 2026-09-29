"""
Indexación de PDFs en PostgreSQL con pgvector.

Flujo por cada PDF:
    1. Extraer texto página por página (pdf_extractor)
    2. Detectar dominio según la carpeta (chunker.detect_domain)
    3. Fragmentar en chunks (chunker.split_into_chunks)
    4. Clasificar cada chunk según la taxonomía
    (chunker.detect_dimension_and_indicator)
    5. Generar embeddings (embedder.embed_documents)
    6. Insertar en Postgres: sources, documents, document_chunks
    (db.Database)

Modo dry-run: si se activa, no toca la BD. Solo reporta qué haría.
Útil cuando Postgres aún no está disponible.
"""
import os
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

from ingestion.pdf_extractor import extract_text_from_pdf, walk_pdfs
from ingestion.taxonomy_loader import (
    load_taxonomy,
    build_search_index,
    list_domains,
)
from ingestion.chunker import (
    split_into_chunks,
    detect_domain,
    detect_dimension_and_indicator,
)
from ingestion.embedder import embed_documents

load_dotenv()


# Configuración
PDFS_DIR = os.getenv("PDFS_DIR", "data/pdfs")
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS"))
EMBED_BATCH_SIZE = 32



# Mapeo dominio → nombre de fuente (para tabla sources)

DOMAIN_TO_SOURCE = {
    "colombia": "DANE",
    "cepal": "CEPAL",
    "ophi": "OPHI",
    "pnud": "PNUD",
}

# Intenta extraer un año de 4 dígitos del nombre del archivo.
def extract_year_from_filename(file_name: str) -> Optional[int]:

    import re
    matches = re.findall(r"\b(19|20)\d{2}\b", file_name)
    if matches:
        # Devuelve el último año encontrado (suele ser el más relevante)
        years = re.findall(r"\b((?:19|20)\d{2})\b", file_name)
        if years:
            return int(years[-1])
    return None

# Limpia el nombre de un archivo para usarlo como título.
def clean_title_from_filename(file_name: str) -> str:

    base = Path(file_name).stem
    # Reemplazar guiones y guiones bajos por espacios
    title = base.replace("_", " ").replace("-", " ")
    # Colapsar espacios
    title = " ".join(title.split())
    return title

# Procesamiento de un solo PDF

def process_pdf(
    pdf_path: str,
    taxonomy: dict,
    search_index: dict,
    valid_domains: list[str],
    show_progress: bool = False,
) -> dict:
    #  Indexa todos los PDFs de una carpeta.
    
    file_name = os.path.basename(pdf_path)
    domain = detect_domain(pdf_path, valid_domains)
    source_name = DOMAIN_TO_SOURCE.get(domain, domain.upper())
    title = clean_title_from_filename(file_name)
    publication_year = extract_year_from_filename(file_name)

    # Extraer texto del PDF
    pages = extract_text_from_pdf(pdf_path)

    # Procesar cada página: fragmentar + clasificar
    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        chunks_text = split_into_chunks(
            text,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )

        for chunk_text in chunks_text:
            dimension, indicator = detect_dimension_and_indicator(
                chunk_text, domain, search_index
            )
            all_chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_text,
                "page_number": page_number,
                "domain": domain,
                "dimension": dimension,
                "indicator": indicator,
                "embedding": None,  # se llena abajo en batch
            })
            chunk_index += 1

    # Generar embeddings en batch para todo el PDF
    if all_chunks:
        texts = [c["content"] for c in all_chunks]
        vectors = embed_documents(texts, show_progress=show_progress)
        for chunk, vec in zip(all_chunks, vectors):
            chunk["embedding"] = vec

    return {
        "file_name": file_name,
        "domain": domain,
        "source_name": source_name,
        "title": title,
        "publication_year": publication_year,
        "chunks": all_chunks,
    }


# Indexación completa

def index_all_pdfs(
    pdfs_dir: str = PDFS_DIR,
    dry_run: bool = False,
    limit: Optional[int] = None,
    force_reindex: bool = False,
    show_progress: bool = True,
) -> dict:
    
    # Indexa todos los PDFs de una carpeta.
    # Cargar taxonomía
    taxonomy = load_taxonomy()
    search_index = build_search_index(taxonomy)
    valid_domains = list_domains(taxonomy)

    # Listar PDFs
    pdf_paths = walk_pdfs(pdfs_dir)
    if limit:
        pdf_paths = pdf_paths[:limit]

    if not pdf_paths:
        print("No se encontraron PDFs.")
        return {}

    print(f"PDFs a procesar: {len(pdf_paths)}")
    print(f"Modo: {'DRY-RUN (no toca BD)' if dry_run else 'REAL (indexa en Postgres)'}")
    print()

    # Conexión a Postgres (solo si no es dry-run)
    db = None
    if not dry_run:
        try:
            from storage.database import Database
            db = Database()
            if not db.ping():
                raise ConnectionError("El ping a Postgres falló.")
            print("Conexión a Postgres OK\n")
        except Exception as e:
            print(f"No se pudo conectar a Postgres: {e}")
            print("   Ejecuta con --dry-run para probar sin BD.")
            return {}

    # Estadísticas
    stats = {
        "pdfs_procesados": 0,
        "pdfs_con_error": 0,
        "chunks_totales": 0,
        "chunks_clasificados": 0,
        "por_dominio": {},
        "por_dimension": {},
    }

    # Iterar PDFs
    iterator = tqdm(pdf_paths, desc="PDFs", unit="pdf") if show_progress else pdf_paths

    for pdf_path in iterator:
        try:
            result = process_pdf(
                pdf_path,
                taxonomy=taxonomy,
                search_index=search_index,
                valid_domains=valid_domains,
                show_progress=False,
            )
        except Exception as e:
            print(f"\n Error procesando {pdf_path}: {e}")
            stats["pdfs_con_error"] += 1
            continue

        if not result["chunks"]:
            continue

        # Guardar en BD (si no es dry-run)
        if not dry_run and db is not None:
            try:
                source_id = db.get_or_create_source(result["source_name"])
                document_id = db.get_or_create_document(
                    source_id=source_id,
                    title=result["title"],
                    publication_year=result["publication_year"],
                )

                if force_reindex:
                    db.delete_chunks_for_document(document_id)

                chunks_to_insert = [
                    {
                        "document_id": document_id,
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        "embedding": c["embedding"],
                        "page_number": c["page_number"],
                        "domain": c["domain"],
                        "dimension": c["dimension"],
                        "indicator": c["indicator"],
                        "metadata": {
                            "file_name": result["file_name"],
                            "title": result["title"],
                        },
                    }
                    for c in result["chunks"]
                ]
                db.insert_chunks(chunks_to_insert)
            except Exception as e:
                print(f"\n Error insertando {result['file_name']} en BD: {e}")
                stats["pdfs_con_error"] += 1
                continue

        # Actualizar estadísticas
        stats["pdfs_procesados"] += 1
        stats["chunks_totales"] += len(result["chunks"])
        stats["chunks_clasificados"] += sum(
            1 for c in result["chunks"] if c["dimension"] is not None
        )

        dom = result["domain"]
        stats["por_dominio"][dom] = stats["por_dominio"].get(dom, 0) + len(result["chunks"])

        for c in result["chunks"]:
            d = c["dimension"] or "no_clasificada"
            stats["por_dimension"][d] = stats["por_dimension"].get(d, 0) + 1

    # Cerrar BD
    if db is not None:
        db.close()

    return stats

# Imprime las estadísticas de indexación de forma legible.
def print_stats(stats: dict) -> None:

    if not stats:
        return

    print("  RESUMEN DE INDEXACIÓN")
    print("-" * 60)
    print(f"PDFs procesados:     {stats['pdfs_procesados']}")
    print(f"PDFs con error:      {stats['pdfs_con_error']}")
    print(f"Chunks totales:      {stats['chunks_totales']}")
    print(f"Chunks clasificados: {stats['chunks_clasificados']}")

    if stats["chunks_totales"] > 0:
        pct = 100 * stats["chunks_clasificados"] / stats["chunks_totales"]
        print(f"  → {pct:.1f}% clasificados en alguna dimensión")

    print("\nChunks por dominio:")
    for dom, count in sorted(stats["por_dominio"].items(), key=lambda x: -x[1]):
        print(f"  {dom}: {count}")

    print("\nChunks por dimensión:")
    for dim, count in sorted(stats["por_dimension"].items(), key=lambda x: -x[1]):
        print(f"  {dim}: {count}")


# Punto de entrada (script)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Indexa PDFs en PostgreSQL con pgvector."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No toca la BD. Solo reporta qué haría.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesar solo los primeros N PDFs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borrar chunks previos antes de insertar (reindexar).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Ocultar barra de progreso.",
    )
    args = parser.parse_args()

    print("  INDEXACIÓN DE PDFs — RAG IPM")
    print("-" * 60)
    print(f"Inicio: {datetime.now().isoformat(timespec='seconds')}")

    stats = index_all_pdfs(
        dry_run=args.dry_run,
        limit=args.limit,
        force_reindex=args.force,
        show_progress=not args.no_progress,
    )
    print_stats(stats)

    print(f"\nFin: {datetime.now().isoformat(timespec='seconds')}")