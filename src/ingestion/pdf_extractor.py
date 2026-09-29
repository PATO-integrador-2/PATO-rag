"""
Extracción de texto desde archivos PDF.
Usa PyMuPDF (nombre moderno del módulo: pymupdf; antes: fitz).
Devuelve el texto página por página, preservando el número de página para trazabilidad.
"""
import os

# Importar PyMuPDF con fallback para versiones antiguas
try:
    import pymupdf  # API moderna (versión >= 1.24)
    _open_pdf = pymupdf.open
except ImportError:
    import fitz   # API antigua, sigue funcionando
    _open_pdf = fitz.open


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extrae el texto de un PDF página por página.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Lista de diccionarios: [{"page": 1, "text": "..."}, ...]
        Solo incluye páginas con contenido no vacío.
    """
    try:
        document = _open_pdf(pdf_path)
    except Exception as e:
        print(f"❌ No se pudo abrir {pdf_path}: {e}")
        return []

    pages = []
    for page_number, page in enumerate(document, start=1):
        try:
            text = page.get_text()
        except Exception as e:
            print(f"⚠️  Error extrayendo página {page_number} de {pdf_path}: {e}")
            text = ""
        if text.strip():
            pages.append({"page": page_number, "text": text})
    document.close()
    return pages


def walk_pdfs(folder: str) -> list[str]:
    """
    Recorre recursivamente una carpeta y devuelve todas las rutas de PDFs.

    Args:
        folder: Ruta de la carpeta raíz.

    Returns:
        Lista ordenada de rutas a archivos .pdf
    """
    paths = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                paths.append(os.path.join(root, file_name))
    return sorted(paths)


if __name__ == "__main__":
    # Prueba con los PDFs disponibles en data/pdfs
    from dotenv import load_dotenv
    load_dotenv()

    pdf_folder = os.getenv("PDFS_DIR", "data/pdfs")
    pdfs = walk_pdfs(pdf_folder)
    print(f"Encontrados {len(pdfs)} PDFs en {pdf_folder}")

    if pdfs:
        pages = extract_text_from_pdf(pdfs[0])
        print(f"Primer PDF: {pdfs[0]}")
        print(f"Páginas con texto: {len(pages)}")
        if pages:
            print(f"Muestra: {pages[0]['text'][:300]}...")