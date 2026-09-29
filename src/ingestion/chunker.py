"""
Fragmentación (chunking) de texto y clasificación de fragmentos
según la taxonomía de dominios, dimensiones e indicadores.
"""
import re
from core.text_normalizer import normalize
from core.tokenizer_utils import split_by_tokens, count_tokens

def split_into_chunks(text: str, max_tokens: int = 400, overlap_tokens: int = 40) -> list[str]:
    """
    Divide un texto en fragmentos respetando el límite de tokens del modelo.

    Args:
        text: texto de una página del PDF.
        max_tokens: máximo de tokens por fragmento (< 512 para nomic-embed).
        overlap_tokens: solapamiento entre fragmentos subdivididos.

    Returns:
        Lista de fragmentos de texto.
    """
    return split_by_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def detect_domain(pdf_path: str, valid_domains: list[str]) -> str:
    """
    Detecta el dominio al que pertenece un PDF según su ruta.
    Ej: 'data/pdfs/colombia/x.pdf' → 'colombia'

    Args:
        pdf_path: Ruta del PDF.
        valid_domains: Lista de dominios válidos según la taxonomía.

    Returns:
        Nombre del dominio detectado, o 'general' si no coincide con ninguno.
    """
    normalized_path = normalize(pdf_path)
    for domain in valid_domains:
        if normalize(domain) in normalized_path:
            return domain
    return "general"


def detect_dimension_and_indicator(
    text: str,
    domain: str,
    search_index: dict
) -> tuple[str | None, str | None]:
    """
    Dado un fragmento de texto y su dominio, identifica qué dimensión
    e indicador menciona, usando el índice invertido.

    Prioriza coincidencias con indicadores (más específicos) sobre
    coincidencias con nombres de dimensión.

    Args:
        text: Fragmento de texto.
        domain: Dominio del PDF del que proviene.
        search_index: Índice invertido construido por build_search_index().

    Returns:
        Tupla (dimension_key, indicator) o (None, None) si no se detecta.
    """
    normalized_text = normalize(text)
    entries = search_index.get(domain, [])

    # Primera pasada: buscar indicadores (más específicos)
    for dim_key, indicator, term in entries:
        if indicator is None:
            continue
        if term and term in normalized_text:
            return dim_key, indicator

    # Segunda pasada: buscar solo por nombre de dimensión
    for dim_key, indicator, term in entries:
        if indicator is not None:
            continue
        if term and term in normalized_text:
            return dim_key, None

    return None, None


if __name__ == "__main__":
    # Prueba de detección de dimensión/indicador
    from ingestion.taxonomy_loader import load_taxonomy, build_search_index

    taxonomy = load_taxonomy()
    search_index = build_search_index(taxonomy)

    samples = [
        ("Un hogar presenta hacinamiento crítico cuando hay tres o más personas por cuarto.", "colombia"),
        ("El bajo logro educativo se mide por los años de educación del jefe de hogar.", "colombia"),
        ("La inasistencia escolar afecta a niños entre 6 y 16 años.", "colombia"),
        ("Esto es un texto cualquiera sin nada relevante.", "colombia"),
    ]

    for text, domain in samples:
        dim, ind = detect_dimension_and_indicator(text, domain, search_index)
        print(f"\nTexto: {text[:60]}...")
        print(f"  → dimensión: {dim} | indicador: {ind}")