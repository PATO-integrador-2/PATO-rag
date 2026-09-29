"""
Módulo de embeddings usando Ollama.

Convierte texto a vectores de 768 dimensiones con el modelo
nomic-embed-text-v2-moe.

IMPORTANTE: este modelo requiere prefijos según el uso:
    - "search_document: " para documentos a indexar
    - "search_query: "    para preguntas de búsqueda
Sin los prefijos, la calidad de recuperación cae mucho.
"""
import os
import sys
from pathlib import Path
from core.tokenizer_utils import count_tokens, get_tokenizer

# Asegurar que la raíz del proyecto esté en sys.path (para ejecutar como script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Iterable, Optional

import ollama
from dotenv import load_dotenv

load_dotenv()


# Configuración

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
EXPECTED_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Prefijos obligatorios del modelo
PREFIX_DOCUMENT = "search_document: "
PREFIX_QUERY = "search_query: "



# Funciones principales

def embed_documents(
    texts: Iterable[str],
    show_progress: bool = False,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Convierte una lista de textos de documentos en embeddings.

    Cada texto recibe el prefijo "search_document: " antes de vectorizarse.

    Args:
        texts: iterable de strings (fragmentos de PDF).
        show_progress: si True, muestra barra de progreso (útil para lotes grandes).
        batch_size: número de textos por lote antes de reportar progreso.

    Returns:
        Lista de vectores (list[float]), uno por cada texto de entrada.
    """
    texts = list(texts)
    if not texts:
        return []

    # Aplicar prefijo a cada texto
    prefixed = [PREFIX_DOCUMENT + t for t in texts]

    # Procesar con o sin barra de progreso
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(0, len(prefixed), batch_size),
                desc="Embeddings",
                unit="batch",
            )
        except ImportError:
            iterator = range(0, len(prefixed), batch_size)
    else:
        iterator = range(0, len(prefixed), batch_size)

    vectors: list[list[float]] = []
    for i in iterator:
        batch = prefixed[i:i + batch_size]
        batch_vectors = _embed_batch(batch)
        vectors.extend(batch_vectors)

    return vectors


def embed_query(query: str) -> list[float]:
    """
    Convierte una pregunta del usuario en su embedding.

    Aplica el prefijo "search_query: " antes de vectorizar.

    Args:
        query: texto de la pregunta.

    Returns:
        Vector de 768 dimensiones.
    """
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    prefixed = PREFIX_QUERY + query.strip()
    response = ollama.embeddings(model=EMBED_MODEL, prompt=prefixed)
    vector = response.get("embedding", [])

    if len(vector) != EXPECTED_DIM:
        raise RuntimeError(
            f"Dimensión inesperada: {len(vector)} (se esperaba {EXPECTED_DIM}). "
            f"Verifica el modelo '{EMBED_MODEL}'."
        )
    return vector



# Interno: lotes

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embebe un lote de textos. Ollama no expone batch nativo en la
    librería Python, así que se procesa uno por uno.
    """
    MAX_TOKENS = 500
    vectors = []
    for text in texts:
        # Truncado defensivo
        if count_tokens(text) > MAX_TOKENS:
            tok = get_tokenizer()
            ids = tok.encode(text, add_special_tokens=False)[:MAX_TOKENS]
            text = tok.decode(ids, skip_special_tokens=True)
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        except Exception as e:
            raise RuntimeError(f"Error generando embedding: {e}") from e
        vector = response.get("embedding", [])
        if len(vector) != EXPECTED_DIM:
            raise RuntimeError(
                f"Dimensión inesperada: {len(vector)} (se esperaba {EXPECTED_DIM})"
            )
        vectors.append(vector)
    return vectors



# Utilidades públicas

def check_model_available() -> bool:
    """Verifica que el modelo de embeddings esté descargado en Ollama."""
    try:
        models = ollama.list()
        if isinstance(models, dict):
            names = [m.get("name", "") for m in models.get("models", [])]
        else:
            names = [getattr(m, "model", str(m)) for m in models]
        base = EMBED_MODEL.split(":")[0]
        return any(base in n for n in names)
    except Exception:
        return False



# Prueba directa

if __name__ == "__main__":
    print("=" * 60)
    print("  PRUEBA DEL MÓDULO DE EMBEDDINGS")
    print("=" * 60)
    print(f"Modelo: {EMBED_MODEL}")
    print(f"Dimensión esperada: {EXPECTED_DIM}")

    if not check_model_available():
        print(f"\n El modelo '{EMBED_MODEL}' no está disponible en Ollama.")
        print(f"   Ejecuta: ollama pull {EMBED_MODEL}")
        raise SystemExit(1)
    print(" Modelo disponible en Ollama")

    # Prueba 1: documento
    print("\n--- Prueba: embed_documents ---")
    docs = [
        "Un hogar presenta hacinamiento crítico cuando hay tres o más personas por cuarto.",
        "El bajo logro educativo se mide por los años de educación del jefe de hogar.",
    ]
    doc_vectors = embed_documents(docs)
    print(f"   Generados {len(doc_vectors)} vectores")
    print(f"   Dimensión del primero: {len(doc_vectors[0])}")
    print(f"   Primeros 5 valores: {doc_vectors[0][:5]}")

    # Prueba 2: consulta
    print("\n--- Prueba: embed_query ---")
    query = "¿Qué es hacinamiento crítico?"
    query_vector = embed_query(query)
    print(f"   Vector de consulta generado")
    print(f"   Dimensión: {len(query_vector)}")
    print(f"   Primeros 5 valores: {query_vector[:5]}")

    # Prueba 3: similitud coseno entre la consulta y cada documento
    print("\n--- Prueba: similitud coseno (para verificar sentido semántico) ---")
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    sims = [cosine_similarity(query_vector, v) for v in doc_vectors]
    for doc, sim in zip(docs, sims):
        print(f"   [{sim:.4f}] {doc[:60]}...")

    print("\nInterpretación: la similitud MÁS ALTA debe corresponder al")
    print("documento más relacionado con la consulta.")