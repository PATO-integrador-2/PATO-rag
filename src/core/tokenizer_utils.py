"""
Utilidades de tokenización para el chunking y los embeddings.

Usa el tokenizador oficial del modelo nomic-embed-text-v2-moe para
contar tokens con precisión y respetar el límite de 512 tokens del modelo.
"""
from functools import lru_cache
from transformers import AutoTokenizer


# Modelo base usado para la tokenización.
# nomic-embed-text-v2-moe se basa en una arquitectura tipo BERT multilingüe,
# por lo que usamos un tokenizador compatible.
# Si más adelante cambias de modelo, actualiza este nombre.
TOKENIZER_NAME = "nomic-ai/nomic-bert-2048"

# Devuelve el tokenizador (cacheado para no recargarlo).
@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

# Cuenta cuántos tokens tiene un texto.
def count_tokens(text: str) -> int:
    if not text:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    # Divide un texto en fragmentos respetando el límite de tokens.
    import re

    # Margen de seguridad: dejamos un pequeño buffer para prefijos
    # ("search_document: ") y posibles tokens especiales
    SAFE_MAX = min(max_tokens, 500)

    tokenizer = get_tokenizer()
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    buffer = ""
    buffer_tokens = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_tokens = count_tokens(paragraph)

        # Caso 1: el párrafo cabe en el buffer actual
        if buffer_tokens + para_tokens <= SAFE_MAX:
            buffer = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            buffer_tokens = count_tokens(buffer)
            continue

        # Caso 2: el párrafo no cabe, pero es pequeño → vaciar buffer y empezar uno nuevo
        if para_tokens <= SAFE_MAX:
            if buffer:
                chunks.append(buffer)
            buffer = paragraph
            buffer_tokens = para_tokens
            continue

        # Caso 3: el párrafo solo ya excede → vaciar buffer y subdividir el párrafo
        if buffer:
            chunks.append(buffer)
            buffer = ""
            buffer_tokens = 0

        # Subdividir por tokens con overlap
        token_ids = tokenizer.encode(paragraph, add_special_tokens=False)
        step = SAFE_MAX - overlap_tokens
        for i in range(0, len(token_ids), step):
            sub_ids = token_ids[i:i + SAFE_MAX]
            sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True)
            chunks.append(sub_text)

    # Vaciar el buffer final
    if buffer:
        chunks.append(buffer)

    return chunks


if __name__ == "__main__":
    # Prueba rápida
    print("Cargando tokenizador...")
    tok = get_tokenizer()
    print(f"✅ Tokenizador cargado: {TOKENIZER_NAME}")

    sample = "Un hogar presenta hacinamiento crítico cuando hay tres o más personas por cuarto."
    print(f"\nTexto: {sample}")
    print(f"Tokens: {count_tokens(sample)}")

    # Texto largo artificial
    long_text = ("La pobreza multidimensional es un fenómeno complejo. " * 100)
    print(f"\nTexto largo: {count_tokens(long_text)} tokens")
    chunks = split_by_tokens(long_text, max_tokens=400, overlap_tokens=40)
    print(f"Dividido en {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3], 1):
        print(f"  Chunk {i}: {count_tokens(c)} tokens")