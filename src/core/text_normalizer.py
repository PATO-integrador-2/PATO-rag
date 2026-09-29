"""
Módulo de normalización de texto.

Se encarga de dejar los textos en una forma canónica para comparaciones:
- Todo en minúsculas
- Sin tildes
- Sin puntuación
- Con espacios colapsados
- Guiones bajos convertidos a espacios

"""
import unicodedata
import re

# Función de normalización
def normalize(text: str) -> str:
    
    if not text:
        return ""
    # Convertir a minúsculas
    text = text.lower()
    # Descomponer caracteres Unicode y eliminar  (tildes, ñ → n)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Convertir guiones bajos a espacios para unificar "acceso_agua" y "acceso agua"
    text = text.replace("_", " ")
    # Eliminar puntuación
    text = re.sub(r"[^\w\s]", " ", text)
    # Colapsar espacios múltiples
    text = " ".join(text.split())
    return text

# Generar variantes de un término para aumentar la probabilidad de match en el texto
def generate_variants(term: str) -> list[str]:
    
    base = term.replace("_", " ").strip()
    base_norm = normalize(base)
    variants = {base_norm}

    # Variante con preposición "a" para términos tipo "acceso X"
    if base_norm.startswith("acceso ") and not base_norm.startswith("acceso a "):
        variants.add(base_norm.replace("acceso ", "acceso a ", 1))

    # Variante con "sin acceso a" para términos tipo "sin X"
    if base_norm.startswith("sin ") and not base_norm.startswith("sin acceso a "):
        variants.add(base_norm.replace("sin ", "sin acceso a ", 1))

    # Singular/plural simple
    if base_norm.endswith("s"):
        variants.add(base_norm[:-1])
    else:
        variants.add(base_norm + "s")

    return list(variants)


if __name__ == "__main__":
    # Pruebas rápidas de normalización
    test_cases = [
        ("Niñez", "ninez"),
        ("HACINAMIENTO CRÍTICO", "hacinamiento critico"),
        ("acceso_internet", "acceso internet"),
        ("Material inadecuado de los pisos", "material inadecuado de los pisos"),
    ]
    for entry, expected in test_cases:
        result = normalize(entry)
        status = "✅" if expected in result else "❌"
        print(f"{status} normalize({entry!r}) = {result!r}")