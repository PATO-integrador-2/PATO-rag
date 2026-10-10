"""
Text normalization module.

Leaves texts in a canonical form for comparisons:
- Lowercase
- No accents
- No punctuation
- Collapsed whitespace
- Underscores converted to spaces
"""
import unicodedata
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def normalize(text: str) -> str:
    """Return a canonical version of the text for keyword matching."""
    if not text:
        return ""
    text = text.lower()
    # Decompose unicode and strip combining marks (tildes, ñ → n)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Underscores → spaces so "acceso_agua" and "acceso agua" match
    text = text.replace("_", " ")
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text

def generate_variants(term: str) -> list[str]:
    """Generate simple variants of a term to increase match probability."""
    base = term.replace("_", " ").strip()
    base_norm = normalize(base)
    variants = {base_norm}

    # Add "a" preposition for "acceso X" → "acceso a X"
    if base_norm.startswith("acceso ") and not base_norm.startswith("acceso a "):
        variants.add(base_norm.replace("acceso ", "acceso a ", 1))

    # Add "sin acceso a" for "sin X"
    if base_norm.startswith("sin ") and not base_norm.startswith("sin acceso a "):
        variants.add(base_norm.replace("sin ", "sin acceso a ", 1))

    # Simple singular/plural
    if base_norm.endswith("s"):
        variants.add(base_norm[:-1])
    else:
        variants.add(base_norm + "s")

    return list(variants)


if __name__ == "__main__":
    test_cases = [
        ("Niñez", "ninez"),
        ("HACINAMIENTO CRÍTICO", "hacinamiento critico"),
        ("acceso_internet", "acceso internet"),
        ("Material inadecuado de los pisos", "material inadecuado de los pisos"),
    ]
    for entry, expected in test_cases:
        result = normalize(entry)
        status = "✓" if expected in result else "X"
        print(f"{status} normalize({entry!r}) = {result!r}")