"""
Carga la taxonomía desde el archivo YAML y expone utilidades
para consultarla y construir un índice invertido de búsqueda.
"""
import os
import yaml
from dotenv import load_dotenv

from core.text_normalizer import normalize

# Cargar variables de entorno
load_dotenv()

TAXONOMY_PATH = os.getenv("TAXONOMY_PATH", "config/taxonomy.yaml")


def load_taxonomy(path: str = TAXONOMY_PATH) -> dict:
    """
    Carga el YAML de taxonomía y valida que tenga la clave raíz 'dominios'.

    Args:
        path: Ruta al archivo YAML.

    Returns:
        Diccionario con la taxonomía completa.

    Raises:
        ValueError: Si el YAML no contiene la clave 'dominios'.
        FileNotFoundError: Si el archivo no existe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de taxonomía: {path}")

    with open(path, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    if "dominios" not in taxonomy:
        raise ValueError("La taxonomía debe tener una clave raíz 'dominios'.")

    # Advertir sobre dominios sin dimensiones definidas
    for domain_key, domain_info in taxonomy["dominios"].items():
        if "dimensiones" not in domain_info:
            print(f"⚠️  El dominio '{domain_key}' no tiene dimensiones definidas.")

    return taxonomy


def list_domains(taxonomy: dict) -> list[str]:
    """Devuelve la lista de dominios disponibles."""
    return list(taxonomy["dominios"].keys())


def list_dimensions(taxonomy: dict, domain: str) -> list[str]:
    """Devuelve las dimensiones de un dominio específico."""
    return list(taxonomy["dominios"].get(domain, {}).get("dimensiones", {}).keys())


def list_indicators(taxonomy: dict, domain: str, dimension: str) -> list[str]:
    """Devuelve los indicadores de una dimensión específica."""
    dimensions = taxonomy["dominios"].get(domain, {}).get("dimensiones", {})
    return dimensions.get(dimension, {}).get("indicadores", [])


def build_search_index(taxonomy: dict) -> dict:
    """
    Construye un índice invertido para clasificación rápida de fragmentos.

    Estructura de retorno:
        {
            "colombia": [
                (dimension_key, indicator_or_None, normalized_term),
                ...
            ],
            ...
        }

    Incluye tanto los nombres de dimensión como los indicadores como términos buscables.

    Args:
        taxonomy: Diccionario de taxonomía cargado.

    Returns:
        Índice invertido por dominio.
    """
    index = {}
    for domain_key, domain_info in taxonomy["dominios"].items():
        entries = []
        for dim_key, dim_info in domain_info.get("dimensiones", {}).items():
            # Nombre legible de la dimensión como término buscable
            dim_name = dim_info.get("nombre", dim_key)
            entries.append((dim_key, None, normalize(dim_name)))
            # También la clave en formato legible
            entries.append((dim_key, None, normalize(dim_key.replace("_", " "))))
            # Y cada indicador
            for indicator in dim_info.get("indicadores", []):
                entries.append((dim_key, indicator, normalize(indicator)))
        index[domain_key] = entries
    return index


if __name__ == "__main__":
    # Prueba de carga y estructura
    taxonomy = load_taxonomy()
    print("Dominios:", list_domains(taxonomy))
    for domain in list_domains(taxonomy):
        print(f"\n{domain}:")
        for dim in list_dimensions(taxonomy, domain):
            indicators = list_indicators(taxonomy, domain, dim)
            print(f"  - {dim}: {len(indicators)} indicadores")