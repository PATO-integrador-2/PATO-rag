"""
Dependencias de la API: autenticación por Bearer token y
acceso compartido a la instancia del Retriever.
"""
import os
from typing import Optional

from fastapi import Header, HTTPException, status
from dotenv import load_dotenv

from retrieval.retriever import Retriever

load_dotenv()

API_KEY = os.getenv("API_KEY", "rag-secret-key-cambiar-en-produccion")


# ============================================================
# Autenticación
# ============================================================
async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """
    Verifica el Bearer token en el header Authorization.
    Formato esperado: 'Bearer <API_KEY>'.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta el header Authorization.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Formato de Authorization inválido. Use 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if parts[1] != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key inválida.",
        )


# ============================================================
# Retriever compartido (singleton)
# ============================================================
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """
    Devuelve la instancia compartida del Retriever.
    Se crea una sola vez en el arranque de la API.
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def shutdown_retriever() -> None:
    """Cierra el retriever al apagar la API."""
    global _retriever
    if _retriever is not None:
        _retriever.close()
        _retriever = None