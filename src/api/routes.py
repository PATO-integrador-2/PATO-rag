"""
Rutas HTTP de la API del RAG.
"""
import os
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import verify_api_key, get_retriever
from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.api.services import process_ask_request
from retrieval.retriever import Retriever


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(retriever: Retriever = Depends(get_retriever)) -> HealthResponse:
    """
    Health check: verifica conexión a Postgres y disponibilidad de Ollama.
    """
    db_ok = False
    try:
        db_ok = retriever.db.ping()
    except Exception:
        pass

    ollama_ok = False
    try:
        import ollama
        ollama.list()
        ollama_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (db_ok and ollama_ok) else "degraded",
        database=db_ok,
        ollama=ollama_ok,
        version=os.getenv("API_VERSION", "1.0.0"),
    )


@router.post(
    "/api/rag/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    tags=["rag"],
    dependencies=[Depends(verify_api_key)],
)
async def ask(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
) -> AskResponse:
    """
    Endpoint principal: recibe una pregunta, recupera contexto de la BD
    vectorial, genera una respuesta con el LLM y persiste la conversación.
    """
    try:
        return process_ask_request(request, retriever, retriever.db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la solicitud: {e}",
        )