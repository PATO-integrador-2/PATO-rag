"""
Health check endpoint.

Reports the status of:
    - The API itself
    - Ollama (LLM backend)
    - The SQL database (clean_data_db)
    - The RAG database (embedding_db)
"""
import time
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
load_dotenv()

from src.api.schemas import HealthResponse, ComponentHealth
from src.storage.data_db import DataDatabase

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

APP_START_TIME = time.time()
API_VERSION = os.getenv("API_VERSION", "1.0.0")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")


def _check_ollama() -> ComponentHealth:
    """Check if Ollama is reachable and list available models."""
    try:
        from ollama import Client
        client = Client(host=OLLAMA_HOST)
        
        response = client.list()
        # Handle both dict and object responses
        if isinstance(response, dict):
            models = response.get("models", [])
            names = [m.get("name", "?") for m in models]
        else:
            names = [getattr(m, "model", str(m)) for m in response]
        if names:
            return ComponentHealth(
                name="ollama",
                status="ok",
                detail=f"{len(names)} modelos disponibles",
            )
        return ComponentHealth(name="ollama", status="degraded", detail="Sin modelos")
    except Exception as exc:
        return ComponentHealth(name="ollama", status="down", detail=str(exc)[:200])


def _check_data_db() -> ComponentHealth:
    """Check clean_data_db connectivity."""
    db = None
    try:
        db = DataDatabase(minconn=1, maxconn=1)
        ok = db.ping()
        if ok:
            return ComponentHealth(name="clean_data_db", status="ok", detail="Postgres OK")
        return ComponentHealth(name="clean_data_db", status="down", detail="Ping falló")
    except Exception as exc:
        return ComponentHealth(name="clean_data_db", status="down", detail=str(exc)[:200])
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


def _check_embedding_db() -> ComponentHealth:
    """Check embedding_db connectivity."""
    try:
        from src.rag.rag_retriever import RagRetriever
        retriever = RagRetriever()
        try:
            ok = retriever.ping()
            if ok:
                return ComponentHealth(name="embedding_db", status="ok", detail="Postgres OK")
            return ComponentHealth(name="embedding_db", status="down", detail="Ping falló")
        finally:
            retriever.close()
    except Exception as exc:
        return ComponentHealth(name="embedding_db", status="down", detail=str(exc)[:200])


@router.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.

    Returns the overall system status and per-component detail.
    """
    components = [
        _check_ollama(),
        _check_data_db(),
        _check_embedding_db(),
    ]

    # Overall status
    if all(c.status == "ok" for c in components):
        overall = "ok"
    elif any(c.status == "down" for c in components):
        overall = "down"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=API_VERSION,
        components=components,
        uptime_seconds=round(time.time() - APP_START_TIME, 1),
    )


@router.get("/version")
def version():
    """Return the API version."""
    return {"version": API_VERSION, "name": "PATO-rag API"}