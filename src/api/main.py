"""
FastAPI application entry point.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health    → system status
    GET  /version   → API version
    POST /ask       → main question-answering endpoint
    GET  /docs      → interactive Swagger UI
    GET  /redoc     → ReDoc documentation
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health, ask

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="PATO-rag API",
    description=(
        "API de IA generativa sobre pobreza multidimensional (IPM) "
        "en Colombia y Latinoamérica.\n\n"
        "Módulos: data (SQL), simulation, rag, recommendation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow all for now; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routes
# ============================================================
app.include_router(health.router)
app.include_router(ask.router)


@app.get("/", tags=["root"])
def root():
    """Root endpoint — quick info."""
    return {
        "name": "PATO-rag API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask",
    }


# ============================================================
# Global error handler
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Error no manejado: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor", "error": str(exc)[:200]},
    )


# ============================================================
# Direct run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )