"""
Punto de entrada de la API HTTP del RAG.
Ejecutar: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.routes import router
from src.api.dependencies import shutdown_retriever, get_retriever

load_dotenv()

API_TITLE = os.getenv("API_TITLE", "RAG IPM API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")



# Ciclo de vida

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos al arrancar y los libera al apagar."""
    print(" Iniciando API del RAG...")
    try:
        # Inicializar el retriever (abre pool de conexiones)
        get_retriever()
        print(" Retriever inicializado")
    except Exception as e:
        print(f"  No se pudo inicializar el retriever: {e}")
        print("   La API arrancará igual, pero las consultas fallarán.")

    yield

    print(" Apagando API...")
    shutdown_retriever()
    print(" Retriever cerrado")



# Aplicación FastAPI

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "API del RAG (Retrieval-Augmented Generation) para consultar "
        "documentos sobre pobreza multidimensional en Latinoamérica."
    ),
    lifespan=lifespan,
)

# CORS (ajustar origins en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          #  En producción, restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )