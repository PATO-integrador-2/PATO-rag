"""
Esquemas Pydantic para la API del RAG.
Definen el contrato de request y response del endpoint /api/rag/ask.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# Request
# ============================================================
class PreviousMessage(BaseModel):
    """Un mensaje previo dentro del contexto de la conversación."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class ContextPayload(BaseModel):
    """Contexto previo enviado opcionalmente por el backend."""
    previous_messages: list[PreviousMessage] = Field(default_factory=list)


class FiltersPayload(BaseModel):
    """Filtros opcionales para la búsqueda vectorial."""
    domain: Optional[str] = None
    dimension: Optional[str] = None
    indicator: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    """Request del endpoint /api/rag/ask."""
    conversation_id: Optional[str] = Field(
        default=None,
        description="UUID de la conversación existente. Null si es la primera pregunta.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="UUID de la sesión del usuario (opcional).",
    )
    user_id: str = Field(..., description="UUID del usuario que hace la pregunta.")
    question: str = Field(..., min_length=1, description="Pregunta del usuario.")
    context: Optional[ContextPayload] = Field(
        default=None,
        description="Contexto previo (si el backend lo tiene).",
    )
    filters: Optional[FiltersPayload] = Field(
        default=None,
        description="Filtros opcionales para la búsqueda.",
    )


# ============================================================
# Response
# ============================================================
class SourceItem(BaseModel):
    """Una fuente citada en la respuesta."""
    source: Optional[str] = None
    document: Optional[str] = None
    page: Optional[int] = None
    dimension: Optional[str] = None
    indicator: Optional[str] = None
    similarity: Optional[float] = None


class AskResponse(BaseModel):
    """Response del endpoint /api/rag/ask."""
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    answer: str
    sources: list[SourceItem]
    detected_domain: Optional[str] = None
    chunks_used: int
    latency_ms: int



# Health

class HealthResponse(BaseModel):
    """Response del endpoint /health."""
    status: str
    database: bool
    ollama: bool
    version: str