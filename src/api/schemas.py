"""
Pydantic schemas for the API.

Defines request/response models for all endpoints.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# Common
# ============================================================
class TokenUsage(BaseModel):
    """Token usage for the request."""
    input_tokens: int = Field(0, description="Tokens consumidos en el prompt")
    output_tokens: int = Field(0, description="Tokens generados en la respuesta")
    total_tokens: int = Field(0, description="Suma de input + output")


class Source(BaseModel):
    """A cited source from the RAG knowledge base."""
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    document_url: Optional[str] = None
    organization: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    preview: Optional[str] = None
    similarity: Optional[float] = None
    rrf_score: Optional[float] = None


# ============================================================
# /ask
# ============================================================
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Pregunta en español",
        examples=["¿Cuál fue el IPM de Huila en 2018?"],
    )
    verbose: bool = Field(
        False,
        description="Si True, incluye detalles internos (módulo crudo, SQL, etc.)",
    )


class AskResponse(BaseModel):
    question: str
    intent: Literal["data", "simulation", "rag", "recommendation"]
    intent_reason: Optional[str] = None
    intent_method: Optional[str] = None
    answer: str
    status: Literal["ok", "no_data", "not_implemented", "error"]
    sources: list[Source] = Field(default_factory=list)
    sql: Optional[dict | str] = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    raw: Optional[dict] = None  # only if verbose=True


# ============================================================
# /health
# ============================================================
class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "down"]
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    components: list[ComponentHealth]
    uptime_seconds: Optional[float] = None