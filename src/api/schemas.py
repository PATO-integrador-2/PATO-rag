"""
Pydantic schemas for the API.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# Common
# ============================================================
class TokenUsage(BaseModel):
    input_tokens: int = Field(0)
    output_tokens: int = Field(0)
    total_tokens: int = Field(0)


class Source(BaseModel):
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
    question: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., description="UUID del usuario (backend lo provee)")
    conversation_id: str = Field(..., description="UUID de la conversación (backend lo provee)")
    verbose: bool = Field(False)


class AskResponse(BaseModel):
    question: str
    intent: Literal["data", "simulation", "rag", "recommendation", "general"]
    intent_reason: Optional[str] = None
    intent_method: Optional[str] = None
    reformulated: bool = False
    effective_question: Optional[str] = None
    answer: str
    status: Literal[
        "ok", "no_data", "no_context", "not_implemented",
        "error", "validation_error", "rejected",
    ]
    sources: list[Source] = Field(default_factory=list)
    sql: Optional[dict | str] = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    raw: Optional[dict] = None


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
    components: list[ComponentHealth] = Field(default_factory=list)
    uptime_seconds: Optional[float] = None


class VersionResponse(BaseModel):
    name: str
    version: str