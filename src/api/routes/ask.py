"""
Ask endpoint.

Sends the question to the Orchestrator and returns a unified response
with token usage.
"""
import logging
import time
from fastapi import APIRouter, HTTPException

from src.api.schemas import AskRequest, AskResponse, TokenUsage, Source
from src.orchestrator.router import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

# Module-level orchestrator (lazy)
_orchestrator: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    """Lazy singleton for the Orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (English/Spanish)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Answer a question using the multi-module orchestrator.

    The orchestrator classifies the question into one of 4 intents:
        - data           → SQL queries against clean_data_db
        - simulation     → IPM scenario simulation
        - rag            → definitional questions (PDFs)
        - recommendation → project proposals
    """
    start = time.time()

    try:
        orch = _get_orchestrator()
        raw = orch.ask(request.question, verbose=request.verbose)
    except Exception as exc:
        logger.exception("Error en el orquestador: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error procesando la pregunta: {exc}")

    # Build token usage estimate from the answer
    # For accurate tokens, each module would need to expose them.
    # We approximate: input = question length, output = answer length.
    input_tokens = _estimate_tokens(request.question)
    output_tokens = _estimate_tokens(raw.get("answer", ""))
    total_tokens = input_tokens + output_tokens

    # Extract sources
    sources = []
    for s in raw.get("sources", []) or []:
        sources.append(Source(**{k: v for k, v in s.items() if k in Source.model_fields}))

    # Build response
    response = AskResponse(
        question=raw.get("question", request.question),
        intent=raw.get("intent", "rag"),
        intent_reason=raw.get("intent_reason"),
        intent_method=raw.get("intent_method"),
        answer=raw.get("answer", ""),
        status=raw.get("status", "ok"),
        sources=sources,
        sql=raw.get("sql"),
        tokens=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
        latency_ms=int((time.time() - start) * 1000),
        raw=raw if request.verbose else None,
    )
    return response