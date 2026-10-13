"""
Ask endpoint with conversation memory.

Flow:
    1. Validate user_id + conversation_id belong together.
    2. Load recent history.
    3. Save user message.
    4. Ask the orchestrator (with history).
    5. Save assistant message (only if status == "ok").
    6. Return response.
"""
import logging
import time
from fastapi import APIRouter, HTTPException

from src.api.schemas import AskRequest, AskResponse, TokenUsage, Source
from src.orchestrator.router import Orchestrator
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

_orchestrator: Orchestrator | None = None
_memory: MemoryManager | None = None


def _get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def _get_memory() -> MemoryManager:
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start = time.time()

    # 1. Validate conversation ownership
    memory = _get_memory()
    if not memory.validate_conversation(request.conversation_id, request.user_id):
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversación no encontrada o no pertenece al usuario indicado."
            ),
        )

    # 2. Load history
    history = memory.load_history(request.conversation_id)
    logger.info(
        "Conversation %s: loaded %d history messages: %s",
        request.conversation_id,
        len(history),
        [m.get("role") for m in history],
    )

    # 3. Save user message
    memory.save_user_message(request.conversation_id, request.question)

    # 4. Ask orchestrator
    try:
        orch = _get_orchestrator()
        raw = orch.ask(
            request.question,
            history=history,
            verbose=request.verbose,
        )
    except Exception as exc:
        logger.exception("Error en orquestador: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error procesando la pregunta: {exc}")

    # 5. Save assistant message only if status == "ok"
    if raw.get("status") == "ok":
        metadata = {
            "intent": raw.get("intent"),
            "intent_reason": raw.get("intent_reason"),
            "intent_method": raw.get("intent_method"),
            "reformulated": raw.get("reformulated"),
            "effective_question": raw.get("effective_question"),
            "tokens": raw.get("tokens"),
            "latency_ms": raw.get("latency_ms"),
            "sources": raw.get("sources", []),
        }
        memory.save_assistant_message(
            request.conversation_id,
            raw.get("answer", ""),
            metadata=metadata,
        )

    # 6. Touch conversation updated_at
    memory.touch_conversation(request.conversation_id)

    # 7. Build response
    sources = []
    for s in raw.get("sources", []) or []:
        sources.append(Source(**{k: v for k, v in s.items() if k in Source.model_fields}))

    tokens_raw = raw.get("tokens", {})
    if isinstance(tokens_raw, dict) and "input_tokens" not in tokens_raw:
        tokens = TokenUsage(
            input_tokens=int(tokens_raw.get("input", 0)),
            output_tokens=int(tokens_raw.get("output", 0)),
            total_tokens=int(tokens_raw.get("input", 0)) + int(tokens_raw.get("output", 0)),
        )
    else:
        tokens = TokenUsage(
            input_tokens=int(tokens_raw.get("input_tokens", 0)),
            output_tokens=int(tokens_raw.get("output_tokens", 0)),
            total_tokens=int(tokens_raw.get("total_tokens", 0)),
        )

    return AskResponse(
        question=raw.get("question", request.question),
        intent=raw.get("intent", "general"),
        intent_reason=raw.get("intent_reason"),
        intent_method=raw.get("intent_method"),
        reformulated=bool(raw.get("reformulated", False)),
        effective_question=raw.get("effective_question"),
        answer=raw.get("answer", ""),
        status=raw.get("status", "ok"),
        sources=sources,
        sql=raw.get("sql"),
        tokens=tokens,
        latency_ms=int((time.time() - start) * 1000),
        raw=raw if request.verbose else None,
    )