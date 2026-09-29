"""
Servicios de la API. Orquestan el flujo completo:
recibir pregunta → recuperar chunks → generar respuesta → persistir.
"""
import time
from typing import Optional

from storage.database import Database
from retrieval.retriever import Retriever
from src.api.schemas import (
    AskRequest,
    AskResponse,
    SourceItem,
    ContextPayload,
)


def _build_context_from_payload(
    payload: Optional[ContextPayload],
    db: Database,
    conversation_id: Optional[str],
    max_messages: int = 10,
) -> list[dict]:
    """
    Construye la lista de mensajes previos para el contexto.
    Estrategia híbrida:
        1. Si el payload trae 'previous_messages', se usan.
        2. Si no, y hay conversation_id, se consultan desde Postgres.
    """
    # 1. Si el backend envió contexto, usarlo
    if payload and payload.previous_messages:
        return [
            {"role": m.role, "content": m.content}
            for m in payload.previous_messages
        ]

    # 2. Si no, consultar Postgres
    if conversation_id:
        try:
            return db.get_conversation_context(conversation_id, max_messages)
        except Exception:
            return []

    return []


def _build_prompt_with_history(
    question: str,
    history: list[dict],
    context_chunks: list[dict],
    retriever: Retriever,
) -> str:
    """
    Construye el prompt final combinando:
    - Historial de conversación (opcional)
    - Chunks recuperados (contexto RAG)
    - Pregunta actual
    """
    # Bloque de historial (si hay)
    history_block = ""
    if history:
        history_lines = []
        for msg in history[-6:]:  # últimos 6 mensajes
            role_label = "Usuario" if msg.get("role") == "user" else "Asistente"
            content = msg.get("content", "").strip()
            if content:
                history_lines.append(f"{role_label}: {content}")
        if history_lines:
            history_block = "\n\nHISTORIAL DE CONVERSACIÓN:\n" + "\n".join(history_lines)

    # Bloque de contexto RAG
    context_block = retriever._build_context(context_chunks)

    # Prompt final
    return f"""Eres un experto en pobreza multidimensional en Latinoamérica.

INSTRUCCIONES ESTRICTAS:
1. Responde SIEMPRE en español, aunque el contexto esté en inglés.
2. Responde ÚNICAMENTE basándote en el contenido de los campos "Texto:" de las fuentes.
3. NO menciones la palabra "similitud", "fuente 1", "fuente 2" ni los números de 
   similitud en tu respuesta. Solo usa el contenido textual.
4. Cita los documentos como "(Documento, pág. X)" cuando sea relevante.
5. Si dos indicadores distintos aparecen en el contexto, NO los mezcles.
6. Si hay historial de conversación, ten en cuenta el hilo para responder.
7. Si la información es parcial, responde con lo que haya y aclara qué falta.
8. Si no encuentras la respuesta, di: "No tengo información suficiente en los 
   documentos para responder esa pregunta."

{history_block}

CONTEXTO:
{context_block}

PREGUNTA: {question}

RESPUESTA (en español, sin mencionar metadatos ni similitudes):"""


def process_ask_request(
    request: AskRequest,
    retriever: Retriever,
    db: Database,
) -> AskResponse:
    """
    Flujo completo de una pregunta:
        1. Determinar/crear conversación
        2. Recuperar contexto previo
        3. Recuperar chunks relevantes
        4. Generar respuesta con Qwen
        5. Persistir mensajes (user + assistant)
        6. Devolver respuesta
    """
    start_time = time.time()

    # 1. Determinar conversación
    conversation_id = request.conversation_id
    if not conversation_id:
        # Crear conversación nueva
        name = request.question[:100]  # primeras palabras como título
        conversation_id = db.create_conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            name=name,
        )
    else:
        # Verificar que existe
        if not db.conversation_exists(conversation_id):
            raise ValueError(f"Conversación {conversation_id} no existe o fue eliminada.")

    # 2. Recuperar contexto previo
    history = _build_context_from_payload(
        request.context,
        db,
        conversation_id,
        max_messages=10,
    )

    # 3. Recuperar chunks relevantes
    filters = request.filters
    top_k = filters.top_k if filters else 5
    domain = filters.domain if filters else None
    dimension = filters.dimension if filters else None
    indicator = filters.indicator if filters else None

    chunks = retriever.retrieve(
        question=request.question,
        top_k=top_k,
        domain=domain,
        dimension=dimension,
        indicator=indicator,
    )

    # 4. Generar respuesta
    if not chunks:
        answer = "No encontré información relevante en los documentos indexados."
        detected_domain = None
    else:
        prompt = _build_prompt_with_history(
            request.question, history, chunks, retriever
        )
        answer = retriever._generate(prompt)
        detected_domain = domain or retriever.detect_domain(request.question)

    latency_ms = int((time.time() - start_time) * 1000)

    # 5. Persistir mensajes
    next_pos = db.get_next_message_position(conversation_id)

    user_msg_id = db.save_conversation_message(
        conversation_id=conversation_id,
        role="user",
        content=request.question,
        position=next_pos,
        metadata=None,
    )

    assistant_metadata = {
        "model": retriever.__class__.__module__,  # ajustar si quieres el nombre del modelo
        "chunks_used": len(chunks),
        "detected_domain": detected_domain,
        "latency_ms": latency_ms,
        "sources": [
            {
                "source": c.get("source_name"),
                "document": c.get("document_title"),
                "page": c.get("page_number"),
                "dimension": c.get("dimension"),
                "indicator": c.get("indicator"),
                "similarity": round(c.get("similarity", 0), 4),
            }
            for c in chunks
        ],
    }

    assistant_msg_id = db.save_conversation_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        position=next_pos + 1,
        metadata=assistant_metadata,
    )

    # 6. Preparar respuesta
    sources = [
        SourceItem(
            source=c.get("source_name"),
            document=c.get("document_title"),
            page=c.get("page_number"),
            dimension=c.get("dimension"),
            indicator=c.get("indicator"),
            similarity=round(c.get("similarity", 0), 4),
        )
        for c in chunks
    ]

    return AskResponse(
        conversation_id=conversation_id,
        user_message_id=user_msg_id,
        assistant_message_id=assistant_msg_id,
        answer=answer,
        sources=sources,
        detected_domain=detected_domain,
        chunks_used=len(chunks),
        latency_ms=latency_ms,
    )