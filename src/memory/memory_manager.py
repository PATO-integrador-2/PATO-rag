"""
Memory manager.

Coordinates loading history and saving messages. Used by the API layer.
"""
import logging
from typing import Optional

from src.memory.conversation_repository import ConversationRepository
from src.memory.memory_config import MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)


class MemoryManager:
    """Load history and save messages for a conversation."""

    def __init__(self, repo: Optional[ConversationRepository] = None):
        self.repo = repo or ConversationRepository()

    def close(self):
        if self.repo:
            self.repo.close()

    def validate_conversation(self, conversation_id: str, user_id: str) -> bool:
        conv = self.repo.get_conversation(conversation_id, user_id)
        return conv is not None

    def load_history(
        self,
        conversation_id: str,
        limit: int = MAX_HISTORY_MESSAGES,
    ) -> list[dict]:
        try:
            rows = self.repo.get_messages(conversation_id, limit=limit)
            return [
                {"role": r["role"], "content": r["content"]}
                for r in rows
                if r["role"] in ("user", "assistant")
            ]
        except Exception as exc:
            logger.error("Error cargando historial: %s", exc)
            return []

    def save_user_message(self, conversation_id: str, content: str) -> None:
        try:
            self.repo.save_user_message(conversation_id, content)
        except Exception as exc:
            logger.error("Error guardando mensaje del usuario: %s", exc)

    def save_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        try:
            self.repo.save_assistant_message(conversation_id, content, metadata)
        except Exception as exc:
            logger.error("Error guardando mensaje del asistente: %s", exc)

    def touch_conversation(self, conversation_id: str) -> None:
        try:
            self.repo.touch_conversation(conversation_id)
        except Exception:
            pass