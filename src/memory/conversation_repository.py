"""
Repository for conversations and messages.

All operations are scoped to a user_id (multi-tenant safe).
"""
import logging
from typing import Optional

from src.storage.users_db import UsersDatabase

from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


class ConversationRepository:
    """CRUD operations for ai_conversation and ai_conversation_message."""

    def __init__(self, db: Optional[UsersDatabase] = None):
        self.db = db or UsersDatabase()
        self.schema = self.db.schema

    def close(self):
        if self.db:
            self.db.close()

    # ----------------------------------------------------------
    # Conversations
    # ----------------------------------------------------------
    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[dict]:
        sql = f"""
            SELECT id, user_id, name, is_deleted, created_at
            FROM {self.schema}.ai_conversation
            WHERE id = %(cid)s
              AND user_id = %(uid)s
              AND is_deleted = FALSE
        """
        try:
            rows = self.db.execute_select(sql, {
                "cid": conversation_id,
                "uid": user_id,
            })
            return rows[0] if rows else None
        except Exception as exc:
            logger.error("Error cargando conversación %s: %s", conversation_id, exc)
            return None

    def touch_conversation(self, conversation_id: str) -> None:
        'Para cuando agregen update_at'
        pass

    # ----------------------------------------------------------
    # Messages
    # ----------------------------------------------------------
    def get_messages(self, conversation_id: str, limit: int = 10) -> list[dict]:
        sql = f"""
            SELECT id, role, content, position, metadata, created_at
            FROM (
                SELECT *
                FROM {self.schema}.ai_conversation_message
                WHERE conversation_id = %(cid)s
                  AND is_deleted = FALSE
                ORDER BY position DESC
                LIMIT %(limit)s
            ) sub
            ORDER BY position ASC
        """
        try:
            return self.db.execute_select(sql, {
                "cid": conversation_id,
                "limit": limit,
            })
        except Exception as exc:
            logger.error("Error cargando mensajes: %s", exc)
            return []

    def _next_position(self, conversation_id: str) -> int:
        sql = f"""
            SELECT COALESCE(MAX(position), 0) + 1 AS next_pos
            FROM {self.schema}.ai_conversation_message
            WHERE conversation_id = %(cid)s
        """
        try:
            rows = self.db.execute_select(sql, {"cid": conversation_id})
            return int(rows[0]["next_pos"]) if rows else 1
        except Exception as exc:
            logger.error("Error calculando posición: %s", exc)
            return 1

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        position = self._next_position(conversation_id)
        sql = f"""
            INSERT INTO {self.schema}.ai_conversation_message
                (conversation_id, role, content, position, metadata)
            VALUES
                (%(cid)s, %(role)s, %(content)s, %(position)s, %(metadata)s)
            RETURNING id, position, created_at
        """
        try:
            # psycopg v3 requires wrapping dicts for JSONB columns
            metadata_json = Jsonb(metadata) if metadata is not None else None
            return self.db.execute_write(sql, {
                "cid": conversation_id,
                "role": role,
                "content": content,
                "position": position,
            "metadata": metadata_json,
            }, returning=True)
        except Exception as exc:
            logger.error("Error guardando mensaje: %s", exc)
            return None

    def save_user_message(self, conversation_id: str, content: str) -> Optional[dict]:
        return self.save_message(conversation_id, "user", content, metadata=None)

    def save_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        return self.save_message(conversation_id, "assistant", content, metadata=metadata)