"""
Configuration for conversation memory.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Users database (user_management_db)
# ============================================================
USERS_DB_HOST = os.getenv("USERS_DB_HOST", "localhost")
USERS_DB_PORT = int(os.getenv("USERS_DB_PORT", "5432"))
USERS_DB_NAME = os.getenv("USERS_DB_NAME", "user_management_db")
USERS_DB_USER = os.getenv("USERS_DB_USER")
USERS_DB_PASSWORD = os.getenv("USERS_DB_PASSWORD")
USERS_DB_SCHEMA = os.getenv("USERS_DB_SCHEMA", "public")


# ============================================================
# Memory behavior
# ============================================================
MAX_HISTORY_MESSAGES = int(os.getenv("MEMORY_MAX_HISTORY_MESSAGES", "6"))
MAX_MESSAGE_CHARS = int(os.getenv("MEMORY_MAX_MESSAGE_CHARS", "500"))

ENABLE_REFORMULATION = os.getenv("MEMORY_ENABLE_REFORMULATION", "true").lower() in {
    "1", "true", "yes", "on"
}

REFORMULATION_MODEL = os.getenv(
    "OLLAMA_REFORMULATION_MODEL",
    os.getenv("OLLAMA_INTENT_MODEL", "qwen2.5:1.5b"),
)
REFORMULATION_MAX_TOKENS = int(os.getenv("MEMORY_REFORMULATION_MAX_TOKENS", "100"))
REFORMULATION_TEMPERATURE = float(os.getenv("MEMORY_REFORMULATION_TEMPERATURE", "0.0"))


if __name__ == "__main__":
    print("Memory configuration:")
    print(f"  Users DB: {USERS_DB_HOST}:{USERS_DB_PORT}/{USERS_DB_NAME}")
    print(f"  Max history messages: {MAX_HISTORY_MESSAGES}")
    print(f"  Reformulation: {ENABLE_REFORMULATION}")
    print(f"  Reformulation model: {REFORMULATION_MODEL}")