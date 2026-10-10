"""
Configuration for the Recommendation module.

The module combines:
    - RAG retrieval (knowledge.document_chunks) for methodology and
      policy context.
    - qwen2.5:3b to draft concrete project proposals.

No simulation, no invented numbers.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Intent classifier
# ============================================================
INTENT_LLM_MODEL = os.getenv(
    "OLLAMA_RECOMMENDATION_INTENT_MODEL",
    os.getenv("OLLAMA_INTENT_MODEL", "qwen2.5:1.5b"),
)
INTENT_MAX_TOKENS = int(os.getenv("RECOMMENDATION_INTENT_MAX_TOKENS", "150"))
INTENT_TEMPERATURE = float(os.getenv("RECOMMENDATION_INTENT_TEMPERATURE", "0.0"))


# ============================================================
# Retrieval (delegated to the RAG retriever)
# ============================================================
RETRIEVAL_TOP_K = int(os.getenv("RECOMMENDATION_RETRIEVAL_TOP_K", "3"))


# ============================================================
# Answer generation
# ============================================================
ANSWER_LLM_MODEL = os.getenv(
    "OLLAMA_RECOMMENDATION_ANSWER_MODEL",
    "qwen2.5:3b",
)
ANSWER_MAX_TOKENS = int(os.getenv("RECOMMENDATION_ANSWER_MAX_TOKENS", "300"))
ANSWER_TEMPERATURE = float(os.getenv("RECOMMENDATION_ANSWER_TEMPERATURE", "0.3"))
CONTEXT_MAX_CHARS = int(os.getenv("RECOMMENDATION_CONTEXT_MAX_CHARS", "2500"))


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    print("Recommendation configuration:")
    print(f"  Intent LLM:  {INTENT_LLM_MODEL}")
    print(f"  Answer LLM:  {ANSWER_LLM_MODEL}")
    print(f"  Retrieval:   top_{RETRIEVAL_TOP_K}")
    print(f"  Context max: {CONTEXT_MAX_CHARS} chars")