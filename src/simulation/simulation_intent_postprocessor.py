"""
Deterministic post-processor for simulation intents.

qwen2.5:1.5b sometimes confuses:
    - reduce_multiple with compare_improvements
    - increase_multiple with compare_worsenings

The 7b model handles these correctly, but we want the speed of the 1.5b.
This post-processor fixes those cases with pure-Python rules based on
unambiguous textual cues.

Cues:
    MULTIPLE  → "y", "además", "simultáneamente", "junto con", "al mismo tiempo"
    COMPARE   → "vs", "o", "comparado con", "cuál de", "qué ... más"

The post-processor fires ONLY when the LLM classified as COMPARE but the
question clearly has MULTIPLE cues, or vice versa.
"""
import re
import logging

from src.core.text_normalizer import normalize

logger = logging.getLogger(__name__)


# Signals for "multiple simultaneous changes"
_MULTIPLE_PATTERNS = [
    r"\by\b",
    r"\bademas\b",
    r"\bsimultaneamente\b",
    r"\bjunto con\b",
    r"\bal mismo tiempo\b",
]

# Signals for "comparison between two options"
_COMPARE_PATTERNS = [
    r"\bvs\b",
    r"\bcomparado con\b",
    r"\bcomparada con\b",
    r"\bque reduce mas\b",
    r"\bque aumenta mas\b",
    r"\bcual reduce mas\b",
    r"\bcual aumenta mas\b",
    r"\bcual de\b",
    r"\bo\b.*\b(cual|que)\b",
]


def _has_multiple_cue(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _MULTIPLE_PATTERNS)


def _has_compare_cue(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _COMPARE_PATTERNS)


def postprocess(question: str, classified: dict) -> dict:
    """
    Override the LLM's intent when textual cues contradict it.
    """
    intent = classified.get("intent", "no_data")
    params = classified.get("params", {})

    if intent not in {
        "compare_improvements", "reduce_multiple",
        "compare_worsenings", "increase_multiple",
    }:
        return classified

    q_norm = normalize(question)
    has_multiple = _has_multiple_cue(q_norm)
    has_compare = _has_compare_cue(q_norm)

    # If both cues present, prefer the LLM (usually it gets it right when
    # the wording is ambiguous).
    if has_multiple == has_compare:
        return classified

    # Case A: the LLM said COMPARE but the question is MULTIPLE
    if has_multiple and not has_compare:
        if intent in ("compare_improvements", "compare_worsenings"):
            new_intent = (
                "reduce_multiple" if intent == "compare_improvements"
                else "increase_multiple"
            )
            # Rebuild the "changes" list from deprivation_a/deprivation_b
            dep_a = params.get("deprivation_a")
            dep_b = params.get("deprivation_b")
            delta = params.get("delta_pct", -10 if "improvements" in intent else 10)

            changes = []
            if dep_a:
                changes.append({"deprivation": dep_a, "delta_pct": delta})
            if dep_b:
                changes.append({"deprivation": dep_b, "delta_pct": delta})

            if not changes:
                return {
                    "intent": "no_data",
                    "params": {},
                    "postprocess_note": "empty_changes_after_fix",
                }

            new_params = {
                "changes": changes,
                "entity_type": params.get("entity_type", "national"),
            }
            for k in ("entity_name", "year"):
                if k in params:
                    new_params[k] = params[k]

            logger.info(
                "Postprocess fixed %s → %s (multiple cue)", intent, new_intent
            )
            return {
                "intent": new_intent,
                "params": new_params,
                "postprocess_note": f"fixed_{intent}_to_{new_intent}",
            }

    # Case B: the LLM said MULTIPLE but the question is COMPARE
    if has_compare and not has_multiple:
        if intent in ("reduce_multiple", "increase_multiple"):
            changes = params.get("changes", [])
            if len(changes) == 2:
                # Rebuild compare params
                new_intent = (
                    "compare_improvements" if intent == "reduce_multiple"
                    else "compare_worsenings"
                )
                new_params = {
                    "deprivation_a": changes[0]["deprivation"],
                    "deprivation_b": changes[1]["deprivation"],
                    "delta_pct": changes[0].get("delta_pct", -10),
                    "entity_type": params.get("entity_type", "national"),
                }
                for k in ("entity_name", "year"):
                    if k in params:
                        new_params[k] = params[k]

                logger.info(
                    "Postprocess fixed %s → %s (compare cue)", intent, new_intent
                )
                return {
                    "intent": new_intent,
                    "params": new_params,
                    "postprocess_note": f"fixed_{intent}_to_{new_intent}",
                }

    return classified