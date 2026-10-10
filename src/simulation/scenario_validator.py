"""
Scenario validator.

Rejects scenarios that are logically impossible or out of range,
before running the (relatively) expensive simulation.
"""
import logging
from typing import Optional

from src.simulation.ipm_weights import DANE_WEIGHTS, DANE_THRESHOLD

logger = logging.getLogger(__name__)


# Range checks
MAX_DELTA_PCT = 100.0
MIN_DELTA_PCT = -100.0


class ScenarioValidator:
    """Validates a simulation intent before executing it."""

    def validate(self, intent: str, params: dict) -> tuple[bool, str]:
        """
        Returns (ok, reason).
        If ok is False, reason explains why.
        """
        # Delta range
        if "delta_pct" in params:
            d = params["delta_pct"]
            if not (MIN_DELTA_PCT <= d <= MAX_DELTA_PCT):
                return False, (
                    f"El cambio porcentual ({d}%) está fuera del rango "
                    f"[{MIN_DELTA_PCT}, {MAX_DELTA_PCT}]."
                )

        # Deprivation exists
        for key in ("deprivation", "deprivation_a", "deprivation_b"):
            if key in params and params[key] not in DANE_WEIGHTS:
                return False, f"Privación desconocida: {params[key]}"

        # Changes list
        if "changes" in params:
            changes = params["changes"]
            if not changes:
                return False, "La lista de cambios está vacía."
            for c in changes:
                if c.get("deprivation") not in DANE_WEIGHTS:
                    return False, f"Privación desconocida en cambios: {c.get('deprivation')}"
                d = c.get("delta_pct", 0)
                if not (MIN_DELTA_PCT <= d <= MAX_DELTA_PCT):
                    return False, f"Cambio fuera de rango: {d}%"

        # Target IPM sanity
        if "target_ipm" in params:
            t = params["target_ipm"]
            if not (0 <= t <= 100):
                return False, f"IPM objetivo fuera de rango: {t}%"
            if intent == "target_ipm" and t >= 100:
                return False, "El IPM objetivo debe ser menor a 100%."
            if intent == "target_ipm_increase" and t <= 0:
                return False, "El IPM objetivo debe ser mayor a 0%."

        # Dimension exists
        if "dimension" in params:
            dim = params["dimension"]
            if dim not in DANE_WEIGHTS and not isinstance(dim, str):
                return False, f"Dimensión inválida: {dim}"

        return True, ""