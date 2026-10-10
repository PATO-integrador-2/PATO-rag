"""
Deterministic answer builder for simulations.

Takes the numeric result of a simulation and formats a Spanish response.
The LLM is used only for the wording of the closing sentence (optional),
or we can use a pure Python template.

For speed and to avoid hallucination, we use pure Python templates here.
"""
import logging

from src.simulation.ipm_weights import (
    DANE_DEPRIVATION_LABELS,
    DANE_DIMENSIONS,
    DANE_THRESHOLD,
)

logger = logging.getLogger(__name__)


THRESHOLD_PCT = DANE_THRESHOLD * 100  # ≈ 33.33


class SimulationAnswerBuilder:
    """Formats a Spanish answer from a simulation result."""

    def build(
        self,
        intent: str,
        params: dict,
        result: dict,
    ) -> str:
        method = getattr(self, f"_format_{intent}", None)
        if method is None:
            return self._format_generic(intent, params, result)
        try:
            return method(params, result)
        except Exception as exc:
            logger.exception("Error formateando intent %s: %s", intent, exc)
            return self._format_generic(intent, params, result)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _entity_label(params: dict) -> str:
        et = params.get("entity_type", "national")
        name = params.get("entity_name")
        if et == "national" or not name:
            return "Colombia"
        return name.title()

    @staticmethod
    def _weighted_selection_note() -> str:
        return (
            "Para la simulación se asume que el cambio afecta primero a los "
            "hogares menos vulnerables (aquellos con menos privaciones), "
            "que son los más fáciles de sacar de la pobreza con políticas públicas."
        )

    def _impact_block(self, result: dict) -> str:
        """Common impact block without repeating the entity."""
        return (
            f"el IPM pasaría de {result['ipm_before']:.2f}% a "
            f"{result['ipm_after']:.2f}%, un cambio de "
            f"{result['delta_ipm_pp']:+.2f} puntos porcentuales. "
            f"La incidencia (proporción de hogares pobres) pasaría de "
            f"{result['incidence_before']:.2f}% a {result['incidence_after']:.2f}%. "
            f"La simulación afectaría a {result['n_affected']} hogares de la "
            f"muestra ({result['n_hogares_total']} hogares en total). "
            f"{self._weighted_selection_note()} "
            f"Según datos del DANE."
        )

    # ----------------------------------------------------------
    # Improvement formatters
    # ----------------------------------------------------------
    def _format_reduce_one_deprivation(self, params: dict, result: dict) -> str:
        dep = DANE_DEPRIVATION_LABELS.get(params["deprivation"], params["deprivation"])
        delta = abs(params.get("delta_pct", 10))
        entity = self._entity_label(params)
        return (
            f"Si {dep} se reduce un {delta:.0f}% en {entity}, "
            + self._impact_block(result)
        )

    def _format_eliminate_one_deprivation(self, params: dict, result: dict) -> str:
        dep = DANE_DEPRIVATION_LABELS.get(params["deprivation"], params["deprivation"])
        entity = self._entity_label(params)
        return (
            f"Si se elimina por completo {dep} en {entity}, "
            + self._impact_block(result)
        )

    def _format_reduce_multiple(self, params: dict, result: dict) -> str:
        changes = params.get("changes", [])
        if not changes:
            return self._format_generic("reduce_multiple", params, result)

        parts = []
        for c in changes:
            dep = DANE_DEPRIVATION_LABELS.get(c["deprivation"], c["deprivation"])
            parts.append(f"{dep} un {abs(c['delta_pct']):.0f}%")

        if len(parts) == 1:
            changes_txt = parts[0]
        elif len(parts) == 2:
            changes_txt = f"{parts[0]} y {parts[1]}"
        else:
            changes_txt = ", ".join(parts[:-1]) + " y " + parts[-1]

        entity = self._entity_label(params)
        return (
            f"Si se reducen simultáneamente {changes_txt} en {entity}, "
            + self._impact_block(result)
        )

    def _format_target_ipm(self, params: dict, result: dict) -> str:
        dep = DANE_DEPRIVATION_LABELS.get(params["deprivation"], params["deprivation"])
        target = params["target_ipm"]
        actual = result["ipm_before"]
        entity = self._entity_label(params)

        if actual <= target:
            return (
                f"El IPM actual de {entity} ({actual:.2f}%) ya está por debajo "
                f"de la meta ({target:.2f}%). No se requiere cambio."
            )

        if not result.get("target_reached", False):
            return (
                f"El IPM actual de {entity} es {actual:.2f}% y la meta es "
                f"{target:.2f}%. Incluso reduciendo {dep} en un 100%, el IPM "
                f"solo llega a {result['ipm_after']:.2f}%. La meta no es "
                f"alcanzable únicamente con cambios en esa privación. "
                f"{self._weighted_selection_note()}"
            )

        return (
            f"Para que el IPM de {entity} baje de {actual:.2f}% a "
            f"{target:.2f}%, se requiere reducir {dep} en aproximadamente un "
            f"{abs(result['delta_required_pct']):.1f}%. "
            + self._impact_block(result)
        )

    def _format_compare_improvements(self, params: dict, result: dict) -> str:
        a = DANE_DEPRIVATION_LABELS.get(params["deprivation_a"], params["deprivation_a"])
        b = DANE_DEPRIVATION_LABELS.get(params["deprivation_b"], params["deprivation_b"])
        delta = abs(params["delta_pct"])
        ra = result.get("result_a", {})
        rb = result.get("result_b", {})
        winner = "A" if ra.get("delta_ipm_pp", 0) < rb.get("delta_ipm_pp", 0) else "B"
        w_label = a if winner == "A" else b
        return (
            f"Comparando reducir {a} vs {b} en un {delta:.0f}%:\n\n"
            f"- Reducir {a}: IPM {ra.get('ipm_before', 0):.2f}% → "
            f"{ra.get('ipm_after', 0):.2f}% ({ra.get('delta_ipm_pp', 0):+.2f} pp)\n"
            f"- Reducir {b}: IPM {rb.get('ipm_before', 0):.2f}% → "
            f"{rb.get('ipm_after', 0):.2f}% ({rb.get('delta_ipm_pp', 0):+.2f} pp)\n\n"
            f"Reducir {w_label} tiene mayor impacto sobre el IPM. "
            f"{self._weighted_selection_note()} Según datos del DANE."
        )

    def _format_dimension_impact(self, params: dict, result: dict) -> str:
        dim = params["dimension"]
        from src.simulation.ipm_weights import DANE_DIMENSIONS
        aliases = {
            "Educacion": "Educación",
            "Ninez y Juventud": "Niñez y Juventud",
            "Ninez": "Niñez y Juventud",
        }
        dim = aliases.get(dim, dim)
        delta = abs(params["delta_pct"])
        deprivations = DANE_DIMENSIONS.get(dim, [])
        depriv_txt = ", ".join(DANE_DEPRIVATION_LABELS.get(d, d) for d in deprivations)
        entity = self._entity_label(params)
        return (
            f"Si todas las privaciones de la dimensión {dim} ({depriv_txt}) "
            f"mejoran un {delta:.0f}% en {entity}, "
            + self._impact_block(result)
        )

    # ----------------------------------------------------------
    # Worsening formatters
    # ----------------------------------------------------------
    def _format_increase_one_deprivation(self, params: dict, result: dict) -> str:
        dep = DANE_DEPRIVATION_LABELS.get(params["deprivation"], params["deprivation"])
        delta = abs(params.get("delta_pct", 10))
        entity = self._entity_label(params)
        return (
            f"Si {dep} aumenta un {delta:.0f}% en {entity}, "
            + self._impact_block(result)
        )

    def _format_worsen_all_in_dimension(self, params: dict, result: dict) -> str:
        dim = params["dimension"]
        delta = abs(params["delta_pct"])
        entity = self._entity_label(params)
        return (
            f"Si todas las privaciones de la dimensión {dim} empeoran un "
            f"{delta:.0f}% en {entity}, "
            + self._impact_block(result)
        )

    def _format_increase_multiple(self, params: dict, result: dict) -> str:
        changes = params.get("changes", [])
        if not changes:
            return self._format_generic("increase_multiple", params, result)

        parts = []
        for c in changes:
            dep = DANE_DEPRIVATION_LABELS.get(c["deprivation"], c["deprivation"])
            parts.append(f"{dep} un {abs(c['delta_pct']):.0f}%")

        if len(parts) == 1:
            changes_txt = parts[0]
        elif len(parts) == 2:
            changes_txt = f"{parts[0]} y {parts[1]}"
        else:
            changes_txt = ", ".join(parts[:-1]) + " y " + parts[-1]

        entity = self._entity_label(params)
        return (
            f"Si se incrementan simultáneamente {changes_txt} en {entity}, "
            + self._impact_block(result)
        )

    def _format_target_ipm_increase(self, params: dict, result: dict) -> str:
        dep = DANE_DEPRIVATION_LABELS.get(params["deprivation"], params["deprivation"])
        target = params["target_ipm"]
        actual = result["ipm_before"]
        entity = self._entity_label(params)

        if actual >= target:
            return (
                f"El IPM actual de {entity} ({actual:.2f}%) ya supera la meta "
                f"({target:.2f}%). No se requiere cambio."
            )

        if not result.get("target_reached", False):
            return (
                f"El IPM actual de {entity} es {actual:.2f}% y la meta es "
                f"{target:.2f}%. Incluso aumentando {dep} en un 100%, el IPM "
                f"solo llega a {result['ipm_after']:.2f}%. La meta no es "
                f"alcanzable únicamente con cambios en esa privación."
            )

        return (
            f"Para que el IPM de {entity} suba de {actual:.2f}% a "
            f"{target:.2f}%, se requiere aumentar {dep} en aproximadamente un "
            f"{abs(result['delta_required_pct']):.1f}%. "
            + self._impact_block(result)
        )

    def _format_compare_worsenings(self, params: dict, result: dict) -> str:
        a = DANE_DEPRIVATION_LABELS.get(params["deprivation_a"], params["deprivation_a"])
        b = DANE_DEPRIVATION_LABELS.get(params["deprivation_b"], params["deprivation_b"])
        delta = abs(params["delta_pct"])
        ra = result.get("result_a", {})
        rb = result.get("result_b", {})
        winner = "A" if ra.get("delta_ipm_pp", 0) > rb.get("delta_ipm_pp", 0) else "B"
        w_label = a if winner == "A" else b
        return (
            f"Comparando aumentar {a} vs {b} en un {delta:.0f}%:\n\n"
            f"- Aumentar {a}: IPM {ra.get('ipm_before', 0):.2f}% → "
            f"{ra.get('ipm_after', 0):.2f}% ({ra.get('delta_ipm_pp', 0):+.2f} pp)\n"
            f"- Aumentar {b}: IPM {rb.get('ipm_before', 0):.2f}% → "
            f"{rb.get('ipm_after', 0):.2f}% ({rb.get('delta_ipm_pp', 0):+.2f} pp)\n\n"
            f"Aumentar {w_label} tiene mayor impacto sobre el IPM. "
            f"{self._weighted_selection_note()} Según datos del DANE."
        )

    def _format_generic(self, intent: str, params: dict, result: dict) -> str:
        return (
            f"Simulación completada. IPM antes: {result.get('ipm_before', 0):.2f}%, "
            f"después: {result.get('ipm_after', 0):.2f}% "
            f"({result.get('delta_ipm_pp', 0):+.2f} pp). Según datos del DANE."
        )


if __name__ == "__main__":
    b = SimulationAnswerBuilder()
    print(b.build("reduce_one_deprivation", {
        "deprivation": "dep_critical_overcrowding",
        "delta_pct": -10,
        "entity_type": "department",
        "entity_name": "Huila",
    }, {
        "ipm_before": 25.35, "ipm_after": 25.31,
        "incidence_before": 27.20, "incidence_after": 27.20,
        "intensity_before": 37.55, "intensity_after": 37.55,
        "delta_ipm_pp": -0.04, "delta_incidence_pp": 0.0,
        "n_hogares_total": 386, "n_poor_before": 105, "n_poor_after": 105,
        "n_affected": 4,
    }))