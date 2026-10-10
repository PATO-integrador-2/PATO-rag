"""
Deterministic answer builder.

Takes the raw rows from the DB and formats them into a Spanish
response. No LLM involved — pure Python templates.

This eliminates hallucinations like "IPM 2418%" or "según datos del DANE"
when the data came from CEPAL.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AnswerBuilder:
    """Formats DB rows into a Spanish answer, per intent."""

    def build(self, intent: str, params: dict, rows: list[dict], source: str) -> str:
        """
        Return the Spanish answer. Falls back to a generic message if the
        intent has no template.
        """
        method_name = f"_format_{intent}"
        formatter = getattr(self, method_name, None)
        if formatter is None:
            return self._format_generic(intent, params, rows, source)
        try:
            return formatter(params, rows, source)
        except Exception as exc:
            logger.exception("Error formateando intent %s: %s", intent, exc)
            return self._format_generic(intent, params, rows, source)

    # ============================================================
    # Formatters per intent
    # ============================================================

    def _format_ipm_department_year(self, params: dict, rows: list[dict], source: str) -> str:
        dep = params.get("department", "ese departamento").title()
        if not rows:
            return f"No encontré datos del IPM para {dep} en el año solicitado."
        row = rows[0]
        year = row["year"]
        val = row["ipm_pct"]
        return (
            f"El Índice de Pobreza Multidimensional (IPM) de {dep} en {year} "
            f"fue del {val:.2f}%, según datos del DANE."
        )

    def _format_ipm_national_latest(self, params: dict, rows: list[dict], source: str) -> str:
        if not rows:
            return "No encontré datos recientes del IPM nacional."
        row = rows[0]
        return (
            f"El IPM nacional más reciente corresponde al año {row['year']}: "
            f"{row['ipm']:.2f}%, según datos del DANE."
        )

    def _format_ipm_national_year(self, params: dict, rows: list[dict], source: str) -> str:
        year = params.get("year")
        if not rows:
            return f"No encontré datos del IPM nacional para el año {year}."
        row = rows[0]
        return (
            f"El IPM nacional en {row['year']} fue del {row['ipm']:.2f}%, "
            f"según datos del DANE."
        )

    def _format_dimension_top_in_domain(self, params: dict, rows: list[dict], source: str) -> str:
        domain = params.get("domain", "ese dominio").title()
        if not rows:
            return f"No encontré datos de contribución por dimensión para {domain}."
        row = rows[0]
        return (
            f"En {domain} ({row['year']}), la dimensión que más contribuye al IPM "
            f"es {row['dimension']}, con un {row['pct']:.2f}% del total, "
            f"según datos del DANE."
        )

    def _format_deprivation_top(self, params: dict, rows: list[dict], source: str) -> str:
        if not rows:
            return "No encontré datos sobre privaciones en Colombia."
        row = rows[0]
        return (
            f"La privación más común en Colombia ({row['year']}) es "
            f"{row['variable']}, presente en el {row['pct']:.2f}% de los hogares, "
            f"según datos del DANE."
        )

    def _format_sex_comparison_national(self, params: dict, rows: list[dict], source: str) -> str:
        if not rows:
            return "No encontré datos de IPM por sexo del jefe de hogar."
        year = rows[0]["year"]
        values = {r["sex"]: r["pct"] for r in rows}
        male = values.get("Hombre")
        female = values.get("Mujer")
        if male is None or female is None:
            return "No encontré datos completos de IPM por sexo del jefe de hogar."
        diff = female - male
        comparison = (
            f"los hogares con jefatura femenina presentan un IPM {abs(diff):.2f} puntos "
            f"porcentuales {'mayor' if diff > 0 else 'menor'}"
        )
        return (
            f"En {year}, según datos del DANE, el IPM nacional es del {male:.2f}% "
            f"para hogares con jefe hombre y del {female:.2f}% para hogares con jefa mujer. "
            f"Es decir, {comparison}."
        )

    def _format_household_count_department(self, params: dict, rows: list[dict], source: str) -> str:
        dep = params.get("department", "ese departamento").title()
        if not rows:
            return f"No encontré datos de hogares para {dep}."
        year_str = f" en {params['year']}" if params.get("year") else ""
        total = rows[0]["total_hogares"]
        return (
            f"Según datos del DANE, {dep} registra {total:,} hogares{year_str} "
            f"en la muestra de pobreza multidimensional."
        ).replace(",", ".")

    def _format_ipm_country_year(self, params: dict, rows: list[dict], source: str) -> str:
        country = params.get("country", "ese país").title()
        if not rows:
            return f"No encontré datos del IPM para {country} en el año solicitado."
        row = rows[0]
        return (
            f"El IPM de {country} en {row['year']} fue del "
            f"{row['value_percentage']:.2f}%, según datos de la CEPAL."
        )

    def _format_country_top_poverty(self, params: dict, rows: list[dict], source: str) -> str:
        if not rows:
            return "No encontré datos de pobreza multidimensional por país."
        year = rows[0]["year"]
        lines = [f"Según datos de la CEPAL ({year}), los países con mayor pobreza multidimensional son:"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['country']}: {r['value_percentage']:.2f}%")
        return "\n".join(lines)

    def _format_ipm_country_age_group(self, params: dict, rows: list[dict], source: str) -> str:
        country = params.get("country", "ese país").title()
        if not rows:
            return f"No encontré datos de IPM por grupo de edad para {country}."
        year = rows[0]["year"]
        # Translate English age group labels to Spanish
        translations = {
            "0 to 17 years old": "0 a 17 años",
            "18 to 59 years old": "18 a 59 años",
            "60 years and over": "60 años o más",
            "Total": "Total",
        }
        lines = [f"Según datos de la CEPAL, en {country} ({year}):\n"]
        lines.append("| Grupo de edad | IPM |")
        lines.append("|---|---|")
        for r in rows:
            age = translations.get(r["age_group"], r["age_group"])
            lines.append(f"| {age} | {r['value_percentage']:.2f}% |")
        return "\n".join(lines)

    def _format_country_vs_latam(self, params: dict, rows: list[dict], source: str) -> str:
        # For BOTH intents, rows come as {"dane": [...], "cepal": [...]}
        dane_rows = rows.get("dane", []) if isinstance(rows, dict) else []
        cepal_rows = rows.get("cepal", []) if isinstance(rows, dict) else []

        if not dane_rows or not cepal_rows:
            return "No encontré datos suficientes para comparar Colombia con la región."

        dane = dane_rows[0]
        cepal = cepal_rows[0]
        avg_regional = cepal["avg_pct"]
        n_countries = cepal["n_countries"]
        colombia_val = dane["ipm"]
        diff = colombia_val - avg_regional
        comparison = (
            f"{abs(diff):.2f} puntos porcentuales por debajo"
            if diff < 0 else
            f"{abs(diff):.2f} puntos porcentuales por encima"
        )
        return (
            f"Según datos del DANE, el IPM nacional de Colombia en {dane['year']} "
            f"es del {colombia_val:.2f}%. Para el mismo periodo, el promedio regional "
            f"de {n_countries} países latinoamericanos (CEPAL) es del {avg_regional:.2f}%. "
            f"Colombia se encuentra {comparison} del promedio regional."
        )

    # ============================================================
    # Fallback
    # ============================================================
    def _format_generic(self, intent: str, params: dict, rows, source: str) -> str:
        if not rows:
            return "No encontré datos para esa consulta."
        if isinstance(rows, dict):
            return "No pude formatear la respuesta combinada."
        if len(rows) == 1:
            row = rows[0]
            pairs = ", ".join(f"{k}: {v}" for k, v in row.items())
            return f"Resultado ({source}): {pairs}."
        return f"Se encontraron {len(rows)} resultados ({source})."


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    b = AnswerBuilder()
    print(b.build(
        "ipm_department_year",
        {"department": "huila", "year": 2018},
        [{"year": 2018, "department_name": "Huila", "ipm_pct": 30.39}],
        "DANE",
    ))
    print()
    print(b.build(
        "sex_comparison_national",
        {},
        [
            {"year": 2025, "sex": "Hombre", "pct": 7.2},
            {"year": 2025, "sex": "Mujer", "pct": 9.8},
        ],
        "DANE",
    ))
    print()
    print(b.build(
        "country_top_poverty",
        {},
        [
            {"year": 2025, "country": "Ecuador", "value_percentage": 20.53},
            {"year": 2025, "country": "Colombia", "value_percentage": 4.0},
            {"year": 2025, "country": "Costa Rica", "value_percentage": 2.5},
        ],
        "CEPAL",
    ))
    print()
    print(b.build(
        "country_vs_latam",
        {"country": "colombia"},
        {
            "dane": [{"year": 2025, "domain": "Nacional", "ipm": 9.9}],
            "cepal": [{"year": 2025, "avg_pct": 8.5, "n_countries": 11}],
        },
        "BOTH",
    ))