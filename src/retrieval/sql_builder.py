"""
Deterministic SQL builder.

Takes a structured intent (from sql_intent_classifier) and produces a
parameterized SQL query. No LLM involved — pure Python templates.

Each template decides:
  - which view to use
  - which columns
  - whether to multiply by 100 or not (per column!)
  - which filters to apply

This eliminates the entire class of "column invention" and
"wrong scale" bugs.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# SQL templates
# ============================================================
# Each template is a dict with:
#   - sql: the SQL query (with %(param)s placeholders)
#   - params: the params needed (keys from intent["params"])

TEMPLATES: dict[str, dict] = {

    # ----------------------------------------------------------
    # DANE — Department IPM by year
    # ----------------------------------------------------------
    "ipm_department_year": {
        "sql": """
            SELECT year, department_name,
                   AVG(ipm_score) * 100 AS ipm_pct
            FROM vw_ipm_household_departmental
            WHERE department_name ILIKE %(department)s
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_ipm_household_departmental))
            GROUP BY year, department_name
            ORDER BY year DESC
            LIMIT 1
        """,
        "params": ["department", "year"],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # DANE — National IPM, latest year
    # ----------------------------------------------------------
    "ipm_national_latest": {
        "sql": """
            SELECT year, domain, ipm
            FROM vw_ipm_incidence
            WHERE domain = 'Nacional'
            ORDER BY year DESC
            LIMIT 1
        """,
        "params": [],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # DANE — National IPM by year
    # ----------------------------------------------------------
    "ipm_national_year": {
        "sql": """
            SELECT year, domain, ipm
            FROM vw_ipm_incidence
            WHERE domain = 'Nacional'
              AND year = %(year)s
            LIMIT 1
        """,
        "params": ["year"],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # DANE — Top dimension in a domain
    # ----------------------------------------------------------
    "dimension_top_in_domain": {
        "sql": """
            SELECT year, domain, dimension,
                   percentage AS pct
            FROM vw_ipm_contribution_dimension
            WHERE domain ILIKE %(domain)s
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_ipm_contribution_dimension))
            ORDER BY percentage DESC
            LIMIT 1
        """,
        "params": ["domain", "year"],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # DANE — Most common deprivation (national)
    # ----------------------------------------------------------
    "deprivation_top": {
        "sql": """
            SELECT year, domain, variable,
                   percentage AS pct
            FROM vw_ipm_deprivations_variable
            WHERE domain = 'Nacional'
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_ipm_deprivations_variable))
            ORDER BY percentage DESC
            LIMIT 1
        """,
        "params": ["year"],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # DANE — Sex comparison (household head)
    # ----------------------------------------------------------
    "sex_comparison_national": {
        "sql": """
            SELECT year, sex, percentage AS pct
            FROM vw_ipm_incidence_household_head_sex
            WHERE domain = 'Nacional'
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_ipm_incidence_household_head_sex))
            ORDER BY sex
        """,
        "params": ["year"],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # DANE — Household count by department
    # ----------------------------------------------------------
    "household_count_department": {
        "sql": """
            SELECT COUNT(*) AS total_hogares
            FROM vw_ipm_household_departmental
            WHERE department_name ILIKE %(department)s
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_ipm_household_departmental))
        """,
        "params": ["department", "year"],
        "source": "DANE",
    },

    # ----------------------------------------------------------
    # CEPAL — IPM by country and year
    # ----------------------------------------------------------
    "ipm_country_year": {
        "sql": """
            SELECT year, country, geographic_area,
                   measurement_type, value_percentage
            FROM vw_latam_ipm_national_population
            WHERE country ILIKE %(country)s
              AND geographic_area = 'National'
              AND measurement_type = 'Adjusted incidence (M0)'
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_latam_ipm_national_population))
            LIMIT 1
        """,
        "params": ["country", "year"],
        "source": "CEPAL",
    },

    # ----------------------------------------------------------
    # CEPAL — Top countries by poverty
    # ----------------------------------------------------------
    "country_top_poverty": {
        "sql": """
            SELECT year, country, value_percentage
            FROM vw_latam_ipm_national_population
            WHERE geographic_area = 'National'
              AND measurement_type = 'Adjusted incidence (M0)'
              AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_latam_ipm_national_population))
            ORDER BY value_percentage DESC
            LIMIT 5
        """,
        "params": ["year"],
        "source": "CEPAL",
    },

    # ----------------------------------------------------------
    # CEPAL — IPM by age group in a country
    # ----------------------------------------------------------
    "ipm_country_age_group": {
        "sql": """
            SELECT year, country, age_group, geographic_area,
                    value_percentage
            FROM vw_latam_ipm_age_group
            WHERE country ILIKE %(country)s
                AND geographic_area = 'National'
                AND year = COALESCE(%(year)s, (SELECT MAX(year) FROM vw_latam_ipm_age_group))
            ORDER BY age_group
        """,
        "params": ["country", "year"],
        "source": "CEPAL",
    },

    # ----------------------------------------------------------
    # BOTH — Country vs Latin America
    # ----------------------------------------------------------
    # This intent produces TWO queries: one DANE, one CEPAL.
    "country_vs_latam": {
        "sql_dane": """
            SELECT year, domain, ipm
            FROM vw_ipm_incidence
            WHERE domain = 'Nacional'
            ORDER BY year DESC
            LIMIT 1
        """,
        "sql_cepal": """
            SELECT MAX(year) AS year,
                AVG(value_percentage) AS avg_pct,
                COUNT(DISTINCT country) AS n_countries
            FROM vw_latam_ipm_national_population
            WHERE geographic_area = 'National'
                AND measurement_type = 'Adjusted incidence (M0)'
                AND year = (SELECT MAX(year) FROM vw_latam_ipm_national_population)
        """,
        "params": [],
        "source": "BOTH",
    },
}

# ============================================================
# SQL builder
# ============================================================
class SqlBuilder:
    """Builds SQL from a structured intent."""

    def build(self, intent: str, params: dict) -> dict:
        """
        Return:
            {
              "ok": bool,
              "error": str | None,
              "source": "DANE" | "CEPAL" | "BOTH",
              "queries": {"main": (sql, sql_params)}  # or
                         {"dane": (...), "cepal": (...)}
            }
        """
        template = TEMPLATES.get(intent)
        if not template:
            return {
                "ok": False,
                "error": f"Intención sin plantilla: {intent}",
                "source": None,
                "queries": {},
            }

        source = template.get("source")

        # Single-query intents
        if "sql" in template:
            try:
                bound = self._bind_params(template, params)
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "source": source,
                    "queries": {},
                }
            return {
                "ok": True,
                "error": None,
                "source": source,
                "queries": {"main": bound},
            }

        # BOTH intents (two queries)
        if "sql_dane" in template and "sql_cepal" in template:
            dane = (template["sql_dane"], {})
            cepal = (template["sql_cepal"], {})
            return {
                "ok": True,
                "error": None,
                "source": "BOTH",
                "queries": {"dane": dane, "cepal": cepal},
            }

        return {
            "ok": False,
            "error": f"Plantilla malformada para intent: {intent}",
            "source": source,
            "queries": {},
        }

    # ---------- Internals ----------

    @staticmethod
    def _bind_params(template: dict, params: dict) -> tuple[str, dict]:
        """
        Prepare the psycopg parameter dict. Values are never interpolated
        into the SQL string — they go through %(name)s placeholders.
        """
        required = template.get("params", [])
        bound: dict = {}

        for key in required:
            if key == "department":
                value = params.get("department")
                if not value:
                    raise ValueError("Falta el parámetro 'department'.")
                # ILIKE with wildcards
                bound["department"] = f"%{value}%"

            elif key == "country":
                value = params.get("country")
                if not value:
                    raise ValueError("Falta el parámetro 'country'.")
                bound["country"] = f"%{value}%"

            elif key == "domain":
                value = params.get("domain")
                if not value:
                    raise ValueError("Falta el parámetro 'domain'.")
                bound["domain"] = f"%{value}%"

            elif key == "year":
                # Optional: None is OK (COALESCE handles it)
                bound["year"] = params.get("year")

            else:
                bound[key] = params.get(key)

        # Normalize the SQL: collapse whitespace
        sql = " ".join(template["sql"].split())
        return sql, bound


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    builder = SqlBuilder()

    test_intents = [
        ("ipm_department_year", {"department": "huila", "year": 2018}),
        ("ipm_department_year", {"department": "antioquia", "year": None}),
        ("ipm_national_latest", {}),
        ("ipm_national_year", {"year": 2020}),
        ("dimension_top_in_domain", {"domain": "caribe", "year": None}),
        ("deprivation_top", {}),
        ("sex_comparison_national", {}),
        ("household_count_department", {"department": "antioquia", "year": 2022}),
        ("ipm_country_year", {"country": "mexico", "year": 2020}),
        ("country_top_poverty", {}),
        ("ipm_country_age_group", {"country": "mexico", "year": None}),
        ("country_vs_latam", {"country": "colombia"}),
    ]

    for intent, params in test_intents:
        result = builder.build(intent, params)
        print(f"\n=== {intent} {params} ===")
        print(f"  ok: {result['ok']}  source: {result['source']}")
        if result.get("error"):
            print(f"  error: {result['error']}")
        for name, (sql, sql_params) in result["queries"].items():
            print(f"  [{name}] SQL: {sql[:120]}...")
            print(f"  [{name}] params: {sql_params}")