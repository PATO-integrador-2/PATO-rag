"""
CEPAL simulator.

Simulates the impact of changing a deprivation's contribution on the
IPM of a CEPAL country, using the linear contribution model:

    ΔIPM = IPM_current × (contribution_i / 100) × Δ_i

Handles different data shapes:
    - Some countries have Adjusted incidence (M0) → use directly.
    - Some countries have Incidence (H) and Intensity (A) → H × A.
    - Some countries only have Incidence (H) → use H as approximation.
    - Some countries have contributions but no IPM → not simulated.

Supported countries: those present in BOTH
    vw_latam_ipm_national_population AND
    vw_latam_ipm_deprivation_contribution.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.data_db import DataDatabase

logger = logging.getLogger(__name__)


class CepalSimulator:
    """Simulates scenario changes for CEPAL countries."""

    def __init__(self, db: Optional[DataDatabase] = None):
        self.db = db or DataDatabase()

    def close(self):
        if self.db:
            self.db.close()

    # ----------------------------------------------------------
    # Country IPM + contributions (aligned by year)
    # ----------------------------------------------------------
    def load_country_data(
        self,
        country: str,
        year: Optional[int] = None,
    ) -> dict:
        """
        Load the country's IPM and contributions for a given year.

        If `year` is None, use the most recent year available in BOTH
        the IPM table and the contributions table.
        """
        resolved_year = self._resolve_year(country, year)
        if resolved_year is None:
            return {
                "country": country,
                "year": None,
                "ipm": None,
                "contributions": {},
                "available": False,
                "reason": "No hay años en común entre IPM y contribuciones.",
            }

        ipm_data = self._load_ipm(country, resolved_year)
        if ipm_data is None:
            return {
                "country": country,
                "year": resolved_year,
                "ipm": None,
                "contributions": {},
                "available": False,
                "reason": f"Sin datos de IPM para {country} en {resolved_year}.",
            }

        contributions = self._load_contributions(country, resolved_year)

        return {
            "country": country,
            "year": resolved_year,
            "ipm": ipm_data["ipm"],
            "method": ipm_data["method"],
            "contributions": contributions,
            "available": True,
        }

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _resolve_year(
        self,
        country: str,
        year: Optional[int],
    ) -> Optional[int]:
        """
        Return a year present in BOTH IPM and contributions tables.
        """
        sql = """
            SELECT ipm.year
            FROM (
                SELECT DISTINCT year
                FROM vw_latam_ipm_national_population
                WHERE country ILIKE %(country)s
                  AND geographic_area = 'National'
            ) ipm
            INNER JOIN (
                SELECT DISTINCT year
                FROM vw_latam_ipm_deprivation_contribution
                WHERE country ILIKE %(country)s
            ) contrib ON contrib.year = ipm.year
            ORDER BY ipm.year DESC
        """
        try:
            rows = self.db.execute_select(
                sql, params={"country": f"%{country}%"}, timeout_seconds=15
            )
        except Exception as exc:
            logger.error("Error resolviendo año de %s: %s", country, exc)
            return None

        if not rows:
            return None

        years = [r["year"] for r in rows]
        if year is not None and year in years:
            return year
        return years[0]

    def _load_ipm(self, country: str, year: int) -> Optional[dict]:
        """
        Load the IPM for a country-year.

        Priority:
            1. Adjusted incidence (M0) — used directly.
            2. Incidence (H) × Intensity (A) — computed (as percentage).
            3. Incidence (H) alone — used as approximation.
        """
        sql = """
            SELECT measurement_type, value_percentage
            FROM vw_latam_ipm_national_population
            WHERE country ILIKE %(country)s
              AND geographic_area = 'National'
              AND year = %(year)s
        """
        try:
            rows = self.db.execute_select(
                sql,
                params={"country": f"%{country}%", "year": year},
                timeout_seconds=15,
            )
        except Exception as exc:
            logger.error("Error cargando IPM de %s/%s: %s", country, year, exc)
            return None

        if not rows:
            return None

        by_type = {
            r["measurement_type"]: float(r["value_percentage"]) for r in rows
        }

        # 1. M0 available: use directly (already in %)
        if "Adjusted incidence (M0)" in by_type:
            return {
                "ipm": by_type["Adjusted incidence (M0)"],
                "method": "Adjusted incidence (M0)",
            }

        # 2. H and A available: compute M0 = H × A
        #    H is already a percentage (0-100)
        #    A is a fraction (0-1)
        #    So: M0_% = H × A
        if "Incidence (H)" in by_type and "Intensity (A)" in by_type:
            h = by_type["Incidence (H)"]
            a = by_type["Intensity (A)"]
            return {
                "ipm": h * a,
                "method": "Calculated (H × A)",
            }

        # 3. Only H: use as approximation
        if "Incidence (H)" in by_type:
            return {
                "ipm": by_type["Incidence (H)"],
                "method": "Incidence (H) — aproximación",
            }

        return None

    def _load_contributions(self, country: str, year: int) -> dict[str, float]:
        sql = """
            SELECT deprivation, value_percentage
            FROM vw_latam_ipm_deprivation_contribution
            WHERE country ILIKE %(country)s
              AND year = %(year)s
            ORDER BY value_percentage DESC
        """
        try:
            rows = self.db.execute_select(
                sql,
                params={"country": f"%{country}%", "year": year},
                timeout_seconds=15,
            )
        except Exception as exc:
            logger.error("Error cargando contribuciones de %s/%s: %s", country, year, exc)
            return {}

        return {
            r["deprivation"]: float(r["value_percentage"])
            for r in rows
        }

    # ----------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------
    def simulate(
        self,
        country: str,
        changes: list[dict],
        year: Optional[int] = None,
    ) -> dict:
        data = self.load_country_data(country, year)

        if not data["available"] or data["ipm"] is None:
            return {
                "country": country,
                "year": data.get("year"),
                "ipm_before": None,
                "ipm_after": None,
                "delta_ipm_pp": None,
                "changes_applied": [],
                "available": False,
                "reason": data.get("reason", "Sin datos disponibles."),
            }

        ipm_before = data["ipm"]
        contributions = data["contributions"]

        total_delta = 0.0
        applied = []

        for change in changes:
            dep = change.get("deprivation")
            delta_pct = change.get("delta_pct", 0)

            if dep not in contributions:
                logger.warning(
                    "Privación '%s' no encontrada para %s. Ignorada.",
                    dep, country,
                )
                continue

            contrib_pct = contributions[dep]
            delta_ipm = ipm_before * (contrib_pct / 100.0) * (delta_pct / 100.0)

            total_delta += delta_ipm
            applied.append({
                "deprivation": dep,
                "delta_pct": delta_pct,
                "contribution_pct": contrib_pct,
                "delta_ipm_pp": round(delta_ipm, 4),
            })

        ipm_after = max(0.0, min(100.0, ipm_before + total_delta))

        return {
            "country": country,
            "year": data["year"],
            "ipm_before": round(ipm_before, 4),
            "ipm_after": round(ipm_after, 4),
            "delta_ipm_pp": round(total_delta, 4),
            "changes_applied": applied,
            "method": data["method"],
            "available": True,
        }

    # ----------------------------------------------------------
    # Helper: list countries available for simulation
    # ----------------------------------------------------------
    def list_available_countries(self) -> list[str]:
        sql = """
            SELECT DISTINCT ipm.country
            FROM vw_latam_ipm_national_population ipm
            INNER JOIN vw_latam_ipm_deprivation_contribution contrib
                ON contrib.country = ipm.country
            WHERE lower(ipm.country) NOT LIKE '%%latin america%%'
              AND lower(ipm.country) NOT LIKE '%%promedio%%'
              AND lower(ipm.country) NOT LIKE '%%average%%'
              AND lower(ipm.country) NOT LIKE '%%total%%'
            ORDER BY ipm.country
        """
        try:
            rows = self.db.execute_select(sql, timeout_seconds=15)
            return [r["country"] for r in rows]
        except Exception as exc:
            logger.error("Error listando países: %s", exc)
            return []

    def top_deprivations(
        self,
        country: str,
        year: Optional[int] = None,
        limit: int = 5,
    ) -> list[dict]:
        data = self.load_country_data(country, year)
        if not data["available"]:
            return []
        contribs = sorted(
            data["contributions"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"deprivation": dep, "contribution_pct": contrib}
            for dep, contrib in contribs[:limit]
        ]


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sim = CepalSimulator()
    try:
        print("=" * 70)
        print("  PAÍSES DISPONIBLES PARA SIMULACIÓN")
        print("=" * 70)
        countries = sim.list_available_countries()
        for c in countries:
            print(f"  - {c}")

        print()
        print("=" * 70)
        print("  TEST: México 2020")
        print("=" * 70)
        data = sim.load_country_data("mexico", 2020)
        if data["available"]:
            print(f"País: {data['country']}")
            print(f"Año: {data['year']}")
            print(f"IPM: {data['ipm']:.4f}%")
            print(f"Método: {data['method']}")
            print(f"Privaciones: {len(data['contributions'])}")

            result = sim.simulate(
                "mexico",
                [{"deprivation": "Internet access", "delta_pct": -20}],
                year=2020,
            )
            print(f"\nSimulación (Internet -20%):")
            print(f"  IPM antes:   {result['ipm_before']:.4f}%")
            print(f"  IPM después: {result['ipm_after']:.4f}%")
            print(f"  ΔIPM:        {result['delta_ipm_pp']:+.4f} pp")
        else:
            print(f"México: {data.get('reason')}")

        print()
        print("=" * 70)
        print("  TEST: Chile")
        print("=" * 70)
        data = sim.load_country_data("chile", None)
        if data["available"]:
            print(f"País: {data['country']}")
            print(f"Año: {data['year']}")
            print(f"IPM: {data['ipm']:.4f}%")
            print(f"Método: {data['method']}")
            print(f"Privaciones: {len(data['contributions'])}")

            result = sim.simulate(
                "chile",
                [{"deprivation": "Internet access", "delta_pct": -20}],
            )
            print(f"\nSimulación (Internet -20%):")
            print(f"  IPM antes:   {result['ipm_before']:.4f}%")
            print(f"  IPM después: {result['ipm_after']:.4f}%")
            print(f"  ΔIPM:        {result['delta_ipm_pp']:+.4f} pp")
        else:
            print(f"Chile: {data.get('reason')}")

    finally:
        sim.close()