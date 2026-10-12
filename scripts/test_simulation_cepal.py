"""Test battery for CEPAL simulation."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.cepal_simulator import CepalSimulator


def main():
    sim = CepalSimulator()
    try:
        countries = ["mexico", "chile", "argentina", "colombia", "peru"]
        for country in countries:
            print(f"\n{'=' * 70}")
            print(f"  {country.title()}")
            print(f"{'=' * 70}")

            data = sim.load_country_data(country, None)
            if not data["available"]:
                print(f"  Sin datos disponibles")
                continue

            print(f"  IPM actual ({data['year']}): {data['ipm']:.2f}%")
            print(f"  Método: {data['method']}")
            print(f"  Top 3 privaciones:")

            top = sim.top_deprivations(country, limit=3)
            for t in top:
                print(f"    - {t['deprivation']:<30} {t['contribution_pct']:.2f}%")

            # Simulate reducing the top deprivation by 20%
            top_dep = top[0]["deprivation"] if top else None
            if top_dep:
                result = sim.simulate(
                    country,
                    [{"deprivation": top_dep, "delta_pct": -20}],
                )
                print(f"\n  Simulación: {top_dep} baja 20%")
                print(f"    IPM antes:   {result['ipm_before']:.4f}%")
                print(f"    IPM después: {result['ipm_after']:.4f}%")
                print(f"    ΔIPM:        {result['delta_ipm_pp']:+.4f} pp")

    finally:
        sim.close()


if __name__ == "__main__":
    main()