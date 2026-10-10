"""
Microsimulator for IPM scenarios — NumPy-optimized version.

Loads household-level data from clean_data_db, applies scenario changes
with a weighted random selection biased toward less vulnerable households,
and recomputes the IPM.

NumPy optimization:
    - The 15 dep_* columns and the ipm_score are stored as a 2D boolean
      matrix and a 1D float array, respectively.
    - Summing weighted deprivations becomes a matrix-vector product.
    - Baseline and simulated aggregates are vectorized.

The math is exact:
    ipm_household = Σ (weight_i × deprivation_i)
    poor = ipm_household >= threshold

Weighted-random selection:
    When we say "reduce deprivation X by Y%", we simulate that the Y%
    of households MOST EASILY HELPED (i.e. those with the fewest existing
    deprivations) are the ones that stop having the deprivation. This is a
    policy-realistic assumption: it's easier to lift households that are
    close to the threshold.
"""
import sys
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.data_db import DataDatabase
from src.simulation.ipm_weights import (
    DANE_WEIGHTS,
    DANE_THRESHOLD,
    DANE_DIMENSIONS,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Column order — must match the DANE_WEIGHTS dict order
DEP_COLUMNS: list[str] = list(DANE_WEIGHTS.keys())
WEIGHTS_ARRAY: np.ndarray = np.array([DANE_WEIGHTS[c] for c in DEP_COLUMNS], dtype=np.float64)
N_DEPRIVATIONS: int = len(DEP_COLUMNS)
THRESHOLD_FLOAT: float = float(DANE_THRESHOLD)
_DEP_INDEX: dict[str, int] = {c: i for i, c in enumerate(DEP_COLUMNS)}


class Microsimulator:
    """Household-level IPM simulator (NumPy-backed)."""

    def __init__(self, db: Optional[DataDatabase] = None):
        self.db = db or DataDatabase()

    def close(self):
        if self.db:
            self.db.close()

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------
    def load_households(
        self,
        entity_type: str = "national",
        entity_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> dict:
        """
        Load households as a dict of numpy arrays:
            {
                "matrix": (n, 15) bool,
                "metadata": list[dict] with year, department_name, region,
                             household_members, ipm_score (from DB)
            }
        """
        dep_cols = ", ".join(DEP_COLUMNS)
        base_cols = f"year, department_name, region, household_members, ipm_score, {dep_cols}"

        where = []
        params: dict = {}

        if entity_type == "department" and entity_name:
            where.append("department_name ILIKE %(entity)s")
            params["entity"] = f"%{entity_name}%"

        if year is not None:
            where.append("year = %(year)s")
            params["year"] = year
        else:
            where.append("year = (SELECT MAX(year) FROM vw_ipm_household_departmental)")

        where_sql = " AND ".join(where)

        sql = f"""
            SELECT {base_cols}
            FROM vw_ipm_household_departmental
            WHERE {where_sql}
        """

        rows = self.db.execute_select(sql, params=params, timeout_seconds=60)

        if not rows:
            return {"matrix": np.zeros((0, N_DEPRIVATIONS), dtype=bool), "metadata": []}

        # Build the boolean matrix and metadata list
        n = len(rows)
        matrix = np.zeros((n, N_DEPRIVATIONS), dtype=bool)
        metadata = []

        for i, r in enumerate(rows):
            for j, col in enumerate(DEP_COLUMNS):
                matrix[i, j] = bool(r.get(col))
            metadata.append({
                "year": r["year"],
                "department_name": r["department_name"],
                "region": r["region"],
                "household_members": int(r["household_members"] or 0),
                "ipm_score_db": float(r["ipm_score"]) if r["ipm_score"] is not None else 0.0,
            })

        return {"matrix": matrix, "metadata": metadata}

    # ----------------------------------------------------------
    # Vectorized IPM computation
    # ----------------------------------------------------------
    @staticmethod
    def compute_ipm_scores(matrix: np.ndarray) -> np.ndarray:
        """Return (n,) float array of ipm_scores from a (n, 15) bool matrix."""
        # matrix is bool; weighted sum = matrix @ weights
        return matrix.astype(np.float64) @ WEIGHTS_ARRAY

    @staticmethod
    def compute_aggregate(scores: np.ndarray) -> dict:
        """Aggregate stats from a (n,) array of ipm_scores."""
        n = len(scores)
        if n == 0:
            return {"ipm": 0.0, "incidence": 0.0, "intensity": 0.0, "n": 0, "n_poor": 0}

        ipm = float(scores.mean() * 100)
        poor_mask = scores >= THRESHOLD_FLOAT
        n_poor = int(poor_mask.sum())
        incidence = n_poor / n * 100
        intensity = float(scores[poor_mask].mean() * 100) if n_poor else 0.0

        return {
            "ipm": round(ipm, 4),
            "incidence": round(incidence, 4),
            "intensity": round(intensity, 4),
            "n": n,
            "n_poor": n_poor,
        }

    # ----------------------------------------------------------
    # Weighted sampling (vectorized)
    # ----------------------------------------------------------
    @staticmethod
    def _weighted_sample_indices(
        candidate_indices: np.ndarray,
        weights: np.ndarray,
        n_to_pick: int,
        rng: random.Random,
    ) -> np.ndarray:
        """
        Weighted random sampling without replacement.

        For small n_to_pick relative to candidate count, this uses an
        iterative approach. For our sizes it's fast enough.
        """
        if n_to_pick >= len(candidate_indices):
            return candidate_indices

        # Use numpy RNG seeded from our deterministic seed
        np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        probs = weights / weights.sum()
        selected = np_rng.choice(candidate_indices, size=n_to_pick, replace=False, p=probs)
        return selected

    # ----------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------
    def simulate(
        self,
        households: dict,
        changes: list[dict],
    ) -> dict:
        """
        Apply changes to the household matrix and compute aggregate impact.

        changes: list of dicts like
            {"deprivation": "dep_critical_overcrowding", "delta_pct": -10}
        """
        matrix = households["matrix"]
        if matrix.shape[0] == 0:
            return self._empty_result()

        baseline_scores = self.compute_ipm_scores(matrix)
        baseline = self.compute_aggregate(baseline_scores)

        sim_matrix = matrix.copy()
        rng = random.Random(RANDOM_SEED)
        n_affected_total = 0
        deprivations_changed = []

        for change in changes:
            dep = change["deprivation"]
            delta = change["delta_pct"]
            if dep not in _DEP_INDEX:
                continue
            col = _DEP_INDEX[dep]
            target_fraction = abs(delta) / 100.0

            if delta < 0:
                # Reduce: pick households that HAVE the deprivation,
                # weighted by lower vulnerability (fewer total deprivations)
                has_dep = np.where(sim_matrix[:, col])[0]
                if len(has_dep) == 0:
                    continue
                n_to_change = int(round(len(has_dep) * target_fraction))
                if n_to_change == 0:
                    continue

                n_dep_per_household = sim_matrix[has_dep].sum(axis=1)
                weights = 1.0 / (1.0 + n_dep_per_household.astype(np.float64))
                selected = self._weighted_sample_indices(has_dep, weights, n_to_change, rng)
                sim_matrix[selected, col] = False

                n_affected_total += len(selected)
                deprivations_changed.append({
                    "deprivation": dep,
                    "delta_pct": delta,
                    "n_hogares_afectados": int(len(selected)),
                    "action": "reduced",
                })
            else:
                # Increase: pick households that DO NOT have the deprivation,
                # weighted by higher vulnerability (more total deprivations)
                not_has = np.where(~sim_matrix[:, col])[0]
                if len(not_has) == 0:
                    continue
                n_to_change = int(round(len(not_has) * target_fraction))
                if n_to_change == 0:
                    continue

                n_dep_per_household = sim_matrix[not_has].sum(axis=1)
                weights = 1.0 + n_dep_per_household.astype(np.float64)
                selected = self._weighted_sample_indices(not_has, weights, n_to_change, rng)
                sim_matrix[selected, col] = True

                n_affected_total += len(selected)
                deprivations_changed.append({
                    "deprivation": dep,
                    "delta_pct": delta,
                    "n_hogares_afectados": int(len(selected)),
                    "action": "increased",
                })

        after_scores = self.compute_ipm_scores(sim_matrix)
        after = self.compute_aggregate(after_scores)

        return {
            "ipm_before": baseline["ipm"],
            "ipm_after": after["ipm"],
            "incidence_before": baseline["incidence"],
            "incidence_after": after["incidence"],
            "intensity_before": baseline["intensity"],
            "intensity_after": after["intensity"],
            "delta_ipm_pp": round(after["ipm"] - baseline["ipm"], 4),
            "delta_incidence_pp": round(after["incidence"] - baseline["incidence"], 4),
            "n_hogares_total": baseline["n"],
            "n_poor_before": baseline["n_poor"],
            "n_poor_after": after["n_poor"],
            "n_affected": n_affected_total,
            "deprivations_changed": deprivations_changed,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "ipm_before": 0.0, "ipm_after": 0.0,
            "incidence_before": 0.0, "incidence_after": 0.0,
            "intensity_before": 0.0, "intensity_after": 0.0,
            "delta_ipm_pp": 0.0, "delta_incidence_pp": 0.0,
            "n_hogares_total": 0, "n_poor_before": 0, "n_poor_after": 0,
            "n_affected": 0, "deprivations_changed": [],
        }


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    sim = Microsimulator()
    try:
        print("Cargando hogares de Huila...")
        t0 = time.time()
        hogares = sim.load_households(entity_type="department", entity_name="huila")
        print(f"  {hogares['matrix'].shape[0]} hogares cargados en {(time.time()-t0)*1000:.0f} ms")
        print(f"  IPM actual: {sim.compute_aggregate(sim.compute_ipm_scores(hogares['matrix']))['ipm']:.4f}%")

        print("\nSimulando: reducir hacinamiento crítico 10%...")
        t0 = time.time()
        result = sim.simulate(hogares, [
            {"deprivation": "dep_critical_overcrowding", "delta_pct": -10},
        ])
        print(f"  Tiempo: {(time.time()-t0)*1000:.0f} ms")
        for k, v in result.items():
            if k != "deprivations_changed":
                print(f"  {k}: {v}")
    finally:
        sim.close()