
from __future__ import annotations

from typing import List, Sequence, Tuple, Dict, Optional

import numpy as np
import pulp as pl

__all__ = [
    "FacilityLocationSolver",
    "HoltWintersForecaster",
]

# ----------------------------------------------------------------------------
# 1. OPTIMIZATION MODEL – FACILITY LOCATION
# ----------------------------------------------------------------------------
class FacilityLocationSolver:



    def __init__(
        self,
        fixed_costs: Sequence[float],
        transport_costs: Sequence[Sequence[float]],
        demand: Sequence[float],
        capacity: Optional[Sequence[float]] = None,
        solver: str = "PULP_CBC_CMD",
    ) -> None:
        self.f = np.asarray(fixed_costs, dtype=float)
        self.c = np.asarray(transport_costs, dtype=float)
        self.d = np.asarray(demand, dtype=float)
        self.cap = None if capacity is None else np.asarray(capacity, dtype=float)

        # Dimensions
        self.n, self.m = self.c.shape
        assert self.f.shape[0] == self.n, "fixed_costs length mismatch"
        assert self.d.shape[0] == self.m, "demand length mismatch"
        if self.cap is not None:
            assert self.cap.shape[0] == self.n, "capacity length mismatch"

        self._solver_name = solver
        self._prob: Optional[pl.LpProblem] = None
        self._y: Dict[Tuple[int], pl.LpVariable] = {}
        self._x: Dict[Tuple[int, int], pl.LpVariable] = {}
        self._status: Optional[str] = None

    # ---------------------------------------------------------------------
    # Model construction & solution
    # ---------------------------------------------------------------------
    def _build(self) -> None:
        prob = pl.LpProblem("FacilityLocation", pl.LpMinimize)

        # Decision variables
        self._y = {i: pl.LpVariable(f"y_{i}", cat=pl.LpBinary) for i in range(self.n)}
        self._x = {
            (i, j): pl.LpVariable(f"x_{i}_{j}", lowBound=0)
            for i in range(self.n)
            for j in range(self.m)
        }

        # Objective
        prob += (
            pl.lpSum(self.f[i] * self._y[i] for i in range(self.n))
            + pl.lpSum(self.c[i, j] * self._x[i, j] for i in range(self.n) for j in range(self.m))
        )

        # Demand satisfaction
        for j in range(self.m):
            prob += (
                pl.lpSum(self._x[i, j] for i in range(self.n)) >= self.d[j],
                f"Demand_{j}",
            )

        # Capacity constraints (if any)
        if self.cap is not None:
            for i in range(self.n):
                prob += (
                    pl.lpSum(self._x[i, j] for j in range(self.m)) <= self.cap[i] * self._y[i],
                    f"Capacity_{i}",
                )

        self._prob = prob

    def solve(self, time_limit: Optional[int] = None, msg: bool = False) -> str:
        if self._prob is None:
            self._build()
        
        # Fix: Create the solver correctly
        if time_limit is not None:
            solver = pl.PULP_CBC_CMD(timeLimit=time_limit, msg=msg)
        else:
            solver = pl.PULP_CBC_CMD(msg=msg)
        
        self._prob.solve(solver)
        self._status = pl.LpStatus[self._prob.status]
        return self._status


    # ---------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------
    def open_facilities(self) -> List[int]:
        self._ensure_solved()
        return [i for i, var in self._y.items() if var.value() > 0.5]

    def assignment_matrix(self) -> np.ndarray:
        """Quantity shipped from facility i to demand j."""
        self._ensure_solved()
        x = np.zeros((self.n, self.m))
        for (i, j), var in self._x.items():
            x[i, j] = var.value() or 0.0
        return x

    def total_cost(self) -> float:
        self._ensure_solved()
        return pl.value(self._prob.objective)

    def _ensure_solved(self) -> None:
        if self._status is None:
            raise RuntimeError("Model not yet solved. Call 'solve()' first.")
        if self._status != "Optimal":
            raise RuntimeError(f"Solver status: {self._status}")

# ----------------------------------------------------------------------------
# 2. FORECASTING MODEL – HOLT‑WINTERS ADDITIVE
# ----------------------------------------------------------------------------
class HoltWintersForecaster:


    def __init__(
        self,
        season_length: int,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        assert season_length >= 1, "season_length must be ≥ 1"
        self.m = season_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # State after fitting
        self._level: float | None = None
        self._trend: float | None = None
        self._season: List[float] | None = None
        self._fitted = False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self, series: Sequence[float], grid: Sequence[float] = (0.2, 0.4, 0.6, 0.8)) -> None:
        y = np.asarray(series, dtype=float)
        n = len(y)
        assert n >= 2 * self.m, "Need at least two full seasons of data"

        # Parameter selection if not provided
        if None in (self.alpha, self.beta, self.gamma):
            best_mse = float("inf")
            best_params = (0.2, 0.1, 0.1)
            for a in grid:
                for b in grid:
                    for g in grid:
                        lvl, trd, seas, mse = self._hw_update(y, a, b, g, update_only=True)
                        if mse < best_mse:
                            best_mse, best_params = mse, (a, b, g)
            self.alpha, self.beta, self.gamma = best_params

        # Final fit with chosen parameters, store state
        lvl, trd, seas, _ = self._hw_update(y, self.alpha, self.beta, self.gamma, update_only=False)
        self._level, self._trend, self._season = lvl, trd, seas
        self._fitted = True

    def forecast(self, horizon: int) -> np.ndarray:
        """Return point forecasts for `horizon` steps ahead."""
        if not self._fitted:
            raise RuntimeError("Call 'fit' before forecasting")
        fcast = np.empty(horizon)
        for k in range(1, horizon + 1):
            idx = (k - 1) % self.m
            fcast[k - 1] = self._level + k * self._trend + self._season[idx]
        return fcast

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initial_components(self, y: np.ndarray) -> Tuple[float, float, List[float]]:
        """Compute initial level, trend, and seasonal components."""
        season_averages = [np.mean(y[i : i + self.m]) for i in range(0, self.m * 2, self.m)]
        overall_avg = np.mean(season_averages)
        season_init = [np.mean(y[i::self.m]) - overall_avg for i in range(self.m)]
        level_init = overall_avg
        trend_init = (season_averages[1] - season_averages[0]) / self.m
        return level_init, trend_init, season_init

    def _hw_update(
        self,
        y: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        update_only: bool = False,
    ) -> Tuple[float, float, List[float], float]:
        level, trend, season = self._initial_components(y)
        season = list(season)
        m = self.m
        mse_accum = 0.0
        for t in range(len(y)):
            s_idx = t % m
            if t == 0:
                fitted = level + season[s_idx]
            else:
                fitted = (level + trend) + season[s_idx]
            error = y[t] - fitted
            mse_accum += error ** 2

            # Update components
            prev_level = level
            level = alpha * (y[t] - season[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            season[s_idx] = gamma * (y[t] - level) + (1 - gamma) * season[s_idx]

        mse = mse_accum / len(y)
        return level, trend, season, mse

# ----------------------------------------------------------------------------
# Example usage (can be removed if integrating as library)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # ----------------------- Facility Location Demo ---------------------
    np.random.seed(0)
    n_fac = 5
    n_dem = 8
    fixed = np.random.uniform(10, 50, size=n_fac)
    tcost = np.random.uniform(1, 20, size=(n_fac, n_dem))
    dem = np.random.uniform(5, 15, size=n_dem)
    cap = np.random.uniform(20, 40, size=n_fac)

    fl = FacilityLocationSolver(fixed, tcost, dem, cap)
    status = fl.solve(msg=False)
    print("Status:", status)
    print("Open facilities:", fl.open_facilities())
    print("Total cost:", fl.total_cost())

    # ----------------------- Holt‑Winters Demo ---------------------------
    # Synthetic monthly data with seasonality
    months = 60
    season_len = 12
    base = np.linspace(50, 70, months)
    season = 10 * np.sin(2 * np.pi * np.arange(months) / season_len)
    noise = np.random.normal(0, 2, months)
    series = base + season + noise

    hw = HoltWintersForecaster(season_length=season_len)
    hw.fit(series)
    print("Next‑year forecast:", hw.forecast(season_len))
