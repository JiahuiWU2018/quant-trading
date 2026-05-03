"""Equal risk contribution (risk parity) optimiser.

Risk parity targets equal marginal risk contribution from each asset,
making no assumptions about expected returns. This makes it more robust
to estimation error than MVO.

The problem is solved as a convex optimisation (Spinu 2013 formulation)
using the same CVXPY solver fallback chain as mean_variance.py.

Reference:
    Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights."
    SSRN: https://ssrn.com/abstract=2297383
"""

import logging
import time

import numpy as np

from quant_trading.optim.mean_variance import DEFAULT_SOLVER_CHAIN, OptimResult

logger = logging.getLogger(__name__)


def risk_parity_optimize(
    cov: np.ndarray,
    *,
    risk_budgets: np.ndarray | None = None,
    solver_chain: list[str] | None = None,
) -> OptimResult:
    """Equal (or budgeted) risk contribution portfolio.

    Finds weights such that each asset contributes equally to total portfolio
    risk (or in proportion to ``risk_budgets`` if supplied).

    The optimisation is formulated as:
        minimise  sum_i [ w_i * (Σw)_i / wᵀΣw - b_i ]²
    which is equivalent to Spinu's log-barrier approach for equal RC.

    We use the convex reformulation:
        minimise  (1/2) wᵀΣw - bᵀ log(w)
    subject to w > 0

    Then normalise: w ← w / sum(w).

    Args:
        cov: Covariance matrix, shape (n, n). Must be positive definite.
        risk_budgets: Desired risk contribution fractions, shape (n,).
            Must sum to 1. Defaults to equal (1/n each).
        solver_chain: Ordered list of solvers to try. Defaults to
            DEFAULT_SOLVER_CHAIN (OSQP → ECOS → SCS → MOSEK).
            Note: ECOS handles logarithmic objectives best; place it first
            for risk parity if speed matters.

    Returns:
        OptimResult with optimal weights. expected_return is NaN since
        risk parity does not use expected returns.

    Raises:
        ValueError: If cov is not square or risk_budgets do not sum to 1.
    """
    import cvxpy as cp  # type: ignore[import]

    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError(f"cov must be square, got shape {cov.shape}.")

    if risk_budgets is None:
        risk_budgets = np.ones(n) / n
    else:
        risk_budgets = np.asarray(risk_budgets, dtype=float)
        if abs(risk_budgets.sum() - 1.0) > 1e-6:
            raise ValueError(
                f"risk_budgets must sum to 1, got {risk_budgets.sum():.6f}."
            )
        if np.any(risk_budgets <= 0):
            raise ValueError("All risk_budgets must be strictly positive.")

    solver_chain = solver_chain or ["ECOS", "SCS", "MOSEK"]

    # Convex log-barrier formulation (Spinu 2013)
    w = cp.Variable(n, pos=True)   # positivity constraint built-in
    objective = cp.Minimize(
        0.5 * cp.quad_form(w, cov) - risk_budgets @ cp.log(w)
    )
    problem = cp.Problem(objective)

    t0 = time.perf_counter()
    last_exc: Exception | None = None
    used_solver = solver_chain[0]

    for solver_name in solver_chain:
        try:
            problem.solve(solver=solver_name)
            if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                used_solver = solver_name
                break
            logger.debug(
                "Risk parity: solver %s returned status=%s — trying next.",
                solver_name,
                problem.status,
            )
        except Exception as exc:
            logger.debug("Risk parity: solver %s raised %s — trying next.", solver_name, exc)
            last_exc = exc
            used_solver = solver_name
            continue

    solve_time_ms = (time.perf_counter() - t0) * 1000

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        logger.warning(
            "Risk parity: all solvers failed (status=%s). Returning equal-weight fallback.",
            problem.status,
        )
        weights = np.ones(n) / n
        exp_vol = float(np.sqrt(weights @ cov @ weights))
        return OptimResult(
            weights=weights,
            status=problem.status or "infeasible",
            solver_used=used_solver,
            expected_return=float("nan"),
            expected_vol=exp_vol,
            sharpe=float("nan"),
            solve_time_ms=solve_time_ms,
            diagnostics={"fallback": "equal_weight", "last_error": str(last_exc)},
        )

    # Normalise weights
    raw = w.value
    weights = raw / raw.sum()

    exp_vol = float(np.sqrt(weights @ cov @ weights))

    # Compute actual risk contributions for diagnostics
    marginal_risk = cov @ weights
    risk_contrib = weights * marginal_risk / exp_vol
    risk_contrib_pct = risk_contrib / risk_contrib.sum()

    logger.info(
        "Risk parity solved: solver=%s status=%s n=%d vol=%.4f time=%.1fms",
        used_solver,
        problem.status,
        n,
        exp_vol,
        solve_time_ms,
    )

    return OptimResult(
        weights=weights,
        status=problem.status,
        solver_used=used_solver,
        expected_return=float("nan"),   # risk parity makes no return assumptions
        expected_vol=exp_vol,
        sharpe=float("nan"),
        solve_time_ms=solve_time_ms,
        diagnostics={
            "risk_contributions": risk_contrib_pct,
            "risk_budgets": risk_budgets,
            "problem_value": problem.value,
        },
    )
