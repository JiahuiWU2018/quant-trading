"""Core CVXPY mean-variance optimiser with solver fallback chain.

Supports three problem formulations:
  1. risk_aversion  — maximise expected_return - λ * portfolio_variance
  2. target_return  — minimise variance subject to return >= target
  3. target_vol     — maximise expected_return subject to vol <= target

All three share the same solver fallback chain:
  OSQP → ECOS → SCS → MOSEK (optional, requires license)

The OptimResult dataclass captures solver diagnostics so failures can be
investigated without re-running the full optimisation.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Ordered solver preference. MOSEK is tried last — only available if installed.
DEFAULT_SOLVER_CHAIN = ["OSQP", "ECOS", "SCS", "MOSEK"]


@dataclass
class OptimResult:
    """Output of a portfolio optimisation run.

    Attributes:
        weights: Optimal portfolio weights, shape (n,).
        status: CVXPY solver status string ("optimal", "infeasible", etc.).
        solver_used: Name of the solver that produced this result.
        expected_return: Portfolio expected return (annualised).
        expected_vol: Portfolio expected volatility (annualised).
        sharpe: Expected return / expected vol (no risk-free rate adjustment here).
        solve_time_ms: Wall-clock solve time in milliseconds.
        diagnostics: Raw CVXPY problem value and solver metadata.
    """

    weights: np.ndarray
    status: str
    solver_used: str
    expected_return: float
    expected_vol: float
    sharpe: float
    solve_time_ms: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        """True if the solver reached a provably optimal solution."""
        return self.status in ("optimal", "optimal_inaccurate")


def mean_variance_optimize(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    *,
    risk_aversion: float | None = 1.0,
    target_return: float | None = None,
    target_vol: float | None = None,
    constraints: list | None = None,
    solver_chain: list[str] | None = None,
) -> OptimResult:
    """Mean-variance portfolio optimisation via CVXPY.

    Exactly one of ``risk_aversion``, ``target_return``, or ``target_vol``
    should be the active formulation. If multiple are provided, precedence
    is: target_vol > target_return > risk_aversion.

    Args:
        expected_returns: Expected return vector, shape (n,). Units should be
            consistent with the covariance matrix (e.g. both annualised).
        cov: Covariance matrix, shape (n, n). Must be positive semi-definite.
            Use covariance.nearest_positive_definite() if needed.
        risk_aversion: λ in ``maximise μᵀw - λ · wᵀΣw``. Higher values
            penalise variance more. Active when target_return and target_vol
            are both None.
        target_return: Minimise variance subject to ``μᵀw >= target_return``.
        target_vol: Maximise expected return subject to ``||L w||₂ ≤ target_vol``
            where L = chol(Σ). This is a standard SOCP constraint.
        constraints: List of additional CVXPY constraints *or* callables of the
            form ``(w: cp.Variable) -> list[cp.Constraint]``.
            If None, defaults to long-only + fully-invested.
            If provided, these replace (not augment) the defaults — include
            fully_invested() / long_only() explicitly if still desired.
        solver_chain: Ordered list of solver names to try. Defaults to
            DEFAULT_SOLVER_CHAIN (OSQP → ECOS → SCS → MOSEK).

    Returns:
        OptimResult with optimal weights and solver diagnostics.
        If all solvers fail or the problem is infeasible, returns an
        OptimResult with status="infeasible" and equal weights as fallback.

    Raises:
        ValueError: If expected_returns and cov have incompatible shapes.
    """
    import cvxpy as cp  # type: ignore[import]
    from quant_trading.optim.constraints import fully_invested, long_only

    n = len(expected_returns)
    if cov.shape != (n, n):
        raise ValueError(
            f"expected_returns has length {n} but cov has shape {cov.shape}."
        )

    solver_chain = solver_chain or DEFAULT_SOLVER_CHAIN
    w = cp.Variable(n)

    # Resolve constraints: support raw CVXPY constraints and factory callables
    if constraints is None:
        resolved_constraints = [*long_only(w), *fully_invested(w)]
    else:
        resolved_constraints = []
        for c in constraints:
            if callable(c):
                resolved_constraints.extend(c(w))
            else:
                resolved_constraints.append(c)

    # Build objective
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov)

    if target_vol is not None:
        # Maximise return subject to vol constraint.
        # Use Cholesky decomposition to express the SOCP cone constraint in
        # DCP-compliant form: ||L w||₂ ≤ target_vol  (L = chol(Σ)).
        objective = cp.Maximize(portfolio_return)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fall back to nearest PD if cov is numerically singular
            from quant_trading.optim.covariance import nearest_positive_definite
            L = np.linalg.cholesky(nearest_positive_definite(cov))
        constraints = resolved_constraints + [cp.norm(L @ w, 2) <= target_vol]
    elif target_return is not None:
        # Minimise variance subject to return constraint
        objective = cp.Minimize(portfolio_variance)
        constraints = resolved_constraints + [portfolio_return >= target_return]
    else:
        # Risk-aversion utility maximisation
        lam = risk_aversion if risk_aversion is not None else 1.0
        objective = cp.Maximize(portfolio_return - lam * portfolio_variance)
        constraints = resolved_constraints

    problem = cp.Problem(objective, constraints)

    # Try each solver in the fallback chain
    t0 = time.perf_counter()
    last_exc: Exception | None = None
    for solver_name in solver_chain:
        try:
            problem.solve(solver=solver_name, warm_start=True)
            if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                break
            logger.debug("Solver %s returned status=%s — trying next.", solver_name, problem.status)
        except Exception as exc:
            logger.debug("Solver %s raised %s — trying next.", solver_name, exc)
            last_exc = exc
            continue
    else:
        solver_name = solver_chain[-1]   # last attempted

    solve_time_ms = (time.perf_counter() - t0) * 1000

    # Fallback to equal weights if all solvers fail
    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        logger.warning(
            "All solvers failed or returned infeasible (status=%s). "
            "Returning equal-weight fallback. Last error: %s",
            problem.status,
            last_exc,
        )
        weights = np.ones(n) / n
        exp_ret = float(expected_returns @ weights)
        exp_vol = float(np.sqrt(weights @ cov @ weights))
        return OptimResult(
            weights=weights,
            status=problem.status or "infeasible",
            solver_used=solver_name,
            expected_return=exp_ret,
            expected_vol=exp_vol,
            sharpe=exp_ret / exp_vol if exp_vol > 0 else 0.0,
            solve_time_ms=solve_time_ms,
            diagnostics={"fallback": "equal_weight", "last_error": str(last_exc)},
        )

    weights = np.clip(w.value, 0, None)   # numerical cleanup
    weights /= weights.sum()              # re-normalise after clipping

    exp_ret = float(expected_returns @ weights)
    exp_vol = float(np.sqrt(weights @ cov @ weights))

    logger.info(
        "MVO solved: solver=%s status=%s n=%d ret=%.4f vol=%.4f sharpe=%.2f time=%.1fms",
        solver_name,
        problem.status,
        n,
        exp_ret,
        exp_vol,
        exp_ret / exp_vol if exp_vol > 0 else 0.0,
        solve_time_ms,
    )

    return OptimResult(
        weights=weights,
        status=problem.status,
        solver_used=solver_name,
        expected_return=exp_ret,
        expected_vol=exp_vol,
        sharpe=exp_ret / exp_vol if exp_vol > 0 else 0.0,
        solve_time_ms=solve_time_ms,
        diagnostics={
            "problem_value": problem.value,
            "solver_stats": problem.solver_stats,
        },
    )
