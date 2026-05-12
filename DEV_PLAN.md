# Development Plan — quant-trading

> **Living document.** Update this file as phases complete or scope changes.
> Last updated: April 2026

---

## Architectural Principles

- This is a **public** repository. It contains only **generic, reusable infrastructure**.
- All strategy signals, parameters, model weights, and account configuration live in a **separate private repository** that imports this package.
- Backtrader is used for **backtesting only**.
- `ib_async` is used directly for **live and paper execution** (no Backtrader IBStore).
- CVXPY is the primary optimization interface; solver chain: OSQP → ECOS → SCS → MOSEK (optional).
- All credentials and sensitive values are read from environment variables. Nothing sensitive is ever committed.

---

## Execution Architecture Decision

```
Backtesting:   Strategy (Backtrader) ← BacktestAdapter ← BrokerAdapter ABC
Live/Paper:    Strategy              ← IBKRAdapter     ← BrokerAdapter ABC
                                           ↕
                                       ib_async
                                           ↕
                                    IB Gateway / TWS
```

Both adapters share the same `BrokerAdapter` ABC so strategy code is
broker-agnostic. The private repo wires up the concrete adapter.

---

## Phase 1 — Data Pipeline Infrastructure ✅ (current)

**Goal:** Reliable, normalized, cached data feeds any strategy can consume.

### Scope (deliberately narrow for Phase 1)

- `BaseDataConnector` ABC — defines the interface all future connectors implement
- `YFinanceConnector` — free, no API key, primary source for prototyping
- Cache layer — local parquet cache keyed by (symbol, start, end, freq, source); TTL-based invalidation
- Unified loader — thin wrapper providing a single entry point for all connectors
- `universe.py` — lightweight asset universe manager
- `.env.example` — lists all env vars required across all phases

### Explicitly deferred to Phase 6

- Alpha Vantage connector
- Finnhub connector
- IBKR data connector (added in Phase 3 when IB connection exists anyway)

### Deliverables

```
src/quant_trading/data/
  apis/
    base_connector.py       # BaseDataConnector ABC
    yfinance_connector.py   # concrete implementation
  loaders.py                # fetch_price_history(), fetch_fundamentals()
  cache.py                  # local parquet cache with TTL
  universe.py               # asset universe helpers
.env.example
tests/test_data_loaders.py
```

### Key interfaces

```python
class BaseDataConnector(ABC):
    @abstractmethod
    def fetch_price_history(
        self, symbol: str, start: datetime, end: datetime, freq: str = "1d",
    ) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame: ...

# Unified entry point
def fetch_price_history(
    symbol: str, start: datetime, end: datetime,
    freq: str = "1d", source: str = "yfinance", cache: bool = True,
) -> pd.DataFrame: ...
```

### Done when

- [ ] `YFinanceConnector` fetches and caches daily OHLCV for a given symbol and date range
- [ ] Cache correctly invalidates on TTL expiry
- [ ] Unit tests pass on synthetic/small real universe (no API key required)
- [ ] `.env.example` created

---

## Phase 2 — Backtesting Engine

**Goal:** Strategy-agnostic Backtrader harness. Strategy logic plugs in from private repo.

> Core risk metrics (Sharpe, Sortino, max drawdown, annualized vol) are pulled into this
> phase because they are required for backtest performance reports.

### Deliverables

```
src/quant_trading/
  signals/
    base.py                  # BaseSignal ABC
  backtesting/
    engine.py                # BacktestEngine: wraps Backtrader Cerebro
    base_strategy.py         # BaseStrategy(bt.Strategy) ABC
    sizing.py                # fixed fractional, vol-targeted, equal-weight
    commission.py            # pluggable commission/slippage models
    performance.py           # Sharpe, Sortino, max drawdown, trade log export
  risk/
    metrics.py               # annualized return/vol, Sharpe, Sortino, drawdown
examples/notebooks/example_backtest.ipynb
tests/test_backtesting.py
tests/test_risk_metrics.py
```

### Key interfaces

```python
class BaseSignal(ABC):
    @abstractmethod
    def generate(self, prices: pd.DataFrame) -> pd.Series:
        """Return timestamp-aligned signal Series (+1 / 0 / -1 or continuous score)."""
        ...

class BaseStrategy(bt.Strategy, ABC):
    """Backtrader strategy base. Implementations live in private repo."""
    @abstractmethod
    def generate_signals(self) -> dict[str, float]: ...
```

### Done when

- [ ] `BacktestEngine` runs a trivial buy-and-hold strategy on synthetic data
- [ ] Performance report outputs Sharpe, max drawdown, trade log
- [ ] `BaseSignal` and `BaseStrategy` ABCs defined and documented
- [ ] Example notebook runs end-to-end on synthetic prices

---

## Phase 3 — Paper Trading Integration (IBKR)

**Goal:** Connect to IB Gateway/TWS for paper trading via `ib_async`.

### Safety requirements (non-negotiable)

- `DRY_RUN=true` is the **default**; live order submission requires explicit opt-out
- Hard position and notional limits enforced at order submission time (not just logged)
- Pre-trade checklist runs before any order is submitted
- Reconnect/backoff logic for connectivity drops
- All IB credentials from env vars only: `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`

### Deliverables

```
src/quant_trading/execution/
  base_adapter.py            # BrokerAdapter ABC
  ibkr_adapter.py            # ib_async implementation
  backtest_adapter.py        # Backtrader-backed simulation adapter
  order_manager.py           # order lifecycle, retry, rate limiting
  safety.py                  # kill switch, position limits, dry-run, pre-trade checks
src/quant_trading/data/apis/
  ibkr_connector.py          # historical + live market data via ib_async
.env.example                 # updated with IB vars
tests/test_execution.py      # dry-run / mock tests only; no real IB connection in CI
```

### Key interfaces

```python
class BrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str, order_type: str) -> str: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...
    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...
    @abstractmethod
    def get_account_value(self) -> float: ...
```

### Done when

- [ ] `IBKRAdapter` connects to IB Gateway in dry-run mode and logs orders without submitting
- [ ] Pre-trade safety checks reject orders exceeding position limits
- [ ] `IBKRConnector` fetches historical bars via `ib_async`
- [ ] Manual smoke test passes against IB paper account

---

## Phase 4 — Portfolio Optimization

**Goal:** Generic MVO and constraint utilities any strategy can call.

### Solver chain

```
OSQP (default, no license needed)
  → ECOS (fallback)
  → SCS (fallback)
  → MOSEK (optional; used if installed and license present)
```

### Deliverables

```
src/quant_trading/optim/
  mean_variance.py           # CVXPY MVO with solver fallback chain
  constraints.py             # long-only, bounds, turnover, sector exposure
  covariance.py              # Ledoit-Wolf shrinkage, eigenvalue flooring, rolling estimators
  risk_parity.py             # equal risk contribution (stretch goal)
examples/notebooks/example_optimizer.ipynb
tests/test_optimizer.py
```

### Key interface

```python
def mean_variance_optimizer(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float | None = None,
    target_return: float | None = None,
    bounds: list[tuple[float, float]] | None = None,
    constraints: list | None = None,
    solver: str = "OSQP",
) -> dict:
    """Returns: {"weights", "status", "solver", "solve_time", "diagnostics"}"""
```

### Done when

- [ ] MVO solves a 2-asset problem matching known analytical solution
- [ ] Infeasible constraint sets return a diagnostic dict, not an exception
- [ ] Solver fallback chain works when preferred solver not installed
- [ ] Ledoit-Wolf shrinkage available as a drop-in covariance estimator

---

## Phase 5 — Full Risk Utilities

**Goal:** Complete risk analytics library beyond the basics pulled into Phase 2.

### Deliverables

```
src/quant_trading/risk/
  metrics.py                 # (started Phase 2) — extended here
  tail_risk.py               # historical VaR, parametric VaR, CVaR
  contribution.py            # marginal and percent contribution to portfolio vol
  stress.py                  # scenario shocks, correlation stress tests
  transaction_costs.py       # turnover, slippage estimation
examples/notebooks/example_risk.ipynb
tests/test_risk_metrics.py   # extended
```

### Done when

- [ ] VaR and CVaR match known analytical values on normal synthetic returns
- [ ] Stress scenario applies return shocks and recomputes portfolio metrics
- [ ] Contribution to risk sums correctly across all positions

---

## Phase 6 — Polish, CI, Packaging, Additional Connectors

**Goal:** Installable package; private repo can `pip install quant-trading`.

### Deliverables

- `pyproject.toml` with package metadata and optional extras:
  - `pip install quant-trading` — core (yfinance, cvxpy, backtrader)
  - `pip install quant-trading[ibkr]` — adds `ib_async`
  - `pip install quant-trading[mosek]` — adds `mosek`
- GitHub Actions CI: black, flake8, pytest on Python 3.10 + 3.11
- Additional data connectors (Alpha Vantage, Finnhub) — added here, not earlier
- `CONTRIBUTING.md` and API reference docs
- All example notebooks verified to run on synthetic data with no real credentials

### Done when

- [ ] `pip install -e .` works from a clean virtualenv
- [ ] CI passes on both Python versions
- [ ] Private repo can import and use `BaseSignal`, `BrokerAdapter`, `mean_variance_optimizer`

---

## Full Repository Layout (target)

```
quant-trading/
├── src/quant_trading/
│   ├── data/
│   │   ├── apis/
│   │   │   ├── base_connector.py
│   │   │   ├── yfinance_connector.py
│   │   │   ├── ibkr_connector.py           # added Phase 3
│   │   │   └── alpha_vantage_connector.py  # added Phase 6
│   │   ├── loaders.py
│   │   ├── cache.py
│   │   └── universe.py
│   ├── signals/
│   │   └── base.py
│   ├── backtesting/
│   │   ├── engine.py
│   │   ├── base_strategy.py
│   │   ├── sizing.py
│   │   ├── commission.py
│   │   └── performance.py
│   ├── execution/
│   │   ├── base_adapter.py
│   │   ├── ibkr_adapter.py
│   │   ├── backtest_adapter.py
│   │   ├── order_manager.py
│   │   └── safety.py
│   ├── optim/
│   │   ├── mean_variance.py
│   │   ├── constraints.py
│   │   ├── covariance.py
│   │   └── risk_parity.py
│   ├── risk/
│   │   ├── metrics.py
│   │   ├── tail_risk.py
│   │   ├── contribution.py
│   │   ├── stress.py
│   │   └── transaction_costs.py
│   └── utils/
│       ├── math.py
│       └── finance.py
├── tests/
│   ├── test_data_loaders.py
│   ├── test_backtesting.py
│   ├── test_optimizer.py
│   ├── test_risk_metrics.py
│   └── test_execution.py
├── examples/
│   └── notebooks/
│       ├── example_backtest.ipynb
│       ├── example_optimizer.ipynb
│       └── example_risk.ipynb
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/ci.yml
├── .env.example
├── .gitignore
├── pyproject.toml
├── DEV_PLAN.md
├── CONTRIBUTING.md
└── README.md
```

---

## Milestone Summary

| Phase | Focus | Est. (1 dev) |
|-------|-------|-------------|
| 1 | Data pipeline: yfinance + cache + universe | 2–3 days |
| 2 | Backtesting engine + ABCs + core risk metrics | 3–4 days |
| 3 | IBKR paper trading + safety layer | 3–4 days |
| 4 | Portfolio optimizer (CVXPY MVO) | 2–3 days |
| 5 | Full risk utilities | 2 days |
| 6 | CI, packaging, docs, extra connectors | 1–2 days |

**Total MVP: ~3 weeks** to a pip-installable package the private strategy repo can import.
