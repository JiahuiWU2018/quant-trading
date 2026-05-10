# quant-trading

[![CI](https://github.com/JiawWu/quant-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/JiawWu/quant-trading/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A personal, open-source quantitative trading framework. This repository contains the **generic infrastructure** — data connectors, backtesting engine, portfolio optimization, risk utilities, and IBKR execution. Actual strategy signals and parameters live in a separate private repository.

---

## Architecture

```
quant-trading/  (this repo — public)
├── src/quant_trading/
│   ├── data/          Phase 1 — market & fundamental data connectors
│   ├── backtest/      Phase 2 — event-driven backtesting engine
│   ├── execution/     Phase 3 — Interactive Brokers order management
│   ├── portfolio/     Phase 4 — portfolio optimization (MVO, risk parity, …)
│   └── risk/          Phase 5 — VaR, stress testing, drawdown, rolling metrics
├── tests/             unit + integration tests (159 passing)
└── examples/          Jupyter notebooks demonstrating each module

private-strategies/    (separate private repo — not public)
└── strategies/        concrete signal logic, calibrated parameters
```

---

## Modules

| Module | Phase | Key Capabilities |
|---|---|---|
| `quant_trading.data` | 1 | yfinance market data, parquet caching, fundamental data stubs |
| `quant_trading.backtest` | 2 | event-driven engine, position tracker, performance reporter |
| `quant_trading.execution` | 3 | IBKR order placement, bracket orders, live position sync |
| `quant_trading.portfolio` | 4 | Mean-variance optimisation, risk parity, max diversification, CVaR |
| `quant_trading.risk` | 5 | Historical/parametric VaR, stress tests, Euler risk contribution, rolling analytics, drawdown analysis |

---

## Quick Start

### Install

```bash
git clone https://github.com/<your-username>/quant-trading.git
cd quant-trading

python -m venv .venvqt
source .venvqt/bin/activate

# Core + optional extras
pip install -e ".[backtest,optim,dev]"
```

### Risk utilities example

```python
import numpy as np
import pandas as pd

from quant_trading.risk import (
    GFC_2008,
    COVID_2020,
    compute_drawdowns,
    run_stress_test,
    var_historical,
)

# Synthetic daily returns
rng = np.random.default_rng(42)
returns = pd.Series(rng.normal(0.0005, 0.012, 756))

# Value at Risk
result = var_historical(returns, confidence=0.95)
print(f"95% 1-day VaR: {result.var_pct:.2%}")

# Stress test against named scenarios
weights = {"Equity": 0.6, "Bonds": 0.3, "Commodities": 0.1}
report = run_stress_test(weights, [GFC_2008, COVID_2020])
print(report)

# Drawdown analysis
analysis = compute_drawdowns(returns)
print(f"Max drawdown: {analysis.max_drawdown:.2%}")
```

### Run all tests

```bash
pytest --tb=short -q
```

---

## Configuration

Sensitive values (API keys, IBKR credentials) are read from environment variables — never hardcoded. Copy the template and fill in your values locally:

```bash
cp .env.example .env
# edit .env — this file is gitignored
```

See `.env.example` for all required variables.

---

## Examples

Interactive notebooks live in `examples/notebooks/`:

| Notebook | Covers |
|---|---|
| `example_risk.ipynb` | VaR, stress testing, risk contribution, drawdown, rolling metrics |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code style, branch naming, and the PR checklist.

**Security note:** This is a public repository. Never commit API keys, account IDs, or filled-in `.env` files. See `CONTRIBUTING.md §6` for details.

---

## Roadmap

- [x] Phase 1 — Data pipeline
- [x] Phase 2 — Backtesting engine
- [x] Phase 3 — IBKR execution
- [x] Phase 4 — Portfolio optimization
- [x] Phase 5 — Risk utilities
- [x] Phase 6 — CI, packaging, docs *(this release)*
- [ ] Phase 7 — Paper trading integration & end-to-end smoke test
- [ ] Phase 8 — Strategy plug-in interface (private repo consumes this)

---

## License

[MIT](LICENSE)
