# Contributing

Thank you for your interest in contributing to this project.  
This is a personal quantitative trading framework. Contributions that improve the **generic infrastructure** (data connectors, backtesting engine, portfolio utilities, risk utilities, CI/tooling) are welcome. Strategy-specific logic lives in a separate private repository and is out of scope here.

---

## 1. Development Setup

```bash
# Clone
git clone https://github.com/<your-username>/quant-trading.git
cd quant-trading

# Create and activate a virtual environment
python -m venv .venvqt
source .venvqt/bin/activate        # macOS / Linux
# .venvqt\Scripts\activate         # Windows

# Install in editable mode with all extras
pip install --upgrade pip
pip install -e ".[backtest,optim,dev]"

# Install pre-commit hooks
pre-commit install
```

---

## 2. Running Tests

```bash
# All tests (quick)
pytest --tb=short -q

# With coverage
pytest --tb=short -q --cov=src/quant_trading --cov-report=term-missing

# Single module
pytest tests/test_risk.py -v
```

The CI pipeline enforces a minimum **80% line coverage**. Please ensure new code includes corresponding tests.

---

## 3. Code Style

The project uses **black**, **isort**, and **flake8** (max line length: 100). Pre-commit hooks enforce these automatically on every commit. To run them manually:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
```

Type hints are required on all function signatures. Run mypy to check:

```bash
mypy src/quant_trading --ignore-missing-imports
```

---

## 4. Branch Naming

| Prefix | Purpose |
|--------|---------|
| `feat/` | New feature |
| `fix/` | Bug fix |
| `refactor/` | Internal restructuring (no behaviour change) |
| `docs/` | Documentation only |
| `ci/` | CI/tooling changes |
| `test/` | Test additions / fixes |

Example: `feat/data-connector-finnhub`

---

## 5. Pull Request Checklist

Before opening a PR, confirm:

- [ ] All existing tests still pass (`pytest --tb=short -q`)
- [ ] New functionality has tests
- [ ] Coverage has not dropped below 80%
- [ ] `black`, `isort`, `flake8` pass with no errors
- [ ] No secrets, credentials, account IDs, or personal file paths are included (this is a **public** repo)
- [ ] Strategy-specific logic is NOT included (belongs in the private repository)
- [ ] Docstrings added to all public functions (Google style)
- [ ] `CHANGELOG` or PR description explains what changed and why

---

## 6. Security Reminder

This is a **public repository**. Never commit:

- API keys, tokens, or passwords
- IBKR account IDs or port numbers
- Personal file paths (e.g. `/home/yourname/...`)
- Filled-in `.env` files (only `.env.example` with placeholders)

If you accidentally commit a secret, treat it as **compromised** and rotate it immediately.

---

## 7. Questions

Open a GitHub Issue for bugs or feature requests. For architectural discussions, start a Discussion thread.
