# Paper Trading Setup — IB Gateway

This document is the authoritative runbook for setting up and running paper trading against Interactive Brokers. Follow each section in order the first time. On subsequent sessions, jump to [Daily Startup Checklist](#daily-startup-checklist).

---

## 1. Prerequisites

| Requirement | How to verify |
|---|---|
| IBKR paper trading account | Log into [portal.interactivebrokers.com](https://portal.interactivebrokers.com) — confirm a paper account (ID starts with `DU`) is listed |
| IB Gateway installed | Open `/Applications/IB Gateway.app` — if missing, download from [IBKR downloads](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) |
| Python venv active | `which python` → should show `.venvqt/bin/python` |
| `.env` configured | See [Section 3](#3-env-configuration) |

---

## 2. IB Gateway Configuration (one-time)

### 2.1 Log in

1. Open **IB Gateway**
2. Select **Paper Trading** from the account type dropdown
3. Enter your IBKR username and password
4. Click **Login**

> ⚠️ IB Gateway auto-disconnects after approximately 24 hours. You must log in again each session. Consider enabling auto-restart in Gateway settings if running overnight.

### 2.2 Enable the API (one-time)

In IB Gateway, go to **Configure → Settings → API → Settings** and set:

| Setting | Value |
|---|---|
| Enable ActiveX and Socket Clients | ✅ Checked |
| Socket port | `7497` |
| Allow connections from localhost only | ✅ Checked |
| Read-Only API | ❌ Unchecked (must be off to place orders) |
| Trusted IP Addresses | `127.0.0.1` |

Click **OK** and restart IB Gateway for changes to take effect.

### 2.3 Find your paper account ID

In IB Gateway, click the **Account** menu in the top bar. Your paper account ID is shown there — it starts with `DU` followed by digits (e.g. `DU123456`). You will need this for your `.env` file.

> ⚠️ Never commit your account ID to this repository. Keep it in your local `.env` file only.

---

## 3. `.env` Configuration

Copy the template if you have not already:

```bash
cp .env.example .env
```

Edit `.env` and fill in your values. The IB-related variables are:

```bash
# IB Gateway connection
IB_HOST=127.0.0.1        # always localhost for local Gateway
IB_PORT=7497             # paper trading port (default for IB Gateway)
IB_CLIENT_ID=1           # any integer; must be unique per connected process
IB_CLIENT_ID_TEST=99     # used by integration tests — must differ from IB_CLIENT_ID
IB_ACCOUNT=DU123456      # your paper account ID — replace with your real value

# Safety
DRY_RUN=true             # set to false only when ready to submit paper orders
MAX_POSITION_SIZE=10
MAX_NOTIONAL_PER_ORDER=500
MAX_PORTFOLIO_NOTIONAL=5000
```

Verify `.env` is gitignored:

```bash
git check-ignore -v .env   # must output a match — if not, check .gitignore
```

---

## 4. Load Environment in Terminal

Every terminal session requires the venv and `.env` to be loaded:

```bash
# Activate venv
source .venvqt/bin/activate

# Load .env into shell environment
set -a && source .env && set +a

# Quick sanity check
echo "IB_PORT=$IB_PORT  DRY_RUN=$DRY_RUN"
```

> VS Code users: if `python.terminal.useEnvFile` is enabled in settings, new terminals automatically inject `.env`. You still need to activate the venv manually with `source .venvqt/bin/activate`.

---

## 5. Verify Connectivity

Run the connectivity check script **before every paper trading session**:

```bash
python scripts/verify_ib_connection.py
```

The script runs 7 checks and prints a summary. Expected output when healthy:

```
────────────────────────────────────────────────────────────
  Step 0 — Environment
────────────────────────────────────────────────────────────
  IB_HOST      : 127.0.0.1
  IB_PORT      : 7497
  IB_CLIENT_ID : 1
  IB_ACCOUNT   : DU******
  ✓ Environment variables loaded

────────────────────────────────────────────────────────────
  Step 1 — Import ib_async
────────────────────────────────────────────────────────────
  ib_async version : x.x.x
  ✓ ib_async imported successfully

────────────────────────────────────────────────────────────
  Step 2 — Connect to IB Gateway
────────────────────────────────────────────────────────────
  Server version   : 179
  ✓ Connected to IB Gateway

...

────────────────────────────────────────────────────────────
  Summary
────────────────────────────────────────────────────────────
  All checks passed. IB Gateway connection is healthy.
  You are ready to run paper trading scripts.
```

### Common failures

| Error | Cause | Fix |
|---|---|---|
| `Connection failed` | IB Gateway not running or not logged in | Open IB Gateway and log in |
| `Connection failed: port` | Wrong port | Check `IB_PORT=7497` in `.env` |
| `IB_ACCOUNT is not set` | Missing env var | Add `IB_ACCOUNT=DU...` to `.env` |
| `accountValues() returned empty` | Gateway still loading | Wait 15 seconds after login and retry |
| `reqHistoricalData returned no bars` | IB pacing violation | Wait 10 seconds and retry |
| `No security definition found` | Invalid symbol | Check ticker is a valid US equity on IBKR |

---

## 6. Running Integration Tests

Integration tests validate the full adapter and connector stack against a live paper account. They are skipped in normal `pytest` runs to avoid requiring a live connection in CI.

```bash
# Run all integration tests (IB Gateway must be running)
INTEGRATION_TESTS=1 pytest tests/integration/ -v

# Also run the order submit/cancel test (requires DRY_RUN=false in .env)
INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

Expected output: all 8 tests pass. The order test places a limit order at `$1.00` (far below market — will never fill) and immediately cancels it. Check the **IB Gateway Activity Monitor** to confirm the order appeared and was cancelled.

---

## 7. Daily Startup Checklist

Before every paper trading session:

- [ ] Open IB Gateway and log in with paper account credentials
- [ ] Wait ~15 seconds for account data to load
- [ ] Activate venv: `source .venvqt/bin/activate`
- [ ] Load env: `set -a && source .env && set +a`
- [ ] Run connectivity check: `python scripts/verify_ib_connection.py`
- [ ] Confirm all 7 steps pass before launching any trading script

---

## 8. Market Data Note

Paper trading accounts receive **delayed market data (15 minutes)** by default — not live quotes. This is handled automatically by the framework:

- `IBKRAdapter.connect()` calls `ib.reqMarketDataType(4)` immediately after connecting
- `IBKRConnector.fetch_price_history()` uses `whatToShow="TRADES"` which works with delayed data
- Delayed data is sufficient for end-of-day strategies and backtesting validation

If you subscribe to a live data plan through IBKR, change `reqMarketDataType(4)` to `reqMarketDataType(1)` in `execution/ibkr_adapter.py`. Do not commit this change to the public repository.

---

## 9. Safety Guardrails

The framework has multiple layers of protection against accidental orders:

| Guard | Location | Behaviour |
|---|---|---|
| `DRY_RUN=true` | `.env` / `IBKRAdapter` | Orders are logged but never submitted |
| `MAX_NOTIONAL_PER_ORDER` | `SafetyChecker` | Rejects any single order exceeding this notional |
| `MAX_PORTFOLIO_NOTIONAL` | `SafetyChecker` | Rejects orders if total portfolio notional would be exceeded |
| `MAX_POSITION_SIZE` | `SafetyChecker` | Rejects orders exceeding max shares in one symbol |

**Recommended values for initial paper trading:**

```bash
DRY_RUN=false                  # set false only after verify_ib_connection passes
MAX_NOTIONAL_PER_ORDER=500     # $500 max per order
MAX_PORTFOLIO_NOTIONAL=5000    # $5,000 total exposure cap
MAX_POSITION_SIZE=10           # max 10 shares in any single name
```

---

## 10. Troubleshooting

### IB Gateway disconnects mid-session

IB Gateway disconnects after ~24 hours. Signs: `ConnectionError` in logs, `ib.isConnected()` returns `False`.

Fix: log back into IB Gateway and restart your trading script. The `IBKRAdapter` will reconnect on the next `connect()` call.

Future improvement: automatic reconnect logic is planned for a later phase.

### `clientId already in use`

Two processes are connecting with the same `IB_CLIENT_ID`. Each simultaneous connection must use a unique client ID.

Fix:
- Main adapter: `IB_CLIENT_ID=1`
- Integration tests: `IB_CLIENT_ID_TEST=99`
- Any additional process: use a different integer (e.g. `2`, `3`)

### Orders not appearing in IB Gateway Activity Monitor

Check:
1. `DRY_RUN=false` in `.env` (dry-run orders are never sent to IB)
2. `SafetyChecker` did not reject the order — look for `SafetyError` in logs
3. The symbol is a valid US equity on IBKR
