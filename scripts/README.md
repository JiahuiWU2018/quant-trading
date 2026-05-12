# scripts/

Operational utility scripts. These are run directly from the terminal —
they are not part of the importable `quant_trading` package.

## Prerequisites

All scripts require the venv to be active and `.env` to be loaded:

```bash
source .venvqt/bin/activate
set -a && source .env && set +a
```

## Scripts

| Script | Purpose |
|---|---|
| `verify_ib_connection.py` | Confirm IB Gateway is reachable and account data is readable. Run before every paper trading session. |

## Running `verify_ib_connection.py`

```bash
python scripts/verify_ib_connection.py
```

Expected output when everything is healthy:

```
────────────────────────────────────────────────────────────
  Step 0 — Environment
────────────────────────────────────────────────────────────
  IB_HOST      : 127.0.0.1
  IB_PORT      : 7497
  ...
  ✓ Environment variables loaded

...

────────────────────────────────────────────────────────────
  Summary
────────────────────────────────────────────────────────────
  All checks passed. IB Gateway connection is healthy.
  You are ready to run paper trading scripts.
```

### Common failures

| Error | Fix |
|---|---|
| `Connection failed` | Open IB Gateway, log in, check API is enabled in Settings → API |
| `IB_ACCOUNT not set` | Add `IB_ACCOUNT=DU...` to `.env` |
| `accountValues() returned empty` | Wait 10–15s after Gateway login, then retry |
| `reqHistoricalData returned no bars` | Pacing violation — wait 10s and retry |
