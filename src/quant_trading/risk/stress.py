"""Stress testing: apply historical and hypothetical shock scenarios to a portfolio.

A scenario defines per-asset (or per-factor) return shocks. The module ships a
set of pre-built historical scenarios and supports user-defined ones.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress scenario.

    Attributes:
        name: Human-readable scenario name.
        shocks: Mapping from asset/factor label to return shock (e.g., -0.40 = -40%).
        description: Optional free-text description.
    """

    name: str
    shocks: Dict[str, float]
    description: str = ""


# ---------------------------------------------------------------------------
# Pre-built historical scenarios
# ---------------------------------------------------------------------------

GFC_2008 = StressScenario(
    name="GFC 2008",
    shocks={
        "equities": -0.50,
        "credit": -0.30,
        "commodities": -0.40,
        "real_estate": -0.45,
        "bonds": 0.10,
        "gold": 0.05,
    },
    description=(
        "Global Financial Crisis peak-to-trough drawdown approximations "
        "(Oct 2007 – Mar 2009)."
    ),
)

COVID_2020 = StressScenario(
    name="COVID Crash 2020",
    shocks={
        "equities": -0.34,
        "credit": -0.20,
        "commodities": -0.25,
        "real_estate": -0.20,
        "bonds": 0.04,
        "gold": -0.02,
    },
    description="COVID-19 market crash (Feb 19 – Mar 23, 2020).",
)

RATE_SPIKE_2022 = StressScenario(
    name="Rate Spike 2022",
    shocks={
        "equities": -0.25,
        "bonds": -0.18,
        "real_estate": -0.25,
        "credit": -0.15,
        "commodities": 0.15,
        "gold": -0.02,
    },
    description=(
        "2022 rate-hike cycle drawdowns (Jan – Dec 2022, peak Fed funds hike cycle)."
    ),
)

DOTCOM_2000 = StressScenario(
    name="Dot-com Crash 2000–2002",
    shocks={
        "equities": -0.49,
        "tech": -0.78,
        "bonds": 0.12,
        "gold": 0.10,
        "real_estate": 0.05,
    },
    description="NASDAQ-led equity sell-off (Mar 2000 – Oct 2002).",
)

HISTORICAL_SCENARIOS: list[StressScenario] = [
    GFC_2008,
    COVID_2020,
    RATE_SPIKE_2022,
    DOTCOM_2000,
]


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


def run_stress_test(
    weights: pd.Series,
    scenarios: Optional[list[StressScenario]] = None,
    asset_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Apply stress scenarios to a portfolio and compute scenario P&L.

    Each asset in *weights* is mapped to a scenario shock label via *asset_map*.
    If an asset has no mapping or its label is absent from a scenario's shocks,
    its shock defaults to 0.0 (no change).

    Args:
        weights: Series mapping asset label → portfolio weight (should sum to ~1).
        scenarios: List of StressScenario objects. Defaults to HISTORICAL_SCENARIOS.
        asset_map: Optional mapping from weight labels → scenario shock labels.
            Useful when portfolio uses tickers (e.g., "SPY") while scenarios
            use category keys (e.g., "equities").  If None, labels are matched
            directly against scenario shock keys.

    Returns:
        DataFrame indexed by scenario name with columns:
            - portfolio_return: Weighted scenario portfolio return.
            - [individual asset columns with their scenario shocks]

    Example::

        weights = pd.Series({"SPY": 0.6, "TLT": 0.4})
        asset_map = {"SPY": "equities", "TLT": "bonds"}
        df = run_stress_test(weights, asset_map=asset_map)
    """
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS

    if weights.empty:
        raise ValueError("weights Series is empty.")

    results = []
    for scenario in scenarios:
        row: Dict[str, float] = {"scenario": scenario.name}
        port_return = 0.0
        for asset, w in weights.items():
            label = asset_map.get(str(asset), str(asset)) if asset_map else str(asset)
            shock = scenario.shocks.get(label, 0.0)
            row[str(asset)] = shock
            port_return += w * shock
        row["portfolio_return"] = port_return
        results.append(row)

    df = pd.DataFrame(results).set_index("scenario")
    # Reorder: portfolio_return first, then individual assets
    cols = ["portfolio_return"] + [c for c in df.columns if c != "portfolio_return"]
    return df[cols]


def historical_scenario_from_returns(
    returns: pd.DataFrame,
    start: str,
    end: str,
    name: str,
    description: str = "",
) -> StressScenario:
    """Build a StressScenario from realized returns over a date window.

    Args:
        returns: DataFrame of asset returns (columns = assets, index = dates).
        start: Start date string, e.g., "2008-09-01".
        end: End date string, e.g., "2009-03-31".
        name: Name for the resulting scenario.
        description: Optional description.

    Returns:
        StressScenario with compounded returns for each asset over [start, end].

    Raises:
        ValueError: If the date window produces an empty slice.
    """
    window = returns.loc[start:end]
    if window.empty:
        raise ValueError(
            f"No return data found between {start} and {end}. "
            "Check that the returns DataFrame covers this period."
        )
    compounded = (1 + window).prod() - 1
    shocks = compounded.to_dict()
    return StressScenario(name=name, shocks=shocks, description=description)
