"""Enhanced drawdown analytics.

Extends the basic max_drawdown from metrics.py with:
- Full drawdown series (underwater chart data)
- Per-drawdown episode table (start, trough, end, depth, duration, recovery)
- Calmar ratio
- Recovery time statistics
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrawdownEpisode:
    """A single drawdown episode.

    Attributes:
        peak_date: Date of the equity peak at the start of the drawdown.
        trough_date: Date of the maximum loss within the episode.
        recovery_date: Date the equity recovered to the prior peak, or None if
            the series ends before recovery.
        max_drawdown: Maximum drawdown depth (negative decimal, e.g., -0.25).
        duration_days: Calendar days from peak to trough.
        recovery_days: Calendar days from trough to recovery, or None.
    """

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    max_drawdown: float
    duration_days: int
    recovery_days: Optional[int]


@dataclass
class DrawdownAnalysis:
    """Full drawdown analysis output.

    Attributes:
        underwater: Series of drawdown values (≤ 0) indexed by date (underwater chart).
        episodes: List of all identified drawdown episodes.
        max_drawdown: Worst single drawdown across all episodes.
        avg_drawdown: Average drawdown depth across all episodes.
        max_duration_days: Longest peak-to-trough duration in calendar days.
        max_recovery_days: Longest trough-to-recovery duration (None if unrecovered).
        calmar_ratio: Annualized return / abs(max drawdown). None if max_dd = 0.
    """

    underwater: pd.Series
    episodes: list[DrawdownEpisode]
    max_drawdown: float
    avg_drawdown: float
    max_duration_days: int
    max_recovery_days: Optional[int]
    calmar_ratio: Optional[float]


def compute_drawdowns(
    equity_curve: pd.Series,
    ann_return: Optional[float] = None,
) -> DrawdownAnalysis:
    """Perform full drawdown analysis on an equity curve.

    Args:
        equity_curve: Series of portfolio values (or cumulative returns) indexed
            by dates. Values must be positive.
        ann_return: Annualized return to use for the Calmar ratio. If None, the
            Calmar ratio is computed directly from the equity curve.

    Returns:
        DrawdownAnalysis with the underwater series, episode table, and summary stats.

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values.
    """
    if equity_curve.empty:
        raise ValueError("equity_curve Series is empty.")
    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain strictly positive values.")

    # --- Underwater series ------------------------------------------------
    running_max = equity_curve.cummax()
    underwater = (equity_curve - running_max) / running_max  # ≤ 0

    # --- Episode detection ------------------------------------------------
    episodes: list[DrawdownEpisode] = []
    in_drawdown = False
    peak_date: Optional[pd.Timestamp] = None
    trough_date: Optional[pd.Timestamp] = None
    trough_val: float = 0.0

    for date, dd_val in underwater.items():
        if not in_drawdown:
            if dd_val < 0:
                in_drawdown = True
                # Peak is the last index where the running_max was achieved
                peak_date = equity_curve[:date].idxmax()
                trough_date = date
                trough_val = dd_val
        else:
            if dd_val < trough_val:
                trough_date = date
                trough_val = dd_val
            elif dd_val == 0.0:
                # Recovered
                recovery_date = date
                duration_days = (trough_date - peak_date).days
                recovery_days = (recovery_date - trough_date).days
                episodes.append(
                    DrawdownEpisode(
                        peak_date=peak_date,
                        trough_date=trough_date,
                        recovery_date=recovery_date,
                        max_drawdown=trough_val,
                        duration_days=duration_days,
                        recovery_days=recovery_days,
                    )
                )
                in_drawdown = False
                peak_date = None
                trough_date = None
                trough_val = 0.0

    # Handle open (unrecovered) drawdown at end of series
    if in_drawdown and trough_date is not None and peak_date is not None:
        duration_days = (trough_date - peak_date).days
        episodes.append(
            DrawdownEpisode(
                peak_date=peak_date,
                trough_date=trough_date,
                recovery_date=None,
                max_drawdown=trough_val,
                duration_days=duration_days,
                recovery_days=None,
            )
        )

    # --- Summary stats ----------------------------------------------------
    if episodes:
        max_dd = min(e.max_drawdown for e in episodes)
        avg_dd = float(np.mean([e.max_drawdown for e in episodes]))
        max_dur = max(e.duration_days for e in episodes)
        recovered = [e.recovery_days for e in episodes if e.recovery_days is not None]
        max_rec = max(recovered) if recovered else None
    else:
        max_dd = 0.0
        avg_dd = 0.0
        max_dur = 0
        max_rec = None

    # --- Calmar ratio -----------------------------------------------------
    if max_dd == 0.0:
        calmar: Optional[float] = None
    else:
        if ann_return is None:
            n_years = len(equity_curve) / 252
            ann_return = (
                (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1
                if n_years > 0
                else 0.0
            )
        calmar = ann_return / abs(max_dd)

    return DrawdownAnalysis(
        underwater=underwater,
        episodes=episodes,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_duration_days=max_dur,
        max_recovery_days=max_rec,
        calmar_ratio=calmar,
    )


def episodes_to_dataframe(analysis: DrawdownAnalysis) -> pd.DataFrame:
    """Convert DrawdownAnalysis episodes to a tidy DataFrame.

    Args:
        analysis: DrawdownAnalysis from :func:`compute_drawdowns`.

    Returns:
        DataFrame with one row per episode and columns:
            peak_date, trough_date, recovery_date, max_drawdown,
            duration_days, recovery_days.
    """
    if not analysis.episodes:
        return pd.DataFrame(
            columns=[
                "peak_date",
                "trough_date",
                "recovery_date",
                "max_drawdown",
                "duration_days",
                "recovery_days",
            ]
        )
    rows = [
        {
            "peak_date": e.peak_date,
            "trough_date": e.trough_date,
            "recovery_date": e.recovery_date,
            "max_drawdown": e.max_drawdown,
            "duration_days": e.duration_days,
            "recovery_days": e.recovery_days,
        }
        for e in analysis.episodes
    ]
    return pd.DataFrame(rows)
