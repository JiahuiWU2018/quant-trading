"""Risk analysis and performance metrics."""

from quant_trading.risk.metrics import (
    annualized_return,
    annualized_volatility,
    compute_metrics,
    drawdown_duration,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quant_trading.risk.var import (
    VaRResult,
    var_historical,
    var_parametric_normal,
    var_parametric_t,
    var_summary,
)
from quant_trading.risk.stress import (
    StressScenario,
    HISTORICAL_SCENARIOS,
    GFC_2008,
    COVID_2020,
    RATE_SPIKE_2022,
    DOTCOM_2000,
    run_stress_test,
    historical_scenario_from_returns,
)
from quant_trading.risk.contribution import (
    RiskContributionResult,
    risk_contribution,
    risk_contribution_from_returns,
    component_var,
)
from quant_trading.risk.drawdown import (
    DrawdownEpisode,
    DrawdownAnalysis,
    compute_drawdowns,
    episodes_to_dataframe,
)
from quant_trading.risk.rolling import (
    rolling_volatility,
    rolling_sharpe,
    rolling_sortino,
    rolling_beta,
    rolling_var,
    rolling_mean_correlation,
)

__all__ = [
    # metrics
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_duration",
    "compute_metrics",
    # var
    "VaRResult",
    "var_historical",
    "var_parametric_normal",
    "var_parametric_t",
    "var_summary",
    # stress
    "StressScenario",
    "HISTORICAL_SCENARIOS",
    "GFC_2008",
    "COVID_2020",
    "RATE_SPIKE_2022",
    "DOTCOM_2000",
    "run_stress_test",
    "historical_scenario_from_returns",
    # contribution
    "RiskContributionResult",
    "risk_contribution",
    "risk_contribution_from_returns",
    "component_var",
    # drawdown
    "DrawdownEpisode",
    "DrawdownAnalysis",
    "compute_drawdowns",
    "episodes_to_dataframe",
    # rolling
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_beta",
    "rolling_var",
    "rolling_mean_correlation",
]
