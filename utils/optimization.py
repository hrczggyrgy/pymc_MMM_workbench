"""Budget optimization for marketing mix modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from numpy.typing import NDArray

from utils.transformations import steady_state_adstock, hill_saturation


def expected_response(
    spend: float,
    beta_draws: NDArray[np.floating],
    decay: float,
    strength: float,
    midpoint: float,
    y_std: float,
) -> NDArray[np.floating]:
    """
    Compute expected response distribution for a given spend level.

    Uses steady-state adstock approximation for optimization speed.

    Parameters
    ----------
    spend : float
        Average-period spend.
    beta_draws : ndarray (n_draws,)
        Posterior draws for the channel coefficient.
    decay : float
        Adstock decay parameter.
    strength : float
        Saturation strength (Hill parameter).
    midpoint : float
        Saturation midpoint.
    y_std : float
        Target standard deviation (for re-scaling).

    Returns
    -------
    ndarray (n_draws,)
        Predicted incremental response per draw.
    """
    adstocked = steady_state_adstock(spend, decay)
    saturated = hill_saturation(adstocked, strength, midpoint)
    return beta_draws * saturated * y_std


def _compute_objective(
    x: NDArray[np.floating],
    channels: list[str],
    result: dict,
    n_draws: int,
    risk_aversion: float = 0.0,
) -> float:
    """
    Compute negative objective value for optimization.

    Parameters
    ----------
    x : array
        Spend allocation per channel.
    channels : list
        Channel names.
    result : dict
        Model result dictionary.
    n_draws : int
        Number of posterior draws to use.
    risk_aversion : float
        Risk aversion parameter (0 = risk-neutral, >0 = risk-averse).
        Objective = -(mean - risk_aversion * variance).
    """
    total_mean = 0.0
    total_var = 0.0
    
    for i, c in enumerate(channels):
        p = result["params"][c]
        beta = result["beta_draws"][:n_draws, result["features"].index(c)]
        resp = expected_response(
            x[i], beta, p["decay"], p["strength"], p["midpoint"], result["y_std"]
        )
        total_mean += resp.mean()
        total_var += resp.var()
    
    if risk_aversion > 0:
        return -(total_mean - risk_aversion * total_var)
    return -total_mean


def optimize_budget(
    total_budget: float,
    minimums: dict[str, float],
    maximums: dict[str, float],
    result: dict,
    n_draws: int = 300,
    method: str = "SLSQP",
    risk_aversion: float = 0.0,
) -> tuple[dict[str, float], NDArray[np.floating]]:
    """
    Optimize budget allocation across channels to maximize expected response.

    Parameters
    ----------
    total_budget : float
        Total budget to allocate (average period).
    minimums : dict
        Minimum spend per channel.
    maximums : dict
        Maximum spend per channel.
    result : dict
        Model result dictionary from fit_bayesian_mmm.
    n_draws : int
        Number of posterior draws to use for expectation.
    method : str
        Optimization method (SLSQP, trust-constr).
    risk_aversion : float
        Risk aversion parameter (0 = risk-neutral, >0 = risk-averse).
        Maximizes: mean_response - risk_aversion * variance_response.

    Returns
    -------
    allocation : dict
        Recommended spend per channel.
    lift_samples : ndarray
        Posterior lift samples vs current allocation.
    """
    channels = result["channels"]
    n_channels = len(channels)

    lower = np.array([minimums[c] for c in channels])
    upper = np.array([maximums[c] for c in channels])

    # Validate per-channel constraints
    for c in channels:
        if minimums[c] > maximums[c]:
            raise ValueError(
                f"Minimum spend for channel '{c}' ({minimums[c]:,.0f}) exceeds maximum ({maximums[c]:,.0f})"
            )
        if minimums[c] < 0:
            raise ValueError(f"Minimum spend for channel '{c}' cannot be negative")
        if maximums[c] < 0:
            raise ValueError(f"Maximum spend for channel '{c}' cannot be negative")

    if lower.sum() > total_budget + 1e-6:
        raise ValueError(
            f"Sum of minimums ({lower.sum():,.0f}) exceeds total budget ({total_budget:,.0f})"
        )
    if upper.sum() < total_budget - 1e-6:
        raise ValueError(
            f"Sum of maximums ({upper.sum():,.0f}) is less than total budget ({total_budget:,.0f})"
        )

    # Current allocation for comparison
    current = np.array([result["current_spend"][c] for c in channels])

    # Initial guess: proportional to current spend, clipped to bounds
    x0 = np.clip(current, lower, upper)
    # Adjust to meet budget constraint
    diff = total_budget - x0.sum()
    if abs(diff) > 1e-6:
        # Distribute difference proportionally to available slack
        slack_up = upper - x0
        slack_down = x0 - lower
        if diff > 0:
            total_slack = slack_up.sum()
            if total_slack > 0:
                x0 += diff * slack_up / total_slack
        else:
            total_slack = slack_down.sum()
            if total_slack > 0:
                x0 += diff * slack_down / total_slack
    x0 = np.clip(x0, lower, upper)

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda x: x.sum() - total_budget},
    ]
    bounds = Bounds(lower, upper)

    def objective(x: NDArray[np.floating]) -> float:
        return _compute_objective(x, channels, result, n_draws, risk_aversion)

    solution = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not solution.success:
        # Try with a different method as fallback
        if method != "trust-constr":
            return optimize_budget(
                total_budget, minimums, maximums, result, n_draws, "trust-constr", risk_aversion
            )
        raise ValueError(f"Optimization failed: {solution.message}")

    allocation = {c: float(solution.x[i]) for i, c in enumerate(channels)}

    # Compute lift vs current
    lift_samples = np.zeros(n_draws)
    for i, c in enumerate(channels):
        p = result["params"][c]
        beta = result["beta_draws"][:n_draws, result["features"].index(c)]
        current_resp = expected_response(
            current[i], beta, p["decay"], p["strength"], p["midpoint"], result["y_std"]
        )
        new_resp = expected_response(
            solution.x[i], beta, p["decay"], p["strength"], p["midpoint"], result["y_std"]
        )
        lift_samples += new_resp - current_resp

    return allocation, lift_samples


def compute_channel_roi(
    allocation: dict[str, float],
    result: dict,
    n_draws: int = 300,
) -> pd.DataFrame:
    """
    Compute ROI and marginal ROAS for each channel at the optimal allocation.

    Returns
    -------
    DataFrame with columns: Channel, Spend, Expected_Response, ROI, Marginal_ROAS
    """
    channels = result["channels"]
    rows = []

    for c in channels:
        spend = allocation[c]
        p = result["params"][c]
        beta = result["beta_draws"][:n_draws, result["features"].index(c)]

        resp_samples = expected_response(
            spend, beta, p["decay"], p["strength"], p["midpoint"], result["y_std"]
        )
        expected_resp = resp_samples.mean()

        # Marginal ROAS: derivative at current spend
        adstocked = steady_state_adstock(spend, p["decay"])
        strength = p["strength"]
        midpoint = p["midpoint"]
        power = max(float(strength), 0.1)

        # d(saturation)/d(adstocked)
        num = power * adstocked ** (power - 1) * midpoint**power
        den = (adstocked**power + midpoint**power) ** 2
        hill_grad = num / (den + 1e-9)

        # d(adstocked)/d(spend) = 1/(1-decay)
        marginal = hill_grad / (1 - p["decay"] + 1e-6)

        # Expected marginal response per draw
        marginal_resp = beta * marginal * result["y_std"]
        marginal_roas = marginal_resp.mean()

        roi = expected_resp / spend if spend > 0 else np.nan

        rows.append(
            {
                "Channel": c,
                "Spend": spend,
                "Expected_Response": expected_resp,
                "ROI": roi,
                "Marginal_ROAS": marginal_roas,
            }
        )

    return pd.DataFrame(rows)


def response_curve_data(
    channel: str,
    result: dict,
    max_multiplier: float = 3.0,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate response curve data for visualization.

    Returns
    -------
    spend_vals : ndarray
        Spend values from 0 to max_multiplier * current_spend.
    mean_response : ndarray
        Mean predicted response.
    low_response : ndarray
        5th percentile response.
    high_response : ndarray
        95th percentile response.
    """
    current = result["current_spend"][channel]
    max_spend = current * max_multiplier
    spend_vals = np.linspace(0, max_spend, n_points)

    p = result["params"][channel]
    beta_draws = result["beta_draws"][:, result["features"].index(channel)]
    n_draws = min(500, len(beta_draws))

    responses = np.zeros((n_draws, n_points))
    for i, spend in enumerate(spend_vals):
        responses[:, i] = expected_response(
            spend, beta_draws[:n_draws], p["decay"], p["strength"], p["midpoint"], result["y_std"]
        )

    mean_response = responses.mean(axis=0)
    low_response = np.quantile(responses, 0.05, axis=0)
    high_response = np.quantile(responses, 0.95, axis=0)

    return spend_vals, mean_response, low_response, high_response