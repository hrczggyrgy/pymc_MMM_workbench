"""Budget optimization for marketing mix modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, NonlinearConstraint
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
    # beta_draws are on standardized features, so multiply by standardized feature
    # For optimization we approximate: response = beta * (saturated / feature_std) * y_std
    # Since we don't have feature_std here, we use the raw scaled response
    # This is an approximation - the model page computes exact contributions
    return beta_draws * saturated * y_std


def optimize_budget(
    total_budget: float,
    minimums: dict[str, float],
    maximums: dict[str, float],
    result: dict,
    n_draws: int = 300,
    method: str = "SLSQP",
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

    def objective(x: NDArray[np.floating]) -> float:
        """Negative expected total response (minimize = maximize response)."""
        total = 0.0
        for i, c in enumerate(channels):
            p = result["params"][c]
            beta = result["beta_draws"][:n_draws, result["features"].index(c)]
            resp = expected_response(
                x[i], beta, p["decay"], p["strength"], p["midpoint"], result["y_std"]
            )
            total += resp.mean()
        return -total

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
                total_budget, minimums, maximums, result, n_draws, "trust-constr"
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