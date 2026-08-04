"""Adstock and saturation transformations for marketing mix modeling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def geometric_adstock(
    values: NDArray[np.floating],
    decay: float = 0.5,
    l_max: int = 8,
) -> NDArray[np.floating]:
    """
    Apply geometric adstock (carryover) transformation.

    The geometric adstock model assumes that the effect of advertising
    decays exponentially over time: each period retains a fraction `decay`
    of the previous period's effect.

    Parameters
    ----------
    values : array-like
        Raw media spend or impressions per period.
    decay : float, default 0.5
        Retention rate per period (0 = no carryover, 1 = permanent).
    l_max : int, default 8
        Maximum lag periods to consider (truncates the infinite sum).

    Returns
    -------
    ndarray
        Adstocked values with same length as input.

    Notes
    -----
    The transformation computes:
        adstock_t = sum_{k=0}^{l_max} decay^k * values_{t-k}
    """
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return x
    weights = decay ** np.arange(l_max + 1)
    return np.convolve(x, weights, mode="full")[: len(x)]


def hill_saturation(
    values: NDArray[np.floating],
    strength: float = 1.5,
    midpoint: float | None = None,
) -> NDArray[np.floating]:
    """
    Apply Hill (sigmoid) saturation transformation.

    Models diminishing returns: each additional unit of spend produces
    less incremental response at higher spend levels.

    Parameters
    ----------
    values : array-like
        Input values (typically adstocked spend).
    strength : float, default 1.5
        Shape parameter controlling steepness (>0).
        Higher = more abrupt transition to saturation.
    midpoint : float, optional
        Spend level at which response reaches 50% of maximum.
        If None, uses median of positive values.

    Returns
    -------
    ndarray
        Saturated values in [0, 1], same length as input.

    Notes
    -----
    The Hill function: f(x) = x^strength / (x^strength + midpoint^strength)
    """
    x = np.maximum(np.asarray(values, dtype=float), 0)
    if midpoint is None:
        midpoint = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
    midpoint = max(float(midpoint), 1e-6)
    power = max(float(strength), 0.1)
    return x**power / (x**power + midpoint**power + 1e-9)


def transform_media(
    values: NDArray[np.floating],
    decay: float = 0.5,
    l_max: int = 8,
    strength: float = 1.5,
    midpoint: float | None = None,
) -> NDArray[np.floating]:
    """
    Apply full media transformation: adstock then saturation.

    This is the standard MMM pipeline for media variables.

    Parameters
    ----------
    values : array-like
        Raw media spend.
    decay : float
        Adstock decay parameter.
    l_max : int
        Adstock carryover window.
    strength : float
        Saturation strength (Hill parameter).
    midpoint : float, optional
        Saturation midpoint.

    Returns
    -------
    ndarray
        Transformed media feature ready for regression.
    """
    adstocked = geometric_adstock(values, decay, l_max)
    return hill_saturation(adstocked, strength, midpoint)


def marginal_response(
    values: NDArray[np.floating],
    decay: float = 0.5,
    l_max: int = 8,
    strength: float = 1.5,
    midpoint: float | None = None,
) -> NDArray[np.floating]:
    """
    Compute marginal response (derivative) of the full transformation.

    Shows how much additional response each additional unit of spend generates.
    """
    x = np.asarray(values, dtype=float)
    if midpoint is None:
        midpoint = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
    midpoint = max(float(midpoint), 1e-6)
    power = max(float(strength), 0.1)

    # For steady-state approximation: adstocked = x / (1 - decay)
    adstocked = x / (1 - decay + 1e-6)

    # Derivative of Hill function
    num = power * adstocked ** (power - 1) * midpoint**power
    den = (adstocked**power + midpoint**power) ** 2
    hill_grad = num / (den + 1e-9)

    # Chain rule: d(saturation)/d(spend) = d(saturation)/d(adstocked) * d(adstocked)/d(spend)
    # d(adstocked)/d(spend) = 1 / (1 - decay) for steady-state
    return hill_grad / (1 - decay + 1e-6)


def steady_state_adstock(spend: float, decay: float) -> float:
    """Compute steady-state adstocked value for constant spend."""
    return spend / (1 - decay + 1e-6)


def effective_spend_range(
    decay: float,
    l_max: int,
    threshold: float = 0.01,
) -> int:
    """
    Calculate effective number of periods where carryover exceeds threshold.

    Useful for understanding how far back the adstock "remembers."
    """
    k = 0
    while decay**k > threshold and k <= l_max:
        k += 1
    return min(k, l_max)