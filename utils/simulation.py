"""Synthetic marketing data generation for demo and testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.transformations import geometric_adstock, hill_saturation


@st.cache_data
def generate_demo_data(
    n_periods: int = 104,
    seed: int = 42,
    freq: str = "W-SUN",
    start_date: str = "2024-01-07",
) -> pd.DataFrame:
    """
    Generate realistic synthetic marketing mix data.

    Parameters
    ----------
    n_periods : int
        Number of time periods (weeks by default).
    seed : int
        Random seed for reproducibility.
    freq : str
        Pandas frequency string for date range.
    start_date : str
        Start date for the time series.

    Returns
    -------
    pd.DataFrame
        Columns: date, search, social, video, display, price, promo, sales
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)

    # Channel spend with seasonal patterns and noise
    channels = {
        "search": np.clip(
            rng.gamma(5, 1500, n_periods) * (1 + 0.15 * np.sin(t / 8)), 500, None
        ),
        "social": np.clip(
            rng.gamma(4, 1200, n_periods) * (1 + 0.2 * np.cos(t / 10)), 300, None
        ),
        "video": np.clip(
            rng.gamma(3, 2100, n_periods) * (1 + 0.25 * np.sin(t / 15)), 300, None
        ),
        "display": np.clip(
            rng.gamma(4, 700, n_periods), 150, None
        ),
    }

    # Control variables
    price = 100 + 2 * np.sin(t / 12) + rng.normal(0, 0.8, n_periods)
    promo = rng.binomial(1, 0.14, n_periods)

    # Base sales with trend, seasonality, and control effects
    sales = (
        52000
        + 110 * t
        + 3500 * np.sin(2 * np.pi * t / 52)
        - 230 * price
        + 8500 * promo
    )

    # Channel contributions with different adstock/saturation profiles
    channel_params = [
        ("search", 0.58, 1.35, 23000, 11000),
        ("social", 0.42, 1.7, 18000, 7000),
        ("video", 0.72, 1.2, 32000, 14000),
        ("display", 0.30, 1.9, 10000, 4000),
    ]

    for name, decay, strength, midpoint, beta in channel_params:
        transformed = hill_saturation(
            geometric_adstock(channels[name], decay, 8), strength, midpoint
        )
        sales += beta * transformed

    # Add observation noise
    sales += rng.normal(0, 2500, n_periods)
    sales = np.maximum(sales, 1000)

    return pd.DataFrame(
        {
            "date": pd.date_range(start_date, periods=n_periods, freq=freq),
            **channels,
            "price": price,
            "promo": promo,
            "sales": sales,
        }
    )


def generate_channel_response_curve(
    channel_name: str,
    max_spend: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the theoretical response curve for a channel.

    Returns
    -------
    x : np.ndarray
        Spend values from 0 to max_spend.
    y : np.ndarray
        Expected response (contribution to sales).
    """
    # Default params per channel (matching demo data)
    params = {
        "search": {"decay": 0.58, "strength": 1.35, "midpoint": 23000, "beta": 11000},
        "social": {"decay": 0.42, "strength": 1.7, "midpoint": 18000, "beta": 7000},
        "video": {"decay": 0.72, "strength": 1.2, "midpoint": 32000, "beta": 14000},
        "display": {"decay": 0.30, "strength": 1.9, "midpoint": 10000, "beta": 4000},
    }

    p = params.get(channel_name, params["search"])
    x = np.linspace(0, max_spend, n_points)
    adstocked = x / (1 - p["decay"] + 1e-6)  # steady-state adstock
    saturated = hill_saturation(adstocked, p["strength"], p["midpoint"])
    y = p["beta"] * saturated
    return x, y