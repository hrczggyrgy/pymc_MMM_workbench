"""Bayesian MMM modeling with PyMC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import streamlit as st
from numpy.typing import NDArray

from utils.transformations import geometric_adstock, hill_saturation, transform_media
from utils.optimization import optimize_budget


def prepare_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    channels: list[str],
    controls: list[str],
    decay: float,
    l_max: int,
    strength: float,
    midpoint_scale: float,
    seasonality: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    NDArray[np.floating],
    pd.Series,
    pd.Series,
    dict,
]:
    """
    Prepare model features: transform media, add controls, trend, seasonality.

    Returns
    -------
    data : DataFrame
        Sorted, cleaned data.
    X : DataFrame
        Raw features (transformed media + controls + trend + seasonality).
    X_scaled : DataFrame
        Standardized features.
    y : ndarray
        Target values.
    means, stds : Series
        Feature scaling parameters.
    params : dict
        Transformation parameters per channel.
    """
    data = df.copy().sort_values(date_col).dropna(subset=[target_col] + channels).reset_index(drop=True)

    media = {}
    params = {}

    for channel in channels:
        raw = data[channel].clip(lower=0).to_numpy(float)
        midpoint = max(
            np.median(raw[raw > 0]) * midpoint_scale if np.any(raw > 0) else 1, 1
        )
        media[channel] = transform_media(raw, decay, l_max, strength, midpoint)
        params[channel] = {
            "decay": decay,
            "l_max": l_max,
            "strength": strength,
            "midpoint": midpoint,
        }

    X = pd.DataFrame(media, index=data.index)

    for col in controls:
        if col in data:
            X[col] = pd.to_numeric(data[col], errors="coerce").fillna(data[col].median())

    X["trend"] = np.linspace(0, 1, len(data))

    if seasonality:
        week = pd.to_datetime(data[date_col]).dt.isocalendar().week.astype(float).to_numpy()
        X["sin_annual"] = np.sin(2 * np.pi * week / 52)
        X["cos_annual"] = np.cos(2 * np.pi * week / 52)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    means, stds = X.mean(), X.std().replace(0, 1)
    X_scaled = (X - means) / stds

    y = pd.to_numeric(data[target_col], errors="coerce").to_numpy(float)

    return data, X, X_scaled, y, means, stds, params


def fit_bayesian_mmm(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    channels: list[str],
    controls: list[str],
    decay: float = 0.5,
    l_max: int = 8,
    strength: float = 1.5,
    midpoint_scale: float = 1.0,
    seasonality: bool = True,
    prior_scale: float = 1.0,
    draws: int = 500,
    tune: int = 500,
    seed: int = 42,
) -> dict:
    """
    Fit a Bayesian MMM using PyMC.

    Model:
        y ~ Normal(alpha + X_scaled @ beta, sigma)
        alpha ~ Normal(0, 1.5)
        beta ~ Normal(0, prior_scale)
        sigma ~ HalfNormal(1)

    Returns
    -------
    dict with keys:
        idata, prediction, summary, contrib, features, beta_draws,
        means, stds, y_mean, y_std, params, channels, current_spend
    """
    data, X, X_scaled, y, means, stds, params = prepare_features(
        df, date_col, target_col, channels, controls, decay, l_max, strength, midpoint_scale, seasonality
    )

    y_mean, y_std = y.mean(), max(y.std(), 1.0)
    y_scaled = (y - y_mean) / y_std

    coords = {"feature": X.columns.tolist(), "obs": np.arange(len(data))}

    with pm.Model(coords=coords):
        x_data = pm.Data("x", X_scaled.to_numpy(), dims=("obs", "feature"))

        alpha = pm.Normal("alpha", 0, 1.5)
        beta = pm.Normal("beta", 0, prior_scale, dims="feature")
        sigma = pm.HalfNormal("sigma", 1)

        mu = pm.Deterministic("mu", alpha + pm.math.dot(x_data, beta), dims="obs")
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y_scaled, dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            target_accept=0.9,
            progressbar=False,
            random_seed=seed,
            return_inferencedata=True,
        )

    # Posterior predictions
    posterior_mu = idata.posterior["mu"].stack(sample=("chain", "draw")).values
    pred_samples = posterior_mu * y_std + y_mean

    prediction = pd.DataFrame(
        {
            "date": data[date_col],
            "observed": y,
            "mean": pred_samples.mean(axis=1),
            "low": np.quantile(pred_samples, 0.05, axis=1),
            "high": np.quantile(pred_samples, 0.95, axis=1),
        }
    )

    # Posterior summary
    beta_draws = idata.posterior["beta"].stack(sample=("chain", "draw")).transpose("sample", "feature").values
    feature_names = X.columns.tolist()
    summary = az.summary(idata, var_names=["alpha", "beta", "sigma"], hdi_prob=0.9).reset_index().rename(columns={"index": "parameter"})

    # Channel contributions (on original scale)
    contrib = pd.DataFrame(
        {
            c: beta_draws[:, feature_names.index(c)].mean() * X_scaled[c].to_numpy() * y_std
            for c in channels
        }
    )

    return {
        "idata": idata,
        "prediction": prediction,
        "summary": summary,
        "contrib": contrib,
        "features": feature_names,
        "beta_draws": beta_draws,
        "means": means,
        "stds": stds,
        "y_mean": y_mean,
        "y_std": y_std,
        "params": params,
        "channels": channels,
        "current_spend": {c: float(data[c].mean()) for c in channels},
    }


def response_samples(
    spend: float,
    channel: str,
    result: dict,
    n: int = 600,
) -> NDArray[np.floating]:
    """
    Generate posterior response samples for a channel at a given spend level.

    Uses the exact transformation from the fitted model.
    """
    p = result["params"][channel]
    # Steady-state adstock approximation for scenario planning
    adstocked = spend / (1 - p["decay"] + 1e-6)
    transformed = hill_saturation(np.array([adstocked]), p["strength"], p["midpoint"])[0]
    scaled = (transformed - result["means"][channel]) / result["stds"][channel]
    beta = result["beta_draws"][:n, result["features"].index(channel)]
    return beta * scaled * result["y_std"]


def scenario_lift(plan: dict[str, float], result: dict) -> NDArray[np.floating]:
    """
    Compute incremental lift for a spend plan vs current baseline.

    Returns posterior samples of incremental outcome.
    """
    n_draws = min(600, len(result["beta_draws"]))
    baseline = np.zeros(n_draws)
    scenario = np.zeros(n_draws)

    for c in result["channels"]:
        baseline += response_samples(result["current_spend"][c], c, result, n_draws)
        scenario += response_samples(plan[c], c, result, n_draws)

    return scenario - baseline


# Re-export optimization functions
__all__ = [
    "generate_demo_data",
    "geometric_adstock",
    "hill_saturation",
    "transform_media",
    "prepare_features",
    "fit_bayesian_mmm",
    "response_samples",
    "scenario_lift",
    "optimize_budget",
]