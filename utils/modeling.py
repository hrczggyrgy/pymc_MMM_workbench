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
    decay: dict[str, float] | float,
    l_max: int,
    strength: dict[str, float] | float,
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

    Parameters
    ----------
    decay : dict or float
        Per-channel adstock decay, or single shared value
    strength : dict or float
        Per-channel saturation strength, or single shared value

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

    # Normalize to per-channel dicts
    if isinstance(decay, (int, float)):
        decay = {c: float(decay) for c in channels}
    if isinstance(strength, (int, float)):
        strength = {c: float(strength) for c in channels}

    for channel in channels:
        raw = data[channel].clip(lower=0).to_numpy(float)
        midpoint = max(
            np.median(raw[raw > 0]) * midpoint_scale if np.any(raw > 0) else 1, 1
        )
        media[channel] = transform_media(raw, decay[channel], l_max, strength[channel], midpoint)
        params[channel] = {
            "decay": decay[channel],
            "l_max": l_max,
            "strength": strength[channel],
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
    decay: dict[str, float] | float = 0.5,
    l_max: int = 8,
    strength: dict[str, float] | float = 1.5,
    midpoint_scale: float = 1.0,
    seasonality: bool = True,
    prior_scale: float = 1.0,
    draws: int = 500,
    tune: int = 500,
    seed: int = 42,
    test_size: float = 0.0,
) -> dict:
    """
    Fit a Bayesian MMM using PyMC.

    Model:
        y ~ Normal(alpha + X_scaled @ beta, sigma)
        alpha ~ Normal(0, 1.5)
        beta ~ Normal(0, prior_scale)
        sigma ~ HalfNormal(1)

    Parameters
    ----------
    decay : dict or float
        Per-channel adstock decay, or single shared value
    strength : dict or float
        Per-channel saturation strength, or single shared value
    test_size : float
        Fraction of data to hold out for out-of-sample validation (0.0 to 0.5)

    Returns
    -------
    dict with keys:
        idata, prediction, summary, contrib, features, beta_draws,
        means, stds, y_mean, y_std, params, channels, current_spend
        If test_size > 0: also includes test_prediction, test_metrics
    """
    data, X, X_scaled, y, means, stds, params = prepare_features(
        df, date_col, target_col, channels, controls, decay, l_max, strength, midpoint_scale, seasonality
    )

    y_mean, y_std = y.mean(), max(y.std(), 1.0)
    y_scaled = (y - y_mean) / y_std

    # Train/test split
    n_obs = len(data)
    test_n = int(n_obs * test_size)
    train_n = n_obs - test_n
    
    if test_n > 0:
        # Split data chronologically
        train_idx = slice(0, train_n)
        test_idx = slice(train_n, n_obs)
        
        X_train = X_scaled.iloc[train_idx]
        y_train = y_scaled[train_idx]
        X_test = X_scaled.iloc[test_idx]
        y_test = y_scaled[test_idx]
        dates_train = data[date_col].iloc[train_idx].values
        dates_test = data[date_col].iloc[test_idx].values
        y_test_orig = y[test_idx]
    else:
        X_train = X_scaled
        y_train = y_scaled
        X_test = None
        y_test = None
        dates_train = data[date_col].values
        dates_test = None
        y_test_orig = None
        train_n = n_obs

    coords = {"feature": X.columns.tolist(), "obs": np.arange(train_n)}

    with pm.Model(coords=coords):
        x_data = pm.Data("x", X_train.to_numpy(), dims=("obs", "feature"))

        alpha = pm.Normal("alpha", 0, 1.5)
        beta = pm.Normal("beta", 0, prior_scale, dims="feature")
        sigma = pm.HalfNormal("sigma", 1)

        mu = pm.Deterministic("mu", alpha + pm.math.dot(x_data, beta), dims="obs")
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y_train, dims="obs")

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

    # Posterior predictions on training data
    posterior_mu = idata.posterior["mu"].stack(sample=("chain", "draw")).values
    pred_samples = posterior_mu * y_std + y_mean

    prediction = pd.DataFrame(
        {
            "date": dates_train,
            "observed": y[:train_n] if test_n > 0 else y,
            "mean": pred_samples.mean(axis=1),
            "low": np.quantile(pred_samples, 0.05, axis=1),
            "high": np.quantile(pred_samples, 0.95, axis=1),
        }
    )

    # Out-of-sample predictions if test set provided
    test_prediction = None
    test_metrics = None
    if test_n > 0:
        # Use posterior predictive for test data - create new model with test coords
        test_coords = {"feature": X.columns.tolist(), "obs": np.arange(test_n)}
        with pm.Model(coords=test_coords) as test_model:
            x_test_data = pm.Data("x_test", X_test.to_numpy(), dims=("obs", "feature"))
            
            alpha = pm.Normal("alpha", 0, 1.5)
            beta = pm.Normal("beta", 0, prior_scale, dims="feature")
            sigma = pm.HalfNormal("sigma", 1)

            mu_test = pm.Deterministic("mu_test", alpha + pm.math.dot(x_test_data, beta), dims="obs")
            # Use posterior samples for prediction
            pm.Normal("likelihood_test", mu=mu_test, sigma=sigma, observed=y_test, dims="obs")

            # Sample from posterior predictive
            post_pred = pm.sample_posterior_predictive(
                idata, var_names=["mu_test"], random_seed=seed, progressbar=False
            )
        
        test_mu = post_pred.posterior_predictive["mu_test"].stack(sample=("chain", "draw")).values
        test_pred_samples = test_mu * y_std + y_mean

        test_prediction = pd.DataFrame(
            {
                "date": dates_test,
                "observed": y_test_orig,
                "mean": test_pred_samples.mean(axis=1),
                "low": np.quantile(test_pred_samples, 0.05, axis=1),
                "high": np.quantile(test_pred_samples, 0.95, axis=1),
            }
        )

        # Calculate test metrics
        test_rmse = np.sqrt(((test_prediction.observed - test_prediction["mean"]) ** 2).mean())
        test_mape = np.mean(np.abs((test_prediction.observed - test_prediction["mean"]) / test_prediction.observed)) * 100
        test_r2 = 1 - ((test_prediction.observed - test_prediction["mean"]) ** 2).sum() / ((test_prediction.observed - test_prediction.observed.mean()) ** 2).sum()
        
        test_metrics = {
            "rmse": test_rmse,
            "mape": test_mape,
            "r2": test_r2,
            "n_test": test_n,
        }

    # Posterior summary
    beta_draws = idata.posterior["beta"].stack(sample=("chain", "draw")).transpose("sample", "feature").values
    feature_names = X.columns.tolist()
    summary = az.summary(idata, var_names=["alpha", "beta", "sigma"], ci_prob=0.9).reset_index().rename(columns={"index": "parameter"})

    # Channel contributions (on original scale)
    contrib = pd.DataFrame(
        {
            c: beta_draws[:, feature_names.index(c)].mean() * X_scaled[c].to_numpy() * y_std
            for c in channels
        }
    )

    result = {
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
    
    if test_n > 0:
        result["test_prediction"] = test_prediction
        result["test_metrics"] = test_metrics
        result["train_n"] = train_n
        result["test_n"] = test_n

    return result


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
    "fit_bayesian_mmm_cached",
    "response_samples",
    "scenario_lift",
    "optimize_budget",
]


@st.cache_resource(show_spinner=False)
def fit_bayesian_mmm_cached(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    channels: tuple[str, ...],
    controls: tuple[str, ...],
    decay: tuple[float, ...],
    l_max: int,
    strength: tuple[float, ...],
    midpoint_scale: float,
    seasonality: bool,
    prior_scale: float,
    draws: int,
    tune: int,
    seed: int = 42,
    test_size: float = 0.0,
) -> dict:
    """
    Cached version of fit_bayesian_mmm that takes hashable arguments.
    
    Converts dict/list params to tuples for cache key hashing.
    """
    # Convert tuples back to dicts for the actual fitting function
    channels_list = list(channels)
    controls_list = list(controls)
    decay_dict = {c: d for c, d in zip(channels_list, decay)}
    strength_dict = {c: s for c, s in zip(channels_list, strength)}
    
    return fit_bayesian_mmm(
        df, date_col, target_col, channels_list, controls_list,
        decay_dict, l_max, strength_dict, midpoint_scale,
        seasonality, prior_scale, draws, tune, seed, test_size
    )