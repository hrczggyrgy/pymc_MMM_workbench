"""Utilities package for PyMC MMM Workbench."""

from utils.data import (
    ensure_demo_data,
    set_data,
    detect_schema,
    validate_data,
    configured_data,
)
from utils.transformations import (
    geometric_adstock,
    hill_saturation,
    transform_media,
    marginal_response,
    steady_state_adstock,
)
from utils.simulation import generate_demo_data, generate_channel_response_curve
from utils.modeling import (
    fit_bayesian_mmm,
    response_samples,
    scenario_lift,
    prepare_features,
)
from utils.optimization import (
    optimize_budget,
    compute_channel_roi,
    response_curve_data,
)
from utils.plotting import line_with_band, allocation_chart

__all__ = [
    "ensure_demo_data",
    "set_data",
    "detect_schema",
    "validate_data",
    "configured_data",
    "geometric_adstock",
    "hill_saturation",
    "transform_media",
    "marginal_response",
    "steady_state_adstock",
    "generate_demo_data",
    "generate_channel_response_curve",
    "fit_bayesian_mmm",
    "response_samples",
    "scenario_lift",
    "prepare_features",
    "optimize_budget",
    "compute_channel_roi",
    "response_curve_data",
    "line_with_band",
    "allocation_chart",
]