"""Utilities package for PyMC MMM Workbench.

Lazy imports for heavy modules (pymc, arviz, streamlit caching) to avoid import-time issues.
"""

from utils.data import (
    ensure_demo_data,
    set_data,
    detect_schema,
    validate_data,
    validate_and_clean_upload,
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

# Lazy imports for plotting (requires streamlit runtime for caching)
def _lazy_import_plotting():
    from utils.plotting import (
        line_with_band,
        allocation_chart,
        get_channel_color,
        get_channel_colors,
        CHANNEL_COLORS,
        response_curve_plot,
        posterior_density_plot,
    )
    return (
        line_with_band,
        allocation_chart,
        get_channel_color,
        get_channel_colors,
        CHANNEL_COLORS,
        response_curve_plot,
        posterior_density_plot,
    )

# Lazy imports for modeling and optimization (require pymc/arviz)
def _lazy_import_modeling():
    from utils.modeling import (
        fit_bayesian_mmm,
        fit_bayesian_mmm_cached,
        response_samples,
        scenario_lift,
        prepare_features,
    )
    return fit_bayesian_mmm, fit_bayesian_mmm_cached, response_samples, scenario_lift, prepare_features

def _lazy_import_optimization():
    from utils.optimization import (
        optimize_budget,
        compute_channel_roi,
        response_curve_data,
    )
    return optimize_budget, compute_channel_roi, response_curve_data


__all__ = [
    "ensure_demo_data",
    "set_data",
    "detect_schema",
    "validate_data",
    "validate_and_clean_upload",
    "configured_data",
    "geometric_adstock",
    "hill_saturation",
    "transform_media",
    "marginal_response",
    "steady_state_adstock",
    "generate_demo_data",
    "generate_channel_response_curve",
    "_lazy_import_plotting",
    "_lazy_import_modeling",
    "_lazy_import_optimization",
]