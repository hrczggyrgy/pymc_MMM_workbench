# PyMC MMM Workbench

A production-quality Streamlit workbench for Bayesian Marketing Mix Modeling. Explore adstock and saturation interactively, fit a real PyMC Bayesian model with full posterior uncertainty, test what-if scenarios, and optimize budget allocation — all in a guided, educational interface.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-red)
![PyMC](https://img.shields.io/badge/pymc-5.16%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

| Page | Purpose |
|------|---------|
| **Home** | Guided overview, workflow, glossary, and quick start |
| **Data** | CSV upload, schema detection, validation, quick visualizations |
| **Effect Explorer** | Interactive adstock/saturation curves with live parameter sliders |
| **Model** | Bayesian MMM with PyMC — posterior summaries, diagnostics, trace plots, out-of-sample validation |
| **Scenarios** | What-if spend changes with full posterior lift distributions |
| **Optimization** | Constrained budget allocation maximizing expected outcome |

### Key Capabilities

- **Real Bayesian inference** via PyMC (NUTS sampler, 2 chains, R-hat/ESS diagnostics)
- **Per-channel adstock (geometric carryover)** and **Hill saturation (diminishing returns)** with interactive exploration
- **Full posterior uncertainty** propagated to predictions, scenarios, and optimization
- **Demo mode** with realistic synthetic data (104 weeks, 4 channels, controls, seasonality)
- **Constraint-aware optimization** (SLSQP with bounds + equality constraint)
- **Out-of-sample validation** with holdout metrics (RMSE, MAPE, R²)
- **Model caching** with `st.cache_resource` for instant re-runs on same config
- **Exportable results** (CSV allocation, posterior summaries)

## Quick Start

```bash
# Clone and enter directory
cd pymc_MMM_workbench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 — click **"Try Demo Data"** on the Home page to see the full workflow instantly.

**Live demo:** https://pymcmmmworkbench.streamlit.app/

## Data Format

Prepare a CSV with:

| Column Type | Example Names | Required |
|-------------|---------------|----------|
| Date | `date`, `week`, `ds` | Yes |
| Target | `sales`, `revenue`, `conversions` | Yes |
| Channel Spends | `search`, `social`, `video`, `display`, `tv`, `radio` | ≥2 |
| Controls (optional) | `price`, `promo`, `holiday` | No |

- **Frequency**: Weekly (recommended) or daily
- **Minimum periods**: 30 (52+ recommended for seasonality)
- **No missing dates** (auto-sorted if unsorted)
- **Spend ≥ 0** (negative values flagged)

### Example CSV

```csv
date,sales,search,social,video,display,price,promo
2024-01-07,55000,12000,8000,15000,3000,99.5,0
2024-01-14,56200,11500,8200,14800,3100,99.5,0
2024-01-21,57100,13000,9000,16000,2900,98.0,1
...
```

## Workflow Guide

### 1. Data → Load & Validate
- Upload CSV or click **Try Demo Data**
- Auto-detects date/target/channels/controls
- Review validation warnings (missing values, duplicates, negative spend, short series)
- Save configuration

### 2. Effect Explorer → Understand Transformations
- Select a channel
- Adjust **adstock decay** (carryover), **window**, **saturation strength**, **midpoint**
- See live: raw spend → adstocked → saturated → marginal response
- Presets for Search/Video/Display profiles

### 3. Model → Fit Bayesian MMM
- Configure per-channel decay/strength (or use shared defaults)
- Set prior strength, seasonality, carryover window
- Choose holdout fraction for out-of-sample validation
- Choose **Fast demo** (250 draws) or **Production** (700+ draws)
- Click **Fit Bayesian MMM** — results cached automatically
- Review:
  - Observed vs posterior prediction with 90% credible band (train + test)
  - Out-of-sample metrics (RMSE, MAPE, R²) if holdout enabled
  - Posterior summary table (R-hat, ESS, HDI)
  - Channel coefficient posterior densities
  - Stacked contribution over time
  - Sampling diagnostics (trace plots, pair plots, residuals)

### 4. Scenarios → Test What-If Questions
- Adjust per-channel % change from baseline
- Quick presets: +10% all, shift to video, cut display, fixed budget rebalance
- See:
  - Expected incremental lift with 90% credible interval
  - Probability of positive lift
  - Channel-level decomposition with uncertainty
  - Response curves with baseline/scenario markers
- Save scenarios for comparison

### 5. Optimization → Allocate Budget
- Set total budget and per-channel min/max constraints
- Presets: ±25%, ±50%, wide, tight
- Click **Find Recommended Allocation**
- Results:
  - Current vs recommended allocation table + chart
  - Expected improvement with credible interval
  - ROI and marginal ROAS per channel
  - Response curves with constraint bounds
  - Budget sensitivity curve (outcome vs total budget)
- Download CSV

## Architecture

```
pymc_MMM_workbench/
├── app.py                    # Entry point, page config
├── runtime.txt               # Python 3.11 for Streamlit Cloud
├── requirements.txt          # Pinned dependencies
├── requirements-lock.txt     # Frozen pip freeze for reproducibility
├── LICENSE                   # MIT License
├── pages/                    # Streamlit multipage app
│   ├── 1_Home.py
│   ├── 2_Data.py
│   ├── 3_Effect_Explorer.py
│   ├── 4_Model.py
│   ├── 5_Scenarios.py
│   └── 6_Optimization.py
├── utils/
│   ├── __init__.py           # Public API exports
│   ├── data.py               # Loading, validation, schema detection
│   ├── simulation.py         # Synthetic data generation (cached)
│   ├── transformations.py    # Adstock, saturation, marginal response
│   ├── modeling.py           # PyMC model, fitting, predictions (cached)
│   ├── optimization.py       # Budget optimization, ROI, response curves
│   └── plotting.py           # Plotly chart builders (cached)
├── .streamlit/config.toml    # Streamlit theme & server config
└── .gitignore
```

### Core Modules

| Module | Responsibility |
|--------|----------------|
| `simulation.py` | `generate_demo_data()` — realistic synthetic MMM data |
| `transformations.py` | `geometric_adstock()`, `hill_saturation()`, `marginal_response()` |
| `modeling.py` | `fit_bayesian_mmm()`, `fit_bayesian_mmm_cached()`, `scenario_lift()`, `response_samples()` |
| `optimization.py` | `optimize_budget()`, `compute_channel_roi()`, `response_curve_data()` |
| `data.py` | CSV loading, schema detection, validation |
| `plotting.py` | Reusable Plotly figures with `@st.cache_data` |

## Model Specification

**Likelihood:**
```
sales_t ~ Normal(μ_t, σ)
```

**Linear Predictor:**
```
μ_t = α + Σ_c β_c · f_c(spend_c,t) + γ · controls_t + δ · trend_t + seasonality_t
```

**Media Transformation (fixed hyperparameters, not estimated):**
```
f_c(spend) = Hill( Adstock(spend; λ_c, L) ; α_c, midpoint_c )
Adstock(spend; λ, L) = Σ_{k=0}^L λ^k · spend_{t-k}
Hill(x; α, m) = x^α / (x^α + m^α)
```

**Priors:**
```
α ~ Normal(0, 1.5)                    # Intercept (standardized scale)
β_c ~ Normal(0, prior_scale)          # Channel coefficients (standardized scale)
σ ~ HalfNormal(1)                     # Observation noise
```

**Out-of-sample validation:**
- Chronological train/test split (last `test_size` fraction)
- Posterior predictive on test set via `sample_posterior_predictive`
- Metrics: RMSE, MAPE, R² on held-out data

**Notes:**
- Adstock/saturation parameters (λ, α, midpoint) are **fixed hyperparameters** set in UI
- This keeps the model fast and identifiable; for full hierarchical estimation see PyMC-Marketing
- Features are standardized before regression; coefficients are on standardized scale

## Performance Notes

| Mode | Draws | Chains | Time (4 channels, 104 weeks) |
|------|-------|--------|------------------------------|
| Fast Demo | 250 | 2 | ~30-60 seconds |
| Production | 700 | 2 | ~2-4 minutes |
| Thorough | 2000 | 4 | ~10-15 minutes |

- **Model caching**: Same configuration → instant load via `st.cache_resource`
- **First run** compiles PyMC model (slower); subsequent runs faster
- Use `cores=1` for reproducibility; increase for speed if needed
- For production use: increase draws, check R-hat < 1.01, ESS > 400

## Caveats & Limitations

⚠️ **This is an analytical prototype, not causal proof.**

- **Correlation ≠ Causation**: MMM identifies associations; validate with experiments (geo tests, holdouts)
- **Data quality is critical**: Spend definitions, aggregation level, missing data all affect results
- **Model assumptions**: Fixed adstock/saturation, no channel interactions, steady-state optimization
- **Extrapolation risk**: Response curves beyond observed spend range are highly uncertain
- **Short time series**: <52 weeks → unstable seasonality; <30 weeks → unreliable posteriors
- **Prior sensitivity**: Check results with different prior scales

**Before operational decisions:**
1. Holdout validation (last 8-12 weeks)
2. Calibration experiments (geo lift tests)
3. Expert review of coefficient signs/magnitudes
4. Sensitivity analysis (priors, hyperparameters, data windows)

## Extending the Workbench

### Add a new channel type
1. Update `simulation.py` with realistic spend patterns
2. Add preset in `Effect Explorer` page
3. Model handles arbitrary channel names automatically

### Custom saturation function
```python
# In transformations.py
def michaelis_menten(x, vmax, km):
    return vmax * x / (km + x)
```

### Hierarchical adstock/saturation (PyMC-Marketing style)
Replace fixed hyperparameters with priors in `modeling.py`:
```python
decay = pm.Beta("decay", 2, 2, dims="channel")
strength = pm.Gamma("strength", 2, 1, dims="channel")
```

### Add interaction terms
In `prepare_features()`:
```python
X["search_x_social"] = X["search"] * X["social"]
```

## Testing & CI

```bash
# Run tests
pytest tests/

# Lint
ruff check .
```

GitHub Actions workflow runs pytest + ruff on every push.

## References & Prior Art

- **PyMC-Marketing** — MMM reference implementation with per-channel hierarchical adstock/saturation
  - Case study: https://www.pymc-marketing.io/en/0.15.1/notebooks/mmm/mmm_case_study.html
  - Explainer app: https://pymc-marketing-explainer.streamlit.app/
- **PyMC** — Bayesian modeling in Python
- **ArviZ** — Exploratory analysis of Bayesian models
- **Streamlit** — Rapid data app framework
- **Plotly** — Interactive visualizations

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built for analysts and stakeholders who want to understand, not just use, their marketing mix model.**# Force rebuild
# Force rebuild Tue Aug  4 05:01:15 PM CEST 2026
# Rebuild Tue Aug  4 05:02:48 PM CEST 2026
# Rebuild 
# Rebuild Tue Aug  4 05:10:52 PM CEST 2026
# Force rebuild Tue Aug  4 05:12:40 PM CEST 2026
