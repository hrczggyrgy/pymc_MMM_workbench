# PyMC MMM Workbench

A production-quality Streamlit workbench for Bayesian Marketing Mix Modeling. Explore adstock and saturation interactively, fit a real PyMC Bayesian model with full posterior uncertainty, test what-if scenarios, and optimize budget allocation — all in a guided, educational interface.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-red)
![PyMC](https://img.shields.io/badge/pymc-5.10%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

| Page | Purpose |
|------|---------|
| **Home** | Guided overview, workflow, glossary, and quick start |
| **Data** | CSV upload, schema detection, validation, quick visualizations |
| **Effect Explorer** | Interactive adstock/saturation curves with live parameter sliders |
| **Model** | Bayesian MMM with PyMC — posterior summaries, diagnostics, trace plots |
| **Scenarios** | What-if spend changes with full posterior lift distributions |
| **Optimization** | Constrained budget allocation maximizing expected outcome |

### Key Capabilities

- **Real Bayesian inference** via PyMC (NUTS sampler, 2 chains, R-hat/ESS diagnostics)
- **Adstock (geometric carryover)** and **Hill saturation (diminishing returns)** with interactive exploration
- **Full posterior uncertainty** propagated to predictions, scenarios, and optimization
- **Demo mode** with realistic synthetic data (104 weeks, 4 channels, controls, seasonality)
- **Constraint-aware optimization** (SLSQP with bounds + equality constraint)
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
- Configure: shared decay/strength, prior strength, seasonality, carryover window
- Choose **Fast demo** (250 draws) or **Production** (700+ draws)
- Click **Fit Bayesian MMM** — watch progress spinner
- Review:
  - Observed vs posterior prediction with 90% credible band
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
│   ├── modeling.py           # PyMC model, fitting, predictions
│   ├── optimization.py       # Budget optimization, ROI, response curves
│   └── plotting.py           # Plotly chart builders (cached)
├── requirements.txt
├── .gitignore
└── README.md
```

### Core Modules

| Module | Responsibility |
|--------|----------------|
| `simulation.py` | `generate_demo_data()` — realistic synthetic MMM data |
| `transformations.py` | `geometric_adstock()`, `hill_saturation()`, `marginal_response()` |
| `modeling.py` | `fit_bayesian_mmm()`, `scenario_lift()`, `response_samples()` |
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

**Media Transformation (fixed, not estimated):**
```
f_c(spend) = Hill( Adstock(spend; λ_c, L) ; α_c, midpoint_c )
Adstock(spend; λ, L) = Σ_{k=0}^L λ^k · spend_{t-k}
Hill(x; α, m) = x^α / (x^α + m^α)
```

**Priors:**
```
α ~ Normal(0, 1.5)           # Intercept (standardized scale)
β_c ~ Normal(0, prior_scale) # Channel coefficients (standardized scale)
σ ~ HalfNormal(1)            # Observation noise
```

**Notes:**
- Adstock/saturation parameters (λ, α, midpoint) are **fixed hyperparameters** set in UI, not estimated
- This keeps the model fast and identifiable; for full hierarchical estimation see PyMC-Marketing
- Features are standardized before regression; coefficients are on standardized scale

## Performance Notes

| Mode | Draws | Chains | Time (4 channels, 104 weeks) |
|------|-------|--------|------------------------------|
| Fast Demo | 250 | 2 | ~30-60 seconds |
| Production | 700 | 2 | ~2-4 minutes |
| Thorough | 2000 | 4 | ~10-15 minutes |

- **First run** compiles PyMC model (slower); subsequent runs faster
- Use `cores=1` for reproducibility; increase for speed if needed
- For production use: increase draws, check R-hat < 1.01, ESS > 400

## Caveats & Limitations

⚠️ **This is an analytical prototype, not causal proof.**

- **Correlation ≠ Causation**: MMM identifies associations; validate with experiments (geo tests, holdouts)
- **Data quality is critical**: Spend definitions, aggregation level, missing data all affect results
- **Model assumptions**: Shared decay/strength, no channel interactions, steady-state optimization
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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pymc` | `pip install pymc>=5.10` (requires Python 3.10+) |
| Sampling very slow | Reduce draws, use `cores=2`, check for divergences |
| R-hat > 1.01 | Increase tune/draws, reparameterize, stronger priors |
| Optimization fails | Check constraint feasibility (min_sum ≤ budget ≤ max_sum) |
| Demo data not loading | Restart app, check `utils/simulation.py` imports |
| Charts not rendering | Ensure Plotly 5.18+, clear browser cache |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PyMC](https://www.pymc.io/) — Bayesian modeling in Python
- [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) — MMM reference implementation
- [ArviZ](https://arviz-devs.github.io/arviz/) — Exploratory analysis of Bayesian models
- [Streamlit](https://streamlit.io/) — Rapid data app framework
- [Plotly](https://plotly.com/python/) — Interactive visualizations

---

**Built for analysts and stakeholders who want to understand, not just use, their marketing mix model.**