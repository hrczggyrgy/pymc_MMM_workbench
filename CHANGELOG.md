# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Visual polish: bordered containers, consistent chart margins, Material icons
- `st.status()` for model fitting with progress updates
- Sidebar page status badges (model fitted, optimization done)
- Footer with attribution ("Built with PyMC · Streamlit · Plotly")
- Favicon matching logo
- `st.container(border=True)` for section grouping across all pages
- `st.columns(gap="large")` for consistent metric spacing
- `st.divider()` standardized; removed excess horizontal rules

### Fixed
- Effect Explorer KeyError:2 bug (dict vs tuple indexing in lazy imports)
- Scenarios page defensive check for missing channels in plan
- Effect Explorer defensive check for channel in model result
- Scenarios page defensive check for plan channels in model result
- Import safety: lazy imports now return dicts (not tuples) to prevent positional unpacking bugs

### Changed
- Sentence casing for headers and labels ("Fit Bayesian MMM" not "Fit Bayesian Mmm")
- Consistent Plotly margins: `margin=dict(l=10, r=10, t=40, b=10)` across all charts
- Consistent `st.columns(gap="large")` for KPI metrics
- `st.divider()` used consistently; excess horizontal rules removed
- `st.columns(gap="large")` for all metric rows
- `st.columns(..., vertical_alignment="bottom")` on slider-heavy pages

### Added
- `st.status()` for model fitting with sub-steps
- `st.badge()` for page completion state in sidebar
- `st.logo()` with MDI chart-line icon
- Favicon matching logo (MDI chart-line)
- Footer caption: "Built with PyMC · Streamlit · Plotly"
- GitHub Actions CI workflow (pytest + ruff + syntax checks)
- `requirements-lock.txt` frozen from `pip freeze`

### Security
- `.gitignore` excludes `.streamlit/secrets.toml`, `.venv/`, `__pycache__/`, `*.log`

## [0.1.0] - 2026-08-04

### Added
- Initial release: PyMC MMM Workbench with 6-page Streamlit app
- Home, Data, Effect Explorer, Model, Scenarios, Optimization pages
- Per-channel adstock/saturation parameters
- Model caching with `st.cache_resource`
- Out-of-sample validation (holdout metrics)
- Risk-aware optimization with variance penalty
- Comprehensive test suite (38 tests)
- GitHub Actions CI (pytest + ruff + syntax checks)
- Comprehensive README with architecture, model spec, workflow guide
- MIT License
- `requirements-lock.txt` for reproducible builds