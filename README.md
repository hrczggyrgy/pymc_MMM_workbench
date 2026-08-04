# PyMC MMM Workbench

A Streamlit workbench for Bayesian marketing mix modeling: synthetic demo data, CSV upload and validation, interactive adstock and saturation concepts, a real PyMC posterior model, scenario planning, and constrained budget optimization.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data schema

Use a daily or weekly CSV with a parseable `date`, a numeric target such as `sales`, at least two numeric channel spend columns, and optional controls such as `price` and `promo`.

## Caveats

This is a portfolio-quality analytical prototype, not causal proof. Validate data definitions, holdout performance, calibration experiments, and business constraints before operational decisions. The fast demo model uses 250 tune and 250 posterior draws; use more draws for a substantive analysis.
