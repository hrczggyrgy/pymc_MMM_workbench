from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from utils.simulation import generate_demo_data

DATE_HINTS = {"date", "week", "day", "ds", "timestamp"}
TARGET_HINTS = {"sales", "revenue", "conversions", "orders", "target", "outcome"}
CONTROL_HINTS = {"price", "promo", "promotion", "holiday", "trend", "seasonality"}

MAX_CHANNELS_WARNING = 15
HIGH_CORRELATION_THRESHOLD = 0.85
OUTLIER_RATIO_THRESHOLD = 100  # max/median ratio
NEAR_CONSTANT_CV_THRESHOLD = 0.01  # coefficient of variation


def ensure_demo_data() -> None:
    if "data" not in st.session_state:
        st.session_state.data = generate_demo_data()
        st.session_state.source = "Built-in synthetic demo"
    st.session_state.setdefault("date_col", "date")
    st.session_state.setdefault("target_col", "sales")
    st.session_state.setdefault("channel_cols", ["search", "social", "video", "display"])
    st.session_state.setdefault("control_cols", ["price", "promo"])


def set_data(df: pd.DataFrame, source: str) -> None:
    st.session_state.data = df.copy()
    st.session_state.source = source
    date_col, target_col, channels, controls = detect_schema(df)
    st.session_state.date_col = date_col
    st.session_state.target_col = target_col
    st.session_state.channel_cols = channels
    st.session_state.control_cols = controls
    st.session_state.pop("model_result", None)


def detect_schema(df: pd.DataFrame):
    """Purely structural schema detection: treat all numeric non-target non-control columns as channel candidates."""
    cols = list(df.columns)
    lowered = {c: c.lower().strip() for c in cols}
    
    date_col = next((c for c in cols if lowered[c] in DATE_HINTS), cols[0])
    target_col = next((c for c in cols if lowered[c] in TARGET_HINTS), None)
    numeric = df.select_dtypes(include="number").columns.tolist()
    
    if target_col is None:
        target_col = numeric[-1] if numeric else cols[-1]
    
    controls = [c for c in numeric if lowered[c] in CONTROL_HINTS and c != target_col]
    
    # All remaining numeric columns (not target, not controls) are channel candidates
    channels = [c for c in numeric if c not in controls and c != target_col]
    
    return date_col, target_col, channels, controls


def validate_and_clean_upload(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    channels: list[str],
    controls: list[str],
) -> dict:
    """
    Comprehensive validation and cleaning of uploaded data.
    
    Returns dict with:
        usable: bool - whether data is usable for modeling
        errors: list[str] - blocking issues
        warnings: list[str] - non-blocking issues
        cleaned_df: pd.DataFrame - cleaned dataframe
        excluded_channels: list[str] - channels excluded from modeling
        channel_info: dict - per-channel diagnostics
    """
    errors = []
    warnings = []
    excluded_channels = []
    channel_info = {}
    cleaned_df = df.copy()
    
    # 1. Check for duplicate column names
    dup_cols = cleaned_df.columns[cleaned_df.columns.duplicated()].tolist()
    if dup_cols:
        errors.append(f"Duplicate column names detected: {dup_cols}. Please rename columns to be unique.")
    
    # 2. Clean candidate channel columns (strip currency symbols, coerce to numeric)
    for col in channels:
        if col not in cleaned_df.columns:
            warnings.append(f"Channel column '{col}' not found in data.")
            continue
            
        original_series = cleaned_df[col]
        original_dtype = str(original_series.dtype)
        
        # Strip currency symbols and commas
        if original_series.dtype == object:
            cleaned_series = original_series.astype(str).str.replace(r"[$,€£,\s]", "", regex=True)
            # Count non-numeric values
            non_numeric_mask = pd.to_numeric(cleaned_series, errors="coerce").isna() & cleaned_series.notna()
            non_numeric_count = int(non_numeric_mask.sum())
            if non_numeric_count > 0:
                warnings.append(f"Channel '{col}': {non_numeric_count} of {len(cleaned_series)} values could not be parsed as numbers and were set to NaN.")
            cleaned_df[col] = pd.to_numeric(cleaned_series, errors="coerce")
        else:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
    
    # 3. Validate date column
    if date_col not in cleaned_df.columns:
        errors.append(f"Date column '{date_col}' not found in data.")
    else:
        try:
            cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors="coerce")
            nat_count = cleaned_df[date_col].isna().sum()
            if nat_count > 0:
                warnings.append(f"Date column '{date_col}': {nat_count} rows have unparseable dates and were set to NaT.")
        except Exception as e:
            errors.append(f"Date column '{date_col}' could not be parsed: {e}")
    
    # 4. Validate target column
    if target_col not in cleaned_df.columns:
        errors.append(f"Target column '{target_col}' not found in data.")
    else:
        cleaned_df[target_col] = pd.to_numeric(cleaned_df[target_col], errors="coerce")
    
    # 5. Check control columns
    for col in controls:
        if col not in cleaned_df.columns:
            warnings.append(f"Control column '{col}' not found in data.")
        else:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
    
    # 5b. Drop rows where date or target is NaN
    before_drop = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=[date_col, target_col]).reset_index(drop=True)
    dropped = before_drop - len(cleaned_df)
    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows with missing date or target values.")
    
    if len(cleaned_df) == 0:
        errors.append("No valid rows remain after removing missing dates/targets.")
        return {
            "usable": False,
            "errors": errors,
            "warnings": warnings,
            "cleaned_df": cleaned_df,
            "excluded_channels": channels,
            "channel_info": {},
        }
    
    # 6. Sort by date
    cleaned_df = cleaned_df.sort_values(date_col).reset_index(drop=True)
    
    # 7. Date continuity and frequency checks
    if date_col in cleaned_df.columns:
        dates = cleaned_df[date_col]
        if dates.duplicated().any():
            warnings.append(f"Duplicate dates detected ({dates.duplicated().sum()}). Consider aggregating.")
        
        # Check frequency
        try:
            inferred_freq = pd.infer_freq(dates)
            if inferred_freq is None:
                warnings.append("Date frequency could not be inferred (irregular intervals). Seasonality features may be unreliable.")
            elif not inferred_freq.startswith(('W', 'D', 'M')):
                warnings.append(f"Detected frequency '{inferred_freq}' - model assumes weekly/daily data.")
        except Exception:
            pass
        
        # Check for large gaps
        diff = dates.diff().dt.days
        if len(diff) > 1:
            max_gap = diff.max()
            median_gap = diff.median()
            if pd.notna(max_gap) and pd.notna(median_gap) and max_gap > median_gap * 3:
                warnings.append(f"Large gap detected in dates (max gap: {max_gap} days vs median {median_gap} days).")
    
    # 8. Per-channel diagnostics
    valid_channels = []
    for col in channels:
        if col not in cleaned_df.columns:
            warnings.append(f"Channel '{col}' not found in cleaned data.")
            excluded_channels.append(col)
            continue
        
        series = cleaned_df[col]
        info = {
            "original_dtype": str(df[col].dtype) if col in df.columns else "unknown",
            "n_total": len(series),
            "n_missing": int(series.isna().sum()),
            "n_zero": int((series == 0).sum()),
            "n_negative": int((series < 0).sum()),
            "mean": float(series.mean()) if series.notna().any() else 0.0,
            "median": float(series.median()) if series.notna().any() else 0.0,
            "std": float(series.std()) if series.notna().any() else 0.0,
            "min": float(series.min()) if series.notna().any() else 0.0,
            "max": float(series.max()) if series.notna().any() else 0.0,
        }
        
        # All-zero or near-constant check
        if series.notna().any():
            non_zero = series[series > 0]
            if len(non_zero) == 0:
                warnings.append(f"Channel '{col}': all values are zero. Channel excluded from modeling.")
                excluded_channels.append(col)
            elif info["std"] / (abs(info["mean"]) + 1e-9) < NEAR_CONSTANT_CV_THRESHOLD:
                warnings.append(f"Channel '{col}': near-constant values (CV < {NEAR_CONSTANT_CV_THRESHOLD}). Channel excluded from modeling.")
                excluded_channels.append(col)
            elif info["max"] / (abs(info["median"]) + 1e-9) > OUTLIER_RATIO_THRESHOLD:
                warnings.append(f"Channel '{col}': extreme outlier detected (max/median > {OUTLIER_RATIO_THRESHOLD}). Check for unit errors.")
            elif info["n_negative"] > 0:
                warnings.append(f"Channel '{col}': {info['n_negative']} negative spend values detected (clipped to zero).")
                # Clip negative values
                cleaned_df.loc[cleaned_df[col] < 0, col] = 0.0
            
            # Check for long zero stretches (intermittent spend)
            zero_stretch = (series == 0).astype(int).groupby((series != 0).cumsum()).sum()
            if len(zero_stretch) > 0 and zero_stretch.max() > len(series) * 0.5:
                warnings.append(f"Channel '{col}': long stretch of zero spend ({int(zero_stretch.max())} consecutive periods). Intermittent spend may affect adstock estimation.")
        else:
            warnings.append(f"Channel '{col}': all values missing. Channel excluded.")
            excluded_channels.append(col)
        
        if col not in excluded_channels:
            valid_channels.append(col)
        
        channel_info[col] = info
    
    # 9. Channel count checks
    if len(valid_channels) < 2:
        errors.append(f"Need at least 2 valid channels for modeling, found {len(valid_channels)}.")
    elif len(valid_channels) > MAX_CHANNELS_WARNING:
        warnings.append(f"Many channels ({len(valid_channels)}). Modeling may be slow and interpretation difficult. Consider combining channels.")
    
    # 10. Correlation check between valid channels
    if len(valid_channels) >= 2:
        chan_df = cleaned_df[valid_channels]
        corr_matrix = chan_df.corr()
        for i, c1 in enumerate(valid_channels):
            for c2 in valid_channels[i+1:]:
                corr_val = corr_matrix.loc[c1, c2]
                if abs(corr_val) > HIGH_CORRELATION_THRESHOLD:
                    warnings.append(f"High correlation ({corr_val:.2f}) between '{c1}' and '{c2}'. Model may struggle to separate their effects.")
    
    # 11. Data length check
    if len(cleaned_df) < 30:
        warnings.append(f"Only {len(cleaned_df)} periods. Posterior estimates will be unstable. Recommended minimum: 52+ periods.")
    
    return {
        "usable": len(errors) == 0 and len(valid_channels) >= 2,
        "errors": errors,
        "warnings": warnings,
        "cleaned_df": cleaned_df,
        "excluded_channels": excluded_channels,
        "channel_info": channel_info,
        "valid_channels": valid_channels,
    }


def validate_data(df: pd.DataFrame, date_col: str, target_col: str, channels: list[str]) -> list[tuple[str, str]]:
    """Legacy validation function - kept for compatibility."""
    issues = []
    if date_col not in df.columns or target_col not in df.columns:
        return [("error", "Choose valid date and target columns.")]
    if len(channels) < 2:
        issues.append(("warning", "Select at least two media channels for a useful mix model."))
    if len(df) < 30:
        issues.append(("warning", "Fewer than 30 periods: posterior estimates will be unstable."))
    if df.isna().any().any():
        issues.append(("warning", f"Missing values detected: {int(df.isna().sum().sum())}. Fill or remove them before fitting."))
    if df[date_col].duplicated().any():
        issues.append(("warning", "Duplicate dates detected."))
    for col in channels:
        if col in df and (pd.to_numeric(df[col], errors="coerce") < 0).any():
            issues.append(("warning", f"Negative spend detected in `{col}`."))
    try:
        dates = pd.to_datetime(df[date_col])
        if not dates.is_monotonic_increasing:
            issues.append(("info", "Dates are not sorted; the app will sort them before modeling."))
    except Exception:
        issues.append(("warning", "Date column could not be parsed consistently."))
    return issues


def configured_data() -> pd.DataFrame:
    df = st.session_state.data.copy()
    date_col = st.session_state.date_col
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.sort_values(date_col).reset_index(drop=True)
