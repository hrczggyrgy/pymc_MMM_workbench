from __future__ import annotations
import pandas as pd
import streamlit as st
from utils.modeling import generate_demo_data

DATE_HINTS = {"date", "week", "day", "ds", "timestamp"}
TARGET_HINTS = {"sales", "revenue", "conversions", "orders", "target", "outcome"}
CONTROL_HINTS = {"price", "promo", "promotion", "holiday", "trend", "seasonality"}

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
    cols = list(df.columns)
    lowered = {c: c.lower().strip() for c in cols}
    date_col = next((c for c in cols if lowered[c] in DATE_HINTS), cols[0])
    target_col = next((c for c in cols if lowered[c] in TARGET_HINTS), None)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if target_col is None:
        target_col = numeric[-1] if numeric else cols[-1]
    controls = [c for c in numeric if lowered[c] in CONTROL_HINTS and c != target_col]
    channels = [c for c in numeric if c not in controls and c != target_col and ("spend" in lowered[c] or c in ["search", "social", "video", "display", "tv", "radio", "email"])]
    if len(channels) < 2:
        channels = [c for c in numeric if c not in controls and c != target_col][:4]
    return date_col, target_col, channels, controls

def validate_data(df: pd.DataFrame, date_col: str, target_col: str, channels: list[str]) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
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
