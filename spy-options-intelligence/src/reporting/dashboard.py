# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Streamlit 3-tab dashboard for SPY Options Intelligence.

Tabs:
  1. Options Movement  — date-range picker, events table, bar chart + histogram
  2. Space Utilization — latest space report, horizontal bar chart + savings table
  3. Hardware Usage    — hardware JSONs, CPU/memory timelines + per-command table

Launch:
    streamlit run src/reporting/dashboard.py
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Streamlit is imported at module level — this module is only executed when
# `streamlit run` is used, so the import is intentional.
try:
    import streamlit as st
except ImportError as exc:
    raise ImportError(
        "streamlit is required for the dashboard. "
        "Install it with: pip install streamlit>=1.30.0"
    ) from exc

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("data")
_REPORTS_DIR = _DATA_ROOT / "reports"
_FORWARD_DIR = _REPORTS_DIR / "options_forward"
_SPACE_DIR = _REPORTS_DIR / "space"
_HARDWARE_DIR = _REPORTS_DIR / "hardware"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_forward(start: str, end: str) -> pd.DataFrame:
    """Load forward-scan CSVs filtered to the given date range."""
    if not _FORWARD_DIR.exists():
        return pd.DataFrame()

    dfs = []
    for csv_path in sorted(_FORWARD_DIR.glob("*_forward.csv")):
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    if "date" in df.columns:
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.reset_index(drop=True)


def _load_latest_space() -> Optional[dict]:
    """Load the most recent space JSON report."""
    if not _SPACE_DIR.exists():
        return None
    jsons = sorted(_SPACE_DIR.glob("*_space.json"))
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        return json.load(f)


def _load_hardware(selected_date: str) -> pd.DataFrame:
    """Load all hardware JSONs for a given date."""
    if not _HARDWARE_DIR.exists():
        return pd.DataFrame()

    records = []
    for path in sorted(_HARDWARE_DIR.glob(f"{selected_date}_*.json")):
        try:
            with open(path) as f:
                records.append(json.load(f))
        except Exception:
            pass

    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _tab_options_movement() -> None:
    """Render the Options Movement tab."""
    st.header("Options Movement")

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=30),
            key="mv_start",
        )
    with col2:
        end = st.date_input("End date", value=date.today(), key="mv_end")

    if start > end:
        st.error("Start date must be before end date.")
        return

    df = _load_forward(str(start), str(end))

    if df.empty:
        st.info("No forward scan reports found for the selected range. Run `scan-options-forward` first.")
        return

    st.subheader(f"{len(df)} qualifying entries  |  {df['date'].nunique() if 'date' in df.columns else 0} days")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    if "minutes_to_trigger" in df.columns:
        c1.metric("Median mins to trigger", f"{df['minutes_to_trigger'].median():.0f} min")
    if "gain_pct" in df.columns:
        c2.metric("Median gain at trigger", f"{df['gain_pct'].median():.1f}%")
    if "above_20pct_duration_min" in df.columns:
        c3.metric("Median ≥20% duration", f"{df['above_20pct_duration_min'].median():.0f} min")
    if "above_10pct_duration_min" in df.columns:
        c4.metric("Median ≥10% duration", f"{df['above_10pct_duration_min'].median():.0f} min")

    # Events table
    st.subheader("Events")
    st.dataframe(df, use_container_width=True)

    # Bar chart: events per day
    if "date" in df.columns:
        st.subheader("Events per Day")
        daily = df.groupby("date").size().reset_index(name="count")
        st.bar_chart(daily.set_index("date")["count"])

    # Histogram: gain_pct distribution
    if "gain_pct" in df.columns and not df["gain_pct"].isna().all():
        st.subheader("Gain % Distribution")
        import altair as alt

        hist_df = df[["gain_pct"]].dropna()
        chart = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(
                alt.X("gain_pct:Q", bin=alt.Bin(maxbins=30), title="Gain %"),
                alt.Y("count():Q", title="Events"),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def _tab_space_utilization() -> None:
    """Render the Space Utilization tab."""
    st.header("Space Utilization")

    data = _load_latest_space()
    if data is None:
        st.info("No space reports found. Run `report-space` first.")
        return

    total_mb = data.get("total_mb", data.get("total_bytes", 0) / 1024 ** 2)
    total_files = data.get("total_files", 0)

    c1, c2 = st.columns(2)
    c1.metric("Total Size", f"{total_mb:.1f} MB")
    c2.metric("Total Files", str(total_files))

    # Build DataFrame for horizontal bar chart
    tree = data.get("tree_mb", data.get("tree", {}))
    rows = []
    for key, val in tree.items():
        mb = val.get("mb", round(val.get("bytes", 0) / 1024 ** 2, 3))
        files = val.get("files", 0)
        rows.append({"directory": key, "size_mb": mb, "files": files})

    if not rows:
        st.info("No directory data in report.")
        return

    df = pd.DataFrame(rows).sort_values("size_mb", ascending=False).head(20)

    st.subheader("Size by Directory (Top 20)")
    st.bar_chart(df.set_index("directory")["size_mb"])

    st.subheader("Directory Details")
    st.dataframe(df, use_container_width=True)


def _tab_hardware_usage() -> None:
    """Render the Hardware Usage tab."""
    st.header("Hardware Usage")

    # Date picker
    selected_date = st.date_input(
        "Select date",
        value=date.today(),
        key="hw_date",
    )

    df = _load_hardware(str(selected_date))

    if df.empty:
        st.info(
            f"No hardware metrics for {selected_date}. "
            "Run a command with hardware tracking enabled first."
        )
        return

    # Per-command summary
    st.subheader("Per-Command Summary")
    cols_to_show = [
        c for c in [
            "command", "elapsed_sec", "cpu_pct_avg",
            "mem_rss_start_mb", "mem_rss_end_mb", "mem_delta_mb",
            "disk_read_mb", "disk_write_mb",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols_to_show], use_container_width=True)

    # CPU timeline
    if "command" in df.columns and "cpu_pct_avg" in df.columns:
        st.subheader("CPU Usage per Command")
        cpu_df = df[["command", "cpu_pct_avg"]].set_index("command")
        st.bar_chart(cpu_df)

    # Memory timeline
    if "command" in df.columns and "mem_delta_mb" in df.columns:
        st.subheader("Memory Delta per Command (MB)")
        mem_df = df[["command", "mem_delta_mb"]].set_index("command")
        st.bar_chart(mem_df)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="SPY Options Intelligence",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("SPY Options Intelligence Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "Options Movement",
        "Space Utilization",
        "Hardware Usage",
    ])

    with tab1:
        _tab_options_movement()
    with tab2:
        _tab_space_utilization()
    with tab3:
        _tab_hardware_usage()


if __name__ == "__main__":
    main()
