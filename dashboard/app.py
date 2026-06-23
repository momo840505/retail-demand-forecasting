from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
DASHBOARD_DATA_DIRECTORY = PROJECT_ROOT / "dashboard" / "data"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from retail_forecasting.replenishment import (  # noqa: E402
    ReplenishmentInputs,
    calculate_replenishment_plan,
)


st.set_page_config(
    page_title="Retail Atelier",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Editorial retail palette
# =============================================================================
INK = "#191715"
INK_SOFT = "#312C28"
PLUM = "#5B2A3C"
CLAY = "#C96D4B"
OCHRE = "#C89A43"
OLIVE = "#778064"
ROSE = "#B85C72"
CREAM = "#F7F3ED"
PAPER = "#FFFDFC"
SAND = "#E8DED2"
MIST = "#EEE9E3"
MUTED = "#746E68"
BORDER = "#DDD3C8"
WHITE = "#FFFFFF"
GOOD = "#50775A"
WARN = "#B67A2E"
BAD = "#A9463D"
FORECAST = "#6B4A57"
HISTORY = "#A49A91"


st.html(
    f"""
<style>
:root {{
    --ink: {INK};
    --ink-soft: {INK_SOFT};
    --plum: {PLUM};
    --clay: {CLAY};
    --ochre: {OCHRE};
    --olive: {OLIVE};
    --rose: {ROSE};
    --cream: {CREAM};
    --paper: {PAPER};
    --sand: {SAND};
    --mist: {MIST};
    --muted: {MUTED};
    --border: {BORDER};
}}

html,
body,
[class*="css"] {{
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 90% 0%, rgba(200,154,67,.10), transparent 23%),
        radial-gradient(circle at 10% 12%, rgba(91,42,60,.055), transparent 21%),
        linear-gradient(180deg, #FBF8F4 0%, var(--cream) 50%, #F2ECE5 100%);
}}

[data-testid="stHeader"] {{
    background: rgba(251,248,244,.84);
    backdrop-filter: blur(14px);
}}

#MainMenu,
footer {{
    visibility: hidden;
}}

.block-container {{
    max-width: 1480px;
    padding: 1.9rem 2rem 3rem;
}}

[data-testid="stSidebar"] {{
    background:
        radial-gradient(circle at 20% 0%, rgba(200,154,67,.16), transparent 22%),
        linear-gradient(180deg, #1C1917 0%, #151311 100%);
    border-right: 1px solid rgba(255,255,255,.06);
}}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1.1rem;
}}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {{
    color: #F8F3ED !important;
}}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: rgba(248,243,237,.58) !important;
}}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background: rgba(255,253,250,.97) !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 12px !important;
}}

[data-testid="stSidebar"] div[data-baseweb="select"] span {{
    color: var(--ink) !important;
}}

[data-testid="stMetric"] {{
    min-height: 118px;
    padding: 1rem 1.05rem;
    border: 1px solid rgba(221,211,200,.96);
    border-radius: 16px;
    background: rgba(255,253,250,.94);
    box-shadow: 0 14px 30px rgba(49,44,40,.05);
}}

[data-testid="stMetricLabel"] {{
    color: var(--muted);
    font-size: .66rem;
    font-weight: 850;
    letter-spacing: .085em;
    text-transform: uppercase;
}}

[data-testid="stMetricValue"] {{
    color: var(--ink);
    font-size: 1.66rem;
    font-weight: 900;
    letter-spacing: -.035em;
}}

[data-testid="stMetricDelta"] {{
    font-weight: 760;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid rgba(221,211,200,.96) !important;
    border-radius: 18px !important;
    background: rgba(255,253,250,.92) !important;
    box-shadow: 0 16px 34px rgba(49,44,40,.045);
}}

div[data-baseweb="tab-list"] {{
    gap: .28rem;
    padding: .30rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(255,253,250,.96);
    box-shadow: 0 10px 24px rgba(49,44,40,.04);
}}

button[data-baseweb="tab"] {{
    height: 43px;
    padding: 0 1rem;
    border-radius: 10px;
    color: var(--muted);
    font-weight: 800;
    font-size: .81rem;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: #FFFDF9;
    background: var(--ink);
}}

[data-testid="stSegmentedControl"] button {{
    border-color: var(--border) !important;
    color: var(--ink-soft) !important;
    font-weight: 760 !important;
}}

[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
    background: var(--ink) !important;
    color: white !important;
}}

div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stSelectSlider"] {{
    border-radius: 11px !important;
    border-color: var(--border) !important;
}}

.stDownloadButton button {{
    width: 100%;
    border-radius: 11px !important;
    border: 1px solid var(--ink) !important;
    background: var(--ink) !important;
    color: white !important;
    font-weight: 820 !important;
}}

.stDownloadButton button:hover {{
    background: var(--plum) !important;
    border-color: var(--plum) !important;
}}

/* Sidebar */
.brand {{
    display: flex;
    align-items: center;
    gap: .78rem;
    padding: .15rem 0 1rem;
}}

.brand-mark {{
    display: grid;
    place-items: center;
    width: 48px;
    height: 48px;
    border-radius: 15px;
    color: #1C1917;
    background: linear-gradient(145deg, #E4C88C, #C89A43);
    box-shadow: 0 16px 30px rgba(200,154,67,.18);
    font-size: 1.15rem;
    font-weight: 950;
}}

.brand-title {{
    color: #FFF9F2;
    font-size: 1rem;
    font-weight: 920;
    line-height: 1.08;
}}

.brand-subtitle {{
    margin-top: .22rem;
    color: rgba(255,249,242,.60);
    font-size: .71rem;
}}

.side-card {{
    margin-top: .58rem;
    padding: .92rem .95rem;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    background: rgba(255,255,255,.045);
}}

.side-label {{
    color: rgba(255,249,242,.52);
    font-size: .63rem;
    font-weight: 900;
    letter-spacing: .095em;
    text-transform: uppercase;
    margin-bottom: .35rem;
}}

.side-value {{
    color: #FFF9F2;
    font-size: .81rem;
    line-height: 1.52;
    font-weight: 720;
}}

/* Masthead */
.masthead {{
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(360px, .75fr);
    gap: 1.5rem;
    min-height: 250px;
    margin: .35rem 0 1.25rem;
    padding: 2.1rem 2.2rem;
    border-radius: 22px;
    background: var(--paper);
    border: 1px solid var(--border);
    box-shadow: 0 24px 56px rgba(49,44,40,.085);
}}

.masthead::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 8px;
    height: 100%;
    background: linear-gradient(180deg, var(--plum), var(--clay), var(--ochre));
}}

.masthead::after {{
    content: "";
    position: absolute;
    right: -110px;
    top: -120px;
    width: 330px;
    height: 330px;
    border-radius: 50%;
    border: 1px solid rgba(91,42,60,.08);
    box-shadow:
        0 0 0 38px rgba(200,154,67,.045),
        0 0 0 82px rgba(201,109,75,.028);
}}

.masthead-main,
.masthead-rail {{
    position: relative;
    z-index: 1;
}}

.eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    color: var(--plum);
    font-size: .68rem;
    font-weight: 900;
    letter-spacing: .10em;
    text-transform: uppercase;
}}

.masthead-title {{
    margin-top: .9rem;
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.8rem, 4.7vw, 4.45rem);
    line-height: .95;
    letter-spacing: -.055em;
    font-weight: 700;
}}

.masthead-copy {{
    max-width: 760px;
    margin-top: .9rem;
    color: var(--muted);
    font-size: .98rem;
    line-height: 1.65;
}}

.context-chip {{
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    margin-top: 1rem;
    padding: .64rem .82rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: #FAF5EE;
    color: var(--ink);
    font-size: .76rem;
    font-weight: 760;
}}

.score-rail {{
    height: 100%;
    padding: 1.05rem;
    border-radius: 16px;
    background: var(--ink);
    color: white;
}}

.score-item {{
    padding: .8rem .85rem;
    border-radius: 12px;
    background: rgba(255,255,255,.045);
}}

.score-item + .score-item {{
    margin-top: .65rem;
}}

.score-label {{
    color: rgba(255,255,255,.56);
    font-size: .61rem;
    font-weight: 900;
    letter-spacing: .095em;
    text-transform: uppercase;
}}

.score-number {{
    margin-top: .18rem;
    color: #FFFDF9;
    font-size: 1.55rem;
    line-height: 1.08;
    font-weight: 920;
    letter-spacing: -.035em;
}}

.score-note {{
    margin-top: .2rem;
    color: rgba(255,255,255,.63);
    font-size: .68rem;
    line-height: 1.4;
}}

/* Sections */
.section-head {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.25rem 0 .78rem;
}}

.section-copy {{
    display: flex;
    align-items: flex-start;
    gap: .72rem;
}}

.section-index {{
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
    border-radius: 50%;
    background: var(--ink);
    color: #FFFDF9;
    font-family: Georgia, "Times New Roman", serif;
    font-size: .82rem;
    font-weight: 700;
}}

.section-title {{
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.25rem;
    line-height: 1.2;
    font-weight: 700;
}}

.section-subtitle {{
    margin-top: .18rem;
    color: var(--muted);
    font-size: .8rem;
    line-height: 1.45;
}}

.stat-card {{
    position: relative;
    height: 100%;
    min-height: 124px;
    padding: 1rem 1.05rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--paper);
    box-shadow: 0 12px 26px rgba(49,44,40,.045);
    transition: transform .18s ease, box-shadow .18s ease;
}}

.stat-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 18px 34px rgba(49,44,40,.085);
}}

.stat-accent {{
    position: absolute;
    right: 14px;
    top: 14px;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    font-weight: 900;
}}

.stat-label {{
    color: var(--muted);
    font-size: .64rem;
    font-weight: 900;
    letter-spacing: .09em;
    text-transform: uppercase;
}}

.stat-value {{
    margin-top: .65rem;
    color: var(--ink);
    font-size: 1.72rem;
    line-height: 1.05;
    letter-spacing: -.04em;
    font-weight: 920;
}}

.stat-caption {{
    margin-top: .38rem;
    max-width: 85%;
    color: var(--muted);
    font-size: .72rem;
    line-height: 1.42;
}}

.insight-stack {{
    display: grid;
    gap: .72rem;
}}

.insight-card {{
    padding: .95rem 1rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--paper);
}}

.insight-kicker {{
    color: var(--muted);
    font-size: .61rem;
    font-weight: 900;
    letter-spacing: .09em;
    text-transform: uppercase;
}}

.insight-value {{
    margin-top: .18rem;
    color: var(--ink);
    font-size: 1.05rem;
    font-weight: 880;
}}

.insight-note {{
    margin-top: .18rem;
    color: var(--muted);
    font-size: .72rem;
    line-height: 1.42;
}}

.selection-note {{
    margin-top: .75rem;
    padding: .82rem .9rem;
    border-radius: 12px;
    background: #F3EEE8;
    border-left: 4px solid var(--ochre);
    color: var(--ink-soft);
    font-size: .78rem;
    line-height: 1.5;
}}

.calendar-grid {{
    display: grid;
    grid-template-columns: repeat(8, minmax(0, 1fr));
    gap: .62rem;
}}

.day-tile {{
    min-height: 112px;
    padding: .78rem;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: var(--paper);
    transition: transform .17s ease, border-color .17s ease;
}}

.day-tile:hover {{
    transform: translateY(-3px);
    border-color: rgba(91,42,60,.34);
}}

.day-date {{
    color: var(--muted);
    font-size: .63rem;
    font-weight: 850;
    letter-spacing: .07em;
    text-transform: uppercase;
}}

.day-value {{
    margin-top: .28rem;
    color: var(--ink);
    font-size: 1.05rem;
    font-weight: 900;
}}

.day-promo {{
    margin-top: .2rem;
    color: var(--muted);
    font-size: .66rem;
}}

.day-bar {{
    height: 5px;
    margin-top: .72rem;
    overflow: hidden;
    border-radius: 999px;
    background: var(--mist);
}}

.day-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--clay), var(--plum));
}}

.decision-card {{
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(260px, .8fr);
    gap: 1.1rem;
    padding: 1.25rem;
    border-radius: 18px;
    background: var(--ink);
    color: white;
}}

.decision-kicker {{
    color: rgba(255,255,255,.55);
    font-size: .62rem;
    font-weight: 900;
    letter-spacing: .1em;
    text-transform: uppercase;
}}

.decision-title {{
    margin-top: .3rem;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2rem;
    line-height: 1.05;
    font-weight: 700;
}}

.decision-copy {{
    margin-top: .45rem;
    color: rgba(255,255,255,.68);
    font-size: .78rem;
    line-height: 1.5;
}}

.risk-chip {{
    display: inline-flex;
    margin-top: .75rem;
    padding: .4rem .62rem;
    border-radius: 999px;
    font-size: .67rem;
    font-weight: 850;
    letter-spacing: .05em;
    text-transform: uppercase;
}}

.risk-critical {{ background: #FBE4E2; color: #8B2822; }}
.risk-high {{ background: #FFE8D7; color: #9A4C21; }}
.risk-moderate {{ background: #FFF1CF; color: #875B13; }}
.risk-low {{ background: #E8F1E7; color: #3F6949; }}

.decision-metrics {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .65rem;
}}

.decision-metric {{
    padding: .8rem;
    border-radius: 12px;
    background: rgba(255,255,255,.055);
}}

.decision-metric-label {{
    color: rgba(255,255,255,.48);
    font-size: .58rem;
    font-weight: 900;
    letter-spacing: .08em;
    text-transform: uppercase;
}}

.decision-metric-value {{
    margin-top: .18rem;
    color: white;
    font-size: 1rem;
    font-weight: 880;
}}

.formula-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .7rem;
}}

.formula-card {{
    position: relative;
    padding: .95rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--paper);
}}

.formula-step {{
    color: var(--plum);
    font-size: .62rem;
    font-weight: 900;
    letter-spacing: .08em;
    text-transform: uppercase;
}}

.formula-value {{
    margin-top: .28rem;
    color: var(--ink);
    font-size: 1.25rem;
    font-weight: 900;
}}

.formula-copy {{
    margin-top: .24rem;
    color: var(--muted);
    font-size: .68rem;
    line-height: 1.4;
}}

.period-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .7rem;
}}

.period-card {{
    padding: .95rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--paper);
}}

.period-label {{
    color: var(--muted);
    font-size: .62rem;
    font-weight: 900;
    letter-spacing: .08em;
    text-transform: uppercase;
}}

.period-number {{
    margin-top: .28rem;
    color: var(--ink);
    font-size: 1.3rem;
    font-weight: 900;
}}

.period-note {{
    margin-top: .22rem;
    color: var(--muted);
    font-size: .68rem;
}}

.feature-list {{
    display: grid;
    gap: .72rem;
}}

.feature-row {{
    display: grid;
    grid-template-columns: minmax(220px, .9fr) minmax(0, 1.5fr) 85px;
    align-items: center;
    gap: .8rem;
}}

.feature-name {{
    color: var(--ink);
    font-size: .78rem;
    font-weight: 760;
}}

.feature-track {{
    height: 9px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--mist);
}}

.feature-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--ochre), var(--clay), var(--plum));
}}

.feature-score {{
    color: var(--muted);
    font-size: .72rem;
    text-align: right;
}}

.step-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .75rem;
}}

.step-card {{
    min-height: 145px;
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: 15px;
    background: var(--paper);
}}

.step-number {{
    display: grid;
    place-items: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--ink);
    color: white;
    font-family: Georgia, "Times New Roman", serif;
    font-size: .73rem;
    font-weight: 700;
}}

.step-title {{
    margin-top: .72rem;
    color: var(--ink);
    font-size: .84rem;
    font-weight: 850;
}}

.step-copy {{
    margin-top: .3rem;
    color: var(--muted);
    font-size: .72rem;
    line-height: 1.48;
}}

.footer-note {{
    padding: 1.1rem 0 .2rem;
    color: var(--muted);
    text-align: center;
    font-size: .71rem;
}}

@media (max-width: 1120px) {{
    .calendar-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .formula-row, .period-grid, .step-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

@media (max-width: 900px) {{
    .block-container {{ padding: 1.45rem .95rem 2.6rem; }}
    .masthead {{ grid-template-columns: 1fr; padding: 1.45rem; border-radius: 18px; }}
    .masthead-title {{ font-size: 2.55rem; }}
    .score-rail {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .55rem; }}
    .score-item + .score-item {{ margin-top: 0; }}
    .decision-card {{ grid-template-columns: 1fr; }}
}}

@media (max-width: 640px) {{
    .calendar-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .formula-row, .period-grid, .step-grid {{ grid-template-columns: 1fr; }}
    .score-rail {{ grid-template-columns: 1fr; }}
    .feature-row {{ grid-template-columns: 1fr; gap: .35rem; }}
    .feature-score {{ text-align: left; }}
}}
</style>
"""
)


# =============================================================================
# Data and labels
# =============================================================================
@st.cache_data(show_spinner="Loading retail planning data...")
def load_dashboard_data() -> dict[str, object]:
    paths = {
        "forecast": DASHBOARD_DATA_DIRECTORY / "forecast.parquet",
        "actuals": DASHBOARD_DATA_DIRECTORY / "recent_actuals.parquet",
        "stores": DASHBOARD_DATA_DIRECTORY / "stores.csv",
        "comparison": DASHBOARD_DATA_DIRECTORY / "model_comparison.csv",
        "folds": DASHBOARD_DATA_DIRECTORY / "fold_metrics.csv",
        "importance": DASHBOARD_DATA_DIRECTORY / "feature_importance.csv",
        "summary": DASHBOARD_DATA_DIRECTORY / "model_summary.csv",
        "manifest": DASHBOARD_DATA_DIRECTORY / "dashboard_manifest.json",
    }

    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Dashboard data is incomplete. Run "
            "`python scripts/prepare_dashboard_data.py`.\n\n"
            + "\n".join(missing_files)
        )

    forecast_data = pd.read_parquet(paths["forecast"])
    actual_data = pd.read_parquet(paths["actuals"])
    forecast_data["date"] = pd.to_datetime(forecast_data["date"])
    actual_data["date"] = pd.to_datetime(actual_data["date"])

    return {
        "forecast": forecast_data,
        "actuals": actual_data,
        "stores": pd.read_csv(paths["stores"]),
        "comparison": pd.read_csv(paths["comparison"]),
        "folds": pd.read_csv(paths["folds"]),
        "importance": pd.read_csv(paths["importance"]),
        "summary": pd.read_csv(paths["summary"]),
        "manifest": json.loads(paths["manifest"].read_text(encoding="utf-8")),
    }


MODEL_LABELS = {
    "xgboost_log_target_nested": "Current forecasting model",
    "shifted_28_day_mean": "Recent 28-day average",
    "weekly_seasonal_naive": "Repeat last week's pattern",
    "lag_16": "Use sales from 16 days ago",
    "lag_364": "Use sales from one year ago",
    "zero": "Predict no sales",
}

FEATURE_LABELS = {
    "sales_rolling_mean_7_shift_16": "Recent 7-day demand level",
    "sales_lag_21": "Sales 21 days earlier",
    "sales_rolling_mean_28_shift_16": "Recent 28-day demand level",
    "sales_lag_28": "Sales 28 days earlier",
    "is_year_start": "Start-of-year pattern",
    "sales_lag_35": "Sales 35 days earlier",
    "onpromotion": "Planned promotion",
    "day_of_year": "Time of year",
    "sales_rolling_std_28_shift_16": "Recent demand variation",
    "is_weekend": "Weekend",
    "sales_lag_16": "Sales 16 days earlier",
    "day_of_week": "Day of week",
    "sales_lag_364": "Sales around one year earlier",
    "family": "Product family",
    "is_national_holiday": "National holiday",
}


def compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.1f}"


def section(number: str, title: str, subtitle: str) -> None:
    st.html(
        f'<div class="section-head">'
        f'<div class="section-copy">'
        f'<div class="section-index">{html.escape(number)}</div>'
        f'<div>'
        f'<div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def render_stat_card(
    label: str,
    value: str,
    caption: str,
    symbol: str,
    accent: str,
    accent_background: str,
) -> None:
    st.html(
        f'<div class="stat-card">'
        f'<div class="stat-accent" style="color:{accent};background:{accent_background};">'
        f'{html.escape(symbol)}'
        f'</div>'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        f'<div class="stat-caption">{html.escape(caption)}</div>'
        f'</div>'
    )


def style_chart(
    figure: go.Figure,
    *,
    height: int,
    show_legend: bool = False,
) -> None:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            color=INK_SOFT,
            size=12,
        ),
        margin=dict(l=18, r=18, t=22, b=18),
        hoverlabel=dict(
            bgcolor=INK,
            bordercolor=INK,
            font=dict(color="white"),
        ),
        showlegend=show_legend,
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        automargin=True,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="rgba(221,211,200,.72)",
        zeroline=False,
        tickfont=dict(color=MUTED),
        automargin=True,
    )


def safe_percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0


try:
    dashboard_data = load_dashboard_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

forecast_data = dashboard_data["forecast"]
actual_data = dashboard_data["actuals"]
store_data = dashboard_data["stores"]
model_comparison = dashboard_data["comparison"]
fold_metrics = dashboard_data["folds"]
feature_importance = dashboard_data["importance"]
model_summary = dashboard_data["summary"].iloc[0]
manifest = dashboard_data["manifest"]

if "city" not in forecast_data.columns:
    forecast_data = forecast_data.merge(
        store_data,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )

store_labels = {
    int(row.store_nbr): f"Store {int(row.store_nbr)} · {row.city} · Type {row.type}"
    for row in store_data.itertuples()
}
store_options = sorted(store_labels)
family_options = sorted(forecast_data["family"].astype(str).unique().tolist())


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.html(
        """
        <div class="brand">
            <div class="brand-mark">R</div>
            <div>
                <div class="brand-title">Retail Atelier</div>
                <div class="brand-subtitle">Demand and inventory intelligence</div>
            </div>
        </div>
        """
    )

    st.markdown("#### Store-level planning focus")

    st.caption(
        "Store and product filters update **Store story** and "
        "**Stock decision**. **Network pulse** always shows "
        "the complete retail network."
    )
    selected_store = st.selectbox(
        "Store",
        store_options,
        format_func=lambda value: store_labels[int(value)],
        key="selected_store_widget",
    )
    selected_family = st.selectbox(
        "Product family",
        family_options,
        index=(
            family_options.index("GROCERY I")
            if "GROCERY I" in family_options
            else 0
        ),
        key="selected_family_widget",
    )

    selected_store_row = store_data.loc[
        store_data["store_nbr"].eq(selected_store)
    ].iloc[0]

    st.html(
        f'<div class="side-card">'
        f'<div class="side-label">Selected view</div>'
        f'<div class="side-value">'
        f'{html.escape(str(selected_store_row.city))}, '
        f'{html.escape(str(selected_store_row.state))}<br>'
        f'Type {html.escape(str(selected_store_row.type))} · '
        f'Group {int(selected_store_row.cluster)}<br>'
        f'{html.escape(selected_family)}'
        f'</div>'
        f'</div>'
    )

    st.html(
        f'<div class="side-card">'
        f'<div class="side-label">Forecast window</div>'
        f'<div class="side-value">'
        f'{manifest["forecast_date_start"]} → {manifest["forecast_date_end"]}<br>'
        f'{manifest["forecast_horizon_days"]} days ahead'
        f'</div>'
        f'</div>'
    )

    st.html(
        """
        <div class="side-card">
            <div class="side-label">Recommended flow</div>
            <div class="side-value">
                01 · Read the network pulse<br>
                02 · Inspect one store and product<br>
                03 · Enter current stock<br>
                04 · Review the suggested action
            </div>
        </div>
        """
    )

    st.caption(
        "Planning support only. Human review is required before placing an order."
    )


selected_forecast = forecast_data.loc[
    forecast_data["store_nbr"].eq(selected_store)
    & forecast_data["family"].eq(selected_family)
].sort_values("date")
selected_actuals = actual_data.loc[
    actual_data["store_nbr"].eq(selected_store)
    & actual_data["family"].eq(selected_family)
].sort_values("date")

if selected_forecast.empty:
    st.error("No forecast data was found for the selected store and product family.")
    st.stop()

selected_total = float(selected_forecast["predicted_sales"].sum())
selected_average = float(selected_forecast["predicted_sales"].mean())
selected_peak = selected_forecast.loc[
    selected_forecast["predicted_sales"].idxmax()
]
selected_low = selected_forecast.loc[
    selected_forecast["predicted_sales"].idxmin()
]

best_baseline_label = str(model_summary["best_wape_baseline"])
friendly_best_baseline = MODEL_LABELS.get(best_baseline_label, best_baseline_label)


# =============================================================================
# Editorial masthead
# =============================================================================
st.html(
    f"""
<div class="masthead">
    <div class="masthead-main">
        <div class="eyebrow">Retail planning intelligence</div>
        <div class="masthead-title">Demand, beautifully understood.</div>
        <div class="masthead-copy">
            A focused workspace for understanding the next 16 days of demand,
            spotting what matters, and turning the forecast into a practical stock decision.
        </div>
        <div class="context-chip">
            {html.escape(store_labels[selected_store])} · {html.escape(selected_family)}
        </div>
    </div>

    <div class="masthead-rail">
        <div class="score-rail">
            <div class="score-item">
                <div class="score-label">Combined historical demand error</div>
                <div class="score-number">{float(model_summary['pooled_wape_percentage']):.2f}%</div>
                <div class="score-note">Pooled WAPE across four past 16-day test periods.</div>
            </div>
            <div class="score-item">
                <div class="score-label">Improvement over simple planning</div>
                <div class="score-number">{float(model_summary['wape_improvement_vs_best_baseline_percentage']):.1f}%</div>
                <div class="score-note">Compared with {html.escape(friendly_best_baseline)}.</div>
            </div>
            <div class="score-item">
                <div class="score-label">Technical review score</div>
                <div class="score-number">{float(model_summary['pooled_rmsle']):.4f}</div>
                <div class="score-note">RMSLE retained for technical reviewers.</div>
            </div>
        </div>
    </div>
</div>
"""
)


overview_tab, demand_tab, planner_tab, trust_tab = st.tabs(
    ["Network pulse", "Store story", "Stock decision", "Trust notes"],
    width="stretch",
    key="main_tabs",
    on_change="rerun",
)


# =============================================================================
# Network pulse
# =============================================================================
if overview_tab.open:
    with overview_tab:
        st.caption(
            f"Network scope · All {manifest['forecast_store_count']} stores "
            f"and {manifest['forecast_family_count']} product families. "
            "Sidebar filters affect Store story and Stock decision only."
        )

        daily_network = (
            forecast_data.groupby("date", as_index=False)
            .agg(
                total_predicted_sales=("predicted_sales", "sum"),
                promotion_records=("onpromotion", lambda values: int(values.gt(0).sum())),
            )
            .sort_values("date")
        )
        network_total = float(forecast_data["predicted_sales"].sum())
        network_daily_average = float(daily_network["total_predicted_sales"].mean())
        network_peak = daily_network.loc[
            daily_network["total_predicted_sales"].idxmax()
        ]
        promoted_rows = int(forecast_data["onpromotion"].gt(0).sum())
        peak_above_average = safe_percentage(
            float(network_peak["total_predicted_sales"]) - network_daily_average,
            network_daily_average,
        )

        kpis = st.columns(4, gap="medium")
        with kpis[0]:
            render_stat_card(
                "16-day network demand",
                compact_number(network_total),
                "Expected demand across all stores and product families.",
                "01",
                PLUM,
                "#F0E4E9",
            )
        with kpis[1]:
            render_stat_card(
                "Typical day",
                compact_number(network_daily_average),
                "Average expected demand for one forecast day.",
                "02",
                OCHRE,
                "#F6EBD4",
            )
        with kpis[2]:
            render_stat_card(
                "Peak day",
                pd.Timestamp(network_peak["date"]).strftime("%d %b"),
                f"{peak_above_average:.1f}% above a typical day.",
                "03",
                CLAY,
                "#F4E1D9",
            )
        with kpis[3]:
            render_stat_card(
                "Promotion activity",
                f"{promoted_rows:,}",
                "Store-product-day records linked to planned promotions.",
                "04",
                OLIVE,
                "#E8ECE3",
            )

        main_columns = st.columns([1.55, .45], gap="large")
        with main_columns[0]:
            section("01", "The 16-day pulse", "One clear view of the expected network rhythm.")
            with st.container(border=True):
                pulse_figure = go.Figure()
                pulse_figure.add_trace(
                    go.Scatter(
                        x=daily_network["date"],
                        y=daily_network["total_predicted_sales"],
                        mode="lines+markers",
                        line=dict(color=PLUM, width=3.2, shape="linear"),
                        marker=dict(
                            size=7,
                            color=PAPER,
                            line=dict(color=CLAY, width=2),
                        ),
                        fill="tozeroy",
                        fillcolor="rgba(91,42,60,.07)",
                        hovertemplate=(
                            "<b>%{x|%d %b %Y}</b><br>"
                            "Expected demand: %{y:,.0f}<extra></extra>"
                        ),
                    )
                )
                pulse_figure.add_hline(
                    y=network_daily_average,
                    line_dash="dot",
                    line_color=OCHRE,
                    annotation_text="Typical day",
                    annotation_position="top left",
                )
                pulse_figure.add_annotation(
                    x=network_peak["date"],
                    y=network_peak["total_predicted_sales"],
                    text="Peak",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=CLAY,
                    bgcolor=PAPER,
                    bordercolor=BORDER,
                    borderpad=5,
                )
                pulse_figure.update_xaxes(title=None, tickformat="%d %b")
                pulse_figure.update_yaxes(title="Expected units", tickformat="~s")
                style_chart(pulse_figure, height=420)
                st.plotly_chart(
                    pulse_figure,
                    width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False},
                )

        with main_columns[1]:
            section("02", "What matters now", "A short briefing instead of another chart.")
            top_family = (
                forecast_data.groupby("family")["predicted_sales"].sum().idxmax()
            )
            top_store_number = int(
                forecast_data.groupby("store_nbr")["predicted_sales"].sum().idxmax()
            )
            selected_share = safe_percentage(selected_total, network_total)
            st.html(
                f'<div class="insight-stack">'
                f'<div class="insight-card">'
                f'<div class="insight-kicker">Largest product family</div>'
                f'<div class="insight-value">{html.escape(str(top_family))}</div>'
                f'<div class="insight-note">The single largest contributor to network demand.</div>'
                f'</div>'
                f'<div class="insight-card">'
                f'<div class="insight-kicker">Busiest store</div>'
                f'<div class="insight-value">{html.escape(store_labels[top_store_number])}</div>'
                f'<div class="insight-note">Highest expected total across the forecast window.</div>'
                f'</div>'
                f'<div class="insight-card">'
                f'<div class="insight-kicker">Current selection</div>'
                f'<div class="insight-value">{compact_number(selected_total)}</div>'
                f'<div class="insight-note">{selected_share:.2f}% of the full network forecast.</div>'
                f'</div>'
                f'</div>'
            )

        section(
            "03",
            "Explore the network",
            "Switch the question instead of stacking several similar charts.",
        )
        explore_mode = st.segmented_control(
            "Choose a business question",
            options=["Product mix", "Store ranking", "Promotion pulse"],
            default="Product mix",
            required=True,
            width="stretch",
            label_visibility="collapsed",
            key="overview_explore_mode",
        )
        top_n = 8
        if explore_mode in {"Product mix", "Store ranking"}:
            top_n = st.slider(
                "How many results should be shown?",
                min_value=5,
                max_value=15,
                value=8,
                step=1,
                key="overview_top_n",
            )
        else:
            st.caption(
                "All 16 forecast dates are shown so the promotion-demand pattern is not selectively filtered."
            )

        with st.container(border=True):
            if explore_mode == "Product mix":
                product_summary = (
                    forecast_data.groupby("family", as_index=False)
                    .agg(expected_sales=("predicted_sales", "sum"))
                    .sort_values("expected_sales", ascending=False)
                    .head(top_n)
                    .sort_values("expected_sales")
                )
                product_summary["share"] = product_summary["expected_sales"] / network_total * 100
                product_figure = go.Figure(
                    go.Bar(
                        x=product_summary["expected_sales"],
                        y=product_summary["family"],
                        orientation="h",
                        marker=dict(
                            color=product_summary["expected_sales"],
                            colorscale=[
                                [0.0, "#D9C9BF"],
                                [0.45, "#C89A43"],
                                [0.75, "#C96D4B"],
                                [1.0, "#5B2A3C"],
                            ],
                            showscale=False,
                        ),
                        customdata=np.column_stack([product_summary["share"]]),
                        text=[compact_number(value) for value in product_summary["expected_sales"]],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Expected demand: %{x:,.0f}<br>"
                            "Network share: %{customdata[0]:.1f}%<extra></extra>"
                        ),
                    )
                )
                product_figure.update_xaxes(title="Expected units", tickformat="~s")
                product_figure.update_yaxes(title=None, showgrid=False)
                style_chart(product_figure, height=430)
                event = st.plotly_chart(
                    product_figure,
                    key="product_mix_chart",
                    width="stretch",
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False},
                )
                selected_points = event.selection.points if event else []
                if selected_points:
                    point = selected_points[0]
                    selected_name = point.get("y", "Selected product family")
                    st.html(
                        f'<div class="selection-note">'
                        f'<b>{html.escape(str(selected_name))}</b> selected. '
                        f'Use the sidebar to open this product family in the store-level view.'
                        f'</div>'
                    )
                else:
                    st.caption("Click a bar to highlight a product family.")

            elif explore_mode == "Store ranking":
                store_summary = (
                    forecast_data.groupby("store_nbr", as_index=False)
                    .agg(expected_sales=("predicted_sales", "sum"))
                    .merge(store_data, on="store_nbr", how="left", validate="one_to_one")
                    .sort_values("expected_sales", ascending=False)
                    .head(top_n)
                    .sort_values("expected_sales")
                )
                store_summary["label"] = store_summary.apply(
                    lambda row: f"Store {int(row['store_nbr'])} · {row['city']}",
                    axis=1,
                )
                store_figure = go.Figure()
                for row in store_summary.itertuples():
                    store_figure.add_shape(
                        type="line",
                        x0=0,
                        x1=row.expected_sales,
                        y0=row.label,
                        y1=row.label,
                        line=dict(color="#D9D0C8", width=3),
                    )
                store_figure.add_trace(
                    go.Scatter(
                        x=store_summary["expected_sales"],
                        y=store_summary["label"],
                        mode="markers+text",
                        marker=dict(
                            size=15,
                            color=store_summary["expected_sales"],
                            colorscale=[
                                [0.0, "#C89A43"],
                                [0.6, "#C96D4B"],
                                [1.0, "#5B2A3C"],
                            ],
                            showscale=False,
                            line=dict(color=PAPER, width=2),
                        ),
                        text=[compact_number(value) for value in store_summary["expected_sales"]],
                        textposition="middle right",
                        customdata=store_summary[["store_nbr", "city", "type"]].to_numpy(),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Expected demand: %{x:,.0f}<br>"
                            "Store type: %{customdata[2]}<extra></extra>"
                        ),
                    )
                )
                store_figure.update_xaxes(title="Expected units", tickformat="~s")
                store_figure.update_yaxes(title=None, showgrid=False)
                style_chart(store_figure, height=430)
                event = st.plotly_chart(
                    store_figure,
                    key="store_ranking_chart",
                    width="stretch",
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False},
                )
                selected_points = event.selection.points if event else []
                if selected_points:
                    point = selected_points[0]
                    selected_label = point.get("y", "Selected store")
                    st.html(
                        f'<div class="selection-note">'
                        f'<b>{html.escape(str(selected_label))}</b> selected. '
                        f'Use the sidebar to inspect its product-level forecast.'
                        f'</div>'
                    )
                else:
                    st.caption("Click a point to highlight a store.")

            else:
                promo_data = daily_network.copy()
                promo_data["date_label"] = promo_data["date"].dt.strftime("%d %b")
                promotion_figure = go.Figure(
                    go.Scatter(
                        x=promo_data["promotion_records"],
                        y=promo_data["total_predicted_sales"],
                        mode="markers+text",
                        text=promo_data["date_label"],
                        textposition="top center",
                        marker=dict(
                            size=16,
                            color=CLAY,
                            opacity=0.86,
                            line=dict(color=PAPER, width=2),
                        ),
                        customdata=promo_data[["date_label"]].to_numpy(),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Promotion records: %{x:,.0f}<br>"
                            "Expected demand: %{y:,.0f}<extra></extra>"
                        ),
                    )
                )
                promotion_figure.update_xaxes(title="Promotion records")
                promotion_figure.update_yaxes(title="Expected units", tickformat="~s")
                style_chart(promotion_figure, height=430)
                st.plotly_chart(
                    promotion_figure,
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.caption(
                    "Each point is one forecast date. Move over a point to compare promotion activity and expected demand."
                )


# =============================================================================
# Store story
# =============================================================================
if demand_tab.open:
    with demand_tab:
        section(
            "01",
            "The store story",
            f"{store_labels[selected_store]} · {selected_family}",
        )

        date_options = selected_forecast["date"].tolist()
        selected_date = st.select_slider(
            "Move through the forecast one day at a time",
            options=date_options,
            value=date_options[0],
            format_func=lambda value: pd.Timestamp(value).strftime("%d %b"),
            key="forecast_date_slider",
        )
        selected_day = selected_forecast.loc[
            selected_forecast["date"].eq(selected_date)
        ].iloc[0]
        difference_from_average = float(selected_day["predicted_sales"]) - selected_average
        difference_percentage = safe_percentage(difference_from_average, selected_average)
        day_position = int(
            selected_forecast.reset_index(drop=True).index[
                selected_forecast.reset_index(drop=True)["date"].eq(selected_date)
            ][0]
        ) + 1

        day_columns = st.columns(4, gap="medium")
        with day_columns[0]:
            render_stat_card(
                "Selected day",
                pd.Timestamp(selected_date).strftime("%d %b"),
                f"Day {day_position} of the 16-day forecast.",
                "A",
                PLUM,
                "#F0E4E9",
            )
        with day_columns[1]:
            render_stat_card(
                "Expected demand",
                compact_number(float(selected_day["predicted_sales"])),
                f"{difference_percentage:+.1f}% versus the selected series average.",
                "B",
                CLAY,
                "#F4E1D9",
            )
        with day_columns[2]:
            render_stat_card(
                "Items on promotion",
                f"{int(selected_day['onpromotion']):,}",
                "Items in this family planned for promotion.",
                "C",
                OCHRE,
                "#F6EBD4",
            )
        with day_columns[3]:
            render_stat_card(
                "16-day total",
                compact_number(selected_total),
                f"Daily average: {compact_number(selected_average)}.",
                "D",
                OLIVE,
                "#E8ECE3",
            )

        section(
            "02",
            "Past and future in one view",
            "The future is shaded so the transition is immediately visible.",
        )
        with st.container(border=True):
            story_figure = go.Figure()
            story_figure.add_trace(
                go.Scatter(
                    x=selected_actuals["date"],
                    y=selected_actuals["sales"],
                    mode="lines",
                    name="Recorded sales",
                    line=dict(color=HISTORY, width=1.8),
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        "Recorded sales: %{y:,.2f}<extra></extra>"
                    ),
                )
            )
            story_figure.add_trace(
                go.Scatter(
                    x=selected_forecast["date"],
                    y=selected_forecast["predicted_sales"],
                    mode="lines+markers",
                    name="Expected demand",
                    line=dict(color=FORECAST, width=3),
                    marker=dict(
                        size=7,
                        color=PAPER,
                        line=dict(color=CLAY, width=2),
                    ),
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        "Expected demand: %{y:,.2f}<extra></extra>"
                    ),
                )
            )
            forecast_start = selected_forecast["date"].min()
            forecast_end = selected_forecast["date"].max()
            story_figure.add_vrect(
                x0=forecast_start,
                x1=forecast_end,
                fillcolor="rgba(200,154,67,.08)",
                line_width=0,
                layer="below",
            )
            story_figure.add_vline(
                x=forecast_start,
                line_dash="dot",
                line_color=OCHRE,
            )
            story_figure.add_vline(
                x=selected_date,
                line_width=2,
                line_color=CLAY,
            )
            story_figure.add_annotation(
                x=selected_date,
                y=float(selected_day["predicted_sales"]),
                text=pd.Timestamp(selected_date).strftime("%d %b"),
                showarrow=True,
                arrowhead=2,
                arrowcolor=CLAY,
                bgcolor=PAPER,
                bordercolor=BORDER,
                borderpad=5,
            )
            story_figure.update_layout(
                legend=dict(orientation="h", y=1.04, x=0)
            )
            story_figure.update_xaxes(title=None, tickformat="%d %b")
            story_figure.update_yaxes(title="Sales units", tickformat="~s")
            style_chart(story_figure, height=455, show_legend=True)
            st.plotly_chart(
                story_figure,
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
            )

        section(
            "03",
            "A 16-day calendar",
            "Scan the entire forecast without decoding another chart.",
        )
        maximum_daily_value = float(selected_forecast["predicted_sales"].max())
        tiles: list[str] = []
        for row in selected_forecast.itertuples():
            fill_width = (
                0.0
                if maximum_daily_value == 0
                else float(row.predicted_sales) / maximum_daily_value * 100.0
            )
            tiles.append(
                f'<div class="day-tile">'
                f'<div class="day-date">{pd.Timestamp(row.date).strftime("%a · %d %b")}</div>'
                f'<div class="day-value">{compact_number(float(row.predicted_sales))}</div>'
                f'<div class="day-promo">{int(row.onpromotion):,} promotion items</div>'
                f'<div class="day-bar"><div class="day-fill" style="width:{fill_width:.1f}%;"></div></div>'
                f'</div>'
            )
        st.html(f'<div class="calendar-grid">{"".join(tiles)}</div>')
        st.caption(
            "The short bar shows each day’s expected demand relative to "
            "the highest-demand day in this 16-day forecast."
        )

        show_table = st.toggle(
            "Show the exact daily values",
            value=False,
            key="show_store_table",
        )
        if show_table:
            display_table = selected_forecast[
                ["date", "predicted_sales", "onpromotion"]
            ].copy()
            display_table["date"] = display_table["date"].dt.strftime("%Y-%m-%d")
            display_table.columns = [
                "Date",
                "Expected sales",
                "Products on promotion",
            ]
            table_columns = st.columns([1.2, .8], gap="large")
            with table_columns[0]:
                st.dataframe(
                    display_table,
                    width="stretch",
                    height=430,
                    hide_index=True,
                    column_config={
                        "Expected sales": st.column_config.NumberColumn(format="%.2f"),
                        "Products on promotion": st.column_config.NumberColumn(format="%d"),
                    },
                )
            with table_columns[1]:
                st.download_button(
                    "Download this forecast",
                    data=display_table.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"store_{selected_store}_"
                        f"{selected_family.lower().replace(' ', '_').replace('/', '_')}"
                        "_forecast.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                )
                st.html(
                    f'<div class="selection-note">'
                    f'<b>Highest day:</b> {pd.Timestamp(selected_peak["date"]).strftime("%d %b")} · '
                    f'{float(selected_peak["predicted_sales"]):,.1f} units<br>'
                    f'<b>Lowest day:</b> {pd.Timestamp(selected_low["date"]).strftime("%d %b")} · '
                    f'{float(selected_low["predicted_sales"]):,.1f} units'
                    f'</div>'
                )


# =============================================================================
# Stock decision
# =============================================================================
if planner_tab.open:
    with planner_tab:
        section(
            "01",
            "Build a stock scenario",
            "Use your real stock assumptions; the recommendation updates immediately.",
        )

        with st.expander("Edit stock assumptions", expanded=True):
            input_columns = st.columns(3, gap="large")
            with input_columns[0]:
                st.markdown("##### Stock position")
                current_inventory = st.number_input(
                    "Stock available now",
                    min_value=0.0,
                    value=max(round(selected_average * 4, 0), 0.0),
                    step=1.0,
                    help="Units currently available to sell.",
                )
                inbound_inventory = st.number_input(
                    "Confirmed stock already on the way",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                )
            with input_columns[1]:
                st.markdown("##### Timing")
                lead_time_days = st.number_input(
                    "Days until new stock arrives",
                    min_value=1,
                    max_value=15,
                    value=3,
                    step=1,
                )
                maximum_review_days = max(1, 16 - int(lead_time_days))
                review_period_days = st.number_input(
                    "Days until the next stock review",
                    min_value=1,
                    max_value=maximum_review_days,
                    value=min(7, maximum_review_days),
                    step=1,
                )
                safety_stock_days = st.number_input(
                    "Extra buffer days",
                    min_value=0,
                    max_value=15,
                    value=2,
                    step=1,
                )
            with input_columns[2]:
                st.markdown("##### Ordering rules")
                case_pack_size = st.number_input(
                    "Units in one carton",
                    min_value=1,
                    value=6,
                    step=1,
                )
                minimum_order_quantity = st.number_input(
                    "Smallest order allowed",
                    min_value=0,
                    value=0,
                    step=1,
                )
                st.caption(
                    f"The plan covers {int(lead_time_days) + int(review_period_days)} days."
                )

        try:
            replenishment_plan = calculate_replenishment_plan(
                selected_forecast[["date", "predicted_sales"]],
                ReplenishmentInputs(
                    current_inventory=float(current_inventory),
                    inbound_inventory=float(inbound_inventory),
                    lead_time_days=int(lead_time_days),
                    safety_stock_days=int(safety_stock_days),
                    review_period_days=int(review_period_days),
                    case_pack_size=int(case_pack_size),
                    minimum_order_quantity=int(minimum_order_quantity),
                ),
            )
        except ValueError as error:
            st.error(str(error))
        else:
            risk_name = replenishment_plan.stockout_risk_band
            risk_class = risk_name.lower()
            risk_messages = {
                "Critical": "Current stock may run out before the next delivery can arrive.",
                "High": "Stock may reach delivery day, but the safety margin is too small.",
                "Moderate": "No immediate shortage is expected, but stock is below the preferred level.",
                "Low": "Current and incoming stock should comfortably cover the planned period.",
            }
            order_title = (
                f"Order {replenishment_plan.suggested_order_quantity:,} units"
                if replenishment_plan.reorder_now
                else "No immediate order required"
            )

            section(
                "02",
                "The decision",
                "The recommendation is shown first; the calculation follows below.",
            )
            st.html(
                f'<div class="decision-card">'
                f'<div>'
                f'<div class="decision-kicker">Recommended action</div>'
                f'<div class="decision-title">{html.escape(order_title)}</div>'
                f'<div class="decision-copy">{html.escape(risk_messages[risk_name])}</div>'
                f'<div class="risk-chip risk-{risk_class}">{html.escape(risk_name)} stock risk</div>'
                f'</div>'
                f'<div class="decision-metrics">'
                f'<div class="decision-metric">'
                f'<div class="decision-metric-label">Inventory position</div>'
                f'<div class="decision-metric-value">{replenishment_plan.inventory_position:,.1f}</div>'
                f'</div>'
                f'<div class="decision-metric">'
                f'<div class="decision-metric-label">Order trigger</div>'
                f'<div class="decision-metric-value">{replenishment_plan.reorder_point:,.1f}</div>'
                f'</div>'
                f'<div class="decision-metric">'
                f'<div class="decision-metric-label">Preferred stock</div>'
                f'<div class="decision-metric-value">{replenishment_plan.target_inventory_level:,.1f}</div>'
                f'</div>'
                f'<div class="decision-metric">'
                f'<div class="decision-metric-label">Approx. cover at average demand</div>'
                f'<div class="decision-metric-value">'
                f'{"N/A" if replenishment_plan.days_of_cover is None else f"{replenishment_plan.days_of_cover:.1f} days"}'
                f'</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )

            section(
                "03",
                "Stock runway",
                "Compare what happens with and without the suggested order.",
            )
            show_order_scenario = st.toggle(
                "Show the suggested-order scenario",
                value=True,
                key="show_order_scenario",
            )

            runway_dates = selected_forecast["date"].tolist()
            daily_demand = selected_forecast["predicted_sales"].astype(float).tolist()
            inventory_without_order: list[float] = []
            inventory_with_order: list[float] = []
            without_balance = float(current_inventory)
            with_balance = float(current_inventory)

            # A lead time of N days means the business must first cover N complete
            # forecast days. Delivery is therefore added before demand on day N+1.
            delivery_index = min(int(lead_time_days), len(runway_dates) - 1)

            for index, demand_value in enumerate(daily_demand):
                if index == delivery_index:
                    without_balance += float(inbound_inventory)
                    with_balance += float(inbound_inventory)
                    if replenishment_plan.reorder_now:
                        with_balance += float(replenishment_plan.suggested_order_quantity)

                without_balance -= demand_value
                with_balance -= demand_value
                inventory_without_order.append(without_balance)
                inventory_with_order.append(with_balance)

            runway_figure = go.Figure()
            runway_figure.add_trace(
                go.Scatter(
                    x=runway_dates,
                    y=inventory_without_order,
                    mode="lines+markers",
                    name="Without a new order",
                    line=dict(color=BAD, width=2.5),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>%{x|%d %b}</b><br>"
                        "Projected balance: %{y:,.1f}<extra></extra>"
                    ),
                )
            )
            if show_order_scenario:
                runway_figure.add_trace(
                    go.Scatter(
                        x=runway_dates,
                        y=inventory_with_order,
                        mode="lines+markers",
                        name="After the suggested order",
                        line=dict(color=GOOD, width=3),
                        marker=dict(size=6),
                        fill="tonexty",
                        fillcolor="rgba(80,119,90,.07)",
                        hovertemplate=(
                            "<b>%{x|%d %b}</b><br>"
                            "Projected balance: %{y:,.1f}<extra></extra>"
                        ),
                    )
                )
            runway_figure.add_hline(
                y=0,
                line_dash="dash",
                line_color=CLAY,
                annotation_text="Shortage begins below zero",
                annotation_position="bottom left",
            )
            runway_figure.add_vline(
                x=runway_dates[delivery_index],
                line_dash="dot",
                line_color=OCHRE,
                annotation_text="Delivery arrives before today’s demand",
                annotation_position="top right",
            )
            runway_figure.update_layout(
                legend=dict(orientation="h", y=1.04, x=0)
            )
            runway_figure.update_xaxes(title=None, tickformat="%d %b")
            runway_figure.update_yaxes(title="Projected end-of-day inventory balance", tickformat="~s")
            style_chart(runway_figure, height=430, show_legend=True)
            with st.container(border=True):
                st.plotly_chart(
                    runway_figure,
                    width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False},
                )
                st.caption(
                    "Each point shows the projected balance after that day’s demand. "
                    "Balances below zero represent unmet demand, not physical negative stock. "
                    f"The suggested order is designed for the current {int(lead_time_days) + int(review_period_days)}-day planning cycle, "
                    "not necessarily the entire 16-day display period."
                )

            section(
                "04",
                "How the order is calculated",
                "Four plain-language steps replace the previous waterfall chart.",
            )
            st.html(
                f'<div class="formula-row">'
                f'<div class="formula-card">'
                f'<div class="formula-step">Step 1 · Target</div>'
                f'<div class="formula-value">{replenishment_plan.target_inventory_level:,.0f}</div>'
                f'<div class="formula-copy">Preferred stock after covering demand and the safety buffer.</div>'
                f'</div>'
                f'<div class="formula-card">'
                f'<div class="formula-step">Step 2 · Inventory position</div>'
                f'<div class="formula-value">− {replenishment_plan.inventory_position:,.0f}</div>'
                f'<div class="formula-copy">Current stock plus confirmed stock already on the way.</div>'
                f'</div>'
                f'<div class="formula-card">'
                f'<div class="formula-step">Step 3 · Initial gap</div>'
                f'<div class="formula-value">{replenishment_plan.raw_order_quantity:,.1f}</div>'
                f'<div class="formula-copy">The amount needed before carton and minimum-order rules.</div>'
                f'</div>'
                f'<div class="formula-card">'
                f'<div class="formula-step">Step 4 · Final order</div>'
                f'<div class="formula-value">{replenishment_plan.suggested_order_quantity:,.0f}</div>'
                f'<div class="formula-copy">Final quantity after carton-size and minimum-order rules. Raw gap: {replenishment_plan.raw_order_quantity:,.1f} units.</div>'
                f'</div>'
                f'</div>'
            )


# =============================================================================
# Trust notes
# =============================================================================
if trust_tab.open:
    with trust_tab:
        section(
            "01",
            "What the results mean",
            "Business language first; technical details remain available below.",
        )

        consistency_value = float(model_summary["standard_deviation_fold_rmsle"])
        minimum_fold_wape = float(fold_metrics["wape_percentage"].min())
        maximum_fold_wape = float(fold_metrics["wape_percentage"].max())
        trust_kpis = st.columns(4, gap="medium")
        with trust_kpis[0]:
            render_stat_card(
                "Past test periods",
                str(int(model_summary["fold_count"])),
                "Separate periods used to check the forecast.",
                "01",
                PLUM,
                "#F0E4E9",
            )
        with trust_kpis[1]:
            render_stat_card(
                "Days predicted each time",
                str(int(model_summary["forecast_horizon_days"])),
                "Each test matched the real 16-day forecast horizon.",
                "02",
                OCHRE,
                "#F6EBD4",
            )
        with trust_kpis[2]:
            render_stat_card(
                "Combined demand error",
                f"{float(model_summary['pooled_wape_percentage']):.2f}%",
                "Pooled WAPE across all four historical test periods; lower is better.",
                "03",
                CLAY,
                "#F4E1D9",
            )
        with trust_kpis[3]:
            render_stat_card(
                "WAPE range across tests",
                f"{minimum_fold_wape:.2f}%–{maximum_fold_wape:.2f}%",
                f"RMSLE standard deviation across four tests: {consistency_value:.4f}.",
                "04",
                OLIVE,
                "#E8ECE3",
            )

        section(
            "02",
            "Compare methods",
            "Switch between the business error and the technical score.",
        )
        comparison_metric = st.segmented_control(
            "Choose the comparison metric",
            options=["Business error", "Technical score"],
            default="Business error",
            required=True,
            width="stretch",
            label_visibility="collapsed",
            key="comparison_metric",
        )

        benchmark_data = model_comparison.loc[
            model_comparison["model"].ne("zero")
        ].copy()
        benchmark_data["label"] = benchmark_data["model"].map(MODEL_LABELS)
        if comparison_metric == "Business error":
            metric_column = "pooled_wape_percentage"
            metric_title = "Pooled WAPE (%) · lower is better"
            suffix = "%"
        else:
            metric_column = "pooled_rmsle"
            metric_title = "RMSLE · lower is better"
            suffix = ""
        benchmark_data = benchmark_data.sort_values(metric_column)

        benchmark_figure = go.Figure(
            go.Bar(
                x=benchmark_data[metric_column],
                y=benchmark_data["label"],
                orientation="h",
                marker=dict(
                    color=[
                        PLUM if "Current" in label else "#D7CEC6"
                        for label in benchmark_data["label"]
                    ]
                ),
                text=[
                    f"{value:.1f}{suffix}" if suffix else f"{value:.3f}"
                    for value in benchmark_data[metric_column]
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{metric_title}: %{{x:.3f}}{suffix}<extra></extra>"
                ),
            )
        )
        benchmark_figure.update_xaxes(title=metric_title)
        benchmark_figure.update_yaxes(title=None, showgrid=False)
        style_chart(benchmark_figure, height=400)
        with st.container(border=True):
            st.plotly_chart(
                benchmark_figure,
                width="stretch",
                config={"displayModeBar": False},
            )

        section(
            "03",
            "The four past test periods",
            "Each card is one untouched 16-day period.",
        )
        period_cards: list[str] = []
        mean_wape = float(fold_metrics["wape_percentage"].mean())
        for row in fold_metrics.itertuples():
            status = "Below the four-period average" if float(row.wape_percentage) <= mean_wape else "Above the four-period average"
            period_cards.append(
                f'<div class="period-card">'
                f'<div class="period-label">Past period {int(row.fold_number)}</div>'
                f'<div class="period-number">{float(row.wape_percentage):.2f}%</div>'
                f'<div class="period-note">{html.escape(status)}</div>'
                f'</div>'
            )
        st.html(f'<div class="period-grid">{"".join(period_cards)}</div>')

        section(
            "04",
            "What the model pays attention to",
            "Relative signal strength compared with the strongest signal.",
        )
        feature_count = st.slider(
            "Number of signals to display",
            min_value=5,
            max_value=15,
            value=8,
            step=1,
            key="feature_count_slider",
        )
        feature_data = feature_importance.head(feature_count).copy()
        feature_data["friendly_name"] = feature_data["feature"].map(FEATURE_LABELS).fillna(feature_data["feature"])
        maximum_gain = float(feature_data["mean_gain"].max()) if not feature_data.empty else 1.0
        feature_rows: list[str] = []
        for row in feature_data.itertuples():
            width = float(row.mean_gain) / maximum_gain * 100.0 if maximum_gain else 0.0
            feature_rows.append(
                f'<div class="feature-row">'
                f'<div class="feature-name">{html.escape(str(row.friendly_name))}</div>'
                f'<div class="feature-track"><div class="feature-fill" style="width:{width:.1f}%;"></div></div>'
                f'<div class="feature-score">{width:.0f}% of top</div>'
                f'</div>'
            )
        st.html(f'<div class="feature-list">{"".join(feature_rows)}</div>')
        st.caption("Scores are normalized against the strongest signal. They do not add up to 100% and do not prove cause and effect.")

        section(
            "05",
            "How the forecast was checked",
            "Four safeguards used before the results were published.",
        )
        steps = [
            (
                "01",
                "Tested on later dates",
                "The forecast was checked on periods that came after the information used to build it.",
            ),
            (
                "02",
                "Settings chosen earlier",
                "Model settings were selected before each final test period was scored.",
            ),
            (
                "03",
                "No future sales included",
                "Only information available before the forecast window was used.",
            ),
            (
                "04",
                "Limits shown clearly",
                "The final test answers are not available locally, so only past-test accuracy is claimed.",
            ),
        ]
        step_cards = [
            f'<div class="step-card">'
            f'<div class="step-number">{number}</div>'
            f'<div class="step-title">{html.escape(title)}</div>'
            f'<div class="step-copy">{html.escape(copy)}</div>'
            f'</div>'
            for number, title, copy in steps
        ]
        st.html(f'<div class="step-grid">{"".join(step_cards)}</div>')

        with st.expander("Technical details for reviewers"):
            st.markdown(
                f"""
- Model: XGBoost with a log-transformed sales target
- Main historical score: RMSLE {float(model_summary['pooled_rmsle']):.4f}
- Operational score: WAPE {float(model_summary['pooled_wape_percentage']):.2f}%
- Improvement versus best simple method: {float(model_summary['wape_improvement_vs_best_baseline_percentage']):.2f}%
- Validation: four chronological 16-day outer test windows
- Model-selection window: 16 days inside each training period
- Training history used per fold: 730 days
- Final test labels available locally: No
"""
            )


st.html(
    """
    <div class="footer-note">
        Retail Atelier · Demand and inventory intelligence · Human review required
    </div>
    """
)
