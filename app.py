"""
BrightBlocks ELC — PE Financial Model Dashboard (PHASE 5)
Multi-page Streamlit app with centre selection, isolated scenarios, and production UI.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from copy import deepcopy
import io

from model import (
    EARLWOOD_CONFIG, SCENARIO_PRESETS,
    run_single_centre, run_scenario, apply_scenario_overrides,
    build_monthly_pl, calc_annual_pl, calc_valuation,
    sensitivity_occupancy_fee, sensitivity_dcf, tornado_sensitivity,
    export_results_csv,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BrightBlocks ELC · PE Model",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#0F2B46",
    "accent": "#1B9AAA",
    "accent2": "#06D6A0",
    "warn": "#F4845F",
    "danger": "#E63946",
    "bg": "#F8F9FC",
    "card": "#FFFFFF",
    "text": "#1A1A2E",
    "muted": "#6B7280",
    "revenue": "#1B9AAA",
    "costs": "#E63946",
    "ebitda": "#06D6A0",
    "cash": "#4361EE",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@500&display=swap');
    
    * {{ font-family: 'DM Sans', sans-serif; }}
    
    .stApp {{ background: {COLORS['bg']}; }}
    
    h1, h2, h3 {{ color: {COLORS['primary']} !important; }}
    
    [data-testid="stSidebar"] {{
        background: white;
    }}
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {{
        color: {COLORS['text']} !important;
        font-weight: 600;
    }}
    
    .metric-card {{
        background: {COLORS['card']};
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E5E7EB;
        text-align: center;
    }}
    
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: {COLORS['muted']};
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────

def fmt_currency(val, decimals=0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:,.{max(decimals, 1)}f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:,.{decimals}f}k"
    return f"${val:,.{decimals}f}"


def fmt_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val * 100:,.{decimals}f}%"


def plotly_layout(title="", height=400):
    return dict(
        title=dict(text=title, font=dict(size=14, color=COLORS["primary"]), x=0, xanchor="left"),
        font=dict(family="DM Sans, sans-serif", size=11, color=COLORS["text"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False),
        hovermode="x unified",
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR — CENTRE SELECTOR & BASE CASE INPUTS
# ─────────────────────────────────────────────────────────────

def build_sidebar():
    """Build sidebar with centre selector and base case inputs."""
    with st.sidebar:
        st.markdown("### 🏫 BrightBlocks ELC")
        
