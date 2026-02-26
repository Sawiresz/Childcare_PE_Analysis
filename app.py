"""
BrightBlocks ELC — PE Financial Model Dashboard
Streamlit app. Requires: streamlit, plotly, pandas, numpy
Run: streamlit run app.py
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
    run_scenario_comparison, sensitivity_occupancy_fee,
    sensitivity_occupancy_wages, sensitivity_fee_wages,
    sensitivity_dcf, tornado_sensitivity,
    scenario_comparison_table,
    rent_stress_test, pe_dashboard,
    calc_annual_pl, calc_valuation,
    export_results_csv, export_scenario_comparison_csv, export_all_to_zip,
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
    "bear": "#E63946",
    "base": "#1B9AAA",
    "bull": "#06D6A0",
    "gpmargin": "#8B5CF6",
    "staffing": "#F59E0B",
    "rent": "#EC4899",
    "food": "#10B981",
    "opex": "#6366F1",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {{ background: {COLORS['bg']}; }}

    .main .block-container {{
        padding-top: 1.5rem;
        max-width: 1400px;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'DM Sans', sans-serif !important;
        color: {COLORS['primary']} !important;
        letter-spacing: -0.01em;
    }}

    p, li, span, div {{
        font-family: 'DM Sans', sans-serif;
    }}

    .metric-card {{
        background: {COLORS['card']};
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, transform 0.15s;
        min-height: 110px;
    }}
    .metric-card:hover {{
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }}

    .metric-label {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLORS['muted']};
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLORS['primary']};
        line-height: 1.2;
    }}
    .metric-sub {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        color: {COLORS['muted']};
        margin-top: 0.2rem;
    }}
    .metric-green {{ color: {COLORS['accent2']} !important; }}
    .metric-red {{ color: {COLORS['danger']} !important; }}
    .metric-blue {{ color: {COLORS['cash']} !important; }}

    .header-bar {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #1a3a5c 60%, #234e78 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }}
    .header-bar::after {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(27,154,170,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }}
    .header-bar h1 {{
        color: white !important;
        font-size: 1.5rem;
        margin: 0;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }}
    .header-bar p {{
        color: rgba(255,255,255,0.7);
        font-size: 0.82rem;
        margin: 0.25rem 0 0 0;
        position: relative;
        z-index: 1;
    }}

    .status-badge {{
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.02em;
    }}
    .badge-green {{ background: #D1FAE5; color: #065F46; }}
    .badge-red {{ background: #FEE2E2; color: #991B1B; }}
    .badge-amber {{ background: #FEF3C7; color: #92400E; }}

    [data-testid="stSidebar"] {{
        background: {COLORS['primary']};
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {{
        color: rgba(255,255,255,0.85) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: white;
        border-radius: 12px;
        padding: 0.3rem;
        border: 1px solid #E5E7EB;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    .section-divider {{
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 1.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def fmt_currency(val, prefix="$", decimals=0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "\u2014"
    if abs(val) >= 1_000_000:
        return f"{prefix}{val / 1_000_000:,.{max(decimals, 1)}f}M"
    if abs(val) >= 1_000:
        return f"{prefix}{val / 1_000:,.{decimals}f}k"
    return f"{prefix}{val:,.{decimals}f}"


def fmt_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "\u2014"
    return f"{val * 100:,.{decimals}f}%"


def metric_card(label, value, sub="", color_class=""):
    vc = f"metric-{color_class}" if color_class else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {vc}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def plotly_layout(title="", height=400, y_dollar=True):
    layout = dict(
        title=dict(text=title, font=dict(family="DM Sans, sans-serif", size=15, color=COLORS["primary"]),
                   x=0, xanchor="left", pad=dict(b=10)),
        font=dict(family="DM Sans, sans-serif", size=11, color=COLORS["text"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=55, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(255,255,255,0)"),
        xaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False, linecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False, linecolor="#E5E7EB"),
        hoverlabel=dict(bgcolor="white", bordercolor="#D1D5DB",
                        font=dict(family="DM Sans, sans-serif", size=11)),
        hovermode="x unified",
    )
    if y_dollar:
        layout["yaxis"]["tickprefix"] = "$"
        layout["yaxis"]["tickformat"] = ",.0f"
    return layout


def export_to_excel(result):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        result["annual_pl"].to_excel(w, sheet_name="Annual_PL", index=False)
        result["monthly_pl"].to_excel(w, sheet_name="Monthly_PL", index=False)
        result["cashflow"].to_excel(w, sheet_name="CashFlow", index=False)
        result["balance_sheet"].to_excel(w, sheet_name="Balance_Sheet", index=False)
        result["debt_schedule"].to_excel(w, sheet_name="Debt_Schedule", index=False)
        result["per_child"].to_excel(w, sheet_name="Per_Child", index=False)
        result["weekly_cashflow"].to_excel(w, sheet_name="Weekly_CashFlow", index=False)
        gst = result["gst_bas"].copy()
        gst["quarter"] = gst["quarter"].astype(str)
        gst.to_excel(w, sheet_name="GST_BAS", index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────

def build_sidebar():
    cfg = deepcopy(EARLWOOD_CONFIG)

    with st.sidebar:
        st.markdown("### \U0001F3EB BrightBlocks ELC")
        st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:0.5rem 0 1rem'>",
                    unsafe_allow_html=True)

        scenario = st.selectbox(
            "\U0001F4CC Scenario", ["Base", "Bear (Downside)", "Bull (Upside)"],
            help="Scenario presets override occupancy, fees, and exit multiples")

        with st.expander("\U0001F4CA Revenue", expanded=False):
            cfg["approved_capacity"] = st.number_input(
                "Approved Capacity", 10, 200, cfg["approved_capacity"], step=5)
            cfg["daily_fees"]["36m_plus"] = st.number_input(
                "Daily Fee 36m+ ($)", 80.0, 250.0, cfg["daily_fees"]["36m_plus"], step=1.0)
            cfg["daily_fees"]["24_36m"] = st.number_input(
                "Daily Fee 24\u201336m ($)", 80.0, 250.0, cfg["daily_fees"]["24_36m"], step=1.0)
            cfg["daily_fees"]["0_24m"] = st.number_input(
                "Daily Fee 0\u201324m ($)", 80.0, 300.0, cfg["daily_fees"]["0_24m"], step=1.0)
            cfg["fee_annual_increase"] = st.number_input(
                "Fee Increase (% p.a.)", 0.0, 10.0, cfg["fee_annual_increase"] * 100, 0.5) / 100
            cfg["revenue_collection_rate"] = st.number_input(
                "Collection Rate (%)", 90.0, 100.0, cfg["revenue_collection_rate"] * 100, 0.5) / 100

        with st.expander("\U0001F4C8 Occupancy Ramp", expanded=False):
            cfg["occupancy_ramp"]["m4"] = st.number_input(
                "Month 4 (%)", 0.0, 100.0, cfg["occupancy_ramp"]["m4"] * 100, 5.0) / 100
            cfg["occupancy_ramp"]["m5_6"] = st.number_input(
                "Months 5\u20136 (%)", 0.0, 100.0, cfg["occupancy_ramp"]["m5_6"] * 100, 5.0) / 100
            cfg["occupancy_ramp"]["y2"] = st.number_input(
                "Year 2 (%)", 0.0, 100.0, cfg["occupancy_ramp"]["y2"] * 100, 5.0) / 100
            cfg["occupancy_ramp"]["y3"] = st.number_input(
                "Year 3 (%)", 0.0, 100.0, cfg["occupancy_ramp"]["y3"] * 100, 5.0) / 100
            cfg["occupancy_ramp"]["y4_plus"] = st.number_input(
                "Year 4+ (%)", 0.0, 100.0, cfg["occupancy_ramp"]["y4_plus"] * 100, 2.5) / 100

        with st.expander("\U0001F469\u200D\U0001F3EB Staffing", expanded=False):
            cfg["director_hourly_rate"] = st.number_input(
                "Director Hourly ($)", 30.0, 100.0, cfg["director_hourly_rate"], 1.0)
            cfg["educator_hourly_rate"] = st.number_input(
                "Educator Hourly ($)", 25.0, 80.0, cfg["educator_hourly_rate"], 0.5)
            cfg["support_hourly_rate"] = st.number_input(
                "Support Hourly ($)", 20.0, 60.0, cfg["support_hourly_rate"], 0.5)
            cfg["superannuation_rate"] = st.number_input(
                "Super Rate (%)", 9.0, 15.0, cfg["superannuation_rate"] * 100, 0.5) / 100
            cfg["staff_wage_annual_increase"] = st.number_input(
                "Wage Increase (% p.a.)", 0.0, 10.0, cfg["staff_wage_annual_increase"] * 100, 0.5) / 100
            cfg["director_salary_deferred"] = st.checkbox(
                "Director Salary Deferred", value=cfg["director_salary_deferred"])

        with st.expander("\U0001F3E0 Lease", expanded=False):
            cfg["base_rent_pa"] = st.number_input(
                "Base Rent ($ p.a.)", 50000.0, 500000.0, cfg["base_rent_pa"], 5000.0)
            cfg["rent_escalation"] = st.number_input(
                "Rent Escalation (% p.a.)", 0.0, 10.0, cfg["rent_escalation"] * 100, 0.5) / 100
            cfg["rent_free_months"] = st.number_input(
                "Rent-Free Months", 0, 24, cfg["rent_free_months"], 1)

        with st.expander("\U0001F4B0 Capital & Financing", expanded=False):
            cfg["capital_items"]["operational_seed"] = st.number_input(
                "Operational Seed ($)", 10000.0, 500000.0,
                float(cfg["capital_items"]["operational_seed"]), 5000.0)
            cfg["director_loan_interest_rate"] = st.number_input(
                "Director Loan Rate (%)", 0.0, 15.0,
                cfg["director_loan_interest_rate"] * 100, 0.5) / 100

        with st.expander("\U0001F9EE Valuation", expanded=False):
            cfg["discount_rate"] = st.number_input(
                "Discount Rate (%)", 5.0, 25.0, cfg["discount_rate"] * 100, 0.5) / 100
            cfg["terminal_growth_rate"] = st.number_input(
                "Terminal Growth (%)", 0.0, 5.0, cfg["terminal_growth_rate"] * 100, 0.25) / 100
            exit_mult = st.number_input(
                "Exit Multiple (\u00d7 EBITDA)", 2.0, 15.0, cfg["exit_multiples"][1], 0.5)
            cfg["exit_multiples"] = [exit_mult - 1.0, exit_mult, exit_mult + 1.5]

        with st.expander("\U0001F34E Food & Other", expanded=False):
            cfg["food_cost_per_child_per_day"] = st.number_input(
                "Food Cost / Child / Day ($)", 4.0, 20.0,
                cfg["food_cost_per_child_per_day"], 0.5)
            cfg["cpi_on_expenses"] = st.number_input(
                "CPI on Expenses (%)", 0.0, 8.0, cfg["cpi_on_expenses"] * 100, 0.25) / 100
            cfg["company_tax_rate"] = st.number_input(
                "Company Tax Rate (%)", 0.0, 40.0, cfg["company_tax_rate"] * 100, 1.0) / 100

        # Validation
        errors = []
        if cfg["approved_capacity"] < 1:
            errors.append("Capacity must be \u2265 1")
        if cfg["discount_rate"] <= cfg["terminal_growth_rate"]:
            errors.append("Discount rate must exceed terminal growth rate")
        if any(cfg["daily_fees"][k] <= 0 for k in cfg["daily_fees"]):
            errors.append("Daily fees must be > $0")
        if cfg["revenue_collection_rate"] < 0.5:
            errors.append("Collection rate seems too low (<50%)")
        if cfg["base_rent_pa"] <= 0:
            errors.append("Base rent must be > $0")
        if cfg["occupancy_ramp"]["y4_plus"] < cfg["occupancy_ramp"]["m4"]:
            st.warning("\u26A0\uFE0F Y4+ occupancy is below M4 \u2014 intentional?")

        if errors:
            for e in errors:
                st.error(e)

    return cfg, scenario, len(errors) == 0


# ─────────────────────────────────────────────────────────────
# CACHED MODEL RUNS
# ─────────────────────────────────────────────────────────────

def _cfg_hash(cfg):
    """Convert cfg dict to a hashable tuple for caching."""
    def _flatten(d, prefix=""):
        items = []
        for k, v in sorted(d.items()):
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(_flatten(v, key))
            elif isinstance(v, list):
                items.append((key, tuple(v)))
            else:
                items.append((key, v))
        return items
    return tuple(_flatten(cfg))


def run_model(cfg, scenario_key):
    """Run model with session-state caching based on config hash."""
    key = (_cfg_hash(cfg), scenario_key)
    if "_model_cache" not in st.session_state:
        st.session_state["_model_cache"] = {}
    cache = st.session_state["_model_cache"]
    if key not in cache:
        # Evict old entries to prevent memory bloat
        if len(cache) > 20:
            cache.clear()
        cache[key] = run_scenario(cfg, scenario_key)
    return cache[key]


# ─────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ─────────────────────────────────────────────────────────────

def render_dashboard(result):
    cfg = result["config"]
    centre_name = cfg.get("centre_name", "BrightBlocks ELC")

    st.markdown(f"""
    <div class="header-bar">
        <h1>\U0001F3EB PE Returns Dashboard</h1>
        <p>5-Year Financial Model \u00b7 {centre_name}</p>
    </div>
    """, unsafe_allow_html=True)

    val = result["valuation"]
    annual = result["annual_pl"]
    cap = result["capital_sufficiency"]
    pl = result["monthly_pl"]
    cf = result["cashflow"]

    # ── Top KPI row ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(metric_card(
            "Y5 Revenue", fmt_currency(annual.iloc[-1]["net_revenue"]),
            f"Y1: {fmt_currency(annual.iloc[0]['net_revenue'])}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Y5 EBITDA", fmt_currency(val["y5_ebitda"]),
            f"Margin: {fmt_pct(annual.iloc[-1]['ebitda_margin'])}", "green"), unsafe_allow_html=True)
    with c3:
        irr_str = fmt_pct(val["irr"]) if val["irr"] else "N/A"
        st.markdown(metric_card("IRR", irr_str, "5-year levered", "green"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "MOIC", f"{val['moic']:.2f}\u00d7",
            f"on {fmt_currency(val['total_capital'])} invested", "blue"), unsafe_allow_html=True)
    with c5:
        mid_mult = cfg["exit_multiples"][1]
        ev_key = f"{mid_mult}x"
        ev_val = val["ev_multiples"].get(ev_key, 0)
        st.markdown(metric_card(
            "Enterprise Value", fmt_currency(ev_val),
            f"@ {mid_mult}\u00d7 EBITDA"), unsafe_allow_html=True)
    with c6:
        badge = "badge-green" if cap["sufficient"] else "badge-red"
        status = "SUFFICIENT" if cap["sufficient"] else "INSUFFICIENT"
        weeks = cap["weeks_runway"]
        weeks_str = f"{weeks:.0f} weeks runway" if weeks < 100 else "Strong"
        st.markdown(metric_card(
            "Capital Status",
            f'<span class="status-badge {badge}">{status}</span>',
            f"Buffer: {fmt_currency(cap['buffer_above_reserve'])} \u00b7 {weeks_str}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Revenue vs Costs + Cash Position ──
    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=annual["fiscal_year"].astype(str), y=annual["net_revenue"],
            name="Revenue", marker_color=COLORS["revenue"], opacity=0.9,
            hovertemplate="Revenue: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=annual["fiscal_year"].astype(str),
            y=annual["total_overheads"] + annual["direct_costs"],
            name="Total Costs", marker_color=COLORS["costs"], opacity=0.7,
            hovertemplate="Costs: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=annual["fiscal_year"].astype(str), y=annual["ebitda"],
            name="EBITDA", line=dict(color=COLORS["ebitda"], width=3),
            mode="lines+markers", marker=dict(size=8),
            hovertemplate="EBITDA: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**plotly_layout("Revenue vs Costs vs EBITDA"))
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pl["date"], y=pl["ebitda"].cumsum(),
            fill="tozeroy", fillcolor="rgba(6,214,160,0.12)",
            line=dict(color=COLORS["ebitda"], width=2),
            name="Cumul. EBITDA",
        ))
        cash_dates = cf["date"] if "date" in cf.columns else pl["date"]
        fig2.add_trace(go.Scatter(
            x=cash_dates, y=cf["closing_cash"],
            line=dict(color=COLORS["cash"], width=2.5, dash="dot"),
            name="Closing Cash",
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="#D1D5DB", line_width=1)
        reserve = cfg.get("minimum_operating_reserve", 20000)
        fig2.add_hline(y=reserve, line_dash="dot", line_color=COLORS["warn"], line_width=1,
                       annotation_text=f"Reserve ${reserve / 1000:.0f}k",
                       annotation_position="top right",
                       annotation_font_size=9, annotation_font_color=COLORS["warn"])
        l2 = plotly_layout("Cash & Cumulative EBITDA (60 Months)")
        l2["xaxis"]["tickformat"] = "%b %Y"
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Occupancy + Cost Breakdown ──
    col_a, col_b = st.columns(2)

    with col_a:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=pl["date"], y=pl["occupancy_pct"] * 100,
            line=dict(color=COLORS["accent"], width=2.5),
            fill="tozeroy", fillcolor="rgba(27,154,170,0.08)",
            name="Occupancy %",
        ))
        fig3.add_trace(go.Scatter(
            x=pl["date"], y=pl["children"],
            line=dict(color=COLORS["primary"], width=1.5, dash="dot"),
            name="Children", yaxis="y2",
        ))
        l3 = plotly_layout("Occupancy Ramp", y_dollar=False)
        l3["yaxis"]["ticksuffix"] = "%"
        l3["yaxis"]["range"] = [0, 105]
        l3["yaxis2"] = dict(
            title="Children", overlaying="y", side="right",
            showgrid=False, tickformat=".0f", range=[0, cfg["approved_capacity"] * 1.1])
        l3["xaxis"]["tickformat"] = "%b %Y"
        fig3.update_layout(**l3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        fig4 = make_subplots()
        fig4.add_trace(go.Scatter(
            x=annual["fiscal_year"].astype(str), y=annual["total_staff_cost"],
            name="Staffing", stackgroup="costs",
            line=dict(width=0), fillcolor="rgba(245,158,11,0.5)",
        ))
        fig4.add_trace(go.Scatter(
            x=annual["fiscal_year"].astype(str), y=annual["food_cost"],
            name="Food", stackgroup="costs",
            line=dict(width=0), fillcolor="rgba(16,185,129,0.5)",
        ))
        fig4.add_trace(go.Scatter(
            x=annual["fiscal_year"].astype(str), y=annual["total_occupancy_cost"],
            name="Rent & Occ.", stackgroup="costs",
            line=dict(width=0), fillcolor="rgba(236,72,153,0.5)",
        ))
        fig4.add_trace(go.Scatter(
            x=annual["fiscal_year"].astype(str), y=annual["total_opex"],
            name="Opex", stackgroup="costs",
            line=dict(width=0), fillcolor="rgba(99,102,241,0.5)",
        ))
        fig4.update_layout(**plotly_layout("Annual Cost Breakdown"))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Margins + Returns Waterfall ──
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=pl["date"], y=pl["ebitda_margin"] * 100,
            line=dict(color=COLORS["ebitda"], width=2.5),
            name="EBITDA Margin",
        ))
        fig5.add_trace(go.Scatter(
            x=pl["date"], y=pl["gp_margin"] * 100,
            line=dict(color=COLORS["gpmargin"], width=2, dash="dash"),
            name="GP Margin",
        ))
        active_months = pl[pl["net_revenue"] > 0]
        if len(active_months) > 0:
            fig5.add_trace(go.Scatter(
                x=active_months["date"],
                y=active_months["wages_pct_revenue"] * 100,
                line=dict(color=COLORS["warn"], width=1.5, dash="dot"),
                name="Wages % Rev",
            ))
        l5 = plotly_layout("Margin Analysis (Monthly)", y_dollar=False)
        l5["yaxis"]["ticksuffix"] = "%"
        l5["xaxis"]["tickformat"] = "%b %Y"
        fig5.update_layout(**l5)
        st.plotly_chart(fig5, use_container_width=True)

    with col_m2:
        total_cap = val["total_capital"]
        exit_ev = val["y5_ebitda"] * cfg["exit_multiples"][1]
        cumul_fcf = sum(val["annual_fcf"])

        fig_w = go.Figure(go.Waterfall(
            x=["Capital Invested", "Cumul. FCF (5yr)", "Exit Proceeds", "Total Return"],
            y=[-total_cap, cumul_fcf, exit_ev, 0],
            measure=["absolute", "relative", "relative", "total"],
            connector=dict(line=dict(color="#D1D5DB", width=1)),
            increasing=dict(marker_color=COLORS["accent2"]),
            decreasing=dict(marker_color=COLORS["danger"]),
            totals=dict(marker_color=COLORS["cash"]),
            text=[fmt_currency(-total_cap), fmt_currency(cumul_fcf),
                  fmt_currency(exit_ev), fmt_currency(exit_ev + cumul_fcf - total_cap)],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11),
        ))
        fig_w.update_layout(**plotly_layout("Returns Waterfall"))
        fig_w.update_layout(showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    # ── Row 4: Director Loan + Per-child ──
    col_d1, col_d2 = st.columns(2)

    debt = result["debt_schedule"]
    with col_d1:
        fig_debt = go.Figure()
        debt_x = debt["date"] if "date" in debt.columns else pd.RangeIndex(len(debt))
        fig_debt.add_trace(go.Scatter(
            x=debt_x, y=debt["closing_balance"],
            fill="tozeroy", fillcolor="rgba(244,132,95,0.12)",
            line=dict(color=COLORS["warn"], width=2.5),
            name="Director Loan",
        ))
        ld = plotly_layout("Director Loan Balance")
        if "date" in debt.columns:
            ld["xaxis"]["tickformat"] = "%b %Y"
        fig_debt.update_layout(**ld)
        st.plotly_chart(fig_debt, use_container_width=True)

    with col_d2:
        pc = result["per_child"]
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Bar(
            x=pc["fiscal_year"].astype(str), y=pc["net_revenue_per_child"],
            name="Revenue / Child", marker_color=COLORS["revenue"]))
        fig_pc.add_trace(go.Bar(
            x=pc["fiscal_year"].astype(str), y=pc["ebitda_per_child"],
            name="EBITDA / Child", marker_color=COLORS["ebitda"]))
        fig_pc.update_layout(**plotly_layout("Unit Economics per Child"))
        fig_pc.update_layout(barmode="group")
        st.plotly_chart(fig_pc, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 2: FINANCIALS
# ─────────────────────────────────────────────────────────────

def render_financials(result):
    st.markdown("### \U0001F4CB Financial Statements")

    sub = st.radio(
        "Select view", [
            "Annual P&L", "Monthly P&L", "Cash Flow", "Balance Sheet",
            "Debt Schedule", "Weekly Cash Flow", "GST / BAS"],
        horizontal=True, label_visibility="collapsed")

    if sub == "Annual P&L":
        df = result["annual_pl"].copy()
        df["fiscal_year"] = df["fiscal_year"].astype(str)
        display_cols = [
            "fiscal_year", "net_revenue", "direct_costs", "gross_profit",
            "total_overheads", "ebitda", "ebitda_margin", "ebit",
            "npbt", "tax", "npat"]
        rename = {
            "fiscal_year": "FY", "net_revenue": "Revenue", "direct_costs": "Direct Costs",
            "gross_profit": "Gross Profit", "total_overheads": "Overheads",
            "ebitda": "EBITDA", "ebitda_margin": "EBITDA %", "ebit": "EBIT",
            "npbt": "NPBT", "tax": "Tax", "npat": "NPAT"}
        out = df[display_cols].rename(columns=rename)
        st.dataframe(
            out.style.format({
                "Revenue": "${:,.0f}", "Direct Costs": "${:,.0f}",
                "Gross Profit": "${:,.0f}", "Overheads": "${:,.0f}",
                "EBITDA": "${:,.0f}", "EBITDA %": "{:.1%}",
                "EBIT": "${:,.0f}", "NPBT": "${:,.0f}", "Tax": "${:,.0f}", "NPAT": "${:,.0f}",
            }),
            use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["fiscal_year"], y=df["net_revenue"], name="Revenue",
                             marker_color=COLORS["revenue"]))
        fig.add_trace(go.Scatter(x=df["fiscal_year"], y=df["ebitda"], name="EBITDA",
                                 line=dict(color=COLORS["ebitda"], width=3),
                                 mode="lines+markers", marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=df["fiscal_year"], y=df["npat"], name="NPAT",
                                 line=dict(color=COLORS["cash"], width=2, dash="dash"),
                                 mode="lines+markers", marker=dict(size=6)))
        fig.update_layout(**plotly_layout("Annual P&L Summary", height=350))
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Monthly P&L":
        df = result["monthly_pl"].copy()
        df["date_str"] = df["date"].dt.strftime("%b %Y")
        cols = ["month", "date_str", "occupancy_pct", "children", "net_revenue",
                "total_staff_cost", "food_cost", "total_occupancy_cost",
                "total_opex", "ebitda", "ebitda_margin", "npat"]
        rename = {
            "date_str": "Date", "occupancy_pct": "Occ %", "children": "Kids",
            "net_revenue": "Revenue", "total_staff_cost": "Staff",
            "food_cost": "Food", "total_occupancy_cost": "Occ. Cost",
            "total_opex": "Opex", "ebitda": "EBITDA",
            "ebitda_margin": "EBITDA %", "npat": "NPAT", "month": "M"}
        out = df[cols].rename(columns=rename)
        st.dataframe(
            out.style.format({
                "Occ %": "{:.0%}", "Revenue": "${:,.0f}",
                "Staff": "${:,.0f}", "Food": "${:,.0f}",
                "Occ. Cost": "${:,.0f}", "Opex": "${:,.0f}",
                "EBITDA": "${:,.0f}", "EBITDA %": "{:.1%}", "NPAT": "${:,.0f}",
            }),
            use_container_width=True, hide_index=True, height=500)

    elif sub == "Cash Flow":
        cf = result["cashflow"].copy()
        cf_annual = cf.groupby("fiscal_year").agg({
            "npat": "sum", "da": "sum", "wc_movement": "sum",
            "gst_input_credits": "sum", "operating_cf": "sum",
            "maintenance_capex": "sum", "investing_cf": "sum",
            "loan_repayments": "sum", "financing_cf": "sum",
            "net_cash_movement": "sum",
        }).reset_index()
        cf_annual["closing_cash"] = cf.groupby("fiscal_year")["closing_cash"].last().values
        cf_annual["fiscal_year"] = cf_annual["fiscal_year"].astype(str)
        rename = {
            "fiscal_year": "FY", "npat": "NPAT", "da": "D&A", "wc_movement": "WC Mvmt",
            "gst_input_credits": "GST Credits", "operating_cf": "Operating CF",
            "maintenance_capex": "Capex", "investing_cf": "Investing CF",
            "loan_repayments": "Loan Repay", "financing_cf": "Financing CF",
            "net_cash_movement": "Net Cash Mvmt", "closing_cash": "Closing Cash"}
        out = cf_annual.rename(columns=rename)
        st.dataframe(
            out.style.format({c: "${:,.0f}" for c in out.columns if c != "FY"}),
            use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=cf_annual["fiscal_year"], y=cf_annual["operating_cf"],
                             name="Operating CF", marker_color=COLORS["accent2"]))
        fig.add_trace(go.Bar(x=cf_annual["fiscal_year"], y=cf_annual["investing_cf"],
                             name="Investing CF", marker_color=COLORS["warn"]))
        fig.add_trace(go.Bar(x=cf_annual["fiscal_year"], y=cf_annual["financing_cf"],
                             name="Financing CF", marker_color=COLORS["danger"]))
        fig.add_trace(go.Scatter(x=cf_annual["fiscal_year"], y=cf_annual["closing_cash"],
                                 name="Closing Cash", line=dict(color=COLORS["cash"], width=3),
                                 mode="lines+markers", marker=dict(size=7)))
        fig.update_layout(**plotly_layout("Cash Flow Summary", height=350))
        fig.update_layout(barmode="relative")
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Balance Sheet":
        bs = result["balance_sheet"].copy()
        bs["fiscal_year"] = bs["fiscal_year"].astype(str)
        st.dataframe(
            bs.style.format({c: "${:,.0f}" for c in bs.columns
                             if c not in ["fiscal_year", "balance_check"]}),
            use_container_width=True, hide_index=True)

    elif sub == "Debt Schedule":
        debt = result["debt_schedule"].copy()
        debt["date_str"] = debt["date"].dt.strftime("%b %Y")
        cols = ["month", "date_str", "opening_balance", "interest",
                "repay_eligible", "repayment", "closing_balance"]
        rename = {"date_str": "Date", "opening_balance": "Opening",
                  "interest": "Interest", "repay_eligible": "Eligible",
                  "repayment": "Repayment", "closing_balance": "Closing"}
        out = debt[cols].rename(columns=rename)
        st.dataframe(
            out.style.format({
                "Opening": "${:,.0f}", "Interest": "${:,.0f}",
                "Repayment": "${:,.0f}", "Closing": "${:,.0f}",
            }),
            use_container_width=True, hide_index=True, height=500)

    elif sub == "Weekly Cash Flow":
        wk = result["weekly_cashflow"].copy()
        rag_colors = {"GREEN": "background-color: #D1FAE5",
                      "AMBER": "background-color: #FEF3C7",
                      "RED": "background-color: #FEE2E2"}

        def highlight_rag(val):
            return rag_colors.get(val, "")

        st.dataframe(
            wk.style.format({
                "revenue": "${:,.0f}", "wages": "${:,.0f}",
                "outgoings": "${:,.0f}", "other_opex": "${:,.0f}",
                "net_movement": "${:,.0f}", "cumulative_cash": "${:,.0f}",
                "buffer_above_reserve": "${:,.0f}",
            }).map(highlight_rag, subset=["rag_status"]),
            use_container_width=True, hide_index=True, height=500)

    elif sub == "GST / BAS":
        gst = result["gst_bas"].copy()
        gst["quarter"] = gst["quarter"].astype(str)
        rename = {"quarter": "Quarter", "gst_collected": "GST Collected",
                  "gst_paid_inputs": "GST Inputs", "net_gst_position": "Net GST",
                  "refund": "Refund"}
        out = gst.rename(columns=rename)
        st.dataframe(
            out.style.format({c: "${:,.0f}" for c in out.columns if c != "Quarter"}),
            use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TAB 3: SCENARIOS
# ─────────────────────────────────────────────────────────────

def render_scenarios(cfg):
    st.markdown("### \U0001F500 Scenario Comparison")

    scenarios_data = {}
    for s in ["bear", "base", "bull"]:
        scenarios_data[s] = run_model(cfg, s)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    card_defs = [
        (c1, "\U0001F43B Bear (Downside)", "bear", "red"),
        (c2, "\U0001F4CA Base Case", "base", ""),
        (c3, "\U0001F402 Bull (Upside)", "bull", "green"),
    ]
    for col, label, key, color in card_defs:
        with col:
            v = scenarios_data[key]["valuation"]
            irr_s = fmt_pct(v["irr"]) if v["irr"] else "N/A"
            pb = f'{v["payback_years"]:.1f}yr' if v["payback_years"] else "\u2014"
            st.markdown(metric_card(
                label, f"IRR {irr_s}",
                f"MOIC {v['moic']:.2f}\u00d7 \u00b7 Y5 EBITDA {fmt_currency(v['y5_ebitda'])} \u00b7 PB {pb}",
                color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # EBITDA comparison chart
    fig = go.Figure()
    for s, color in [("bear", COLORS["bear"]), ("base", COLORS["base"]), ("bull", COLORS["bull"])]:
        a = scenarios_data[s]["annual_pl"]
        fig.add_trace(go.Scatter(
            x=a["fiscal_year"].astype(str), y=a["ebitda"],
            name=SCENARIO_PRESETS.get(s, {}).get("label", s.capitalize()),
            line=dict(color=color, width=3),
            mode="lines+markers", marker=dict(size=8),
        ))
    fig.update_layout(**plotly_layout("EBITDA by Scenario (5 Years)"))
    st.plotly_chart(fig, use_container_width=True)

    # Revenue + NPAT side by side
    col_l, col_r = st.columns(2)
    with col_l:
        fig2 = go.Figure()
        for s, color in [("bear", COLORS["bear"]), ("base", COLORS["base"]), ("bull", COLORS["bull"])]:
            a = scenarios_data[s]["annual_pl"]
            fig2.add_trace(go.Bar(
                x=a["fiscal_year"].astype(str), y=a["net_revenue"],
                name=SCENARIO_PRESETS.get(s, {}).get("label", s.capitalize()),
                marker_color=color, opacity=0.85))
        fig2.update_layout(**plotly_layout("Revenue by Scenario"))
        fig2.update_layout(barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        fig3 = go.Figure()
        for s, color in [("bear", COLORS["bear"]), ("base", COLORS["base"]), ("bull", COLORS["bull"])]:
            a = scenarios_data[s]["annual_pl"]
            fig3.add_trace(go.Bar(
                x=a["fiscal_year"].astype(str), y=a["npat"],
                name=SCENARIO_PRESETS.get(s, {}).get("label", s.capitalize()),
                marker_color=color, opacity=0.85))
        fig3.update_layout(**plotly_layout("NPAT by Scenario"))
        fig3.update_layout(barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

    # Full comparison table
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Returns & KPI Summary")

    table = scenario_comparison_table(cfg)
    display = table.drop(columns=["annual_fcf"], errors="ignore").copy()
    display = display.rename(columns={
        "scenario": "Scenario", "y1_revenue": "Y1 Revenue", "y3_revenue": "Y3 Revenue",
        "y5_revenue": "Y5 Revenue", "y1_ebitda": "Y1 EBITDA", "y3_ebitda": "Y3 EBITDA",
        "y5_ebitda": "Y5 EBITDA", "y1_margin": "Y1 Margin", "y5_margin": "Y5 Margin",
        "y1_npat": "Y1 NPAT", "y5_npat": "Y5 NPAT", "y5_staff_pct": "Y5 Staff %",
        "ev_dcf": "EV (DCF)", "ev_multiple": "EV (Multiple)",
        "irr": "IRR", "moic": "MOIC", "payback_years": "Payback",
        "total_capital": "Capital", "y5_closing_cash": "Y5 Cash",
        "capital_sufficient": "Cap. OK",
    })

    fmt_map = {}
    for c in display.columns:
        if c == "Scenario" or c == "Cap. OK":
            continue
        elif "Margin" in c or "Staff %" in c or c == "IRR":
            fmt_map[c] = "{:.1%}"
        elif c == "MOIC":
            fmt_map[c] = "{:.2f}\u00d7"
        elif c == "Payback":
            fmt_map[c] = "{:.1f}"
        else:
            fmt_map[c] = "${:,.0f}"

    st.dataframe(display.style.format(fmt_map, na_rep="\u2014"),
                 use_container_width=True, hide_index=True)

    # Scenario assumptions comparison
    st.markdown("### Scenario Assumptions")
    assumption_rows = []
    for s in ["bear", "base", "bull"]:
        p = SCENARIO_PRESETS.get(s, {})
        c = scenarios_data[s]["config"]
        assumption_rows.append({
            "Scenario": p.get("label", s.capitalize()),
            "M4 Occ": c["occupancy_ramp"]["m4"],
            "Y3 Occ": c["occupancy_ramp"]["y3"],
            "Y4+ Occ": c["occupancy_ramp"]["y4_plus"],
            "Fee 36m+": c["daily_fees"]["36m_plus"],
            "Fee Esc": c["fee_annual_increase"],
            "Ed. Rate": c["educator_hourly_rate"],
            "Wage Esc": c["staff_wage_annual_increase"],
            "CPI": c["cpi_on_expenses"],
            "Rent p.a.": c["base_rent_pa"],
            "Exit Mult": c["exit_multiples"][1],
        })
    assumptions = pd.DataFrame(assumption_rows)
    st.dataframe(
        assumptions.style.format({
            "M4 Occ": "{:.0%}", "Y3 Occ": "{:.0%}", "Y4+ Occ": "{:.0%}",
            "Fee 36m+": "${:.0f}", "Fee Esc": "{:.1%}",
            "Ed. Rate": "${:.2f}", "Wage Esc": "{:.1%}", "CPI": "{:.1%}",
            "Rent p.a.": "${:,.0f}", "Exit Mult": "{:.1f}\u00d7",
        }),
        use_container_width=True, hide_index=True)

    # CSV export
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    csv_data = export_scenario_comparison_csv(cfg)
    st.download_button(
        "\u2B07 Export Scenario Comparison (CSV)",
        data=csv_data, file_name="scenario_comparison.csv",
        mime="text/csv", use_container_width=False)

    # Rent stress test
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### \U0001F3E0 Market Rent Stress Test (Year 6 Renewal)")
    rst = rent_stress_test(cfg)
    st.dataframe(
        rst.style.format({
            "annual_rent": "${:,.0f}", "monthly_rent": "${:,.0f}",
            "vs_base": "${:,.0f}", "ebitda_impact": "${:,.0f}",
        }),
        use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# TAB 4: SENSITIVITY
# ─────────────────────────────────────────────────────────────

def _render_heatmap(grid, title, x_label, y_label, x_fmt_fn, y_fmt_fn, cell_fmt_fn, height=450):
    """Reusable heatmap renderer."""
    fig = go.Figure(data=go.Heatmap(
        z=grid.values,
        x=[x_fmt_fn(c) for c in grid.columns],
        y=[y_fmt_fn(i) for i in grid.index],
        colorscale=[
            [0, COLORS["danger"]], [0.3, "#FEF3C7"],
            [0.5, "#FFFBEB"], [0.7, "#D1FAE5"], [1, COLORS["accent2"]]],
        text=[[cell_fmt_fn(v) for v in row] for row in grid.values],
        texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono, monospace"),
        hovertemplate=f"{y_label}: %{{y}} \u00b7 {x_label}: %{{x}}<br>Value: %{{text}}<extra></extra>",
    ))
    l = plotly_layout(title, height=height, y_dollar=False)
    l["xaxis"] = dict(title=x_label)
    l["yaxis"] = dict(title=y_label)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

    grid_d = grid.copy()
    grid_d.index = [y_fmt_fn(i) for i in grid_d.index]
    grid_d.columns = [x_fmt_fn(c) for c in grid_d.columns]
    st.dataframe(grid_d.style.format("${:,.0f}"), use_container_width=True)


def render_sensitivity(cfg):
    st.markdown("### \U0001F3AF Sensitivity Analysis")

    sub = st.radio(
        "Select analysis", [
            "Tornado (Single-Variable)",
            "Occupancy \u00d7 Fee",
            "Occupancy \u00d7 Wages",
            "Fee \u00d7 Wages",
            "Discount Rate \u00d7 Terminal Growth",
        ], horizontal=True, label_visibility="collapsed")

    # ── TORNADO ──
    if "Tornado" in sub:
        st.caption("Impact of \u00b1 single-variable changes on Year 3 EBITDA")

        with st.spinner("Computing tornado..."):
            tornado = tornado_sensitivity(cfg, target_year=3)

        base_ebitda = tornado.iloc[0]["ebitda"] - tornado.iloc[0]["ebitda_change"]  # approx

        variables = tornado["variable"].unique()
        lows = []
        highs = []
        labels = []
        for var in variables:
            subset = tornado[tornado["variable"] == var]
            low_row = subset[subset["side"] == "low"].iloc[0]
            high_row = subset[subset["side"] == "high"].iloc[0]
            lows.append(low_row["ebitda_change"])
            highs.append(high_row["ebitda_change"])
            labels.append(var)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=lows, orientation="h",
            name="Downside", marker_color=COLORS["danger"], opacity=0.8,
            text=[f"${v:+,.0f}" for v in lows],
            textposition="auto", textfont=dict(size=10),
        ))
        fig.add_trace(go.Bar(
            y=labels, x=highs, orientation="h",
            name="Upside", marker_color=COLORS["accent2"], opacity=0.8,
            text=[f"${v:+,.0f}" for v in highs],
            textposition="auto", textfont=dict(size=10),
        ))
        fig.add_vline(x=0, line_color="#9CA3AF", line_width=1)
        l = plotly_layout("Tornado: Y3 EBITDA Impact (\u00b1 Single Variable)", height=450, y_dollar=False)
        l["xaxis"] = dict(title="\u0394 EBITDA ($)", tickprefix="$", tickformat="+,.0f")
        l["yaxis"] = dict(title="")
        l["barmode"] = "overlay"
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

        # Detail table
        detail = tornado.copy()
        detail = detail.rename(columns={
            "variable": "Variable", "side": "Side", "delta": "Change",
            "base_value": "Base", "adjusted_value": "Adjusted",
            "ebitda": "EBITDA", "ebitda_change": "\u0394 EBITDA"})
        st.dataframe(
            detail[["Variable", "Side", "Base", "Adjusted", "EBITDA", "\u0394 EBITDA"]].style.format({
                "EBITDA": "${:,.0f}", "\u0394 EBITDA": "${:+,.0f}",
            }),
            use_container_width=True, hide_index=True)

    # ── OCCUPANCY x FEE ──
    elif "Occupancy" in sub and "Fee" in sub:
        st.caption("Year 3 EBITDA across occupancy and daily fee combinations")

        col1, col2 = st.columns(2)
        with col1:
            occ_low = st.number_input("Min Occupancy (%)", 30.0, 90.0, 60.0, 5.0) / 100
            occ_high = st.number_input("Max Occupancy (%)", 70.0, 100.0, 100.0, 5.0) / 100
        with col2:
            fee_low = st.number_input("Min Fee ($)", 80.0, 150.0, 125.0, 5.0)
            fee_high = st.number_input("Max Fee ($)", 130.0, 200.0, 155.0, 5.0)

        occs = np.linspace(occ_low, occ_high, 5).tolist()
        fees = [round(f, 0) for f in np.linspace(fee_low, fee_high, 5).tolist()]

        with st.spinner("Computing grid..."):
            grid = sensitivity_occupancy_fee(cfg, occs, fees)

        _render_heatmap(grid, "Y3 EBITDA: Occupancy \u00d7 Fee",
                        "Daily Fee", "Occupancy",
                        lambda c: f"${c:.0f}", lambda i: f"{i:.0%}",
                        lambda v: f"${v:,.0f}")

    # ── OCCUPANCY x WAGES ──
    elif "Wages" in sub and "Occupancy" in sub:
        st.caption("Year 3 EBITDA across occupancy and educator hourly rate")

        base_rate = cfg["educator_hourly_rate"]
        with st.spinner("Computing grid..."):
            occs = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
            wages = [round(base_rate * m, 2) for m in [0.85, 0.92, 1.0, 1.08, 1.15]]
            grid = sensitivity_occupancy_wages(cfg, occs, wages)

        _render_heatmap(grid, "Y3 EBITDA: Occupancy \u00d7 Educator Rate",
                        "Educator Rate ($/hr)", "Occupancy",
                        lambda c: f"${c:.2f}", lambda i: f"{i:.0%}",
                        lambda v: f"${v:,.0f}")

    # ── FEE x WAGES ──
    elif "Fee" in sub and "Wages" in sub:
        st.caption("Year 3 EBITDA across daily fee and educator hourly rate")

        base_rate = cfg["educator_hourly_rate"]
        with st.spinner("Computing grid..."):
            fees = [125, 130, 135, 139, 145, 155]
            wages = [round(base_rate * m, 2) for m in [0.85, 0.92, 1.0, 1.08, 1.15]]
            grid = sensitivity_fee_wages(cfg, fees, wages)

        _render_heatmap(grid, "Y3 EBITDA: Fee \u00d7 Educator Rate",
                        "Educator Rate ($/hr)", "Daily Fee",
                        lambda c: f"${c:.2f}", lambda i: f"${i:.0f}",
                        lambda v: f"${v:,.0f}")

    # ── DCF ──
    else:
        st.caption("DCF Enterprise Value across discount rate and terminal growth rate")

        with st.spinner("Computing DCF grid..."):
            drs = [0.08, 0.10, 0.12, 0.14, 0.16]
            grs = [0.015, 0.020, 0.025, 0.030, 0.035]
            grid = sensitivity_dcf(cfg, drs, grs)

        fig = go.Figure(data=go.Heatmap(
            z=grid.values,
            x=[f"{g:.1%}" for g in grid.columns],
            y=[f"{d:.0%}" for d in grid.index],
            colorscale=[[0, "#FEF3C7"], [0.5, "#D1FAE5"], [1, COLORS["accent"]]],
            text=[[f"${v / 1e6:.2f}M" for v in row] for row in grid.values],
            texttemplate="%{text}",
            textfont=dict(size=12, family="JetBrains Mono, monospace"),
            hovertemplate="DR: %{y} \u00b7 TGR: %{x}<br>EV: %{text}<extra></extra>",
        ))
        l = plotly_layout("DCF Enterprise Value", height=450, y_dollar=False)
        l["xaxis"] = dict(title="Terminal Growth Rate")
        l["yaxis"] = dict(title="Discount Rate")
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

        grid_d = grid.copy()
        grid_d.index = [f"{d:.0%}" for d in grid_d.index]
        grid_d.columns = [f"{g:.1%}" for g in grid_d.columns]
        st.dataframe(grid_d.style.format("${:,.0f}"), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    cfg, scenario_label, valid = build_sidebar()

    if not valid:
        st.error("\u26A0\uFE0F Fix the input errors in the sidebar before proceeding.")
        st.stop()

    scenario_key = {
        "Base": "base",
        "Bear (Downside)": "bear",
        "Bull (Upside)": "bull",
    }[scenario_label]

    with st.spinner("Running model..."):
        result = run_model(cfg, scenario_key)

    col_spacer, col_xlsx, col_csv = st.columns([4, 1, 1])
    with col_xlsx:
        try:
            xlsx_bytes = export_to_excel(result)
            st.download_button(
                "\u2B07 XLSX", data=xlsx_bytes,
                file_name="brightblocks_model_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except Exception:
            st.caption("XLSX needs openpyxl")
    with col_csv:
        try:
            zip_bytes = export_all_to_zip(result, cfg)
            st.download_button(
                "\u2B07 CSV (ZIP)", data=zip_bytes,
                file_name="brightblocks_csvs.zip",
                mime="application/zip",
                use_container_width=True)
        except Exception:
            pass

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001F4CA Dashboard",
        "\U0001F4CB Financials",
        "\U0001F500 Scenarios",
        "\U0001F3AF Sensitivity",
    ])

    with tab1:
        render_dashboard(result)

    with tab2:
        render_financials(result)

    with tab3:
        render_scenarios(cfg)

    with tab4:
        render_sensitivity(cfg)


if __name__ == "__main__":
    main()
