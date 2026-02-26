"""
BrightBlocks ELC — PE Financial Model Dashboard
Streamlit app. Phase 5 Finalisation: Isolated UI layout, strict pre-revenue controls, reporting.
Requires: streamlit, plotly, pandas, numpy
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from copy import deepcopy

from model import EARLWOOD_CONFIG, BEXLEY_CONFIG, run_single_centre

st.set_page_config(page_title="BrightBlocks ELC · PE Model", layout="wide", initial_sidebar_state="expanded")

COLORS = {
    "primary": "#0F2B46", "accent": "#1B9AAA", "accent2": "#06D6A0", "warn": "#F4845F",
    "danger": "#E63946", "bg": "#F8F9FC", "card": "#FFFFFF", "text": "#1A1A2E",
    "muted": "#6B7280", "revenue": "#1B9AAA", "costs": "#E63946", "ebitda": "#06D6A0",
    "cash": "#4361EE"
}

st.markdown("""
<style>
    .stApp { background: #F8F9FC; }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E5E7EB; }
    [data-testid="stSidebar"] * { color: #000000 !important; }
    [data-testid="stSidebar"] .stRadio label { font-weight: 600 !important; }
    .metric-card { background: #FFFFFF; border-radius: 8px; padding: 1.2rem; border: 1px solid #E5E7EB; margin-bottom: 1rem; }
    .metric-label { font-size: 0.75rem; font-weight: 600; color: #6B7280; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0F2B46; }
    .status-green { color: #06D6A0; } .status-red { color: #E63946; }
    .header-box { background: #0F2B46; color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; }
    .header-box h1, .header-box p { color: white !important; margin: 0; }
</style>
""", unsafe_allow_html=True)

def fmt_c(val): return f"${val:,.0f}" if pd.notnull(val) else "—"
def fmt_p(val): return f"{val*100:.1f}%" if pd.notnull(val) else "—"

def get_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=16, color=COLORS["primary"])),
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=50, b=30, l=50, r=20),
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"), yaxis=dict(showgrid=True, gridcolor="#F3F4F6")
    )

# ─────────────────────────────────────────────────────────────
# STATE INIT
# ─────────────────────────────────────────────────────────────
if "centres" not in st.session_state:
    st.session_state["centres"] = {
        "Earlwood": deepcopy(EARLWOOD_CONFIG),
        "Bexley": deepcopy(BEXLEY_CONFIG)
    }
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {
        "Base": {},
        "Upside": {"occ_mod": 1.1, "fee_mod": 1.05, "ect_mod": 1.0, "pre_rev_mod": 0},
        "Downside": {"occ_mod": 0.85, "fee_mod": 0.95, "ect_mod": 1.0, "pre_rev_mod": 1}
    }

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### BrightBlocks ELC")
selected_centre = st.sidebar.selectbox("Select Centre", list(st.session_state["centres"].keys()))
page = st.sidebar.radio("Navigation", [
    "Inputs (Base Case)", 
    "Dashboards", 
    "Scenarios", 
    "Centre Comparison", 
    "Financial Statements"
])

cfg = st.session_state["centres"][selected_centre]

# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────

if page == "Inputs (Base Case)":
    st.markdown(f"## Base Case Inputs: {selected_centre}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time & Pre-Revenue Controls")
        try: c_date = pd.to_datetime(cfg["commencement_date"]).date()
        except: c_date = pd.to_datetime("today").date()
        new_date = st.date_input("Commencement Date", value=c_date)
        cfg["commencement_date"] = new_date.strftime("%Y-%m-%d")
        
        cfg["pre_revenue_months"] = st.number_input("Pre-revenue Period (Months)", 0, 12, cfg.get("pre_revenue_months", 3))
        cfg["support_staff_unpaid_pre_revenue"] = st.checkbox("Support Staff Unpaid during Pre-revenue", value=cfg.get("support_staff_unpaid_pre_revenue", True))

    with col2:
        st.subheader("Capacity & Fees")
        cfg["approved_capacity"] = st.number_input("Approved Capacity", 10, 200, cfg["approved_capacity"])
        cfg["daily_fees"]["36m_plus"] = st.number_input("Blended Daily Fee ($)", 80.0, 250.0, cfg["daily_fees"]["36m_plus"])
    
    st.subheader("Staffing Inputs (Monthly FTEs)")
    st.caption("Adjust ECT and Support staff counts month-by-month. Post-revenue, capacity checks strictly apply based on ECT ratios.")
    
    df_staff = pd.DataFrame({
        "Month": range(1, 61),
        "ECT Staff": cfg.get("ect_staff_counts", [4]*60),
        "Support Staff": cfg.get("support_staff_counts", [1]*60)
    }).set_index("Month").T
    
    edited_staff = st.data_editor(df_staff, use_container_width=True)
    cfg["ect_staff_counts"] = edited_staff.loc["ECT Staff"].tolist()
    cfg["support_staff_counts"] = edited_staff.loc["Support Staff"].tolist()
    
    st.success("Inputs auto-saved to session state.")

elif page == "Dashboards":
    res = run_single_centre(cfg)
    pl = res["monthly_pl"]
    ann = res["annual_pl"]
    val = res["valuation"]
    cap = res["capital_sufficiency"]
    
    st.markdown(f"<div class='header-box'><h1>{cfg['centre_name']}</h1><p>Base Case Financial Dashboard</p></div>", unsafe_allow_html=True)
    
    # Capacity Check Warning
    cap_breaches = pl[~pl["meets_capacity_limits"]]
    if not cap_breaches.empty:
        st.error(f"⚠️ Capacity limits breached in {len(cap_breaches)} months! Check Staffing Inputs vs Occupancy.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Y5 Revenue</div><div class='metric-value'>{fmt_c(ann['net_revenue'].iloc[-1])}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>Y5 EBITDA</div><div class='metric-value status-green'>{fmt_c(val['y5_ebitda'])}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-label'>IRR</div><div class='metric-value'>{fmt_p(val['irr'])}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-label'>MOIC</div><div class='metric-value'>{val['moic']:.2f}x</div></div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=ann["fiscal_year"].astype(str), y=ann["net_revenue"], name="Revenue", marker_color=COLORS["revenue"]))
        fig1.add_trace(go.Scatter(x=ann["fiscal_year"].astype(str), y=ann["ebitda"], name="EBITDA", line=dict(color=COLORS["ebitda"], width=3)))
        fig1.update_layout(**get_layout("Revenue & EBITDA"))
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pl["date"], y=pl["occupancy_pct"]*100, name="Occupancy %", fill="tozeroy"))
        fig2.update_layout(**get_layout("Occupancy Ramp"))
        fig2.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Scenarios":
    st.markdown("## Scenario Analysis (Isolated)")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Upside Overrides")
        u_occ = st.slider("Occupancy Multiplier", 0.8, 1.2, st.session_state["scenarios"]["Upside"]["occ_mod"], key="u_occ")
        u_fee = st.slider("Fee Multiplier", 0.8, 1.2, st.session_state["scenarios"]["Upside"]["fee_mod"], key="u_fee")
    with col_s2:
        st.subheader("Downside Overrides")
        d_occ = st.slider("Occupancy Multiplier", 0.5, 1.0, st.session_state["scenarios"]["Downside"]["occ_mod"], key="d_occ")
        d_fee = st.slider("Fee Multiplier", 0.5, 1.0, st.session_state["scenarios"]["Downside"]["fee_mod"], key="d_fee")
    
    # Apply scenarios safely
    scen_results = {}
    for name, mods in [("Base", {"occ_mod":1, "fee_mod":1}), 
                       ("Upside", {"occ_mod":u_occ, "fee_mod":u_fee}), 
                       ("Downside", {"occ_mod":d_occ, "fee_mod":d_fee})]:
        c = deepcopy(cfg)
        c["occupancy_ramp"]["y3"] = min(1.0, c["occupancy_ramp"]["y3"] * mods["occ_mod"])
        c["occupancy_ramp"]["y4_plus"] = min(1.0, c["occupancy_ramp"]["y4_plus"] * mods["occ_mod"])
        c["daily_fees"]["36m_plus"] *= mods["fee_mod"]
        scen_results[name] = run_single_centre(c)
        
    res_data = []
    for k, v in scen_results.items():
        val = v["valuation"]
        res_data.append({
            "Scenario": k, "IRR": fmt_p(val["irr"]), "MOIC": f"{val['moic']:.2f}x",
            "Y5 EBITDA": fmt_c(val["y5_ebitda"]), "EV": fmt_c(val["ev_multiples"][list(val["ev_multiples"].keys())[1]])
        })
    st.dataframe(pd.DataFrame(res_data), use_container_width=True, hide_index=True)

elif page == "Centre Comparison":
    st.markdown("## Multi-Centre Comparison")
    comp_data = []
    for c_name, c_cfg in st.session_state["centres"].items():
        res = run_single_centre(c_cfg)
        ann = res["annual_pl"]
        val = res["valuation"]
        comp_data.append({
            "Centre": c_name, "Capacity": c_cfg["approved_capacity"],
            "Y5 Revenue": ann["net_revenue"].iloc[-1], "Y5 EBITDA": val["y5_ebitda"],
            "IRR": val["irr"], "MOIC": val["moic"]
        })
    
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp.style.format({
        "Y5 Revenue": "${:,.0f}", "Y5 EBITDA": "${:,.0f}", "IRR": "{:.1%}", "MOIC": "{:.2f}x"
    }), use_container_width=True, hide_index=True)
    
    fig = go.Figure()
    for _, row in df_comp.iterrows():
        fig.add_trace(go.Bar(name=row["Centre"], x=["Y5 Revenue", "Y5 EBITDA"], y=[row["Y5 Revenue"], row["Y5 EBITDA"]]))
    fig.update_layout(**get_layout("Financial Comparison"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Financial Statements":
    st.markdown(f"## Standard Financial Statements: {selected_centre}")
    res = run_single_centre(cfg)
    
    tab1, tab2, tab3 = st.tabs(["Profit & Loss", "Balance Sheet", "Cashflow"])
    
    with tab1:
        pl = res["annual_pl"].copy().rename(columns={"fiscal_year": "FY"})
        disp_cols = ["FY", "net_revenue", "gross_profit", "ebitda", "ebit", "npat"]
        st.dataframe(pl[disp_cols].style.format({c: "${:,.0f}" for c in disp_cols if c!="FY"}), use_container_width=True, hide_index=True)
    
    with tab2:
        bs = res["balance_sheet"].copy().rename(columns={"fiscal_year": "FY"})
        st.dataframe(bs.style.format({c: "${:,.0f}" for c in bs.columns if c!="FY"}), use_container_width=True, hide_index=True)
        
    with tab3:
        cf = res["cashflow"].groupby("fiscal_year").sum(numeric_only=True).reset_index().rename(columns={"fiscal_year": "FY"})
        cf["closing_cash"] = res["cashflow"].groupby("fiscal_year")["closing_cash"].last().values
        cf_cols = ["FY", "operating_cf", "investing_cf", "financing_cf", "net_cash_movement", "closing_cash"]
        st.dataframe(cf[cf_cols].style.format({c: "${:,.0f}" for c in cf_cols if c!="FY"}), use_container_width=True, hide_index=True)
