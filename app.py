"""
BrightBlocks ELC - PE Financial Model Dashboard (Refactored)
Layout: Sidebar = Base Case Inputs | Main = Tabs (Dashboard, Scenarios, Financial Statements)
State: st.session_state["base_config"][centre_id], st.session_state["scenario_configs"][centre_id]
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from copy import deepcopy
import io

from model import (
    EARLWOOD_CONFIG, SCENARIO_PRESETS, CENTRE_REGISTRY,
    run_single_centre, run_scenario, apply_scenario_overrides,
    run_scenario_comparison, sensitivity_occupancy_fee,
    sensitivity_occupancy_wages, sensitivity_fee_wages,
    sensitivity_dcf, tornado_sensitivity,
    scenario_comparison_table, run_portfolio,
    rent_stress_test, pe_dashboard, list_centres, get_centre_config,
    calc_annual_pl, calc_valuation,
    export_results_csv, export_scenario_comparison_csv, export_all_to_zip,
    staffing_blocks_to_monthly_override,
)

st.set_page_config(page_title="BrightBlocks ELC - PE Model", page_icon="\U0001F3EB",
                   layout="wide", initial_sidebar_state="expanded")

C = {"primary":"#0F2B46","accent":"#1B9AAA","accent2":"#06D6A0","warn":"#F4845F",
     "danger":"#E63946","bg":"#F8F9FC","card":"#FFFFFF","text":"#1A1A2E","muted":"#6B7280",
     "revenue":"#1B9AAA","costs":"#E63946","ebitda":"#06D6A0","cash":"#4361EE",
     "bear":"#E63946","base":"#1B9AAA","bull":"#06D6A0","gpmargin":"#8B5CF6",
     "staffing":"#F59E0B","rent":"#EC4899","food":"#10B981","opex":"#6366F1"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: #F8F9FC; }
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }
h1,h2,h3,h4,h5,h6 { font-family: 'DM Sans', sans-serif !important; color: #0F2B46 !important; }
p,li,span,div { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] * { color: #1A1A2E !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stDateInput label {
    color: #374151 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.8rem !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { color: #0F2B46 !important; }
[data-testid="stSidebar"] input { color: #1A1A2E !important; background: #F9FAFB !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #F9FAFB !important; color: #1A1A2E !important; }
.metric-card { background: white; border-radius: 12px; padding: 1.15rem 1.4rem; border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04); min-height: 110px; transition: box-shadow 0.2s; }
.metric-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.08); }
.metric-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #6B7280; margin-bottom: 0.3rem; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #0F2B46; }
.metric-sub { font-size: 0.78rem; color: #6B7280; margin-top: 0.2rem; }
.metric-green { color: #06D6A0 !important; }
.metric-red { color: #E63946 !important; }
.metric-blue { color: #4361EE !important; }
.header-bar { background: linear-gradient(135deg, #0F2B46 0%, #1a3a5c 60%, #234e78 100%);
    color: white; padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; }
.header-bar h1 { color: white !important; font-size: 1.5rem; margin: 0; }
.header-bar p { color: rgba(255,255,255,0.7); font-size: 0.82rem; margin: 0.25rem 0 0 0; }
.status-badge { display: inline-block; padding: 0.2rem 0.75rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.badge-green { background: #D1FAE5; color: #065F46; }
.badge-red { background: #FEE2E2; color: #991B1B; }
.badge-amber { background: #FEF3C7; color: #92400E; }
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: white; border-radius: 12px; padding: 0.3rem; border: 1px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 0.85rem; }
.section-divider { border: none; border-top: 1px solid #E5E7EB; margin: 1.5rem 0; }
.capacity-warn { background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.85rem; color: #92400E; }
.sidebar-section { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────

def fmt_currency(val, prefix="$", decimals=0):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "\u2014"
    if abs(val) >= 1e6: return f"{prefix}{val/1e6:,.{max(decimals,1)}f}M"
    if abs(val) >= 1e3: return f"{prefix}{val/1e3:,.{decimals}f}k"
    return f"{prefix}{val:,.{decimals}f}"

def fmt_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "\u2014"
    return f"{val*100:,.{decimals}f}%"

def metric_card(label, value, sub="", color_class=""):
    vc = f"metric-{color_class}" if color_class else ""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {vc}">{value}</div><div class="metric-sub">{sub}</div></div>'

def plotly_layout(title="", height=400, y_dollar=True):
    l = dict(title=dict(text=title, font=dict(family="DM Sans", size=15, color=C["primary"]), x=0),
        font=dict(family="DM Sans", size=11, color=C["text"]),
        plot_bgcolor="white", paper_bgcolor="white", height=height,
        margin=dict(l=55, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False, linecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False, linecolor="#E5E7EB"),
        hoverlabel=dict(bgcolor="white", bordercolor="#D1D5DB", font=dict(family="DM Sans", size=11)),
        hovermode="x unified")
    if y_dollar: l["yaxis"]["tickprefix"]="$"; l["yaxis"]["tickformat"]=",.0f"
    return l

def export_to_excel(result):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for k,s in [("annual_pl","Annual_PL"),("monthly_pl","Monthly_PL"),("cashflow","CashFlow"),
                     ("balance_sheet","Balance_Sheet"),("debt_schedule","Debt_Schedule"),
                     ("per_child","Per_Child"),("weekly_cashflow","Weekly_CashFlow")]:
            result[k].to_excel(w, sheet_name=s, index=False)
        gst=result["gst_bas"].copy(); gst["quarter"]=gst["quarter"].astype(str)
        gst.to_excel(w, sheet_name="GST_BAS", index=False)
    return buf.getvalue()


# ── Session State Initialisation ──────────────────────────────

def _sd(key, default):
    """session_state.setdefault wrapper."""
    return st.session_state.setdefault(key, default)

def init_state():
    """Initialise all persistent state containers."""
    _sd("base_config", {})
    _sd("scenario_configs", {})
    _sd("active_centre", None)
    _sd("active_tab", 0)

def get_base_config(centre_id):
    """Return persisted base config for centre, or initialise from registry."""
    store = st.session_state["base_config"]
    if centre_id not in store:
        store[centre_id] = get_centre_config(centre_id)
    return store[centre_id]

def get_scenario_configs(centre_id):
    """Return persisted scenario overrides for centre."""
    store = st.session_state["scenario_configs"]
    if centre_id not in store:
        store[centre_id] = {
            "bear": {},
            "base": {},
            "bull": {},
        }
    return store[centre_id]


# ── Sidebar: Centre Selector + Base Case Inputs ──────────────

def _sk(centre_id, field):
    """Generate a stable widget key for sidebar inputs."""
    return f"bc_{centre_id}_{field}"

def build_sidebar():
    with st.sidebar:
        st.markdown("### \U0001F3EB BrightBlocks ELC")
        st.markdown("---")

        centres = list_centres()
        centre_names = [c["name"] for c in centres]
        centre_ids = [c["id"] for c in centres]
        sel_idx = st.selectbox("Centre", range(len(centres)), format_func=lambda i: centre_names[i], key="centre_selector")
        centre_id = centre_ids[sel_idx]
        st.session_state["active_centre"] = centre_id

        cfg = get_base_config(centre_id)

        st.markdown("---")
        st.markdown("#### Timeline & Capacity")
        cfg["commencement_date"] = str(st.date_input("Centre Start Date",
            pd.Timestamp(cfg["commencement_date"]), key=_sk(centre_id,"start_date")))
        cfg["pre_revenue_months"] = st.number_input("Pre-Revenue (months)", 0, 12,
            cfg.get("pre_revenue_months",3), 1, key=_sk(centre_id,"pre_rev"))
        cfg["approved_capacity"] = st.number_input("Approved Capacity", 10, 200,
            cfg["approved_capacity"], 5, key=_sk(centre_id,"capacity"))
        cfg["model_months"] = st.number_input("Model Length (months)", 24, 120,
            cfg["model_months"], 12, key=_sk(centre_id,"model_months"))

        st.markdown("---")
        st.markdown("#### Revenue")
        cfg["daily_fees"]["36m_plus"] = st.number_input("Fee 36m+ ($)", 80.0, 250.0,
            cfg["daily_fees"]["36m_plus"], 1.0, key=_sk(centre_id,"fee36"))
        cfg["daily_fees"]["24_36m"] = st.number_input("Fee 24-36m ($)", 80.0, 250.0,
            cfg["daily_fees"]["24_36m"], 1.0, key=_sk(centre_id,"fee24"))
        cfg["daily_fees"]["0_24m"] = st.number_input("Fee 0-24m ($)", 80.0, 300.0,
            cfg["daily_fees"]["0_24m"], 1.0, key=_sk(centre_id,"fee0"))
        cfg["fee_annual_increase"] = st.number_input("Fee Increase (% p.a.)", 0.0, 10.0,
            cfg["fee_annual_increase"]*100, 0.5, key=_sk(centre_id,"fee_inc")) / 100
        cfg["revenue_collection_rate"] = st.number_input("Collection Rate (%)", 90.0, 100.0,
            cfg["revenue_collection_rate"]*100, 0.5, key=_sk(centre_id,"coll_rate")) / 100

        st.markdown("---")
        st.markdown("#### Occupancy Ramp")
        cfg["occupancy_ramp"]["m4"] = st.number_input("First Month Post-Rev (%)", 0.0, 100.0,
            cfg["occupancy_ramp"]["m4"]*100, 5.0, key=_sk(centre_id,"occ_m4")) / 100
        cfg["occupancy_ramp"]["m5_6"] = st.number_input("Months 2-6 Post-Rev (%)", 0.0, 100.0,
            cfg["occupancy_ramp"]["m5_6"]*100, 5.0, key=_sk(centre_id,"occ_m56")) / 100
        cfg["occupancy_ramp"]["y2"] = st.number_input("Year 2 (%)", 0.0, 100.0,
            cfg["occupancy_ramp"]["y2"]*100, 5.0, key=_sk(centre_id,"occ_y2")) / 100
        cfg["occupancy_ramp"]["y3"] = st.number_input("Year 3 (%)", 0.0, 100.0,
            cfg["occupancy_ramp"]["y3"]*100, 5.0, key=_sk(centre_id,"occ_y3")) / 100
        cfg["occupancy_ramp"]["y4_plus"] = st.number_input("Year 4+ (%)", 0.0, 100.0,
            cfg["occupancy_ramp"]["y4_plus"]*100, 2.5, key=_sk(centre_id,"occ_y4")) / 100

        st.markdown("---")
        st.markdown("#### Staffing Rates")
        cfg["educator_hourly_rate"] = st.number_input("Educator Hourly ($)", 25.0, 80.0,
            cfg["educator_hourly_rate"], 0.5, key=_sk(centre_id,"ed_rate"))
        cfg["support_hourly_rate"] = st.number_input("Support Hourly ($)", 20.0, 60.0,
            cfg["support_hourly_rate"], 0.5, key=_sk(centre_id,"sup_rate"))
        cfg["director_hourly_rate"] = st.number_input("Director Hourly ($)", 30.0, 100.0,
            cfg["director_hourly_rate"], 1.0, key=_sk(centre_id,"dir_rate"))
        cfg["staff_wage_annual_increase"] = st.number_input("Wage Increase (% p.a.)", 0.0, 10.0,
            cfg["staff_wage_annual_increase"]*100, 0.5, key=_sk(centre_id,"wage_inc")) / 100
        cfg["superannuation_rate"] = st.number_input("Super Rate (%)", 9.0, 15.0,
            cfg["superannuation_rate"]*100, 0.5, key=_sk(centre_id,"super")) / 100

        # ── Structured Staffing Blocks ──
        st.markdown("---")
        st.markdown("#### Staffing Overrides")
        st.caption("Pre-revenue & ramp-up staffing levels")

        # ECT blocks
        st.markdown("**ECT Staff**")
        _sd(f"ect_blocks_{centre_id}", [{"count": 1, "months": cfg.get("pre_revenue_months",3)}])
        ect_blocks = st.session_state[f"ect_blocks_{centre_id}"]
        new_ect_blocks = []
        for bi, block in enumerate(ect_blocks):
            ec1, ec2 = st.columns(2)
            with ec1:
                cnt = st.number_input("# ECT", 0, 20, int(block.get("count",1)), 1,
                    key=_sk(centre_id, f"ect_cnt_{bi}"))
            with ec2:
                dur = st.number_input("Months", 1, 24, int(block.get("months",3)), 1,
                    key=_sk(centre_id, f"ect_dur_{bi}"))
            new_ect_blocks.append({"count": cnt, "months": dur})
        if st.button("+ ECT block", key=_sk(centre_id, "ect_add")):
            new_ect_blocks.append({"count": 1, "months": 3})
        st.session_state[f"ect_blocks_{centre_id}"] = new_ect_blocks

        # Support blocks
        st.markdown("**Support Staff**")
        _sd(f"sup_blocks_{centre_id}", [{"count": 0, "months": cfg.get("pre_revenue_months",3)}])
        sup_blocks = st.session_state[f"sup_blocks_{centre_id}"]
        new_sup_blocks = []
        for bi, block in enumerate(sup_blocks):
            sc1, sc2 = st.columns(2)
            with sc1:
                cnt = st.number_input("# Support", 0, 20, int(block.get("count",0)), 1,
                    key=_sk(centre_id, f"sup_cnt_{bi}"))
            with sc2:
                dur = st.number_input("Months", 1, 24, int(block.get("months",3)), 1,
                    key=_sk(centre_id, f"sup_dur_{bi}"))
            new_sup_blocks.append({"count": cnt, "months": dur})
        if st.button("+ Support block", key=_sk(centre_id, "sup_add")):
            new_sup_blocks.append({"count": 0, "months": 3})
        st.session_state[f"sup_blocks_{centre_id}"] = new_sup_blocks

        # Director toggle
        cfg["director_salary_deferred"] = st.checkbox("Director Salary Deferred",
            value=cfg.get("director_salary_deferred", True), key=_sk(centre_id,"dir_deferred"))

        # Convert staffing blocks → monthly overrides
        cfg["ect_monthly_override"] = staffing_blocks_to_monthly_override(new_ect_blocks, start_month=1)
        cfg["support_monthly_override"] = staffing_blocks_to_monthly_override(new_sup_blocks, start_month=1)

        st.markdown("---")
        st.markdown("#### Lease & Occupancy")
        cfg["base_rent_pa"] = st.number_input("Base Rent ($ p.a.)", 50000.0, 500000.0,
            cfg["base_rent_pa"], 5000.0, key=_sk(centre_id,"rent"))
        cfg["rent_escalation"] = st.number_input("Rent Escalation (%)", 0.0, 10.0,
            cfg["rent_escalation"]*100, 0.5, key=_sk(centre_id,"rent_esc")) / 100
        cfg["rent_free_months"] = st.number_input("Rent-Free Months", 0, 24,
            cfg["rent_free_months"], 1, key=_sk(centre_id,"rent_free"))
        cfg["cpi_on_expenses"] = st.number_input("CPI on Expenses (%)", 0.0, 8.0,
            cfg["cpi_on_expenses"]*100, 0.25, key=_sk(centre_id,"cpi")) / 100

        st.markdown("---")
        st.markdown("#### Capital & Valuation")
        cfg["capital_items"]["operational_seed"] = st.number_input("Operational Seed ($)", 10000.0, 500000.0,
            float(cfg["capital_items"]["operational_seed"]), 5000.0, key=_sk(centre_id,"opseed"))
        cfg["director_loan_interest_rate"] = st.number_input("Director Loan Rate (%)", 0.0, 15.0,
            cfg["director_loan_interest_rate"]*100, 0.5, key=_sk(centre_id,"loan_rate")) / 100
        cfg["discount_rate"] = st.number_input("Discount Rate (%)", 5.0, 25.0,
            cfg["discount_rate"]*100, 0.5, key=_sk(centre_id,"disc_rate")) / 100
        cfg["terminal_growth_rate"] = st.number_input("Terminal Growth (%)", 0.0, 5.0,
            cfg["terminal_growth_rate"]*100, 0.25, key=_sk(centre_id,"term_g")) / 100
        exit_mult = st.number_input("Exit Multiple (x EBITDA)", 2.0, 15.0,
            cfg["exit_multiples"][1], 0.5, key=_sk(centre_id,"exit_mult"))
        cfg["exit_multiples"] = [exit_mult - 1.0, exit_mult, exit_mult + 1.5]
        cfg["company_tax_rate"] = st.number_input("Company Tax Rate (%)", 0.0, 40.0,
            cfg["company_tax_rate"]*100, 1.0, key=_sk(centre_id,"tax_rate")) / 100

        # Persist back
        st.session_state["base_config"][centre_id] = cfg

        # Validation
        errors = []
        if cfg["approved_capacity"] < 1: errors.append("Capacity must be >= 1")
        if cfg["discount_rate"] <= cfg["terminal_growth_rate"]: errors.append("Discount rate must exceed terminal growth")
        if any(cfg["daily_fees"][k] <= 0 for k in cfg["daily_fees"]): errors.append("Daily fees must be > $0")

    return centre_id, cfg, errors


# ── Dashboard Page ────────────────────────────────────────────

def render_dashboard(result):
    cfg = result["config"]; pl = result["monthly_pl"]; annual = result["annual_pl"]
    val = result["valuation"]; cap = result["capital_sufficiency"]

    st.markdown(f'<div class="header-bar"><h1>{result["centre_name"]}</h1>'
        f'<p>{cfg["entity"]} | ACN {cfg["acn"]} | {cfg["approved_capacity"]} places | '
        f'Start {cfg["commencement_date"]} | Pre-rev {cfg.get("pre_revenue_months",3)}m</p></div>',
        unsafe_allow_html=True)

    # Capacity compliance warning banner
    breaches = pl[~pl["meets_capacity_limits"]]
    if len(breaches) > 0:
        months_list = ", ".join(str(int(m)) for m in breaches["month"].values[:10])
        st.markdown(f'<div class="capacity-warn">\u26A0\uFE0F Capacity compliance breached in months: '
            f'{months_list}. ECT staffing below required ratio.</div>', unsafe_allow_html=True)

    # KPI cards
    irr_s = fmt_pct(val["irr"]) if val["irr"] else "N/A"
    pb_s = f'{val["payback_years"]:.1f}yr' if val["payback_years"] else "\u2014"
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(metric_card("IRR", irr_s, f"MOIC {val['moic']:.2f}x",
        "green" if val.get("irr") and val["irr"]>0.15 else ""), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Y5 EBITDA", fmt_currency(val["y5_ebitda"]),
        f"Margin {fmt_pct(annual.iloc[-1]['ebitda_margin'])}"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("EV (DCF)", fmt_currency(val["ev_dcf"]),
        f"Payback {pb_s}"), unsafe_allow_html=True)
    with c4:
        badge = "badge-green" if cap["sufficient"] else "badge-red"
        wr = f'{cap["weeks_runway"]:.0f}wk' if cap["weeks_runway"] < 1000 else "Strong"
        st.markdown(metric_card("Capital", fmt_currency(cap["total_capital_committed"]),
            f'<span class="status-badge {badge}">{"OK" if cap["sufficient"] else "SHORT"}</span> Runway {wr}'),
            unsafe_allow_html=True)
    with c5:
        pre_rev = cfg.get("pre_revenue_months",3)
        st.markdown(metric_card("Pre-Revenue", f"{pre_rev}m",
            f"Burn {fmt_currency(cap.get('pre_revenue_burn',0))}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue vs Costs + Cash Position
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=annual["fiscal_year"].astype(str), y=annual["net_revenue"],
            name="Revenue", marker_color=C["revenue"]))
        fig.add_trace(go.Bar(x=annual["fiscal_year"].astype(str),
            y=annual["direct_costs"]+annual["total_overheads"],
            name="Total Costs", marker_color=C["costs"], opacity=0.6))
        fig.add_trace(go.Scatter(x=annual["fiscal_year"].astype(str), y=annual["ebitda"],
            name="EBITDA", line=dict(color=C["ebitda"],width=3), mode="lines+markers"))
        fig.update_layout(**plotly_layout("Revenue vs Costs vs EBITDA"))
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cf = result["cashflow"]
        fig2 = go.Figure()
        cf_x = cf["date"] if "date" in cf.columns else cf.index
        fig2.add_trace(go.Scatter(x=cf_x, y=cf["closing_cash"], fill="tozeroy",
            fillcolor="rgba(67,97,238,0.08)", line=dict(color=C["cash"],width=2.5), name="Closing Cash"))
        reserve = cfg["minimum_operating_reserve"]
        fig2.add_hline(y=reserve, line_dash="dash", line_color="#9CA3AF",
            annotation_text=f"Reserve ${reserve:,.0f}")
        l2 = plotly_layout("Cash Position"); l2["xaxis"]["tickformat"]="%b %Y"
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    # Occupancy + Waterfall
    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=pl["date"], y=pl["occupancy_pct"]*100, fill="tozeroy",
            fillcolor="rgba(27,154,170,0.08)", line=dict(color=C["accent"],width=2.5), name="Occupancy %"))
        l3 = plotly_layout("Occupancy Ramp", y_dollar=False)
        l3["yaxis"]["ticksuffix"]="%"; l3["yaxis"]["range"]=[0,105]
        l3["xaxis"]["tickformat"]="%b %Y"
        fig3.update_layout(**l3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        total_cap = val["total_capital"]
        exit_ev = val["y5_ebitda"]*cfg["exit_multiples"][1]
        cumul_fcf = sum(val["annual_fcf"])
        fig_w = go.Figure(go.Waterfall(
            x=["Capital","Cumul FCF","Exit","Total Return"],
            y=[-total_cap, cumul_fcf, exit_ev, 0],
            measure=["absolute","relative","relative","total"],
            increasing=dict(marker_color=C["accent2"]),
            decreasing=dict(marker_color=C["danger"]),
            totals=dict(marker_color=C["cash"]),
            text=[fmt_currency(-total_cap),fmt_currency(cumul_fcf),
                  fmt_currency(exit_ev),fmt_currency(exit_ev+cumul_fcf-total_cap)],
            textposition="outside", textfont=dict(family="JetBrains Mono",size=11)))
        fig_w.update_layout(**plotly_layout("Returns Waterfall"))
        fig_w.update_layout(showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    # Staffing headcount + Unit economics
    col_e, col_f = st.columns(2)
    with col_e:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=pl["date"], y=pl["educators"], name="ECT",
            line=dict(color=C["staffing"],width=2)))
        fig5.add_trace(go.Scatter(x=pl["date"], y=pl["support_staff"], name="Support",
            line=dict(color=C["accent"],width=2)))
        fig5.add_trace(go.Scatter(x=pl["date"], y=pl["director_active"], name="Director",
            line=dict(color=C["primary"],width=2,dash="dot")))
        l5 = plotly_layout("Staff Headcount", y_dollar=False)
        l5["xaxis"]["tickformat"]="%b %Y"
        fig5.update_layout(**l5)
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        pc = result["per_child"]
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=pc["fiscal_year"].astype(str), y=pc["net_revenue_per_child"],
            name="Revenue/Child", marker_color=C["revenue"]))
        fig6.add_trace(go.Bar(x=pc["fiscal_year"].astype(str), y=pc["ebitda_per_child"],
            name="EBITDA/Child", marker_color=C["ebitda"]))
        fig6.update_layout(**plotly_layout("Unit Economics per Child"))
        fig6.update_layout(barmode="group")
        st.plotly_chart(fig6, use_container_width=True)


# ── Scenarios Page ────────────────────────────────────────────

def render_scenarios(cfg, centre_id):
    st.markdown('<div class="header-bar"><h1>\U0001F500 Scenario Analysis</h1>'
        '<p>Isolated scenario assumptions. Changes here do NOT affect Base Case.</p></div>',
        unsafe_allow_html=True)

    scn_store = get_scenario_configs(centre_id)

    tabs_s = st.tabs(["\U0001F43B Bear", "\U0001F4CA Base", "\U0001F402 Bull"])
    for tab, key in zip(tabs_s, ["bear","base","bull"]):
        with tab:
            preset = SCENARIO_PRESETS.get(key, {})
            merged = apply_scenario_overrides(cfg, preset)
            # Layer any previously saved scenario overrides
            prev = scn_store.get(key, {})
            for pk, pv in prev.items():
                if pk in merged and isinstance(merged[pk], dict) and isinstance(pv, dict):
                    merged[pk].update(pv)
                elif pk != "label":
                    merged[pk] = pv

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**Occupancy**")
                occ_m4 = st.number_input("M4 Occ %", 0.0, 100.0,
                    merged["occupancy_ramp"]["m4"]*100, 5.0, key=f"s_{centre_id}_{key}_m4") / 100
                occ_y3 = st.number_input("Y3 Occ %", 0.0, 100.0,
                    merged["occupancy_ramp"]["y3"]*100, 5.0, key=f"s_{centre_id}_{key}_y3") / 100
                occ_y4 = st.number_input("Y4+ Occ %", 0.0, 100.0,
                    merged["occupancy_ramp"]["y4_plus"]*100, 5.0, key=f"s_{centre_id}_{key}_y4") / 100
            with c2:
                st.markdown("**Fees**")
                fee = st.number_input("Fee 36m+ ($)", 80.0, 250.0,
                    merged["daily_fees"]["36m_plus"], 1.0, key=f"s_{centre_id}_{key}_fee")
                fi = st.number_input("Fee Esc (%)", 0.0, 10.0,
                    merged["fee_annual_increase"]*100, 0.5, key=f"s_{centre_id}_{key}_fi") / 100
            with c3:
                st.markdown("**Wages**")
                wr = st.number_input("Ed. Rate ($)", 25.0, 80.0,
                    merged["educator_hourly_rate"], 0.5, key=f"s_{centre_id}_{key}_wr")
                wi = st.number_input("Wage Esc (%)", 0.0, 10.0,
                    merged["staff_wage_annual_increase"]*100, 0.5, key=f"s_{centre_id}_{key}_wi") / 100
            with c4:
                st.markdown("**Timeline**")
                sd = str(st.date_input("Start Date", pd.Timestamp(merged["commencement_date"]),
                    key=f"s_{centre_id}_{key}_sd"))
                prm = st.number_input("Pre-Rev Months", 0, 12,
                    merged.get("pre_revenue_months",3), 1, key=f"s_{centre_id}_{key}_prm")

            # Staffing overrides for scenarios
            st.markdown("**Staffing Overrides**")
            sc_a, sc_b = st.columns(2)
            with sc_a:
                ect_ov_count = st.number_input("ECT during pre-rev", 0, 20, 1,
                    key=f"s_{centre_id}_{key}_ect_cnt")
            with sc_b:
                sup_ov_count = st.number_input("Support during pre-rev", 0, 20, 0,
                    key=f"s_{centre_id}_{key}_sup_cnt")

            ect_scn_override = staffing_blocks_to_monthly_override(
                [{"count": ect_ov_count, "months": prm}], start_month=1) if ect_ov_count > 0 else None
            sup_scn_override = staffing_blocks_to_monthly_override(
                [{"count": sup_ov_count, "months": prm}], start_month=1) if sup_ov_count > 0 else None

            overrides = {
                "occupancy_ramp": {"m4":occ_m4, "y3":occ_y3, "y4_plus":occ_y4},
                "daily_fees": {"36m_plus":fee},
                "fee_annual_increase": fi,
                "educator_hourly_rate": wr,
                "staff_wage_annual_increase": wi,
                "commencement_date": sd,
                "pre_revenue_months": prm,
            }
            if ect_scn_override:
                overrides["ect_monthly_override"] = ect_scn_override
            if sup_scn_override:
                overrides["support_monthly_override"] = sup_scn_override

            scn_store[key] = overrides

    st.session_state["scenario_configs"][centre_id] = scn_store

    # Run scenarios
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Scenario Comparison")

    scenarios_data = {}
    for key in ["bear","base","bull"]:
        merged_preset = SCENARIO_PRESETS.get(key, {}).copy()
        merged_preset.update(scn_store.get(key, {}))
        scenarios_data[key] = run_scenario(cfg, "base", custom_overrides=merged_preset)

    # KPI cards
    cols = st.columns(3)
    for col, (key, icon, color) in zip(cols,
            [("bear","\U0001F43B","red"),("base","\U0001F4CA",""),("bull","\U0001F402","green")]):
        with col:
            v = scenarios_data[key]["valuation"]
            irr_s = fmt_pct(v["irr"]) if v["irr"] else "N/A"
            cap_s = scenarios_data[key]["capital_sufficiency"]
            wr = f'{cap_s["weeks_runway"]:.0f}wk' if cap_s["weeks_runway"]<1000 else "\u221E"
            st.markdown(metric_card(
                f"{icon} {SCENARIO_PRESETS.get(key,{}).get('label',key.title())}",
                f"IRR {irr_s}",
                f"MOIC {v['moic']:.2f}x | Y5E {fmt_currency(v['y5_ebitda'])} | Runway {wr}",
                color), unsafe_allow_html=True)

    # EBITDA chart
    fig = go.Figure()
    for key, color in [("bear",C["bear"]),("base",C["base"]),("bull",C["bull"])]:
        a = scenarios_data[key]["annual_pl"]
        fig.add_trace(go.Scatter(x=a["fiscal_year"].astype(str), y=a["ebitda"],
            name=SCENARIO_PRESETS.get(key,{}).get("label",key.title()),
            line=dict(color=color,width=3), mode="lines+markers"))
    fig.update_layout(**plotly_layout("EBITDA by Scenario"))
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    rows = []
    for key in ["bear","base","bull"]:
        v = scenarios_data[key]["valuation"]; a = scenarios_data[key]["annual_pl"]
        cap_s = scenarios_data[key]["capital_sufficiency"]
        rows.append({"Scenario":SCENARIO_PRESETS.get(key,{}).get("label",key.title()),
            "Y1 Rev":a.iloc[0]["net_revenue"], "Y5 Rev":a.iloc[-1]["net_revenue"],
            "Y5 EBITDA":v["y5_ebitda"], "Y5 Margin":a.iloc[-1]["ebitda_margin"],
            "EV (DCF)":v["ev_dcf"], "IRR":v["irr"] if v["irr"] else 0,
            "MOIC":v["moic"], "Cash Runway":cap_s["weeks_runway"]})
    summary = pd.DataFrame(rows)
    st.dataframe(summary.style.format({
        "Y1 Rev":"${:,.0f}","Y5 Rev":"${:,.0f}","Y5 EBITDA":"${:,.0f}",
        "Y5 Margin":"{:.1%}","EV (DCF)":"${:,.0f}","IRR":"{:.1%}",
        "MOIC":"{:.2f}x","Cash Runway":"{:.0f} wks"}),
        use_container_width=True, hide_index=True)

    csv_data = export_scenario_comparison_csv(cfg)
    st.download_button("\u2B07 Export Scenario CSV", data=csv_data,
        file_name="scenario_comparison.csv", mime="text/csv")


# ── Financial Statements Page ─────────────────────────────────

def render_financial_statements(result):
    st.markdown('<div class="header-bar"><h1>\U0001F4CB Financial Statements</h1>'
        '<p>Standard format P&L, Balance Sheet, and Cash Flow Statement</p></div>',
        unsafe_allow_html=True)

    cfg = result["config"]
    col_x, col_csv, col_xlsx = st.columns([4,1,1])
    with col_csv:
        try:
            z = export_all_to_zip(result, cfg)
            st.download_button("\u2B07 CSV (ZIP)", data=z, file_name="financials.zip",
                mime="application/zip", use_container_width=True)
        except: pass
    with col_xlsx:
        try:
            xb = export_to_excel(result)
            st.download_button("\u2B07 XLSX", data=xb, file_name="financials.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except: st.caption("Needs openpyxl")

    sub = st.radio("Statement", ["Profit & Loss","Balance Sheet","Cash Flow Statement",
        "Monthly Detail","Debt Schedule","Weekly Cash Flow","GST / BAS"],
        horizontal=True, label_visibility="collapsed")

    def fmt_statement(df):
        styled = df.copy()
        fy_cols = [c for c in styled.columns if c.startswith("FY")]
        fmt_dict = {c: lambda v: f"${v:,.0f}" if pd.notna(v) and isinstance(v,(int,float)) else ""
            for c in fy_cols}
        return styled.style.format(fmt_dict)

    if sub == "Profit & Loss":
        st.markdown("### Profit & Loss Statement")
        st.dataframe(fmt_statement(result["formatted_pl"]),
            use_container_width=True, hide_index=True, height=580)

    elif sub == "Balance Sheet":
        st.markdown("### Balance Sheet")
        st.dataframe(fmt_statement(result["formatted_bs"]),
            use_container_width=True, hide_index=True, height=650)

    elif sub == "Cash Flow Statement":
        st.markdown("### Cash Flow Statement")
        st.dataframe(fmt_statement(result["formatted_cf"]),
            use_container_width=True, hide_index=True, height=650)

    elif sub == "Monthly Detail":
        st.markdown("### Monthly P&L")
        df = result["monthly_pl"].copy()
        df["date"] = df["date"].dt.strftime("%Y-%m")
        cols = ["date","net_revenue","total_staff_cost","food_cost","gross_profit",
                "total_occupancy_cost","total_opex","ebitda","npat","children","occupancy_pct",
                "educators","support_staff","director_active","meets_capacity_limits"]
        out = df[cols].rename(columns={"date":"Month","net_revenue":"Revenue",
            "total_staff_cost":"Staff","food_cost":"Food","gross_profit":"GP",
            "total_occupancy_cost":"Rent & Occ","total_opex":"Opex","ebitda":"EBITDA",
            "npat":"NPAT","children":"Children","occupancy_pct":"Occ %","educators":"ECT",
            "support_staff":"Support","director_active":"Dir","meets_capacity_limits":"Cap OK"})
        st.dataframe(out.style.format({"Revenue":"${:,.0f}","Staff":"${:,.0f}","Food":"${:,.0f}",
            "GP":"${:,.0f}","Rent & Occ":"${:,.0f}","Opex":"${:,.0f}","EBITDA":"${:,.0f}",
            "NPAT":"${:,.0f}","Occ %":"{:.0%}"}),
            use_container_width=True, hide_index=True, height=600)

    elif sub == "Debt Schedule":
        st.markdown("### Director Loan Schedule")
        ds = result["debt_schedule"].copy()
        ds["date"] = ds["date"].dt.strftime("%Y-%m")
        st.dataframe(ds.rename(columns={"date":"Month","opening_balance":"Opening",
            "interest":"Interest","repay_eligible":"Can Repay","repayment":"Repayment",
            "closing_balance":"Closing"}).style.format(
            {"Opening":"${:,.0f}","Interest":"${:,.0f}","Repayment":"${:,.0f}","Closing":"${:,.0f}"}),
            use_container_width=True, hide_index=True, height=600)

    elif sub == "Weekly Cash Flow":
        st.markdown("### Weekly Cash Flow (First 26 Weeks)")
        wk = result["weekly_cashflow"].copy()
        def rag_color(val):
            if val == "RED": return "background-color: #FEE2E2"
            elif val == "AMBER": return "background-color: #FEF3C7"
            return "background-color: #D1FAE5"
        st.dataframe(wk.style.format({"revenue":"${:,.0f}","wages":"${:,.0f}",
            "outgoings":"${:,.0f}","other_opex":"${:,.0f}","net_movement":"${:,.0f}",
            "cumulative_cash":"${:,.0f}","buffer_above_reserve":"${:,.0f}"}).map(
            rag_color, subset=["rag_status"]),
            use_container_width=True, hide_index=True)

    elif sub == "GST / BAS":
        st.markdown("### GST / BAS Quarterly Summary")
        gst = result["gst_bas"].copy(); gst["quarter"] = gst["quarter"].astype(str)
        st.dataframe(gst.style.format({"gst_collected":"${:,.0f}","gst_paid_inputs":"${:,.0f}",
            "net_gst_position":"${:,.0f}","refund":"${:,.0f}"}),
            use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────

def main():
    init_state()
    centre_id, cfg, errors = build_sidebar()

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Main page tabs
    tab_dash, tab_scen, tab_fin = st.tabs([
        "\U0001F4CA Dashboard", "\U0001F500 Scenarios", "\U0001F4CB Financial Statements"])

    with tab_dash:
        with st.spinner("Running model..."):
            result = run_single_centre(deepcopy(cfg))
        render_dashboard(result)

    with tab_scen:
        render_scenarios(deepcopy(cfg), centre_id)

    with tab_fin:
        with st.spinner("Running model..."):
            result = run_single_centre(deepcopy(cfg))
        render_financial_statements(result)

if __name__ == "__main__":
    main()
