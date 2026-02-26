"""
BrightBlocks ELC — Childcare PE Financial Model Engine
Converts Excel model to parameterised Python. No UI code.
Phase 5 Finalisation: Parameterised staffing, pre-revenue controls, strict capacity checks.
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import io

def _irr(cashflows, tol=1e-8, max_iter=1000):
    """Newton's method IRR solver"""
    if not cashflows or all(c == 0 for c in cashflows):
        return None
    signs = [1 if c >= 0 else -1 for c in cashflows if c != 0]
    if len(set(signs)) < 2:
        return None
    r = 0.10
    for _ in range(max_iter):
        npv = sum(c / (1 + r) ** i for i, c in enumerate(cashflows))
        dnpv = sum(-i * c / (1 + r) ** (i + 1) for i, c in enumerate(cashflows))
        if abs(dnpv) < 1e-14:
            break
        r_new = r - npv / dnpv
        if r_new <= -1.0:
            r_new = -0.99
        if abs(r_new) > 1e6:
            return None
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    final_npv = sum(c / (1 + r) ** i for i, c in enumerate(cashflows))
    if abs(final_npv) > 1000:
        return None
    return r

# ─────────────────────────────────────────────────────────────
# DEFAULT CENTRE CONFIGS
# ─────────────────────────────────────────────────────────────

EARLWOOD_CONFIG = {
    "centre_id": "earlwood_001",
    "centre_name": "Bright Blocks ELC – Earlwood",
    "entity": "Bright Blocks ELC – Earlwood Pty Ltd",
    "acn": "685 365 253",
    "address": "Ground Floor, 205 Homer Street, Earlwood NSW 2206",
    "state": "NSW",

    "approved_capacity": 40,
    "commencement_date": "2026-07-01",
    "pre_revenue_months": 3,
    "model_months": 60,
    "biz_days_per_month": 21.67,

    "daily_fees": {"0_24m": 145.0, "24_36m": 142.0, "36m_plus": 139.0},
    "fee_mix_weights": {"0_24m": 0.0, "24_36m": 0.0, "36m_plus": 1.0},
    "fee_annual_increase": 0.03,
    "revenue_collection_rate": 0.98,
    "ccs_eligibility_pct": 0.75,
    "ccs_subsidy_rate": 0.80,
    "ccs_lag_days": 14,

    "seasonality_toggle": False,
    "seasonality_factors": {
        1: 0.85, 2: 0.95, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.95,
        7: 0.90, 8: 0.95, 9: 1.0, 10: 1.0, 11: 1.0, 12: 0.85,
    },

    "occupancy_ramp": {
        "m1_3": 0.0, "m4": 0.70, "m5_6": 0.75, "y2": 0.85, "y3": 0.95, "y4_plus": 0.975, "steady_state": 0.95,
    },

    "director_hourly_rate": 50.0,
    "director_annual_cost": 100350.0,
    "educator_hourly_rate": 38.59,
    "support_hourly_rate": 35.0,
    "hours_per_month": 164.6667,
    "staff_wage_annual_increase": 0.03,
    "superannuation_rate": 0.115,
    "workers_comp_rate": 0.017,
    "educator_child_ratio": 10,
    "staff_turnover_rate": 0.20,
    "recruitment_cost_per_hire": 3000.0,
    
    # Pre-revenue and manual staffing controls
    "ect_staff_counts": [2]*3 + [4]*57,
    "support_staff_counts": [1]*60,
    "support_staff_unpaid_pre_revenue": True,
    
    "director_salary_deferred": True,

    "base_rent_pa": 230000.0,
    "rent_escalation": 0.03,
    "rent_free_months": 6,
    "rent_free_credit": 115000.0,
    "initial_lease_term_years": 5,
    "make_good_provision_pa": 15000.0,

    "outgoings": {
        "strata": 1250, "electricity": 500, "gas": 333, "water": 58,
        "council_rates": 350, "waste": 167, "r_and_m": 500,
    },
    "land_tax_monthly_pre_approval": 1283.0,

    "insurance": {
        "public_liability": 200, "professional_indemnity": 175,
        "management_liability": 150, "contents": 100, "business_interruption": 150,
    },
    "insurance_escalation_rate": 0.06,

    "admin_opex": {
        "accounting": 978, "bank_fees": 50, "cleaning": 1500, "software": 400,
        "pest_control": 83, "phone_internet": 200, "toys_books": 250,
        "training": 500, "marketing": 3300, "extra_curricular": 1000,
        "miscellaneous": 500, "compliance": 250,
    },
    "marketing_pre_opening": 1500.0,
    "cpi_on_expenses": 0.02,

    "food_cost_per_child_per_day": 8.50,
    "food_delivery_monthly": 200.0,

    "capital_items": {
        "bank_guarantee": 126500, "legal_professional": 23000, "signage": 5000,
        "marketing_setup": 23009, "other_pre_opening": 18709,
        "loose_furniture": 35000, "operational_seed": 75000,
    },
    "total_deployed_director_loans": 196218.0,
    "minimum_operating_reserve": 20000.0,
    "maintenance_capex_pa": 5000.0,

    "gst_rate": 0.10,
    "company_tax_rate": 0.25,
    "discount_rate": 0.12,
    "terminal_growth_rate": 0.025,
    "exit_multiples": [4.5, 5.5, 7.0],
    "depreciation_furniture_total": 35000.0,
    "amortisation_pre_opening_total": 196218.0,
    "depreciation_years": 5,

    "director_loan_opening": 196218.0,
    "director_loan_interest_rate": 0.05,
    "director_loan_repayment_monthly": 5000.0,
}

BEXLEY_CONFIG = deepcopy(EARLWOOD_CONFIG)
BEXLEY_CONFIG.update({
    "centre_id": "bexley_002",
    "centre_name": "Bright Blocks ELC – Bexley",
    "approved_capacity": 60,
    "base_rent_pa": 310000.0,
    "ect_staff_counts": [2]*3 + [6]*57,
})


def build_time_series(cfg: dict) -> pd.DataFrame:
    start = pd.Timestamp(cfg["commencement_date"])
    months = cfg["model_months"]
    dates = pd.date_range(start, periods=months, freq="MS")
    df = pd.DataFrame({
        "month": range(1, months + 1),
        "date": dates,
        "calendar_month": dates.month,
        "fiscal_year": [(d.year + 1 if d.month >= 7 else d.year) for d in dates],
    })
    year_boundaries = []
    for i, d in enumerate(dates):
        if i == 0:
            year_boundaries.append(0)
        else:
            delta_months = (d.year - dates[0].year) * 12 + d.month - dates[0].month
            year_boundaries.append(delta_months // 12)
    df["year_from_start"] = year_boundaries
    return df


def calc_occupancy(cfg: dict, ts: pd.DataFrame) -> pd.DataFrame:
    ramp = cfg["occupancy_ramp"]
    cap = cfg["approved_capacity"]
    pre_rev = cfg.get("pre_revenue_months", 0)
    
    occ_pcts = []
    for _, row in ts.iterrows():
        m = row["month"]
        if m <= pre_rev:
            occ = 0.0
        else:
            adj_m = m - pre_rev
            if adj_m <= 3: occ = ramp["m1_3"]
            elif adj_m == 4: occ = ramp["m4"]
            elif adj_m <= 6: occ = ramp["m5_6"]
            elif adj_m <= 12: occ = ramp["m5_6"]
            elif adj_m <= 24: occ = ramp["y2"]
            elif adj_m <= 36: occ = ramp["y3"]
            else: occ = ramp["y4_plus"]
            
        if cfg["seasonality_toggle"] and m > pre_rev:
            occ *= cfg["seasonality_factors"].get(row["calendar_month"], 1.0)
        occ_pcts.append(occ)
        
    df = ts.copy()
    df["occupancy_pct"] = occ_pcts
    df["children"] = [round(cap * o) for o in occ_pcts]
    return df


def calc_revenue(cfg: dict, occ: pd.DataFrame) -> pd.DataFrame:
    df = occ.copy()
    fees = cfg["daily_fees"]
    weights = cfg["fee_mix_weights"]
    blended_base = sum(fees[k] * weights[k] for k in fees)
    biz_days = cfg["biz_days_per_month"]
    collection = cfg["revenue_collection_rate"]
    fee_inc = cfg["fee_annual_increase"]
    pre_rev = cfg.get("pre_revenue_months", 0)

    daily_fee, gross = [], []
    for _, row in df.iterrows():
        m = row["month"]
        yr = row["year_from_start"]
        fee = blended_base * (1 + fee_inc) ** yr
        daily_fee.append(fee)
        gross.append(0.0 if m <= pre_rev else row["children"] * fee * biz_days)
        
    df["daily_fee"] = daily_fee
    df["gross_revenue"] = gross
    df["bad_debt"] = df["gross_revenue"] * (1 - collection)
    df["net_revenue"] = df["gross_revenue"] * collection
    df["ccs_portion"] = df["net_revenue"] * cfg["ccs_eligibility_pct"] * cfg["ccs_subsidy_rate"]
    df["gap_fee_portion"] = df["net_revenue"] - df["ccs_portion"]
    return df


def calc_staffing(cfg: dict, rev: pd.DataFrame) -> pd.DataFrame:
    df = rev.copy()
    ratio = cfg["educator_child_ratio"]
    ed_hr = cfg["educator_hourly_rate"]
    sup_hr = cfg["support_hourly_rate"]
    hrs = cfg["hours_per_month"]
    sup_rate = cfg["superannuation_rate"]
    wc_rate = cfg["workers_comp_rate"]
    wage_inc = cfg["staff_wage_annual_increase"]
    turnover = cfg["staff_turnover_rate"]
    recruit = cfg["recruitment_cost_per_hire"]
    pre_rev = cfg.get("pre_revenue_months", 0)

    ect_list = cfg.get("ect_staff_counts", [0]*60)
    sup_list = cfg.get("support_staff_counts", [0]*60)

    educators, support, director_flag, gross_wages = [], [], [], []
    super_cost, wc_cost, turnover_cost, total_staff = [], [], [], []
    meets_cap = []

    director_triggered = not cfg.get("director_salary_deferred", True)

    for i, row in df.iterrows():
        m = row["month"]
        yr = row["year_from_start"]
        children = row["children"]
        esc = (1 + wage_inc) ** yr
        is_pre = m <= pre_rev

        ed = ect_list[i] if i < len(ect_list) else ect_list[-1]
        sup = sup_list[i] if i < len(sup_list) else sup_list[-1]

        educators.append(ed)
        support.append(sup)

        # Capacity Check
        max_kids = ed * ratio
        if is_pre:
            meets_cap.append(True)
        else:
            meets_cap.append((children <= max_kids) and (children <= cfg["approved_capacity"]))

        # Wages
        ed_wage = ed * ed_hr * hrs * esc
        sup_wage = 0.0 if (is_pre and cfg.get("support_staff_unpaid_pre_revenue", False)) else (sup * sup_hr * hrs * esc)

        # Director logic: deferred & starts when profitable (approx EBITDA > 0)
        if not director_triggered and not is_pre:
            approx_rev = row["net_revenue"]
            approx_staff = (ed_wage + sup_wage) * (1 + sup_rate)
            approx_costs = approx_rev * 0.45
            if approx_rev - approx_staff - approx_costs > 0:
                director_triggered = True

        dir_active = 1 if director_triggered and not is_pre else 0
        director_flag.append(dir_active)
        dir_wage = cfg["director_hourly_rate"] * hrs * esc * dir_active

        total_gross = ed_wage + sup_wage + dir_wage
        gross_wages.append(total_gross)
        s = total_gross * sup_rate
        super_cost.append(s)
        w = total_gross * wc_rate
        wc_cost.append(w)
        
        fte = ed + sup + dir_active
        tc = fte * turnover * recruit / 12 if fte > 0 else 0
        turnover_cost.append(tc)
        total_staff.append(total_gross + s + w + tc)

    df["meets_capacity_limits"] = meets_cap
    df["educators"] = educators
    df["support_staff"] = support
    df["director_active"] = director_flag
    df["total_fte"] = df["educators"] + df["support_staff"] + df["director_active"]
    df["gross_wages"] = gross_wages
    df["super_cost"] = super_cost
    df["wc_cost"] = wc_cost
    df["turnover_cost"] = turnover_cost
    df["total_staff_cost"] = total_staff
    df["wages_pct_revenue"] = df.apply(
        lambda r: r["total_staff_cost"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1
    )
    return df


def calc_rent(cfg: dict, ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()
    base = cfg["base_rent_pa"]
    esc = cfg["rent_escalation"]
    free_months = cfg["rent_free_months"]
    make_good_pa = cfg["make_good_provision_pa"]
    outgoings = cfg["outgoings"]
    land_tax = cfg["land_tax_monthly_pre_approval"]
    cpi = cfg["cpi_on_expenses"]

    rent_payable, outgoings_total, land_tax_col, make_good, total_occ = [], [], [], [], []

    for _, row in df.iterrows():
        m = row["month"]
        yr = row["year_from_start"]
        annual_rent = base * (1 + esc) ** yr
        monthly_rent = annual_rent / 12

        rp = 0.0 if m <= free_months else monthly_rent
        rent_payable.append(rp)

        out_esc = (1 + cpi) ** yr
        out_total = sum(outgoings.values()) * out_esc
        outgoings_total.append(out_total)

        lt = land_tax if m <= 3 else 0.0
        land_tax_col.append(lt)

        mg = make_good_pa / 12
        make_good.append(mg)

        total_occ.append(rp + out_total + lt + mg)

    df["rent_payable"] = rent_payable
    df["outgoings"] = outgoings_total
    df["land_tax"] = land_tax_col
    df["make_good"] = make_good
    df["total_occupancy_cost"] = total_occ
    return df


def calc_opex(cfg: dict, staff: pd.DataFrame) -> pd.DataFrame:
    df = staff.copy()
    ins = cfg["insurance"]
    admin = cfg["admin_opex"]
    ins_esc = cfg["insurance_escalation_rate"]
    cpi = cfg["cpi_on_expenses"]

    insurance_total, admin_total, total_opex = [], [], []

    for _, row in df.iterrows():
        m = row["month"]
        yr = row["year_from_start"]
        ins_mult = (1 + ins_esc) ** yr
        cpi_mult = (1 + cpi) ** yr

        ins_t = sum(ins.values()) * ins_mult
        insurance_total.append(ins_t)

        adm = 0.0
        for k, v in admin.items():
            val = v
            if k == "marketing" and m <= 3: val = cfg["marketing_pre_opening"]
            elif k == "extra_curricular" and m <= 3: val = 0.0
            elif k == "software" and m < 3: val = 0.0
            adm += val * cpi_mult
        admin_total.append(adm)
        total_opex.append(ins_t + adm)

    df["insurance_total"] = insurance_total
    df["admin_total"] = admin_total
    df["director_salary_opex"] = 0.0 # Handled in staffing
    df["total_opex"] = total_opex
    return df


def calc_food(cfg: dict, occ: pd.DataFrame) -> pd.DataFrame:
    df = occ.copy()
    cost_per = cfg["food_cost_per_child_per_day"]
    delivery = cfg["food_delivery_monthly"]
    biz_days = cfg["biz_days_per_month"]
    cpi = cfg["cpi_on_expenses"]

    food_cost = []
    for _, row in df.iterrows():
        yr = row["year_from_start"]
        esc = (1 + cpi) ** yr
        fc = row["children"] * cost_per * esc * biz_days
        dl = delivery * esc if row["children"] > 0 else 0
        food_cost.append(fc + dl)

    df["food_cost"] = food_cost
    return df


def build_monthly_pl(cfg: dict) -> pd.DataFrame:
    ts = build_time_series(cfg)
    occ = calc_occupancy(cfg, ts)
    rev = calc_revenue(cfg, occ)
    staff = calc_staffing(cfg, rev)
    rent = calc_rent(cfg, ts)
    opex = calc_opex(cfg, staff)
    food_df = calc_food(cfg, occ)

    df = pd.DataFrame()
    df["month"] = ts["month"]
    df["date"] = ts["date"]
    df["fiscal_year"] = ts["fiscal_year"]
    df["year_from_start"] = ts["year_from_start"]
    df["occupancy_pct"] = occ["occupancy_pct"]
    df["children"] = occ["children"]
    df["meets_capacity_limits"] = staff["meets_capacity_limits"]

    df["net_revenue"] = rev["net_revenue"]
    df["total_staff_cost"] = staff["total_staff_cost"]
    df["food_cost"] = food_df["food_cost"]
    df["direct_costs"] = df["total_staff_cost"] + df["food_cost"]
    df["gross_profit"] = df["net_revenue"] - df["direct_costs"]
    df["gp_margin"] = df.apply(lambda r: r["gross_profit"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1)

    df["total_occupancy_cost"] = rent["total_occupancy_cost"]
    df["rent_payable"] = rent["rent_payable"]
    df["total_opex"] = opex["total_opex"]
    df["total_overheads"] = df["total_occupancy_cost"] + df["total_opex"]

    df["ebitda"] = df["gross_profit"] - df["total_overheads"]
    df["ebitda_margin"] = df.apply(lambda r: r["ebitda"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1)

    furn_monthly = cfg["depreciation_furniture_total"] / (cfg["depreciation_years"] * 12)
    preop_monthly = cfg["amortisation_pre_opening_total"] / (cfg["depreciation_years"] * 12)
    mg_monthly = cfg["make_good_provision_pa"] / 12
    df["depreciation_furniture"] = furn_monthly
    df["amortisation_pre_opening"] = preop_monthly
    df["depreciation_make_good"] = mg_monthly
    df["total_da"] = furn_monthly + preop_monthly + mg_monthly

    df["ebit"] = df["ebitda"] - df["total_da"]

    loan_rate = cfg["director_loan_interest_rate"]
    loan_bal = cfg["director_loan_opening"]
    interest_col = []
    for _, row in df.iterrows():
        interest = loan_bal * loan_rate / 12
        interest_col.append(interest)
    df["interest_expense"] = interest_col

    df["npbt"] = df["ebit"] - df["interest_expense"]
    df["cumul_npbt"] = df["npbt"].cumsum()

    tax = []
    for _, row in df.iterrows():
        if row["cumul_npbt"] > 0 and row["npbt"] > 0:
            tax.append(row["npbt"] * cfg["company_tax_rate"])
        else:
            tax.append(0.0)
    df["tax"] = tax
    df["npat"] = df["npbt"] - df["tax"]

    df["educators"] = staff["educators"]
    df["support_staff"] = staff["support_staff"]
    df["director_active"] = staff["director_active"]
    df["total_fte"] = staff["total_fte"]
    df["gross_wages"] = staff["gross_wages"]
    df["wages_pct_revenue"] = staff["wages_pct_revenue"]

    return df


def calc_debt_schedule(cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    rate = cfg["director_loan_interest_rate"]
    repay_amt = cfg["director_loan_repayment_monthly"]
    opening = cfg["director_loan_opening"]

    rows = []
    bal = opening
    for _, row in pl.iterrows():
        interest = bal * rate / 12
        can_repay = row["ebitda"] > 0
        repayment = repay_amt if can_repay and bal > repay_amt else (bal + interest if can_repay and bal <= repay_amt else 0)
        closing = bal + interest - repayment
        rows.append({
            "month": row["month"], "date": row["date"],
            "opening_balance": bal, "interest": interest,
            "repay_eligible": can_repay, "repayment": repayment,
            "closing_balance": max(closing, 0),
        })
        bal = max(closing, 0)

    return pd.DataFrame(rows)


def calc_cashflow(cfg: dict, pl: pd.DataFrame, debt: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["month"] = pl["month"]
    df["date"] = pl["date"]
    df["fiscal_year"] = pl["fiscal_year"]

    df["npat"] = pl["npat"]
    df["da"] = pl["total_da"]

    wc_movement = []
    prev_rev, prev_costs = 0, 0
    for _, row in pl.iterrows():
        debtor_change = row["net_revenue"] - prev_rev
        creditor_change = (row["total_staff_cost"] + row["food_cost"] + row["total_opex"]) - prev_costs
        wc_movement.append(creditor_change - debtor_change)
        prev_rev = row["net_revenue"]
        prev_costs = row["total_staff_cost"] + row["food_cost"] + row["total_opex"]
    df["wc_movement"] = wc_movement

    total_purchases = pl["total_occupancy_cost"] + pl["total_opex"] + pl["food_cost"]
    df["gst_input_credits"] = total_purchases * cfg["gst_rate"] / (1 + cfg["gst_rate"])

    df["operating_cf"] = df["npat"] + df["da"] + df["wc_movement"] + df["gst_input_credits"]
    df["maintenance_capex"] = -cfg["maintenance_capex_pa"] / 12
    df["investing_cf"] = df["maintenance_capex"]
    df["loan_repayments"] = -debt["repayment"]
    df["financing_cf"] = df["loan_repayments"]
    df["net_cash_movement"] = df["operating_cf"] + df["investing_cf"] + df["financing_cf"]

    opening = cfg["capital_items"]["operational_seed"]
    cumul = [opening]
    for i in range(len(df)):
        if i == 0:
            cumul[0] = opening + df["net_cash_movement"].iloc[0]
        else:
            cumul.append(cumul[-1] + df["net_cash_movement"].iloc[i])
    df["closing_cash"] = cumul

    return df


def calc_gst_bas(cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    df = pl[["month", "date", "fiscal_year"]].copy()
    total_purchases = pl["total_occupancy_cost"] + pl["total_opex"] + pl["food_cost"]
    df["gst_collected"] = 0.0
    df["gst_paid_inputs"] = total_purchases * cfg["gst_rate"] / (1 + cfg["gst_rate"])
    df["net_gst_position"] = df["gst_collected"] - df["gst_paid_inputs"]
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter").agg({"gst_collected": "sum", "gst_paid_inputs": "sum", "net_gst_position": "sum"}).reset_index()
    quarterly["refund"] = -quarterly["net_gst_position"]
    return quarterly


def calc_balance_sheet(cfg: dict, pl: pd.DataFrame, cf: pd.DataFrame, debt: pd.DataFrame) -> pd.DataFrame:
    furn_total = cfg["depreciation_furniture_total"]
    preop_total = cfg["amortisation_pre_opening_total"]
    dep_years = cfg["depreciation_years"]

    annual = pl.copy()
    fy_groups = annual.groupby("fiscal_year")

    rows = []
    for fy, grp in fy_groups:
        last_idx = grp.index[-1]
        last_month = grp["month"].iloc[-1]
        yr_num = last_month / 12

        cash = cf["closing_cash"].iloc[last_idx]
        debtors = grp["net_revenue"].iloc[-1]
        furniture_net = max(furn_total - (furn_total / dep_years) * min(yr_num, dep_years), 0)
        preop_net = max(preop_total - (preop_total / dep_years) * min(yr_num, dep_years), 0)
        total_assets = cash + debtors + furniture_net + preop_net

        creditors = grp["total_staff_cost"].iloc[-1] + grp["food_cost"].iloc[-1] + grp["total_opex"].iloc[-1]
        dir_loan = debt["closing_balance"].iloc[last_idx]
        make_good = (cfg["make_good_provision_pa"] / 12) * last_month
        tax_payable = grp["tax"].iloc[-1]
        total_liabilities = creditors + dir_loan + make_good + tax_payable
        equity = total_assets - total_liabilities

        rows.append({
            "fiscal_year": fy, "cash": cash, "trade_debtors": debtors,
            "furniture_net": furniture_net, "pre_opening_net": preop_net,
            "total_assets": total_assets, "trade_creditors": creditors, "shareholder_loan": dir_loan,
            "make_good_provision": make_good, "tax_payable": tax_payable,
            "total_liabilities": total_liabilities, "retained_earnings": equity,
            "total_equity": equity, "balance_check": round(total_assets - total_liabilities - equity, 2),
        })

    return pd.DataFrame(rows)


def calc_annual_pl(pl: pd.DataFrame) -> pd.DataFrame:
    agg = pl.groupby("fiscal_year").agg({
        "net_revenue": "sum", "total_staff_cost": "sum", "food_cost": "sum",
        "direct_costs": "sum", "gross_profit": "sum", "total_occupancy_cost": "sum",
        "total_opex": "sum", "total_overheads": "sum", "ebitda": "sum",
        "total_da": "sum", "ebit": "sum", "interest_expense": "sum",
        "npbt": "sum", "tax": "sum", "npat": "sum",
    }).reset_index()

    agg["ebitda_margin"] = agg.apply(lambda r: r["ebitda"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1)
    agg["gp_margin"] = agg.apply(lambda r: r["gross_profit"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1)
    agg["yoy_revenue_growth"] = agg["net_revenue"].pct_change()
    return agg


def calc_valuation(cfg: dict, annual: pd.DataFrame) -> dict:
    dep_years = cfg["depreciation_years"]
    capex = cfg["maintenance_capex_pa"]

    annual_fcf = []
    for _, row in annual.iterrows():
        annual_fcf.append(row["npat"] + row["total_da"] - capex)

    r = cfg["discount_rate"]
    g = cfg["terminal_growth_rate"]

    y5_fcf = annual_fcf[-1] if len(annual_fcf) >= 5 else annual_fcf[-1]
    terminal_value = y5_fcf * (1 + g) / (r - g)

    pv_fcfs = sum(fcf / (1 + r) ** (i + 1) for i, fcf in enumerate(annual_fcf))
    pv_terminal = terminal_value / (1 + r) ** len(annual_fcf)
    ev_dcf = pv_fcfs + pv_terminal

    y5_ebitda = annual["ebitda"].iloc[-1] if len(annual) >= 5 else annual["ebitda"].iloc[-1]
    ev_multiples = {f"{m}x": y5_ebitda * m for m in cfg["exit_multiples"]}

    total_capital = sum(cfg["capital_items"].values())
    base_multiple = cfg["exit_multiples"][1] if len(cfg["exit_multiples"]) > 1 else cfg["exit_multiples"][0]
    ev_exit = y5_ebitda * base_multiple
    moic = ev_exit / total_capital if total_capital > 0 else 0

    irr_flows = [-total_capital] + annual_fcf[:-1] + [annual_fcf[-1] + ev_exit]
    try: irr = _irr(irr_flows)
    except Exception: irr = None

    cumul, payback = 0, None
    for i, fcf in enumerate(annual_fcf):
        cumul += fcf
        if cumul >= total_capital:
            payback = (total_capital / fcf if fcf > 0 else None) if i == 0 else i + ((total_capital - (cumul - fcf)) / fcf if fcf > 0 else 0)
            break

    return {
        "annual_fcf": annual_fcf, "y5_ebitda": y5_ebitda, "y5_fcf": y5_fcf,
        "terminal_value": terminal_value, "pv_fcfs": pv_fcfs, "pv_terminal": pv_terminal,
        "ev_dcf": ev_dcf, "ev_multiples": ev_multiples, "total_capital": total_capital,
        "moic": moic, "irr": irr, "payback_years": payback, "irr_flows": irr_flows,
    }


def calc_capital_sufficiency(cfg: dict, pl: pd.DataFrame) -> dict:
    items = cfg["capital_items"]
    total_capital = sum(items.values())
    opening_cash = items["operational_seed"]
    reserve = cfg["minimum_operating_reserve"]

    pre_rev = pl[pl["month"] <= cfg.get("pre_revenue_months", 0)]
    total_burn = abs(pre_rev["ebitda"].sum()) if len(pre_rev) > 0 else 0

    cash_at_trough = opening_cash - total_burn
    buffer = cash_at_trough - reserve

    return {
        "total_capital_committed": total_capital,
        "opening_operational_cash": opening_cash,
        "minimum_reserve": reserve,
        "pre_revenue_burn": total_burn,
        "cash_at_trough": cash_at_trough,
        "buffer_above_reserve": buffer,
        "sufficient": buffer > 0,
        "weeks_runway": buffer / (total_burn / 13) if total_burn > 0 else float("inf"),
    }


def run_single_centre(cfg: dict = None) -> dict:
    if cfg is None: cfg = deepcopy(EARLWOOD_CONFIG)
    pl = build_monthly_pl(cfg)
    annual = calc_annual_pl(pl)
    debt = calc_debt_schedule(cfg, pl)
    cf = calc_cashflow(cfg, pl, debt)
    bs = calc_balance_sheet(cfg, pl, cf, debt)
    gst = calc_gst_bas(cfg, pl)
    valuation = calc_valuation(cfg, annual)
    capital = calc_capital_sufficiency(cfg, pl)

    return {
        "centre_id": cfg["centre_id"],
        "centre_name": cfg["centre_name"],
        "config": cfg,
        "monthly_pl": pl,
        "annual_pl": annual,
        "debt_schedule": debt,
        "cashflow": cf,
        "balance_sheet": bs,
        "gst_bas": gst,
        "valuation": valuation,
        "capital_sufficiency": capital,
    }
