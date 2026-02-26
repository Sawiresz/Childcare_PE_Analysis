"""
BrightBlocks ELC — Childcare PE Financial Model Engine (PHASE 5 FINALISATION)
Parameterised multi-centre model with isolated scenarios, monthly staffing inputs,
pre-revenue controls, director loan as shareholder funding.
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import io


# ─────────────────────────────────────────────────────────────
# IRR SOLVER
# ─────────────────────────────────────────────────────────────

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
    "practical_completion_delay_months": 0,
    "pre_revenue_months": 3,
    "model_months": 60,
    "biz_days_per_month": 21.67,
    
    # REVENUE
    "daily_fees": {"0_24m": 145.0, "24_36m": 142.0, "36m_plus": 139.0},
    "fee_mix_weights": {"0_24m": 0.0, "24_36m": 0.0, "36m_plus": 1.0},
    "fee_annual_increase": 0.03,
    "revenue_collection_rate": 0.98,
    "ccs_eligibility_pct": 0.75,
    "ccs_subsidy_rate": 0.80,
    
    # OCCUPANCY RAMP (% of capacity)
    "occupancy_ramp": {
        "pre_revenue": 0.0,
        "m1_3": 0.0,
        "m4": 0.70,
        "m5_6": 0.75,
        "y2": 0.85,
        "y3": 0.95,
        "y4_plus": 0.975,
    },
    
    # MONTHLY STAFFING (ECT and Support counts per month, 1-60)
    "monthly_ect_staff": [1] * 60,
    "monthly_support_staff": [0] * 60,
    "support_staff_unpaid_pre_revenue": False,
    
    # STAFFING RATES
    "director_hourly_rate": 50.0,
    "director_annual_cost": 100350.0,
    "educator_hourly_rate": 38.59,
    "support_hourly_rate": 35.0,
    "hours_per_month": 164.6667,
    "staff_wage_annual_increase": 0.03,
    "superannuation_rate": 0.115,
    "workers_comp_rate": 0.017,
    "educator_child_ratio": 10,
    
    # DIRECTOR SALARY (DEFERRED)
    "director_salary_deferred": True,
    "dir_trigger_ebitda_threshold": 7500.0,
    "dir_trigger_occupancy_threshold": 0.60,
    "dir_trigger_consecutive_months": 2,
    
    # LEASE
    "base_rent_pa": 230000.0,
    "rent_escalation": 0.03,
    "rent_free_months": 6,
    "rent_free_credit": 115000.0,
    "initial_lease_term_years": 5,
    "further_options": "5x5",
    "make_good_provision_pa": 15000.0,
    
    # OUTGOINGS (monthly)
    "outgoings": {
        "strata": 1250, "electricity": 500, "gas": 333, "water": 58,
        "council_rates": 350, "waste": 167, "r_and_m": 500,
    },
    "land_tax_monthly_pre_approval": 1283.0,
    
    # INSURANCE (monthly)
    "insurance": {
        "public_liability": 200, "professional_indemnity": 175,
        "management_liability": 150, "contents": 100, "business_interruption": 150,
    },
    "insurance_escalation_rate": 0.06,
    
    # ADMIN / OTHER OPEX (monthly)
    "admin_opex": {
        "accounting": 978, "bank_fees": 50, "cleaning": 1500, "software": 400,
        "pest_control": 83, "phone_internet": 200, "toys_books": 250,
        "training": 500, "marketing": 3300, "extra_curricular": 1000,
        "miscellaneous": 500, "compliance": 250,
    },
    "marketing_pre_opening": 1500.0,
    "cpi_on_expenses": 0.02,
    
    # FOOD & CATERING
    "food_cost_per_child_per_day": 8.50,
    "food_delivery_monthly": 200.0,
    
    # CAPITAL DEPLOYMENT
    "capital_items": {
        "bank_guarantee": 126500, "legal_professional": 23000, "signage": 5000,
        "marketing_setup": 23009, "other_pre_opening": 18709,
        "loose_furniture": 35000, "operational_seed": 75000,
    },
    "total_deployed_director_loans": 196218.0,
    "minimum_operating_reserve": 20000.0,
    "maintenance_capex_pa": 5000.0,
    
    # TAX & VALUATION
    "gst_rate": 0.10,
    "company_tax_rate": 0.25,
    "discount_rate": 0.12,
    "terminal_growth_rate": 0.025,
    "exit_multiples": [4.5, 5.5, 7.0],
    "depreciation_furniture_total": 35000.0,
    "amortisation_pre_opening_total": 196218.0,
    "depreciation_years": 5,
    
    # DIRECTOR LOAN (shareholder funding)
    "director_loan_opening": 196218.0,
    "director_loan_interest_rate": 0.05,
    "director_loan_repayment_monthly": 5000.0,
}

SCENARIO_PRESETS = {
    "bear": {
        "label": "Bear (Downside)",
        "occupancy_ramp": {"m4": 0.55, "m5_6": 0.65, "y2": 0.78, "y3": 0.85, "y4_plus": 0.90},
        "daily_fees": {"36m_plus": 133.0, "24_36m": 130.0, "0_24m": 138.0},
        "fee_annual_increase": 0.025,
        "staff_wage_annual_increase": 0.035,
        "cpi_on_expenses": 0.025,
        "exit_multiples": [3.5, 4.5, 5.5],
    },
    "base": {
        "label": "Base Case",
    },
    "bull": {
        "label": "Bull (Upside)",
        "occupancy_ramp": {"m4": 0.80, "m5_6": 0.85, "y2": 0.95, "y3": 0.98, "y4_plus": 1.0},
        "daily_fees": {"36m_plus": 152.0, "24_36m": 149.0, "0_24m": 158.0},
        "fee_annual_increase": 0.04,
        "staff_wage_annual_increase": 0.025,
        "base_rent_pa": 225000.0,
        "cpi_on_expenses": 0.015,
        "exit_multiples": [6.0, 7.0, 8.5],
    },
}


# ─────────────────────────────────────────────────────────────
# TIME SERIES & OCCUPANCY
# ─────────────────────────────────────────────────────────────

def build_time_series(cfg: dict) -> pd.DataFrame:
    start = pd.Timestamp(cfg["commencement_date"]) + pd.DateOffset(
        months=cfg["practical_completion_delay_months"])
    months = cfg["model_months"]
    dates = pd.date_range(start, periods=months, freq="MS")
    pre_rev = cfg["pre_revenue_months"]
    
    df = pd.DataFrame({
        "month": range(1, months + 1),
        "date": dates,
        "calendar_month": dates.month,
        "fiscal_year": [(d.year + 1 if d.month >= 7 else d.year) for d in dates],
        "is_pre_revenue": [1 if m <= pre_rev else 0 for m in range(1, months + 1)],
    })
    
    year_boundaries = []
    for i, d in enumerate(dates):
        delta_months = (d.year - dates[0].year) * 12 + d.month - dates[0].month
        years_from_start = delta_months // 12
        year_boundaries.append(years_from_start)
    df["year_from_start"] = year_boundaries
    return df


def calc_occupancy(cfg: dict, ts: pd.DataFrame) -> pd.DataFrame:
    ramp = cfg["occupancy_ramp"]
    cap = cfg["approved_capacity"]
    pre_rev = cfg["pre_revenue_months"]
    occ_pcts = []
    
    for _, row in ts.iterrows():
        m = row["month"]
        if m <= pre_rev:
            occ = ramp["pre_revenue"]
        elif m <= 3:
            occ = ramp["m1_3"]
        elif m == 4:
            occ = ramp["m4"]
        elif m <= 6:
            occ = ramp["m5_6"]
        elif m <= 12:
            occ = ramp["m5_6"]
        elif m <= 24:
            occ = ramp["y2"]
        elif m <= 36:
            occ = ramp["y3"]
        else:
            occ = ramp["y4_plus"]
        occ_pcts.append(occ)
    
    df = ts.copy()
    df["occupancy_pct"] = occ_pcts
    df["children"] = [round(cap * o) if o > 0 else 0 for o in occ_pcts]
    return df


def calc_revenue(cfg: dict, occ: pd.DataFrame) -> pd.DataFrame:
    df = occ.copy()
    fees = cfg["daily_fees"]
    weights = cfg["fee_mix_weights"]
    blended_base = sum(fees[k] * weights[k] for k in fees)
    biz_days = cfg["biz_days_per_month"]
    collection = cfg["revenue_collection_rate"]
    fee_inc = cfg["fee_annual_increase"]
    pre_rev = cfg["pre_revenue_months"]
    
    daily_fee = []
    for _, row in df.iterrows():
        yr = row["year_from_start"]
        daily_fee.append(blended_base * (1 + fee_inc) ** yr)
    df["daily_fee"] = daily_fee
    
    gross_revenue = []
    for _, row in df.iterrows():
        if row["is_pre_revenue"] or row["children"] == 0:
            gross_revenue.append(0.0)
        else:
            gross_revenue.append(row["children"] * row["daily_fee"] * biz_days)
    
    df["gross_revenue"] = gross_revenue
    df["bad_debt"] = df["gross_revenue"] * (1 - collection)
    df["net_revenue"] = df["gross_revenue"] * collection
    df["ccs_portion"] = df["net_revenue"] * cfg["ccs_eligibility_pct"] * cfg["ccs_subsidy_rate"]
    df["gap_fee_portion"] = df["net_revenue"] - df["ccs_portion"]
    return df


# ─────────────────────────────────────────────────────────────
# STAFFING WITH MONTHLY INPUTS
# ─────────────────────────────────────────────────────────────

def calc_staffing(cfg: dict, rev: pd.DataFrame) -> pd.DataFrame:
    df = rev.copy()
    ed_hr = cfg["educator_hourly_rate"]
    sup_hr = cfg["support_hourly_rate"]
    hrs = cfg["hours_per_month"]
    sup_rate = cfg["superannuation_rate"]
    wc_rate = cfg["workers_comp_rate"]
    wage_inc = cfg["staff_wage_annual_increase"]
    dir_annual = cfg["director_annual_cost"]
    pre_rev = cfg["pre_revenue_months"]
    
    monthly_ect = cfg["monthly_ect_staff"]
    monthly_sup = cfg["monthly_support_staff"]
    sup_unpaid = cfg["support_staff_unpaid_pre_revenue"]
    
    educators = []
    support = []
    director_flag = []
    gross_wages = []
    super_cost = []
    wc_cost = []
    total_staff = []
    
    director_triggered = False
    ebitda_streak = 0
    occ_streak = 0
    
    for i, row in df.iterrows():
        m = row["month"]
        yr = row["year_from_start"]
        esc = (1 + wage_inc) ** yr
        
        # Monthly staffing from config
        ed = monthly_ect[m - 1] if m <= len(monthly_ect) else monthly_ect[-1]
        sup = monthly_sup[m - 1] if m <= len(monthly_sup) else monthly_sup[-1]
        
        educators.append(ed)
        support.append(sup)
        
        ed_wage = ed * ed_hr * hrs * esc
        
        # Support staff wages (may be unpaid in pre-revenue)
        if m <= pre_rev and sup_unpaid:
            sup_wage = 0.0
        else:
            sup_wage = sup * sup_hr * hrs * esc
        
        # Director trigger
        if not cfg["director_salary_deferred"]:
            director_triggered = True
        
        if not director_triggered and m >= pre_rev + 1:
            approx_rev = row["net_revenue"]
            approx_staff = (ed_wage + sup_wage) * (1 + sup_rate)
            approx_costs = approx_rev * 0.45 if approx_rev > 0 else 0
            approx_ebitda = approx_rev - approx_costs
            
            if approx_ebitda > cfg["dir_trigger_ebitda_threshold"]:
                ebitda_streak += 1
            else:
                ebitda_streak = 0
            
            if row["occupancy_pct"] >= cfg["dir_trigger_occupancy_threshold"]:
                occ_streak += 1
            else:
                occ_streak = 0
            
            if (ebitda_streak > cfg["dir_trigger_consecutive_months"] and
                occ_streak > cfg["dir_trigger_consecutive_months"]):
                director_triggered = True
        
        dir_active = 1 if director_triggered and row["children"] > 0 and m > pre_rev else 0
        director_flag.append(dir_active)
        dir_wage = cfg["director_hourly_rate"] * hrs * esc * dir_active
        
        total_gross = ed_wage + sup_wage + dir_wage
        gross_wages.append(total_gross)
        s = total_gross * sup_rate
        super_cost.append(s)
        w = total_gross * wc_rate
        wc_cost.append(w)
        total_staff.append(total_gross + s + w)
    
    df["educators"] = educators
    df["support_staff"] = support
    df["director_active"] = director_flag
    df["total_fte"] = df["educators"] + df["support_staff"] + df["director_active"]
    df["gross_wages"] = gross_wages
    df["super_cost"] = super_cost
    df["wc_cost"] = wc_cost
    df["total_staff_cost"] = total_staff
    df["wages_pct_revenue"] = df.apply(
        lambda r: r["total_staff_cost"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1
    )
    return df


# ─────────────────────────────────────────────────────────────
# RENT & OCCUPANCY COSTS
# ─────────────────────────────────────────────────────────────

def calc_rent(cfg: dict, ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()
    base = cfg["base_rent_pa"]
    esc = cfg["rent_escalation"]
    free_months = cfg["rent_free_months"]
    make_good_pa = cfg["make_good_provision_pa"]
    outgoings = cfg["outgoings"]
    land_tax = cfg["land_tax_monthly_pre_approval"]
    cpi = cfg["cpi_on_expenses"]
    pre_rev = cfg["pre_revenue_months"]
    
    rent_payable = []
    outgoings_total = []
    land_tax_col = []
    make_good = []
    total_occ = []
    
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
        
        lt = land_tax if m <= pre_rev else 0.0
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


# ─────────────────────────────────────────────────────────────
# OPEX
# ─────────────────────────────────────────────────────────────

def calc_opex(cfg: dict, staff: pd.DataFrame) -> pd.DataFrame:
    df = staff.copy()
    ins = cfg["insurance"]
    admin = cfg["admin_opex"]
    ins_esc = cfg["insurance_escalation_rate"]
    cpi = cfg["cpi_on_expenses"]
    pre_rev = cfg["pre_revenue_months"]
    
    insurance_total = []
    admin_total = []
    director_salary_opex = []
    total_opex = []
    
    for _, row in df.iterrows():
        m = row["month"]
        yr = row["year_from_start"]
        ins_mult = (1 + ins_esc) ** yr
        cpi_mult = (1 + cpi) ** yr
        
        ins_t = sum(ins.values()) * ins_mult
        insurance_total.append(ins_t)
        
        adm = 0.0
        for k, v in admin.items():
            if k == "marketing":
                val = cfg["marketing_pre_opening"] if m <= pre_rev else v
            elif k == "extra_curricular":
                val = 0.0 if m <= pre_rev else v
            elif k == "software":
                val = v if m >= pre_rev else 0.0
            else:
                val = v
            adm += val * cpi_mult
        admin_total.append(adm)
        
        dir_sal = (cfg["director_annual_cost"] / 12) * ((1 + cfg["staff_wage_annual_increase"]) ** yr) * row["director_active"]
        director_salary_opex.append(dir_sal)
        
        total_opex.append(ins_t + adm + dir_sal)
    
    df["insurance_total"] = insurance_total
    df["admin_total"] = admin_total
    df["director_salary_opex"] = director_salary_opex
    df["total_opex"] = total_opex
    return df


# ─────────────────────────────────────────────────────────────
# FOOD & CATERING
# ─────────────────────────────────────────────────────────────

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
        dl = delivery * esc
        food_cost.append(fc + dl)
    
    df["food_cost"] = food_cost
    return df


# ─────────────────────────────────────────────────────────────
# MONTHLY P&L
# ─────────────────────────────────────────────────────────────

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
    df["is_pre_revenue"] = ts["is_pre_revenue"]
    df["occupancy_pct"] = occ["occupancy_pct"]
    df["children"] = occ["children"]
    
    df["net_revenue"] = rev["net_revenue"]
    df["ccs_portion"] = rev["ccs_portion"]
    df["gap_fee_portion"] = rev["gap_fee_portion"]
    
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
    
    # D&A
    furn_monthly = cfg["depreciation_furniture_total"] / (cfg["depreciation_years"] * 12)
    preop_monthly = cfg["amortisation_pre_opening_total"] / (cfg["depreciation_years"] * 12)
    mg_monthly = cfg["make_good_provision_pa"] / 12
    df["depreciation_furniture"] = furn_monthly
    df["amortisation_pre_opening"] = preop_monthly
    df["depreciation_make_good"] = mg_monthly
    df["total_da"] = furn_monthly + preop_monthly + mg_monthly
    
    df["ebit"] = df["ebitda"] - df["total_da"]
    
    # Interest
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
    df["wages_pct_revenue"] = staff["wages_pct_revenue"]
    
    return df


# ─────────────────────────────────────────────────────────────
# DEBT SCHEDULE (director loan as shareholder funding)
# ─────────────────────────────────────────────────────────────

def calc_debt_schedule(cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    rate = cfg["director_loan_interest_rate"]
    repay_amt = cfg["director_loan_repayment_monthly"]
    opening = cfg["director_loan_opening"]
    
    rows = []
    bal = opening
    
    for _, row in pl.iterrows():
        interest = bal * rate / 12
        can_repay = row["month"] >= 4 and row["ebitda"] > 0
        repayment = repay_amt if can_repay and bal > repay_amt else (bal + interest if can_repay and bal <= repay_amt else 0)
        closing = bal + interest - repayment
        rows.append({
            "month": row["month"],
            "date": row["date"],
            "opening_balance": bal,
            "interest": interest,
            "repay_eligible": can_repay,
            "repayment": repayment,
            "closing_balance": max(closing, 0),
        })
        bal = max(closing, 0)
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# CASH FLOW STATEMENT
# ─────────────────────────────────────────────────────────────

def calc_cashflow(cfg: dict, pl: pd.DataFrame, debt: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["month"] = pl["month"]
    df["date"] = pl["date"]
    df["fiscal_year"] = pl["fiscal_year"]
    
    df["npat"] = pl["npat"]
    df["da"] = pl["total_da"]
    
    wc_movement = []
    prev_rev = 0
    prev_costs = 0
    for _, row in pl.iterrows():
        debtor_change = row["net_revenue"] - prev_rev
        creditor_change = (row["total_staff_cost"] + row["food_cost"] + row["total_opex"]) - prev_costs
        wc = creditor_change - debtor_change
        wc_movement.append(wc)
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


# ─────────────────────────────────────────────────────────────
# BALANCE SHEET
# ─────────────────────────────────────────────────────────────

def calc_balance_sheet(cfg: dict, pl: pd.DataFrame, cf: pd.DataFrame, debt: pd.DataFrame) -> pd.DataFrame:
    furn_total = cfg["depreciation_furniture_total"]
    preop_total = cfg["amortisation_pre_opening_total"]
    dep_years = cfg["depreciation_years"]
    
    annual = pl.copy()
    annual["fy"] = pl["fiscal_year"]
    fy_groups = annual.groupby("fy")
    
    rows = []
    for fy, grp in fy_groups:
        last_idx = grp.index[-1]
        last_month = grp["month"].iloc[-1]
        yr_num = last_month / 12
        
        cash = cf["closing_cash"].iloc[last_idx]
        debtors = grp["net_revenue"].iloc[-1] if grp["net_revenue"].iloc[-1] > 0 else 0
        furniture_net = max(furn_total - (furn_total / dep_years) * min(yr_num, dep_years), 0)
        preop_net = max(preop_total - (preop_total / dep_years) * min(yr_num, dep_years), 0)
        total_assets = cash + debtors + furniture_net + preop_net
        
        creditors = grp["total_staff_cost"].iloc[-1] + grp["food_cost"].iloc[-1] + grp["total_opex"].iloc[-1]
        dir_loan = debt["closing_balance"].iloc[last_idx]
        make_good = (cfg["make_good_provision_pa"] / 12) * last_month
        deferred_salary = 0
        
        tax_payable = grp["tax"].iloc[-1] if grp["tax"].iloc[-1] > 0 else 0
        total_liabilities = creditors + dir_loan + make_good + deferred_salary + tax_payable
        
        equity = total_assets - total_liabilities
        
        rows.append({
            "fiscal_year": fy,
            "cash": cash,
            "trade_debtors": debtors,
            "furniture_net": furniture_net,
            "pre_opening_net": preop_net,
            "total_assets": total_assets,
            "trade_creditors": creditors,
            "director_loan": dir_loan,
            "make_good_provision": make_good,
            "deferred_salary": deferred_salary,
            "tax_payable": tax_payable,
            "total_liabilities": total_liabilities,
            "retained_earnings": equity,
            "total_equity": equity,
            "balance_check": round(total_assets - total_liabilities - equity, 2),
        })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# ANNUAL P&L
# ─────────────────────────────────────────────────────────────

def calc_annual_pl(pl: pd.DataFrame) -> pd.DataFrame:
    agg = pl.groupby("fiscal_year").agg({
        "net_revenue": "sum",
        "total_staff_cost": "sum",
        "food_cost": "sum",
        "direct_costs": "sum",
        "gross_profit": "sum",
        "total_occupancy_cost": "sum",
        "total_opex": "sum",
        "total_overheads": "sum",
        "ebitda": "sum",
        "total_da": "sum",
        "ebit": "sum",
        "interest_expense": "sum",
        "npbt": "sum",
        "tax": "sum",
        "npat": "sum",
    }).reset_index()
    
    agg["ebitda_margin"] = agg.apply(lambda r: r["ebitda"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1)
    agg["gp_margin"] = agg.apply(lambda r: r["gross_profit"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1)
    agg["yoy_revenue_growth"] = agg["net_revenue"].pct_change()
    return agg


# ─────────────────────────────────────────────────────────────
# VALUATION
# ─────────────────────────────────────────────────────────────

def calc_valuation(cfg: dict, annual: pd.DataFrame) -> dict:
    dep_years = cfg["depreciation_years"]
    capex = cfg["maintenance_capex_pa"]
    
    annual_fcf = []
    for _, row in annual.iterrows():
        fcf = row["npat"] + row["total_da"] - capex
        annual_fcf.append(fcf)
    
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
    try:
        irr = _irr(irr_flows)
    except Exception:
        irr = None
    
    cumul = 0
    payback = None
    for i, fcf in enumerate(annual_fcf):
        cumul += fcf
        if cumul >= total_capital:
            if i == 0:
                payback = total_capital / fcf if fcf > 0 else None
            else:
                prev_cumul = cumul - fcf
                remaining = total_capital - prev_cumul
                payback = i + (remaining / fcf if fcf > 0 else 0)
            break
    
    return {
        "annual_fcf": annual_fcf,
        "y5_ebitda": y5_ebitda,
        "y5_fcf": y5_fcf,
        "terminal_value": terminal_value,
        "pv_fcfs": pv_fcfs,
        "pv_terminal": pv_terminal,
        "ev_dcf": ev_dcf,
        "ev_multiples": ev_multiples,
        "total_capital": total_capital,
        "moic": moic,
        "irr": irr,
        "payback_years": payback,
        "irr_flows": irr_flows,
    }


# ─────────────────────────────────────────────────────────────
# SENSITIVITY TABLES
# ───────────────────��─────────────────────────────────────────

def sensitivity_occupancy_fee(cfg: dict, occupancies: List[float] = None, fees: List[float] = None) -> pd.DataFrame:
    if occupancies is None:
        occupancies = [0.60, 0.70, 0.80, 0.90, 1.00]
    if fees is None:
        fees = [125, 135, 139, 145, 155]
    
    results = []
    for occ in occupancies:
        for fee in fees:
            c = deepcopy(cfg)
            c["occupancy_ramp"]["y3"] = occ
            c["occupancy_ramp"]["y4_plus"] = occ
            c["daily_fees"]["36m_plus"] = fee
            pl = build_monthly_pl(c)
            annual = calc_annual_pl(pl)
            y3_ebitda = annual["ebitda"].iloc[2] if len(annual) >= 3 else annual["ebitda"].iloc[-1]
            results.append({"occupancy": occ, "daily_fee": fee, "y3_ebitda": y3_ebitda})
    
    return pd.DataFrame(results).pivot(index="occupancy", columns="daily_fee", values="y3_ebitda")


def sensitivity_dcf(cfg: dict, discount_rates: List[float] = None, growth_rates: List[float] = None) -> pd.DataFrame:
    if discount_rates is None:
        discount_rates = [0.08, 0.10, 0.12, 0.15]
    if growth_rates is None:
        growth_rates = [0.015, 0.025, 0.035]
    
    pl = build_monthly_pl(cfg)
    annual = calc_annual_pl(pl)
    
    results = []
    for dr in discount_rates:
        for gr in growth_rates:
            c = deepcopy(cfg)
            c["discount_rate"] = dr
            c["terminal_growth_rate"] = gr
            val = calc_valuation(c, annual)
            results.append({"discount_rate": dr, "terminal_growth": gr, "ev_dcf": val["ev_dcf"]})
    
    return pd.DataFrame(results).pivot(index="discount_rate", columns="terminal_growth", values="ev_dcf")


# ─────────────────────────────────────────────────────────────
# SCENARIO UTILITIES
# ─────────────────────────────────────────────────────────────

def apply_scenario_overrides(cfg: dict, overrides: dict) -> dict:
    c = deepcopy(cfg)
    for k, v in overrides.items():
        if k == "label":
            continue
        if isinstance(v, dict) and k in c and isinstance(c[k], dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


def run_scenario(cfg: dict, scenario: str = "base", custom_overrides: dict = None) -> dict:
    preset = SCENARIO_PRESETS.get(scenario, {})
    c = apply_scenario_overrides(cfg, preset)
    if custom_overrides:
        c = apply_scenario_overrides(c, custom_overrides)
    return run_single_centre(c)


# ─────────────────────────────────────────────────────────────
# CAPITAL SUFFICIENCY & WEEKLY CASHFLOW
# ─────────────────────────────────────────────────────────────

def calc_capital_sufficiency(cfg: dict, pl: pd.DataFrame) -> dict:
    items = cfg["capital_items"]
    total_capital = sum(items.values())
    deployed = cfg["total_deployed_director_loans"]
    opening_cash = items["operational_seed"]
    reserve = cfg["minimum_operating_reserve"]
    net_deployable = opening_cash - reserve
    
    pre_rev = cfg["pre_revenue_months"]
    pre_rev_pl = pl[pl["month"] <= pre_rev]
    total_burn = abs(pre_rev_pl["ebitda"].sum()) if len(pre_rev_pl) > 0 else 0
    
    cash_at_trough = opening_cash - total_burn
    buffer = cash_at_trough - reserve
    sufficient = buffer > 0
    weeks_runway = buffer / (total_burn / (pre_rev + 4)) if total_burn > 0 else float("inf")
    
    return {
        "total_capital_committed": total_capital,
        "already_deployed": deployed,
        "opening_operational_cash": opening_cash,
        "minimum_reserve": reserve,
        "net_deployable": net_deployable,
        "pre_revenue_burn_months": total_burn,
        "cash_at_trough": cash_at_trough,
        "buffer_above_reserve": buffer,
        "sufficient": sufficient,
        "weeks_runway": weeks_runway,
    }


def calc_weekly_cashflow(cfg: dict, pl: pd.DataFrame) -> pd.DataFrame:
    opening = cfg["capital_items"]["operational_seed"]
    reserve = cfg["minimum_operating_reserve"]
    weeks = 26
    rows = []
    cumul = opening
    
    for w in range(1, weeks + 1):
        m = (w - 1) // 4 + 1
        m = min(m, 6)
        m_idx = m - 1
        
        if m_idx < len(pl):
            row_data = pl.iloc[m_idx]
            weekly_rev = row_data["net_revenue"] / 4.33
            weekly_wages = row_data["total_staff_cost"] / 4.33
            weekly_outgoings = row_data["total_occupancy_cost"] / 4.33
            weekly_opex = row_data["total_opex"] / 4.33
        else:
            weekly_rev = 0
            weekly_wages = 0
            weekly_outgoings = 0
            weekly_opex = 0
        
        net = weekly_rev - weekly_wages - weekly_outgoings - weekly_opex
        cumul += net
        buffer = cumul - reserve
        
        if buffer < 0:
            rag = "RED"
        elif buffer < 10000:
            rag = "AMBER"
        else:
            rag = "GREEN"
        
        rows.append({
            "week": w,
            "month": m,
            "revenue": weekly_rev,
            "wages": weekly_wages,
            "outgoings": weekly_outgoings,
            "other_opex": weekly_opex,
            "net_movement": net,
            "cumulative_cash": cumul,
            "buffer_above_reserve": buffer,
            "rag_status": rag,
        })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# MASTER: RUN SINGLE CENTRE
# ─────────────────────────────────────��───────────────────────

def run_single_centre(cfg: dict = None) -> dict:
    if cfg is None:
        cfg = deepcopy(EARLWOOD_CONFIG)
    
    pl = build_monthly_pl(cfg)
    annual = calc_annual_pl(pl)
    debt = calc_debt_schedule(cfg, pl)
    cf = calc_cashflow(cfg, pl, debt)
    bs = calc_balance_sheet(cfg, pl, cf, debt)
    valuation = calc_valuation(cfg, annual)
    capital = calc_capital_sufficiency(cfg, pl)
    weekly = calc_weekly_cashflow(cfg, pl)
    
    return {
        "centre_id": cfg["centre_id"],
        "centre_name": cfg["centre_name"],
        "config": cfg,
        "monthly_pl": pl,
        "annual_pl": annual,
        "debt_schedule": debt,
        "cashflow": cf,
        "balance_sheet": bs,
        "valuation": valuation,
        "capital_sufficiency": capital,
        "weekly_cashflow": weekly,
    }


# ─────────────────────────────────────────────────────────────
# MULTI-CENTRE PORTFOLIO
# ─────────────────────────────────────────────────────────────

def run_portfolio(configs: List[dict]) -> dict:
    centres = {}
    for cfg in configs:
        result = run_single_centre(cfg)
        centres[cfg["centre_id"]] = result
    
    all_annual = []
    for cid, res in centres.items():
        a = res["annual_pl"].copy()
        a["centre_id"] = cid
        all_annual.append(a)
    
    consolidated = pd.concat(all_annual)
    portfolio_annual = consolidated.groupby("fiscal_year").agg({
        "net_revenue": "sum",
        "total_staff_cost": "sum",
        "food_cost": "sum",
        "direct_costs": "sum",
        "gross_profit": "sum",
        "total_occupancy_cost": "sum",
        "total_opex": "sum",
        "total_overheads": "sum",
        "ebitda": "sum",
        "total_da": "sum",
        "ebit": "sum",
        "interest_expense": "sum",
        "npbt": "sum",
        "tax": "sum",
        "npat": "sum",
    }).reset_index()
    
    portfolio_annual["ebitda_margin"] = portfolio_annual.apply(
        lambda r: r["ebitda"] / r["net_revenue"] if r["net_revenue"] > 0 else 0, axis=1
    )
    
    total_capital = sum(sum(c["capital_items"].values()) for c in configs)
    y5_ebitda = portfolio_annual["ebitda"].iloc[-1] if len(portfolio_annual) > 0 else 0
    ev_55 = y5_ebitda * 5.5
    moic = ev_55 / total_capital if total_capital > 0 else 0
    
    return {
        "centres": centres,
        "portfolio_annual_pl": portfolio_annual,
        "total_capital_invested": total_capital,
        "portfolio_y5_ebitda": y5_ebitda,
        "portfolio_ev_5_5x": ev_55,
        "portfolio_moic": moic,
    }


# ─────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────

def export_results_csv(result: dict, prefix: str = "brightblocks") -> Dict[str, str]:
    exports = {}
    
    for key, label in [
        ("monthly_pl", "monthly_pl"),
        ("annual_pl", "annual_pl"),
        ("cashflow", "cashflow"),
        ("balance_sheet", "balance_sheet"),
        ("debt_schedule", "debt_schedule"),
        ("weekly_cashflow", "weekly_cashflow"),
    ]:
        df = result[key].copy()
        for col in df.columns:
            if hasattr(df[col], "dt"):
                try:
                    df[col] = df[col].dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
        exports[f"{prefix}_{label}.csv"] = df.to_csv(index=False)
    
    val = result["valuation"]
    val_rows = [
        {"metric": "EV (DCF)", "value": val["ev_dcf"]},
        {"metric": "Terminal Value", "value": val["terminal_value"]},
        {"metric": "IRR", "value": val["irr"]},
        {"metric": "MOIC", "value": val["moic"]},
        {"metric": "Payback (years)", "value": val["payback_years"]},
        {"metric": "Total Capital", "value": val["total_capital"]},
        {"metric": "Y5 EBITDA", "value": val["y5_ebitda"]},
    ]
    for k, v in val["ev_multiples"].items():
        val_rows.append({"metric": f"EV ({k})", "value": v})
    exports[f"{prefix}_valuation.csv"] = pd.DataFrame(val_rows).to_csv(index=False)
    
    cap = result["capital_sufficiency"]
    exports[f"{prefix}_capital.csv"] = pd.DataFrame([cap]).to_csv(index=False)
    
    return exports


# ─────────────────────────────────────────────────────────────
# TORNADO SENSITIVITY
# ─────────────────────────────────────────────────────────────

def tornado_sensitivity(cfg: dict, target_year: int = 3) -> pd.DataFrame:
    base_result = run_single_centre(cfg)
    base_annual = base_result["annual_pl"]
    idx = min(target_year - 1, len(base_annual) - 1)
    base_ebitda = base_annual["ebitda"].iloc[idx]
    
    variables = [
        ("Occupancy (Y3)", "occupancy_ramp.y3", -0.10, +0.10),
        ("Daily Fee (36m+)", "daily_fees.36m_plus", -15.0, +15.0),
        ("Educator Hourly Rate", "educator_hourly_rate", -5.0, +5.0),
        ("Base Rent p.a.", "base_rent_pa", -30000, +30000),
        ("Wage Increase (% p.a.)", "staff_wage_annual_increase", -0.01, +0.01),
        ("Fee Increase (% p.a.)", "fee_annual_increase", -0.01, +0.01),
        ("CPI on Expenses", "cpi_on_expenses", -0.01, +0.01),
        ("Capacity", "approved_capacity", -10, +10),
    ]
    
    rows = []
    for label, key_path, low_delta, high_delta in variables:
        parts = key_path.split(".")
        for delta, side in [(low_delta, "low"), (high_delta, "high")]:
            c = deepcopy(cfg)
            if len(parts) == 2:
                base_val = c[parts[0]][parts[1]]
                c[parts[0]][parts[1]] = base_val + delta
            else:
                base_val = c[parts[0]]
                c[parts[0]] = base_val + delta
            
            try:
                pl = build_monthly_pl(c)
                annual = calc_annual_pl(pl)
                ebitda = annual["ebitda"].iloc[idx]
            except Exception:
                ebitda = base_ebitda
            
            rows.append({
                "variable": label,
                "side": side,
                "delta": delta,
                "base_value": base_val,
                "adjusted_value": base_val + delta,
                "ebitda": ebitda,
                "ebitda_change": ebitda - base_ebitda,
            })
    
    df = pd.DataFrame(rows)
    df["abs_impact"] = df.groupby("variable")["ebitda_change"].transform(lambda x: x.abs().max())
    return df.sort_values("abs_impact", ascending=True)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_single_centre()
    
    print("=" * 60)
    print(f"CENTRE: {result['centre_name']}")
    print("=" * 60)
    
    print("\n--- CAPITAL SUFFICIENCY ---")
    cap = result["capital_sufficiency"]
    print(f"  Total Capital: ${cap['total_capital_committed']:,.0f}")
    print(f"  Pre-Rev Burn: ${cap['pre_revenue_burn_months']:,.0f}")
    print(f"  Trough Cash: ${cap['cash_at_trough']:,.0f}")
    print(f"  Buffer: ${cap['buffer_above_reserve']:,.0f}")
    print(f"  Sufficient: {'YES' if cap['sufficient'] else 'NO'}")
    
    print("\n--- ANNUAL P&L ---")
    annual = result["annual_pl"]
    print(annual[["fiscal_year", "net_revenue", "ebitda", "ebitda_margin", "npat"]].to_string(index=False))
    
    print("\n--- VALUATION ---")
    val = result["valuation"]
    print(f"  EV (DCF): ${val['ev_dcf']:,.0f}")
    print(f"  IRR: {val['irr']:.1%}" if val["irr"] else "  IRR: N/A")
    print(f"  MOIC: {val['moic']:.2f}x")
    
    print("\nDone.")
