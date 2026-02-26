"""
BrightBlocks ELC - Childcare PE Financial Model Engine (Phase 5)
Fully parameterised. Supports multi-centre, adjustable pre-revenue
period, monthly ECT/Support staffing overrides, capacity compliance
flags, director salary deferral, standard financial statements.
"""
import math, io, zipfile
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from copy import deepcopy

# ── IRR Solver ──────────────────────────────────────────────

def _irr(cashflows, tol=1e-8, max_iter=1000):
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
    return None if abs(final_npv) > 1000 else r

# ── Default Config (Earlwood) ───────────────────────────────

EARLWOOD_CONFIG = {
    "centre_id": "earlwood_001",
    "centre_name": "Bright Blocks ELC \u2013 Earlwood",
    "entity": "Bright Blocks ELC \u2013 Earlwood Pty Ltd",
    "acn": "685 365 253",
    "address": "Ground Floor, 205 Homer Street, Earlwood NSW 2206",
    "state": "NSW",
    "approved_capacity": 40,
    "commencement_date": "2026-07-01",
    "pre_revenue_months": 3,
    "practical_completion_delay_months": 0,
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
    "seasonality_factors": {1:0.85,2:0.95,3:1.0,4:1.0,5:1.0,6:0.95,7:0.90,8:0.95,9:1.0,10:1.0,11:1.0,12:0.85},
    "occupancy_ramp": {"m1_3":0.0,"m4":0.70,"m5_6":0.75,"y2":0.85,"y3":0.95,"y4_plus":0.975,"steady_state":0.95},
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
    "agency_premium": 0.25,
    "vacancy_duration_weeks": 4,
    "ect_monthly_override": None,
    "support_monthly_override": None,
    "support_unpaid_pre_revenue": False,
    "director_salary_deferred": True,
    "dir_trigger_ebitda_threshold": 7500.0,
    "dir_trigger_cash_threshold": 30000.0,
    "dir_trigger_occupancy_threshold": 0.60,
    "dir_trigger_consecutive_months": 2,
    "base_rent_pa": 230000.0,
    "rent_escalation": 0.03,
    "rent_free_months": 6,
    "rent_free_credit": 115000.0,
    "initial_lease_term_years": 5,
    "further_options": "5x5",
    "make_good_provision_pa": 15000.0,
    "outgoings": {"strata":1250,"electricity":500,"gas":333,"water":58,"council_rates":350,"waste":167,"r_and_m":500},
    "land_tax_monthly_pre_approval": 1283.0,
    "insurance": {"public_liability":200,"professional_indemnity":175,"management_liability":150,"contents":100,"business_interruption":150},
    "insurance_escalation_rate": 0.06,
    "admin_opex": {"accounting":978,"bank_fees":50,"cleaning":1500,"software":400,"pest_control":83,"phone_internet":200,"toys_books":250,"training":500,"marketing":3300,"extra_curricular":1000,"miscellaneous":500,"compliance":250},
    "marketing_pre_opening": 1500.0,
    "cpi_on_expenses": 0.02,
    "food_cost_per_child_per_day": 8.50,
    "food_delivery_monthly": 200.0,
    "capital_items": {"bank_guarantee":126500,"legal_professional":23000,"signage":5000,"marketing_setup":23009,"other_pre_opening":18709,"loose_furniture":35000,"operational_seed":75000},
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

# ── Multi-Centre Registry ───────────────────────────────────

def _make_sample_centre_2():
    c = deepcopy(EARLWOOD_CONFIG)
    c.update({"centre_id":"marrickville_002","centre_name":"Bright Blocks ELC \u2013 Marrickville",
              "entity":"Bright Blocks ELC \u2013 Marrickville Pty Ltd","acn":"700 000 001",
              "address":"10 Illawarra Rd, Marrickville NSW 2204","approved_capacity":60,
              "commencement_date":"2027-01-01","pre_revenue_months":4,"base_rent_pa":280000.0})
    c["daily_fees"]["36m_plus"] = 145.0
    c["capital_items"]["operational_seed"] = 90000
    c["capital_items"]["bank_guarantee"] = 140000
    c["director_loan_opening"] = 230000.0
    c["total_deployed_director_loans"] = 230000.0
    c["amortisation_pre_opening_total"] = 230000.0
    return c

CENTRE_REGISTRY = {"earlwood_001": EARLWOOD_CONFIG, "marrickville_002": _make_sample_centre_2()}

def get_centre_config(centre_id): return deepcopy(CENTRE_REGISTRY[centre_id])
def list_centres(): return [{"id":k,"name":v["centre_name"]} for k,v in CENTRE_REGISTRY.items()]
def register_centre(cfg): CENTRE_REGISTRY[cfg["centre_id"]] = cfg

# ── B1: Time Series ─────────────────────────────────────────

def build_time_series(cfg):
    start = pd.Timestamp(cfg["commencement_date"]) + pd.DateOffset(months=cfg.get("practical_completion_delay_months",0))
    months = cfg["model_months"]; pre_rev = cfg.get("pre_revenue_months",3)
    dates = pd.date_range(start, periods=months, freq="MS")
    df = pd.DataFrame({"month":range(1,months+1),"date":dates,"calendar_month":dates.month,
                        "fiscal_year":[(d.year+1 if d.month>=7 else d.year) for d in dates]})
    yb = [0]
    for i in range(1,len(dates)):
        yb.append(((dates[i].year-dates[0].year)*12+dates[i].month-dates[0].month)//12)
    df["year_from_start"] = yb
    df["is_pre_revenue"] = df["month"] <= pre_rev
    return df

# ── B2: Occupancy ───────────────────────────────────────────

def calc_occupancy(cfg, ts):
    ramp = cfg["occupancy_ramp"]; cap = cfg["approved_capacity"]
    pre_rev = cfg.get("pre_revenue_months",3)
    occ_pcts = []
    for _, row in ts.iterrows():
        m = row["month"]
        if m <= pre_rev:
            occ = 0.0
        else:
            mso = m - pre_rev
            if mso == 1: occ = ramp["m4"]
            elif mso <= 3: occ = ramp["m5_6"]
            elif mso <= 9: occ = ramp["m5_6"]
            elif mso <= 21: occ = ramp["y2"]
            elif mso <= 33: occ = ramp["y3"]
            else: occ = ramp["y4_plus"]
        if cfg.get("seasonality_toggle") and occ > 0:
            occ *= cfg["seasonality_factors"].get(row["calendar_month"],1.0)
        occ_pcts.append(min(occ,1.0))
    df = ts.copy(); df["occupancy_pct"] = occ_pcts
    df["children"] = [round(cap*o) for o in occ_pcts]
    return df

# ── B3: Revenue ─────────────────────────────────────────────

def calc_revenue(cfg, occ):
    df = occ.copy()
    fees = cfg["daily_fees"]; weights = cfg["fee_mix_weights"]
    blended = sum(fees[k]*weights[k] for k in fees)
    biz = cfg["biz_days_per_month"]; coll = cfg["revenue_collection_rate"]; fi = cfg["fee_annual_increase"]
    df["daily_fee"] = [blended*(1+fi)**r["year_from_start"] for _,r in df.iterrows()]
    df["gross_revenue"] = df["children"]*df["daily_fee"]*biz
    df["bad_debt"] = df["gross_revenue"]*(1-coll)
    df["net_revenue"] = df["gross_revenue"]*coll
    df["ccs_portion"] = df["net_revenue"]*cfg["ccs_eligibility_pct"]*cfg["ccs_subsidy_rate"]
    df["gap_fee_portion"] = df["net_revenue"]-df["ccs_portion"]
    return df

# -- B4+B5: Staffing (ECT/Support overrides, capacity compliance) --

def calc_staffing(cfg, rev):
    df = rev.copy()
    ratio = cfg["educator_child_ratio"]; ed_hr = cfg["educator_hourly_rate"]
    sup_hr = cfg["support_hourly_rate"]; hrs = cfg["hours_per_month"]
    sup_rate = cfg["superannuation_rate"]; wc_rate = cfg["workers_comp_rate"]
    wage_inc = cfg["staff_wage_annual_increase"]
    turnover = cfg["staff_turnover_rate"]; recruit = cfg["recruitment_cost_per_hire"]
    pre_rev = cfg.get("pre_revenue_months",3)
    ect_ov = cfg.get("ect_monthly_override"); sup_ov = cfg.get("support_monthly_override")
    sup_unpaid = cfg.get("support_unpaid_pre_revenue",False)

    educators=[]; support=[]; director_flag=[]; gross_wages=[]
    super_cost=[]; wc_cost=[]; turnover_cost=[]; total_staff=[]
    meets_capacity=[]
    director_triggered=False; ebitda_streak=0; occ_streak=0; cumul_approx=0.0

    for i, row in df.iterrows():
        m=row["month"]; yr=row["year_from_start"]; children=row["children"]
        is_pre=row["is_pre_revenue"]; esc=(1+wage_inc)**yr

        # ECT count
        if ect_ov and m in ect_ov: ed=int(ect_ov[m])
        elif children==0 and not is_pre: ed=0
        elif is_pre: ed=1
        else: ed=max(1,math.ceil(children/ratio))

        # Support count
        if sup_ov and m in sup_ov: sc=int(sup_ov[m])
        elif children==0 and not is_pre: sc=0
        elif is_pre: sc=0
        else: sc=1

        educators.append(ed); support.append(sc)

        # Capacity compliance (post-revenue only)
        if not is_pre and children>0:
            meets_capacity.append(ed >= max(1,math.ceil(children/ratio)))
        else:
            meets_capacity.append(True)

        ed_wage = ed*ed_hr*hrs*esc
        sup_wage = 0.0 if (is_pre and sup_unpaid) else sc*sup_hr*hrs*esc

        # Director salary trigger
        if not cfg.get("director_salary_deferred",True):
            director_triggered = True
        if not director_triggered and not is_pre:
            approx_ebitda = row["net_revenue"]*0.55 if row["net_revenue"]>0 else 0
            cumul_approx += approx_ebitda - (ed_wage+sup_wage)*(1+sup_rate)
            if approx_ebitda > cfg.get("dir_trigger_ebitda_threshold",7500): ebitda_streak+=1
            else: ebitda_streak=0
            if row["occupancy_pct"] >= cfg.get("dir_trigger_occupancy_threshold",0.60): occ_streak+=1
            else: occ_streak=0
            consec = cfg.get("dir_trigger_consecutive_months",2)
            if ebitda_streak>consec and occ_streak>consec and cumul_approx>0:
                director_triggered = True

        dir_active = 1 if director_triggered and (children>0 or is_pre) else 0
        if cfg.get("director_salary_deferred",True) and not director_triggered: dir_active=0
        director_flag.append(dir_active)
        dir_wage = cfg["director_hourly_rate"]*hrs*esc*dir_active

        tg = ed_wage+sup_wage+dir_wage; gross_wages.append(tg)
        s=tg*sup_rate; super_cost.append(s)
        w=tg*wc_rate; wc_cost.append(w)
        fte=ed+sc+dir_active
        tc = fte*turnover*recruit/12 if fte>0 and not is_pre else 0
        turnover_cost.append(tc); total_staff.append(tg+s+w+tc)

    df["educators"]=educators; df["support_staff"]=support
    df["director_active"]=director_flag
    df["total_fte"]=df["educators"]+df["support_staff"]+df["director_active"]
    df["gross_wages"]=gross_wages; df["super_cost"]=super_cost
    df["wc_cost"]=wc_cost; df["turnover_cost"]=turnover_cost
    df["total_staff_cost"]=total_staff; df["meets_capacity_limits"]=meets_capacity
    df["wages_pct_revenue"]=df.apply(lambda r: r["total_staff_cost"]/r["net_revenue"] if r["net_revenue"]>0 else 0, axis=1)
    return df

# -- B6: Rent & Occupancy --

def calc_rent(cfg, ts):
    df = ts.copy()
    base=cfg["base_rent_pa"]; esc=cfg["rent_escalation"]; free_m=cfg["rent_free_months"]
    mg_pa=cfg["make_good_provision_pa"]; outg=cfg["outgoings"]
    lt=cfg.get("land_tax_monthly_pre_approval",0); cpi=cfg["cpi_on_expenses"]
    pre_rev=cfg.get("pre_revenue_months",3)
    rp=[]; ot=[]; ltc=[]; mg=[]; toc=[]
    for _,row in df.iterrows():
        m=row["month"]; yr=row["year_from_start"]
        mr=base*(1+esc)**yr/12
        r_pay = 0.0 if m<=free_m else mr; rp.append(r_pay)
        oe=(1+cpi)**yr; o_tot=sum(outg.values())*oe; ot.append(o_tot)
        l=lt if m<=pre_rev else 0.0; ltc.append(l)
        m_mg=mg_pa/12; mg.append(m_mg)
        toc.append(r_pay+o_tot+l+m_mg)
    df["rent_payable"]=rp; df["outgoings"]=ot; df["land_tax"]=ltc
    df["make_good"]=mg; df["total_occupancy_cost"]=toc
    return df

# -- B7: Opex --

def calc_opex(cfg, staff):
    df = staff.copy()
    ins=cfg["insurance"]; admin=cfg["admin_opex"]
    ie=cfg["insurance_escalation_rate"]; cpi=cfg["cpi_on_expenses"]
    pre_rev=cfg.get("pre_revenue_months",3)
    it=[]; at=[]; dso=[]; to=[]
    for _,row in df.iterrows():
        m=row["month"]; yr=row["year_from_start"]
        im=(1+ie)**yr; cm=(1+cpi)**yr
        i_t=sum(ins.values())*im; it.append(i_t)
        adm=0.0
        for k,v in admin.items():
            if k=="marketing": val=cfg.get("marketing_pre_opening",1500) if m<=pre_rev else v
            elif k=="extra_curricular": val=0.0 if m<=pre_rev else v
            elif k=="software": val=v if m>=max(3,pre_rev) else 0.0
            else: val=v
            adm+=val*cm
        at.append(adm)
        ds=(cfg["director_annual_cost"]/12)*((1+cfg["staff_wage_annual_increase"])**yr)*row["director_active"]
        dso.append(ds); to.append(i_t+adm+ds)
    df["insurance_total"]=it; df["admin_total"]=at
    df["director_salary_opex"]=dso; df["total_opex"]=to
    return df

# -- B8: Food --

def calc_food(cfg, occ):
    df = occ.copy()
    cp=cfg["food_cost_per_child_per_day"]; dl=cfg["food_delivery_monthly"]
    biz=cfg["biz_days_per_month"]; cpi=cfg["cpi_on_expenses"]
    fc=[]
    for _,row in df.iterrows():
        e=(1+cpi)**row["year_from_start"]
        f=row["children"]*cp*e*biz + (dl*e if row["children"]>0 else 0)
        fc.append(f)
    df["food_cost"]=fc
    return df

# -- B9+B10: Monthly P&L --

def build_monthly_pl(cfg):
    ts=build_time_series(cfg); occ=calc_occupancy(cfg,ts)
    rev=calc_revenue(cfg,occ); staff=calc_staffing(cfg,rev)
    rent=calc_rent(cfg,ts); opex=calc_opex(cfg,staff); food=calc_food(cfg,occ)
    df=pd.DataFrame()
    df["month"]=ts["month"]; df["date"]=ts["date"]
    df["fiscal_year"]=ts["fiscal_year"]; df["year_from_start"]=ts["year_from_start"]
    df["is_pre_revenue"]=ts["is_pre_revenue"]
    df["occupancy_pct"]=occ["occupancy_pct"]; df["children"]=occ["children"]
    df["net_revenue"]=rev["net_revenue"]; df["ccs_portion"]=rev["ccs_portion"]
    df["gap_fee_portion"]=rev["gap_fee_portion"]
    df["total_staff_cost"]=staff["total_staff_cost"]; df["food_cost"]=food["food_cost"]
    df["direct_costs"]=df["total_staff_cost"]+df["food_cost"]
    df["gross_profit"]=df["net_revenue"]-df["direct_costs"]
    df["gp_margin"]=df.apply(lambda r: r["gross_profit"]/r["net_revenue"] if r["net_revenue"]>0 else 0, axis=1)
    df["total_occupancy_cost"]=rent["total_occupancy_cost"]; df["rent_payable"]=rent["rent_payable"]
    df["total_opex"]=opex["total_opex"]
    df["total_overheads"]=df["total_occupancy_cost"]+df["total_opex"]
    df["ebitda"]=df["gross_profit"]-df["total_overheads"]
    df["ebitda_margin"]=df.apply(lambda r: r["ebitda"]/r["net_revenue"] if r["net_revenue"]>0 else 0, axis=1)
    fm=cfg["depreciation_furniture_total"]/(cfg["depreciation_years"]*12)
    pm=cfg["amortisation_pre_opening_total"]/(cfg["depreciation_years"]*12)
    mm=cfg["make_good_provision_pa"]/12
    df["depreciation_furniture"]=fm; df["amortisation_pre_opening"]=pm
    df["depreciation_make_good"]=mm; df["total_da"]=fm+pm+mm
    df["ebit"]=df["ebitda"]-df["total_da"]
    df["interest_expense"]=cfg["director_loan_opening"]*cfg["director_loan_interest_rate"]/12
    df["npbt"]=df["ebit"]-df["interest_expense"]
    df["cumul_npbt"]=df["npbt"].cumsum()
    tax=[]
    for _,row in df.iterrows():
        tax.append(row["npbt"]*cfg["company_tax_rate"] if row["cumul_npbt"]>0 and row["npbt"]>0 else 0.0)
    df["tax"]=tax; df["npat"]=df["npbt"]-df["tax"]
    df["educators"]=staff["educators"]; df["support_staff"]=staff["support_staff"]
    df["director_active"]=staff["director_active"]; df["total_fte"]=staff["total_fte"]
    df["gross_wages"]=staff["gross_wages"]; df["wages_pct_revenue"]=staff["wages_pct_revenue"]
    df["meets_capacity_limits"]=staff["meets_capacity_limits"]
    return df

# -- B11: Debt Schedule --

def calc_debt_schedule(cfg, pl):
    rate=cfg["director_loan_interest_rate"]; repay_amt=cfg["director_loan_repayment_monthly"]
    opening=cfg["director_loan_opening"]; pre_rev=cfg.get("pre_revenue_months",3)
    rows=[]; bal=opening
    for _,row in pl.iterrows():
        interest=bal*rate/12
        can_repay = row["month"]>pre_rev and row["ebitda"]>0
        if can_repay and bal>repay_amt: repayment=repay_amt
        elif can_repay and bal>0: repayment=bal+interest
        else: repayment=0
        closing=max(bal+interest-repayment,0)
        rows.append({"month":row["month"],"date":row["date"],"opening_balance":bal,
                      "interest":interest,"repay_eligible":can_repay,
                      "repayment":repayment,"closing_balance":closing})
        bal=closing
    return pd.DataFrame(rows)

# -- B12: Cash Flow --

def calc_cashflow(cfg, pl, debt):
    df=pd.DataFrame()
    df["month"]=pl["month"]; df["date"]=pl["date"]; df["fiscal_year"]=pl["fiscal_year"]
    df["npat"]=pl["npat"]; df["da"]=pl["total_da"]
    wc=[]; pr=0; pc=0
    for _,row in pl.iterrows():
        dc=row["net_revenue"]-pr; cc=(row["total_staff_cost"]+row["food_cost"]+row["total_opex"])-pc
        wc.append(cc-dc); pr=row["net_revenue"]; pc=row["total_staff_cost"]+row["food_cost"]+row["total_opex"]
    df["wc_movement"]=wc
    tp=pl["total_occupancy_cost"]+pl["total_opex"]+pl["food_cost"]
    df["gst_input_credits"]=tp*cfg["gst_rate"]/(1+cfg["gst_rate"])
    df["operating_cf"]=df["npat"]+df["da"]+df["wc_movement"]+df["gst_input_credits"]
    df["maintenance_capex"]=-cfg["maintenance_capex_pa"]/12
    df["investing_cf"]=df["maintenance_capex"]
    df["loan_repayments"]=-debt["repayment"]
    df["financing_cf"]=df["loan_repayments"]
    df["net_cash_movement"]=df["operating_cf"]+df["investing_cf"]+df["financing_cf"]
    op=cfg["capital_items"]["operational_seed"]
    cum=[op+df["net_cash_movement"].iloc[0]]
    for i in range(1,len(df)): cum.append(cum[-1]+df["net_cash_movement"].iloc[i])
    df["closing_cash"]=cum
    return df

# -- B13: GST/BAS --

def calc_gst_bas(cfg, pl):
    df=pl[["month","date","fiscal_year"]].copy()
    tp=pl["total_occupancy_cost"]+pl["total_opex"]+pl["food_cost"]
    df["gst_collected"]=0.0
    df["gst_paid_inputs"]=tp*cfg["gst_rate"]/(1+cfg["gst_rate"])
    df["net_gst_position"]=df["gst_collected"]-df["gst_paid_inputs"]
    df["quarter"]=df["date"].dt.to_period("Q")
    q=df.groupby("quarter").agg({"gst_collected":"sum","gst_paid_inputs":"sum","net_gst_position":"sum"}).reset_index()
    q["refund"]=-q["net_gst_position"]
    return q

# -- B14: Balance Sheet --

def calc_balance_sheet(cfg, pl, cf, debt):
    ft=cfg["depreciation_furniture_total"]; pt=cfg["amortisation_pre_opening_total"]
    dy=cfg["depreciation_years"]
    annual=pl.copy(); annual["fy"]=pl["fiscal_year"]
    rows=[]
    for fy,grp in annual.groupby("fy"):
        li=grp.index[-1]; lm=grp["month"].iloc[-1]; yn=lm/12
        cash=cf["closing_cash"].iloc[li]; debtors=grp["net_revenue"].iloc[-1]
        fn=max(ft-(ft/dy)*min(yn,dy),0); pn=max(pt-(pt/dy)*min(yn,dy),0)
        ta=cash+debtors+fn+pn
        cred=grp["total_staff_cost"].iloc[-1]+grp["food_cost"].iloc[-1]+grp["total_opex"].iloc[-1]
        dl=debt["closing_balance"].iloc[li]; mg=(cfg["make_good_provision_pa"]/12)*lm
        tp=grp["tax"].iloc[-1]; tl=cred+dl+mg+tp
        eq=ta-tl
        rows.append({"fiscal_year":fy,"cash":cash,"trade_debtors":debtors,
                      "furniture_net":fn,"pre_opening_net":pn,"total_assets":ta,
                      "trade_creditors":cred,"director_loan":dl,
                      "make_good_provision":mg,"tax_payable":tp,
                      "total_liabilities":tl,"retained_earnings":eq,"total_equity":eq,
                      "balance_check":round(ta-tl-eq,2)})
    return pd.DataFrame(rows)

# -- B15: Annual P&L --

def calc_annual_pl(pl):
    agg=pl.groupby("fiscal_year").agg({
        "net_revenue":"sum","total_staff_cost":"sum","food_cost":"sum",
        "direct_costs":"sum","gross_profit":"sum","total_occupancy_cost":"sum",
        "total_opex":"sum","total_overheads":"sum","ebitda":"sum",
        "total_da":"sum","ebit":"sum","interest_expense":"sum",
        "npbt":"sum","tax":"sum","npat":"sum"}).reset_index()
    agg["ebitda_margin"]=agg.apply(lambda r: r["ebitda"]/r["net_revenue"] if r["net_revenue"]>0 else 0, axis=1)
    agg["gp_margin"]=agg.apply(lambda r: r["gross_profit"]/r["net_revenue"] if r["net_revenue"]>0 else 0, axis=1)
    agg["yoy_revenue_growth"]=agg["net_revenue"].pct_change()
    return agg

# -- Standard Financial Statements (formatted) --

def format_pl_statement(annual):
    lines=["Revenue","Cost of Sales \u2013 Staff","Cost of Sales \u2013 Food","Gross Profit","",
           "Occupancy Costs","Admin & Other Opex","EBITDA","",
           "Depreciation & Amortisation","EBIT","Interest Expense",
           "Net Profit Before Tax","Income Tax","Net Profit After Tax"]
    result=pd.DataFrame({"Line Item":lines})
    for _,r in annual.iterrows():
        fy=f"FY{int(r['fiscal_year'])}"
        result[fy]=[r["net_revenue"],-r["total_staff_cost"],-r["food_cost"],r["gross_profit"],None,
                    -r["total_occupancy_cost"],-r["total_opex"],r["ebitda"],None,
                    -r["total_da"],r["ebit"],-r["interest_expense"],
                    r["npbt"],-r["tax"],r["npat"]]
    return result

def format_balance_sheet(bs):
    lines=["ASSETS","Cash & Cash Equivalents","Trade Debtors",
           "Furniture & Equipment (net)","Pre-Opening Costs (net)","Total Assets","",
           "LIABILITIES","Trade Creditors","Director Loan",
           "Make Good Provision","Tax Payable","Total Liabilities","",
           "EQUITY","Retained Earnings","Total Equity"]
    result=pd.DataFrame({"Line Item":lines})
    for _,r in bs.iterrows():
        fy=f"FY{int(r['fiscal_year'])}"
        result[fy]=[None,r["cash"],r["trade_debtors"],r["furniture_net"],r["pre_opening_net"],
                    r["total_assets"],None,None,r["trade_creditors"],r["director_loan"],
                    r["make_good_provision"],r["tax_payable"],r["total_liabilities"],None,
                    None,r["retained_earnings"],r["total_equity"]]
    return result

def format_cashflow_statement(cf):
    cfa=cf.groupby("fiscal_year").agg({"npat":"sum","da":"sum","wc_movement":"sum",
        "gst_input_credits":"sum","operating_cf":"sum","maintenance_capex":"sum",
        "investing_cf":"sum","loan_repayments":"sum","financing_cf":"sum",
        "net_cash_movement":"sum"}).reset_index()
    cfa["closing_cash"]=cf.groupby("fiscal_year")["closing_cash"].last().values
    lines=["OPERATING ACTIVITIES","Net Profit After Tax","Add Back: D&A",
           "Working Capital Movement","GST Input Credits","Net Operating Cash Flow","",
           "INVESTING ACTIVITIES","Maintenance Capex","Net Investing Cash Flow","",
           "FINANCING ACTIVITIES","Loan Repayments","Net Financing Cash Flow","",
           "Net Cash Movement","Closing Cash Balance"]
    result=pd.DataFrame({"Line Item":lines})
    for _,r in cfa.iterrows():
        fy=f"FY{int(r['fiscal_year'])}"
        result[fy]=[None,r["npat"],r["da"],r["wc_movement"],r["gst_input_credits"],
                    r["operating_cf"],None,None,r["maintenance_capex"],r["investing_cf"],None,
                    None,r["loan_repayments"],r["financing_cf"],None,
                    r["net_cash_movement"],r["closing_cash"]]
    return result

# -- B16-B18: Valuation --

def calc_valuation(cfg, annual):
    capex=cfg["maintenance_capex_pa"]
    annual_fcf=[row["npat"]+row["total_da"]-capex for _,row in annual.iterrows()]
    r=cfg["discount_rate"]; g=cfg["terminal_growth_rate"]
    y5f=annual_fcf[-1] if annual_fcf else 0
    tv=y5f*(1+g)/(r-g) if (r-g)>0 else 0
    pvf=sum(f/(1+r)**(i+1) for i,f in enumerate(annual_fcf))
    pvt=tv/(1+r)**len(annual_fcf) if annual_fcf else 0
    ev_dcf=pvf+pvt
    y5e=annual["ebitda"].iloc[-1] if len(annual)>0 else 0
    ev_m={f"{m}x":y5e*m for m in cfg["exit_multiples"]}
    tc=sum(cfg["capital_items"].values())
    bm=cfg["exit_multiples"][1] if len(cfg["exit_multiples"])>1 else cfg["exit_multiples"][0]
    ev_exit=y5e*bm; moic=ev_exit/tc if tc>0 else 0
    irf=[-tc]+annual_fcf[:-1]+[annual_fcf[-1]+ev_exit] if annual_fcf else [-tc]
    try: irr=_irr(irf)
    except: irr=None
    cum=0; pb=None
    for i,f in enumerate(annual_fcf):
        cum+=f
        if cum>=tc:
            pb=i+(tc-(cum-f))/f if f>0 else None; break
    return {"annual_fcf":annual_fcf,"y5_ebitda":y5e,"y5_fcf":y5f,
            "terminal_value":tv,"pv_fcfs":pvf,"pv_terminal":pvt,
            "ev_dcf":ev_dcf,"ev_multiples":ev_m,"total_capital":tc,
            "moic":moic,"irr":irr,"payback_years":pb,"irr_flows":irf}

# -- Sensitivity --

def sensitivity_occupancy_fee(cfg, occupancies=None, fees=None):
    if not occupancies: occupancies=[0.60,0.70,0.80,0.90,1.00]
    if not fees: fees=[125,135,139,145,155]
    res=[]
    for o in occupancies:
        for f in fees:
            c=deepcopy(cfg); c["occupancy_ramp"]["y3"]=o; c["occupancy_ramp"]["y4_plus"]=o
            c["daily_fees"]["36m_plus"]=f; pl=build_monthly_pl(c); a=calc_annual_pl(pl)
            res.append({"occupancy":o,"daily_fee":f,"y3_ebitda":a["ebitda"].iloc[min(2,len(a)-1)]})
    return pd.DataFrame(res).pivot(index="occupancy",columns="daily_fee",values="y3_ebitda")

def sensitivity_dcf(cfg, discount_rates=None, growth_rates=None):
    if not discount_rates: discount_rates=[0.08,0.10,0.12,0.15]
    if not growth_rates: growth_rates=[0.015,0.025,0.035]
    pl=build_monthly_pl(cfg); a=calc_annual_pl(pl); res=[]
    for dr in discount_rates:
        for gr in growth_rates:
            c=deepcopy(cfg); c["discount_rate"]=dr; c["terminal_growth_rate"]=gr
            res.append({"discount_rate":dr,"terminal_growth":gr,"ev_dcf":calc_valuation(c,a)["ev_dcf"]})
    return pd.DataFrame(res).pivot(index="discount_rate",columns="terminal_growth",values="ev_dcf")

def sensitivity_occupancy_wages(cfg, occupancies=None, wage_rates=None):
    if not occupancies: occupancies=[0.70,0.80,0.85,0.90,0.95,1.00]
    if not wage_rates:
        b=cfg["educator_hourly_rate"]; wage_rates=[round(b*m,2) for m in [0.85,0.92,1.0,1.08,1.15]]
    res=[]
    for o in occupancies:
        for w in wage_rates:
            c=deepcopy(cfg); c["occupancy_ramp"]["y3"]=o; c["occupancy_ramp"]["y4_plus"]=o
            c["educator_hourly_rate"]=w; pl=build_monthly_pl(c); a=calc_annual_pl(pl)
            res.append({"occupancy":o,"educator_rate":w,"y3_ebitda":a["ebitda"].iloc[min(2,len(a)-1)]})
    return pd.DataFrame(res).pivot(index="occupancy",columns="educator_rate",values="y3_ebitda")

def sensitivity_fee_wages(cfg, fees=None, wage_rates=None):
    if not fees: fees=[125,130,139,145,155]
    if not wage_rates:
        b=cfg["educator_hourly_rate"]; wage_rates=[round(b*m,2) for m in [0.85,0.92,1.0,1.08,1.15]]
    res=[]
    for f in fees:
        for w in wage_rates:
            c=deepcopy(cfg); c["daily_fees"]["36m_plus"]=f; c["educator_hourly_rate"]=w
            pl=build_monthly_pl(c); a=calc_annual_pl(pl)
            res.append({"daily_fee":f,"educator_rate":w,"y3_ebitda":a["ebitda"].iloc[min(2,len(a)-1)]})
    return pd.DataFrame(res).pivot(index="daily_fee",columns="educator_rate",values="y3_ebitda")

def tornado_sensitivity(cfg, target_year=3):
    br=run_single_centre(cfg); ba=br["annual_pl"]
    idx=min(target_year-1,len(ba)-1); be=ba["ebitda"].iloc[idx]
    variables=[("Occupancy (Y3)","occupancy_ramp.y3",-0.10,+0.10),
        ("Daily Fee (36m+)","daily_fees.36m_plus",-15.0,+15.0),
        ("Educator Rate","educator_hourly_rate",-5.0,+5.0),
        ("Base Rent p.a.","base_rent_pa",-30000,+30000),
        ("Wage Escalation","staff_wage_annual_increase",-0.01,+0.01),
        ("Fee Escalation","fee_annual_increase",-0.01,+0.01),
        ("CPI","cpi_on_expenses",-0.01,+0.01),
        ("Pre-Revenue Months","pre_revenue_months",-1,+2),
        ("Capacity","approved_capacity",-10,+10)]
    rows=[]
    for label,kp,ld,hd in variables:
        parts=kp.split(".")
        for delta,side in [(ld,"low"),(hd,"high")]:
            c=deepcopy(cfg)
            if len(parts)==2: bv=c[parts[0]][parts[1]]; c[parts[0]][parts[1]]=bv+delta
            else: bv=c[parts[0]]; c[parts[0]]=bv+delta
            try: pl=build_monthly_pl(c); a=calc_annual_pl(pl); e=a["ebitda"].iloc[idx]
            except: e=be
            rows.append({"variable":label,"side":side,"delta":delta,"base_value":bv,
                         "adjusted_value":bv+delta,"ebitda":e,"ebitda_change":e-be})
    df=pd.DataFrame(rows)
    df["abs_impact"]=df.groupby("variable")["ebitda_change"].transform(lambda x: x.abs().max())
    return df.sort_values("abs_impact",ascending=True)

# -- Scenarios --

SCENARIO_PRESETS = {
    "bear": {"label":"Bear (Downside)",
        "occupancy_ramp":{"m4":0.55,"m5_6":0.65,"y2":0.78,"y3":0.85,"y4_plus":0.90},
        "daily_fees":{"36m_plus":133.0,"24_36m":130.0,"0_24m":138.0},
        "fee_annual_increase":0.025,"staff_wage_annual_increase":0.035,
        "cpi_on_expenses":0.025,"exit_multiples":[3.5,4.5,5.5],"pre_revenue_months":4},
    "base": {"label":"Base Case"},
    "bull": {"label":"Bull (Upside)",
        "occupancy_ramp":{"m4":0.80,"m5_6":0.85,"y2":0.95,"y3":0.98,"y4_plus":1.0},
        "daily_fees":{"36m_plus":152.0,"24_36m":149.0,"0_24m":158.0},
        "fee_annual_increase":0.04,"staff_wage_annual_increase":0.025,
        "cpi_on_expenses":0.015,"exit_multiples":[6.0,7.0,8.5],"pre_revenue_months":2},
}

def apply_scenario_overrides(cfg, overrides):
    c=deepcopy(cfg)
    for k,v in overrides.items():
        if k=="label": continue
        if isinstance(v,dict) and k in c and isinstance(c[k],dict): c[k].update(v)
        else: c[k]=v
    return c

def run_scenario(cfg, scenario="base", custom_overrides=None):
    preset=SCENARIO_PRESETS.get(scenario,{})
    c=apply_scenario_overrides(cfg,preset)
    if custom_overrides: c=apply_scenario_overrides(c,custom_overrides)
    return run_single_centre(c)

# -- Capital Sufficiency --

def calc_capital_sufficiency(cfg, pl):
    items=cfg["capital_items"]; tc=sum(items.values())
    oc=items["operational_seed"]; res=cfg["minimum_operating_reserve"]
    pre_rev=cfg.get("pre_revenue_months",3)
    prd=pl[pl["month"]<=pre_rev]
    burn=abs(prd["ebitda"].sum()) if len(prd)>0 else 0
    cat=oc-burn; buf=cat-res
    wr=buf/(burn/(pre_rev*4.33)) if burn>0 else float("inf")
    return {"total_capital_committed":tc,"already_deployed":cfg.get("total_deployed_director_loans",0),
            "opening_operational_cash":oc,"minimum_reserve":res,"net_deployable":oc-res,
            "pre_revenue_burn":burn,"pre_revenue_months":pre_rev,
            "cash_at_trough":cat,"buffer_above_reserve":buf,"sufficient":buf>0,"weeks_runway":wr}

# -- Per-child / Weekly / Benchmarks --

def calc_per_child(annual, pl):
    ac=pl.groupby("fiscal_year")["children"].mean()
    pc=annual.copy(); pc["avg_children"]=pc["fiscal_year"].map(ac)
    for col in ["net_revenue","total_staff_cost","food_cost","total_opex","ebitda","npat"]:
        pc[f"{col}_per_child"]=pc.apply(lambda r: r[col]/r["avg_children"] if r["avg_children"]>0 else 0, axis=1)
    return pc[["fiscal_year","avg_children","net_revenue_per_child","total_staff_cost_per_child",
               "food_cost_per_child","total_opex_per_child","ebitda_per_child","npat_per_child"]]

def calc_weekly_cashflow(cfg, pl):
    op=cfg["capital_items"]["operational_seed"]; res=cfg["minimum_operating_reserve"]
    pre_rev=cfg.get("pre_revenue_months",3); rows=[]; cum=op
    for w in range(1,27):
        m=min((w-1)//4+1,6); mi=m-1
        if mi<len(pl):
            rd=pl.iloc[mi]; wr=rd["net_revenue"]/4.33; ww=rd["total_staff_cost"]/4.33
            wo=rd["total_occupancy_cost"]/4.33; wx=rd["total_opex"]/4.33
        else: wr=ww=wo=wx=0
        if m==pre_rev+1 and w<=(pre_rev+1)*4: wr*=0.4
        net=wr-ww-wo-wx; cum+=net; buf=cum-res
        rag="RED" if buf<0 else ("AMBER" if buf<10000 else "GREEN")
        rows.append({"week":w,"month":m,"revenue":wr,"wages":ww,"outgoings":wo,
                      "other_opex":wx,"net_movement":net,"cumulative_cash":cum,
                      "buffer_above_reserve":buf,"rag_status":rag})
    return pd.DataFrame(rows)

def calc_benchmarks(cfg, annual):
    cap=cfg["approved_capacity"]; rows=[]
    for i,label in enumerate(["Year 1","Year 3"]):
        idx=0 if i==0 else min(2,len(annual)-1); r=annual.iloc[idx]
        rows.append({"period":label,"ebitda_margin":r["ebitda_margin"],
            "wages_pct":r["total_staff_cost"]/r["net_revenue"] if r["net_revenue"]>0 else 0,
            "lease_pct":r["total_occupancy_cost"]/r["net_revenue"] if r["net_revenue"]>0 else 0,
            "revenue_per_place":r["net_revenue"]/cap,"ebitda_per_place":r["ebitda"]/cap})
    return pd.DataFrame(rows)

# -- Master: Run Single Centre --

def run_single_centre(cfg=None):
    if cfg is None: cfg=deepcopy(EARLWOOD_CONFIG)
    pl=build_monthly_pl(cfg); annual=calc_annual_pl(pl)
    debt=calc_debt_schedule(cfg,pl); cf=calc_cashflow(cfg,pl,debt)
    bs=calc_balance_sheet(cfg,pl,cf,debt); gst=calc_gst_bas(cfg,pl)
    val=calc_valuation(cfg,annual); cap=calc_capital_sufficiency(cfg,pl)
    pc=calc_per_child(annual,pl); wk=calc_weekly_cashflow(cfg,pl)
    bm=calc_benchmarks(cfg,annual)
    return {"centre_id":cfg["centre_id"],"centre_name":cfg["centre_name"],"config":cfg,
            "monthly_pl":pl,"annual_pl":annual,"debt_schedule":debt,"cashflow":cf,
            "balance_sheet":bs,"gst_bas":gst,"valuation":val,"capital_sufficiency":cap,
            "per_child":pc,"weekly_cashflow":wk,"benchmarks":bm,
            "formatted_pl":format_pl_statement(annual),
            "formatted_bs":format_balance_sheet(bs),
            "formatted_cf":format_cashflow_statement(cf)}

# -- Multi-Centre Portfolio --

def run_portfolio(configs=None):
    if configs is None:
        configs=[deepcopy(v) for v in CENTRE_REGISTRY.values()]
    centres={}
    for cfg in configs:
        centres[cfg["centre_id"]]=run_single_centre(cfg)
    all_a=[]
    for cid,res in centres.items():
        a=res["annual_pl"].copy(); a["centre_id"]=cid; all_a.append(a)
    consol=pd.concat(all_a)
    pa=consol.groupby("fiscal_year").agg({
        "net_revenue":"sum","total_staff_cost":"sum","food_cost":"sum",
        "direct_costs":"sum","gross_profit":"sum","total_occupancy_cost":"sum",
        "total_opex":"sum","total_overheads":"sum","ebitda":"sum",
        "total_da":"sum","ebit":"sum","interest_expense":"sum",
        "npbt":"sum","tax":"sum","npat":"sum"}).reset_index()
    pa["ebitda_margin"]=pa.apply(lambda r: r["ebitda"]/r["net_revenue"] if r["net_revenue"]>0 else 0, axis=1)
    tc=sum(sum(c["capital_items"].values()) for c in configs)
    y5e=pa["ebitda"].iloc[-1] if len(pa)>0 else 0
    return {"centres":centres,"portfolio_annual_pl":pa,
            "total_capital_invested":tc,"portfolio_y5_ebitda":y5e,
            "portfolio_ev_5_5x":y5e*5.5,"portfolio_moic":y5e*5.5/tc if tc>0 else 0}

# -- Scenario comparison table --

def run_scenario_comparison(cfg=None):
    if cfg is None: cfg=deepcopy(EARLWOOD_CONFIG)
    results=[]
    for s in ["bear","base","bull"]:
        res=run_scenario(cfg,s); a=res["annual_pl"]
        for _,row in a.iterrows():
            results.append({"scenario":s,"fiscal_year":row["fiscal_year"],
                "revenue":row["net_revenue"],"ebitda":row["ebitda"],
                "ebitda_margin":row["ebitda_margin"],"npat":row["npat"],
                "staff_cost":row["total_staff_cost"],"occupancy_cost":row["total_occupancy_cost"]})
    return pd.DataFrame(results)

def scenario_comparison_table(cfg, scenarios=None):
    if scenarios is None: scenarios=["bear","base","bull"]
    rows=[]
    for s in scenarios:
        res=run_scenario(cfg,s); a=res["annual_pl"]; v=res["valuation"]
        cap=res["capital_sufficiency"]; cf=res["cashflow"]
        rows.append({"scenario":SCENARIO_PRESETS.get(s,{}).get("label",s.capitalize()),
            "y1_revenue":a.iloc[0]["net_revenue"],
            "y3_revenue":a.iloc[min(2,len(a)-1)]["net_revenue"],
            "y5_revenue":a.iloc[-1]["net_revenue"],
            "y1_ebitda":a.iloc[0]["ebitda"],
            "y3_ebitda":a.iloc[min(2,len(a)-1)]["ebitda"],
            "y5_ebitda":v["y5_ebitda"],
            "y1_margin":a.iloc[0]["ebitda_margin"],
            "y5_margin":a.iloc[-1]["ebitda_margin"],
            "ev_dcf":v["ev_dcf"],
            "irr":v["irr"],"moic":v["moic"],
            "payback_years":v["payback_years"],
            "total_capital":v["total_capital"],
            "y5_closing_cash":cf["closing_cash"].iloc[-1],
            "capital_sufficient":cap["sufficient"],
            "weeks_runway":cap["weeks_runway"],
            "annual_fcf":v["annual_fcf"]})
    return pd.DataFrame(rows)

def pe_dashboard(result):
    annual=result["annual_pl"]; val=result["valuation"]
    debt=result["debt_schedule"]; cf=result["cashflow"]
    kpis=[]
    for i,yl in enumerate(["Year 1","Year 3","Year 5"]):
        idx=min([0,2,len(annual)-1][i],len(annual)-1); r=annual.iloc[idx]
        kpis.append({"period":yl,"revenue":r["net_revenue"],"ebitda":r["ebitda"],
            "ebitda_margin":r["ebitda_margin"],"npat":r["npat"],
            "closing_cash":cf["closing_cash"].iloc[min(idx*12+11,len(cf)-1)],
            "director_loan":debt["closing_balance"].iloc[min(idx*12+11,len(debt)-1)]})
    kdf=pd.DataFrame(kpis)
    kdf.loc[len(kdf)]={"period":"Exit","revenue":None,"ebitda":val["y5_ebitda"],
        "ebitda_margin":None,"npat":None,"closing_cash":None,"director_loan":None}
    return kdf,{"ev_dcf":val["ev_dcf"],"ev_5_5x":val["ev_multiples"].get("5.5x",0),
        "irr":val["irr"],"moic":val["moic"],"payback_years":val["payback_years"],
        "total_capital":val["total_capital"]}

def rent_stress_test(cfg):
    br=cfg["base_rent_pa"]; y6r=br*(1+cfg["rent_escalation"])**5
    scenarios=[("Base (3% CPI yr 6)",1.0),("+10%",1.10),("+20%",1.20),("+30%",1.30),("Market $380k",None)]
    rows=[]
    for label,mult in scenarios:
        ar=y6r*mult if mult else 380000; d=ar-y6r
        rows.append({"scenario":label,"annual_rent":ar,"monthly_rent":ar/12,"vs_base":d,"ebitda_impact":-d})
    return pd.DataFrame(rows)

# -- CSV/ZIP Export --

def export_results_csv(result, prefix="brightblocks"):
    exports={}
    for key,label in [("monthly_pl","monthly_pl"),("annual_pl","annual_pl"),("cashflow","cashflow"),
        ("balance_sheet","balance_sheet"),("debt_schedule","debt_schedule"),
        ("per_child","per_child"),("weekly_cashflow","weekly_cashflow")]:
        df=result[key].copy()
        for col in df.columns:
            if hasattr(df[col],"dt"):
                try: df[col]=df[col].dt.strftime("%Y-%m-%d")
                except: pass
        exports[f"{prefix}_{label}.csv"]=df.to_csv(index=False)
    gst=result["gst_bas"].copy(); gst["quarter"]=gst["quarter"].astype(str)
    exports[f"{prefix}_gst_bas.csv"]=gst.to_csv(index=False)
    val=result["valuation"]
    vr=[{"metric":"EV (DCF)","value":val["ev_dcf"]},{"metric":"IRR","value":val["irr"]},
        {"metric":"MOIC","value":val["moic"]},{"metric":"Total Capital","value":val["total_capital"]}]
    for k,v in val["ev_multiples"].items(): vr.append({"metric":f"EV ({k})","value":v})
    exports[f"{prefix}_valuation.csv"]=pd.DataFrame(vr).to_csv(index=False)
    exports[f"{prefix}_capital.csv"]=pd.DataFrame([result["capital_sufficiency"]]).to_csv(index=False)
    return exports

def export_scenario_comparison_csv(cfg, scenarios=None):
    t=scenario_comparison_table(cfg,scenarios)
    return t.drop(columns=["annual_fcf"],errors="ignore").to_csv(index=False)

def export_all_to_zip(result, cfg):
    csvs=export_results_csv(result)
    csvs["scenario_comparison.csv"]=export_scenario_comparison_csv(cfg)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for name,content in csvs.items(): zf.writestr(name,content)
    return buf.getvalue()

# -- Entry Point --

if __name__ == "__main__":
    result=run_single_centre()
    print(f"Centre: {result['centre_name']}")
    a=result["annual_pl"]; v=result["valuation"]
    print(a[["fiscal_year","net_revenue","ebitda","ebitda_margin","npat"]].to_string(index=False))
    irr_s=f"{v['irr']:.1%}" if v["irr"] else "N/A"
    print(f"IRR: {irr_s}  MOIC: {v['moic']:.2f}x")
    cap=result["capital_sufficiency"]
    print(f"Pre-Rev Months: {cap['pre_revenue_months']}  Runway: {cap['weeks_runway']:.0f} wks")
    print("Done.")
