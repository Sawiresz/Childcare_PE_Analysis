============================================================
BrightBlocks ELC — Childcare PE Financial Model (Phase 5)
============================================================

HOW TO RUN LOCALLY
------------------
1. Install dependencies:
   pip install streamlit plotly pandas numpy openpyxl

2. Place model.py and app.py in the same directory.

3. Run:
   streamlit run app.py

4. Open browser at http://localhost:8501

FOLDER STRUCTURE
----------------
model.py       — Financial model engine. No UI. All calculations.
app.py         — Streamlit dashboard. 5 pages:
                 - Inputs (Base Case assumptions)
                 - Dashboard (KPIs, charts, staffing)
                 - Scenarios (isolated Bear/Base/Bull with overrides)
                 - Centre Comparison (multi-centre portfolio)
                 - Financial Statements (P&L, BS, CF, detail views)
README.txt     — This file.
requirements.txt — Python dependencies.

UI STRUCTURE
------------
Sidebar:
  - Centre selector dropdown (Earlwood, Marrickville, or custom)
  - Page navigation radio buttons
  - Clean white background, black text, always readable

Main Pages:
  1. Inputs
     - All Base Case assumptions editable in 3-column layout
     - Timeline: start date, pre-revenue months, model length
     - Revenue: fees, escalation, collection rate
     - Occupancy ramp (relative to post-pre-revenue)
     - Staffing: hourly rates, super, wage escalation
     - Staffing overrides: monthly ECT/Support count overrides
     - Director salary deferral toggle
     - Support unpaid during pre-revenue toggle
     - Lease, capital, valuation parameters

  2. Dashboard
     - 5 KPI cards: IRR, Y5 EBITDA, EV, Capital/Runway, Pre-Rev
     - Capacity compliance warnings (red banner if breached)
     - Revenue vs Costs vs EBITDA chart
     - Cash position with reserve line
     - Occupancy ramp chart
     - Returns waterfall
     - Staff headcount (ECT/Support/Director)
     - Unit economics per child

  3. Scenarios
     - Bear/Base/Bull tabs with INDEPENDENT overrides
     - Each scenario has: occupancy, fees, wages, start date,
       pre-revenue months
     - Changes do NOT overwrite Base Case
     - Comparison table: IRR, MOIC, EBITDA, cash runway, valuation
     - EBITDA by scenario chart
     - CSV export

  4. Centre Comparison
     - Portfolio model runs all registered centres
     - Side-by-side metrics table
     - Portfolio KPI cards (total capital, EBITDA, EV, MOIC)
     - EBITDA by centre grouped bar chart

  5. Financial Statements
     - Standard Profit & Loss (FY columns, proper line items)
     - Standard Balance Sheet (Assets/Liabilities/Equity)
     - Standard Cash Flow Statement (Operating/Investing/Financing)
     - Monthly Detail with capacity compliance column
     - Debt Schedule
     - Weekly Cash Flow with RAG status highlighting
     - GST / BAS quarterly summary
     - XLSX and CSV ZIP export buttons

PRE-REVENUE PERIOD
------------------
- Controlled by "pre_revenue_months" config key (default: 3)
- Editable on Inputs page and independently per scenario
- During pre-revenue months:
    * Revenue is forced to $0 (occupancy = 0%)
    * Staffing ramp-up is allowed (ECT defaults to 1, Support to 0)
    * Both ECT and Support counts can be overridden per month
    * Support staff can be flagged as unpaid via "support_unpaid_pre_revenue"
    * Costs (rent, insurance, admin, land tax) still accrue
    * No ratio enforcement during pre-revenue
- Post-revenue occupancy ramp begins at month pre_revenue_months + 1
- Pre-revenue burn flows through to:
    * Capital sufficiency / runway calculation
    * Cash flow statement
    * IRR (via annual FCF)
    * Valuation (via DCF)
    * All financial statements

STAFFING LOGIC
--------------
ECT (Early Childhood Teachers):
  - Auto-calculated: ceil(children / educator_child_ratio)
  - Minimum 1 during pre-revenue for setup
  - Can be overridden monthly via ect_monthly_override dict
    e.g. {1:1, 2:2, 3:2} sets ECT count for months 1-3

Support Staff:
  - Auto-calculated: 1 when children > 0, 0 during pre-revenue
  - Can be overridden monthly via support_monthly_override dict
  - Can be flagged as unpaid during pre-revenue period

Capacity Compliance:
  - Column "meets_capacity_limits" (True/False) on every month
  - Post-revenue: True if ECT >= ceil(children / ratio)
  - Pre-revenue: always True (no enforcement)
  - UI shows warning banner when any month is non-compliant

Director Salary:
  - Deferred by default (director_salary_deferred = True)
  - Never accrued or repaid during deferral period
  - Starts only when business is profitable:
    * EBITDA > threshold for N consecutive months
    * Occupancy > threshold for N consecutive months
    * Cumulative approximate NPBT > 0
  - When not deferred: director salary active from month 1

DIRECTOR LOAN
-------------
- Treated as shareholder funding (not bank debt)
- Interest accrues monthly at director_loan_interest_rate
- Repayment only when: month > pre_revenue_months AND EBITDA > 0
- Fixed monthly repayment amount (director_loan_repayment_monthly)
- Withdrawal/wind-down logic unchanged from original model

SCENARIOS (ISOLATED)
--------------------
- Three presets: Bear (Downside), Base Case, Bull (Upside)
- Each scenario tab has independent inputs for:
    * Occupancy (M4, Y3, Y4+)
    * Daily fees and fee escalation
    * Educator hourly rate and wage escalation
    * Start date
    * Pre-revenue period length
- Scenario overrides are applied ON TOP of the Base Case config
- They do NOT modify the Base Case — fully isolated
- Scenario comparison shows: IRR, MOIC, Y5 EBITDA, Cash Runway,
  EV (DCF), Y5 Margin
- Bear scenario defaults: lower occ, lower fees, higher wages,
  longer pre-revenue (4m)
- Bull scenario defaults: higher occ, higher fees, lower wages,
  shorter pre-revenue (2m)

MULTI-CENTRE
-------------
- CENTRE_REGISTRY dict holds all centre configs
- Default: Earlwood (40 places) + Marrickville (60 places, demo)
- Sidebar dropdown selects active centre
- Centre Comparison page runs all centres and shows:
    * Per-centre metrics table
    * Portfolio totals (capital, EBITDA, EV, MOIC)
    * Grouped bar chart

HOW TO ADD A NEW CENTRE
------------------------
1. In model.py, create a config dict (copy EARLWOOD_CONFIG as template)
2. Set unique centre_id, centre_name, and all parameters
3. Register it:
     from model import register_centre
     register_centre(my_new_config)
   Or add directly to CENTRE_REGISTRY dict in model.py
4. The centre will appear in the sidebar dropdown automatically

VALIDATION CHECKLIST vs EXCEL
------------------------------
Base Case (Earlwood, 40 places, Jul 2026 start, 3m pre-rev):
  [ ] Y3 EBITDA:  $238,693
  [ ] Y5 EBITDA:  $296,140
  [ ] MOIC:       5.32x
  [ ] IRR:        ~60% (60.3% — minor diff from occupancy ramp mapping)
  [ ] Pre-revenue months 1-3: Revenue = $0
  [ ] Month 4 occupancy: 70% (28 children)
  [ ] Y4+ occupancy: 97.5% (39 children)
  [ ] Director salary deferred until profitable
  [ ] Rent-free first 6 months
  [ ] Exit multiple: 5.5x applied to Y5 EBITDA

Scenario Presets:
  [ ] Bear: Y5 EBITDA ~$76k, MOIC 1.12x, pre-rev 4m
  [ ] Bull: Y5 EBITDA ~$570k, MOIC 13.04x, pre-rev 2m

Financial Statements:
  [ ] P&L: Revenue, COGS (Staff + Food), GP, Overheads, EBITDA,
           D&A, EBIT, Interest, NPBT, Tax, NPAT — per FY
  [ ] Balance Sheet: Assets (Cash, Debtors, Fixed), Liabilities
           (Creditors, Dir Loan, Provisions), Equity — per FY
  [ ] Cash Flow: Operating, Investing, Financing sections — per FY
  [ ] All three balance: Assets = Liabilities + Equity

TECHNICAL NOTES
---------------
- Fully parameterised: no hardcoded centre logic
- Financial correctness prioritised over UI aesthetics
- Modular reusable functions throughout model.py
- Pandas for all tabular data, Plotly for all charts
- Session state used for config persistence across pages
- Export: XLSX (openpyxl) and CSV ZIP available on Financial Statements page
- Sidebar: white background, black text, all inputs readable
