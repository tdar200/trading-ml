"""Track C golden companies. Numbers are 'GBP millions' unless noted; eps is GBP per share;
share counts are raw shares. 'hand' = values derived by hand on paper (NOT by the oracle) and
asserted independently in test_calc_golden.py. Everything else is checked by the Decimal oracle.
"""
from __future__ import annotations
from decimal import Decimal

M = 1_000_000


def Y(revenue, gross_profit, operating_income, net_income, ocf, capex, cash, short_debt, long_debt,
      current_assets, current_liabilities, equity, assets, shares, eps, *, depreciation=None,
      interest_expense=None, pretax_income=None, tax_expense=None, inventory=None, prepaids=None,
      receivables=None, payables=None, rd=None, buybacks=None, dividends=None, issuance=None,
      acquisitions=None, cogs="auto", drop=()):
    """Positional core statement lines + optional extras. None means: no Fact is written."""
    d = dict(revenue=revenue, gross_profit=gross_profit, operating_income=operating_income, net_income=net_income,
             ocf=ocf, capex=capex, cash=cash, short_debt=short_debt, long_debt=long_debt,
             current_assets=current_assets, current_liabilities=current_liabilities, equity=equity,
             assets=assets, diluted_shares=shares, eps=eps, depreciation=depreciation,
             interest_expense=interest_expense, pretax_income=pretax_income, tax_expense=tax_expense,
             inventory=inventory, prepaids=prepaids, receivables=receivables, payables=payables, rd=rd,
             buybacks=buybacks, dividends=dividends, issuance=issuance, acquisitions=acquisitions)
    if cogs == "auto":
        d["cogs"] = None if revenue is None or gross_profit is None else Decimal(str(revenue)) - Decimal(str(gross_profit))
    else:
        d["cogs"] = cogs
    for k in drop:
        d[k] = None
    return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in d.items() if v is not None}


def R(revenue):
    return {"revenue": revenue}


GOLDEN = {}

# ---------------------------------------------------------------- A: the supplied Company A
GOLDEN["A"] = dict(
    note="supplied Company A: revenue 100m->120m, GP 60, OI 24, NI 18, OCF 25, capex 5, cash 30, debt 20, CA 50, CL 25, equity 90, 10m shares, price 20",
    sector=None, price="20", shares_out=10 * M, forward_eps="2.0",
    years={
        2018: R(60), 2020: R(80),
        2021: Y(90, 40, 14, 9, 15, 3, 15, 5, 25, 35, 20, 70, 130, 10 * M, "0.9", depreciation=4, interest_expense=2,
                pretax_income=12, tax_expense=3, inventory=7, prepaids=1, receivables=11, payables=6),
        2022: Y(100, 48, 18, "13.2", 20, 4, 20, 5, 20, 40, 22, 80, 140, 10 * M, "1.32", depreciation=5, interest_expense="1.5",
                pretax_income="16.5", tax_expense="3.3", inventory=8, prepaids=1, receivables=12, payables=7),
        2023: Y(120, 60, 24, 18, 25, 5, 30, 5, 15, 50, 25, 90, 150, 10 * M, "1.8", depreciation=6, interest_expense="1.5",
                pretax_income="22.5", tax_expense="4.5", inventory=10, prepaids=2, receivables=15, payables=8,
                rd=6, buybacks=2, dividends=3, issuance=1, acquisitions=4),
    },
    hand={  # derived on paper from the brief and the rows above (ROE/ROIC use average balances)
        "revenue_yoy": "0.2", "gross_margin": "0.5", "operating_margin": "0.2", "net_margin": "0.15",
        "fcf": "20000000", "fcf_margin": "0.16666666666666666666", "current_ratio": "2", "debt_equity": "0.22222222222222222222",
        "debt": "20000000", "net_debt": "-10000000", "ebitda_proxy": "30000000", "net_debt_ebitda": "-0.33333333333333333333",
        "interest_coverage": "16", "cash_debt": "1.5", "roe": "0.21176470588235294118",
        "roic": "0.23272727272727272727", "eps_yoy": "0.36363636363636363636", "diluted_shares_yoy": "0",
        "quick_ratio": "1.52", "ocf_net_income": "1.38888888888888888889", "fcf_net_income": "1.11111111111111111111",
    },
    hand_valuation={"market_cap": "200000000", "pe": "11.1111111111111111111", "price_fcf": "10", "fcf_yield": "0.1",
                    "earnings_yield": "0.09", "ps": "1.66666666666666666667", "pb": "2.22222222222222222222",
                    "forward_pe": "10", "enterprise_value": "190000000", "ev_revenue": "1.58333333333333333333",
                    "ev_ebitda": "6.33333333333333333333"},
    hand_flags=[],  # fcf>0, GM 50% vs 48%, accel +0.0889, no dilution, receivables +25% vs sales +20% -> outpace
)
GOLDEN["A"]["hand_flags"] = ["RECEIVABLES_OUTPACE_SALES"]

# ---------------------------------------------------------------- B: high growth
GOLDEN["B"] = dict(
    note="high growth: revenue 40->80->200, accelerating; heavy capex",
    sector=None, price="40", shares_out=50 * M, forward_eps="0.5",
    years={
        2018: R(5), 2020: R(20),
        2021: Y(40, 24, -2, -3, 2, 5, 40, 0, 30, 70, 30, 90, 150, 40 * M, "-0.075", depreciation=3, interest_expense=2,
                pretax_income=-3, tax_expense=0, inventory=3, receivables=6, payables=4),
        2022: Y(80, 52, 4, 3, 8, 6, 60, 0, 40, 100, 40, 160, 250, 45 * M, "0.0667", depreciation=5, interest_expense=2,
                pretax_income=2, tax_expense="0.4", inventory=5, receivables=12, payables=6),
        2023: Y(200, 140, 20, 16, 30, 20, 100, 0, 50, 160, 60, 300, 500, 50 * M, "0.32", depreciation=10, interest_expense=2,
                pretax_income=20, tax_expense=4, inventory=9, receivables=40, payables=12),
    },
    hand={"revenue_yoy": "1.5", "revenue_acceleration": "0.5", "gross_margin": "0.7", "gross_margin_change": "0.05",
          "operating_margin": "0.1", "fcf": "10000000", "fcf_margin": "0.05", "current_ratio": "2.6666666666666666667",
          "revenue_cagr_3y": "1.1544346900318837218", "diluted_shares_yoy": "0.11111111111111111111"},
    hand_flags=["DILUTION", "RECEIVABLES_OUTPACE_SALES"],
)

# ---------------------------------------------------------------- C: low growth, tiny deceleration
GOLDEN["C"] = dict(
    note="low growth: 100,101,102; growth 1% then 0.990099% => mathematically decelerating",
    sector=None, price="10", shares_out=20 * M, forward_eps="1.0",
    years={
        2018: R(95), 2020: R(99),
        2021: Y(100, 30, 10, 6, 9, 4, 20, 5, 25, 50, 30, 100, 200, 20 * M, "0.3", depreciation=5, interest_expense=2,
                pretax_income=8, tax_expense=2, inventory=10, prepaids=1, receivables=12, payables=9),
        2022: Y(101, 30.3, 10.1, 6.06, 9, 4, 20, 5, 25, 50, 30, 100, 200, 20 * M, "0.303", depreciation=5, interest_expense=2,
                pretax_income="8.1", tax_expense="2.04", inventory=10, prepaids=1, receivables=12, payables=9),
        2023: Y(102, "30.6", "10.2", "6.12", 9, 4, 20, 5, 25, 50, 30, 100, 200, 20 * M, "0.306", depreciation=5, interest_expense=2,
                pretax_income="8.2", tax_expense="2.08", inventory=10, prepaids=1, receivables=12, payables=9),
    },
    hand={"revenue_yoy": "0.00990099009900990099", "revenue_acceleration": "-0.00009900990099009901",
          "gross_margin": "0.3", "gross_margin_change": "0", "fcf": "5000000"},
    hand_flags=["REVENUE_DECELERATION"],
)

# ---------------------------------------------------------------- D: negative earnings and negative FCF
GOLDEN["D"] = dict(
    note="loss making: NI -10, pretax -9 => ROIC undefined, P/E undefined, FCF -8",
    sector=None, price="5", shares_out=40 * M, forward_eps="-0.1",
    years={
        2018: R(40), 2020: R(60),
        2021: Y(80, 30, -2, -3, 1, 3, 25, 5, 15, 40, 25, 60, 120, 40 * M, "-0.075", depreciation=4, interest_expense=1,
                pretax_income=-3, tax_expense=0, inventory=8, receivables=10, payables=7),
        2022: Y(90, 33, 2, "1.5", 3, 3, 20, 5, 15, 40, 25, 55, 120, 40 * M, "0.0375", depreciation=4, interest_expense=1,
                pretax_income=1, tax_expense="0.2", inventory=9, receivables=11, payables=7),
        2023: Y(100, 30, -8, -10, -5, 3, 12, 5, 15, 35, 25, 45, 115, 40 * M, "-0.25", depreciation=4, interest_expense=1,
                pretax_income=-9, tax_expense=1, inventory=10, receivables=14, payables=7),
    },
    hand={"fcf": "-8000000", "net_margin": "-0.1", "operating_margin": "-0.08", "fcf_margin": "-0.08",
          "roic": None, "ocf_net_income": None, "fcf_net_income": None, "eps_yoy": "-7.66666666666666666667",
          "gross_margin_change": "-0.06666666666666666667", "revenue_yoy": "0.11111111111111111111"},
    hand_valuation={"pe": None, "price_fcf": None, "forward_pe": None, "earnings_yield": "-0.05", "fcf_yield": "-0.04"},
    hand_flags=["NEGATIVE_FCF", "FALLING_MARGINS", "REVENUE_DECELERATION", "RECEIVABLES_OUTPACE_SALES"],
)

# ---------------------------------------------------------------- E: high debt
GOLDEN["E"] = dict(
    note="high debt: debt 300 vs equity 50 => D/E 6, ND/EBITDA 7.25, coverage 1.2",
    sector=None, price="8", shares_out=10 * M, forward_eps="0.5",
    years={
        2018: R(180), 2020: R(190),
        2021: Y(200, 60, 25, 8, 20, 10, 15, 40, 220, 80, 70, 52, 400, 10 * M, "0.8", depreciation=10, interest_expense=20,
                pretax_income=5, tax_expense=1, inventory=20, receivables=25, payables=15),
        2022: Y(205, 62, 28, 5, 18, 10, 12, 50, 240, 85, 80, 50, 420, 10 * M, "0.5", depreciation=10, interest_expense=22,
                pretax_income=6, tax_expense="1.2", inventory=22, receivables=26, payables=15),
        2023: Y(210, 63, 30, 6, 25, 12, 10, 60, 240, 90, 90, 50, 430, 10 * M, "0.6", depreciation=10, interest_expense=25,
                pretax_income=5, tax_expense=1, inventory=24, receivables=27, payables=16),
    },
    hand={"debt": "300000000", "net_debt": "290000000", "debt_equity": "6", "ebitda_proxy": "40000000",
          "net_debt_ebitda": "7.25", "interest_coverage": "1.2", "current_ratio": "1", "cash_debt": "0.03333333333333333333",
          "debt_change": "10000000"},
    hand_flags=["FALLING_MARGINS", "REVENUE_DECELERATION", "RECEIVABLES_OUTPACE_SALES"],
)

# ---------------------------------------------------------------- F: cash rich (negative net debt)
GOLDEN["F"] = dict(
    note="cash rich: cash 500, debt 50 => net debt -450, cash/debt 10",
    sector=None, price="30", shares_out=10 * M, forward_eps="3.0",
    years={
        2018: R(250), 2020: R(280),
        2021: Y(300, 150, 60, 50, 70, 10, 420, 10, 40, 600, 100, 700, 900, 10 * M, "5", depreciation=10, interest_expense=1,
                pretax_income=60, tax_expense=10, inventory=30, receivables=40, payables=20),
        2022: Y(320, 160, 64, 52, 72, 10, 460, 10, 40, 640, 105, 740, 950, 10 * M, "5.2", depreciation=10, interest_expense=1,
                pretax_income=63, tax_expense="10.5", inventory=31, receivables=42, payables=21),
        2023: Y(340, 170, 68, 54, 75, 10, 500, 10, 40, 680, 110, 780, 1000, 10 * M, "5.4", depreciation=10, interest_expense=1,
                pretax_income=67, tax_expense="13.4", inventory=32, receivables=44, payables=22),
    },
    hand={"debt": "50000000", "net_debt": "-450000000", "cash_debt": "10", "debt_equity": "0.06410256410256410256",
          "net_debt_ebitda": "-5.76923076923076923077", "interest_coverage": "68", "current_ratio": "6.18181818181818181818",
          "fcf": "65000000", "cash_change": "40000000"},
    hand_valuation={"market_cap": "300000000", "enterprise_value": "-150000000", "ev_revenue": "-0.44117647058823529412"},
    hand_flags=["REVENUE_DECELERATION"],
)

# ---------------------------------------------------------------- G: distressed
GOLDEN["G"] = dict(
    note="distressed: negative equity, CR 0.5, coverage 0.4, negative OCF",
    sector=None, price="1", shares_out=30 * M, forward_eps="0",
    years={
        2018: R(150), 2020: R(140),
        2021: Y(130, 30, 5, -5, 2, 4, 10, 30, 70, 40, 60, 30, 150, 30 * M, "-0.1667", depreciation=6, interest_expense=10,
                pretax_income=-5, tax_expense=0, inventory=15, receivables=20, payables=18),
        2022: Y(120, 24, -2, -12, -3, 4, 6, 40, 80, 35, 70, 10, 140, 30 * M, "-0.4", depreciation=6, interest_expense=11,
                pretax_income=-13, tax_expense=0, inventory=16, receivables=21, payables=19),
        2023: Y(100, 15, -4, -30, -10, 4, 3, 50, 80, 30, 60, -20, 120, 30 * M, "-1", depreciation=6, interest_expense=10,
                pretax_income=-14, tax_expense=0, inventory=18, receivables=22, payables=20),
    },
    hand={"current_ratio": "0.5", "debt_equity": None, "roe": None, "interest_coverage": "-0.4", "fcf": "-14000000",
          "net_debt": "127000000", "net_debt_ebitda": "63.5", "roic": None, "revenue_yoy": "-0.16666666666666666667",
          "gross_margin": "0.15", "gross_margin_change": "-0.05", "eps_yoy": None},
    hand_flags=["NEGATIVE_FCF", "FALLING_MARGINS", "REVENUE_DECELERATION", "RECEIVABLES_OUTPACE_SALES"],
)

# ---------------------------------------------------------------- H: heavily diluted
GOLDEN["H"] = dict(
    note="heavily diluted: 100m -> 120m -> 140m weighted diluted shares (+20%, +16.667%)",
    sector=None, price="2", shares_out=140 * M, forward_eps="0.1",
    years={
        2018: R(70), 2020: R(80),
        2021: Y(90, 36, 9, 8, 10, 4, 20, 5, 15, 40, 25, 100, 150, 100 * M, "0.08", depreciation=3, interest_expense=1,
                pretax_income=8, tax_expense=0, inventory=8, receivables=10, payables=6),
        2022: Y(100, 40, 10, 9, 11, 4, 22, 5, 15, 42, 26, 110, 160, 120 * M, "0.075", depreciation=3, interest_expense=1,
                pretax_income=9, tax_expense=0, inventory=9, receivables=11, payables=6),
        2023: Y(110, 44, 11, 10, 12, 4, 24, 5, 15, 44, 27, 120, 170, 140 * M, "0.0714", depreciation=3, interest_expense=1,
                pretax_income=10, tax_expense=0, inventory=10, receivables=12, payables=7),
    },
    hand={"diluted_shares_yoy": "0.16666666666666666667", "revenue_yoy": "0.1", "net_margin": "0.09090909090909090909",
          "eps_yoy": "-0.048", "roic": "0.09821428571428571429", "fcf": "8000000"},
    hand_flags=["DILUTION", "REVENUE_DECELERATION"],
)

# ---------------------------------------------------------------- I: cyclically improving
GOLDEN["I"] = dict(
    note="cyclical improvement: GM 30->35->40, growth 5% -> 15% -> 25%",
    sector=None, price="12", shares_out=10 * M, forward_eps="2.0",
    years={
        2018: R(90), 2020: R(95),
        2021: Y(100, 30, 8, 4, 8, 6, 10, 5, 25, 40, 30, 80, 150, 10 * M, "0.4", depreciation=6, interest_expense=3,
                pretax_income=5, tax_expense=1, inventory=12, receivables=14, payables=9),
        2022: Y(115, 40.25, 14, 8, 12, 6, 12, 5, 25, 45, 30, 85, 155, 10 * M, "0.8", depreciation=6, interest_expense=3,
                pretax_income=11, tax_expense=3, inventory=13, receivables=15, payables=10),
        2023: Y("143.75", "57.5", 25, 16, 22, 6, 20, 5, 25, 60, 32, 95, 165, 10 * M, "1.6", depreciation=6, interest_expense=3,
                pretax_income=22, tax_expense=6, inventory=14, receivables=17, payables=11),
    },
    hand={"revenue_yoy": "0.25", "gross_margin": "0.4", "gross_margin_change": "0.05", "revenue_acceleration": "0.1",
          "eps_yoy": "1", "fcf": "16000000", "operating_margin_change": "0.05217391304347826087"},
    hand_flags=[],
)

# ---------------------------------------------------------------- J: deteriorating
GOLDEN["J"] = dict(
    note="deteriorating: GM 40->35->30, receivables outrunning sales, FCF turning negative",
    sector=None, price="12", shares_out=10 * M, forward_eps="0.5",
    years={
        2018: R(150), 2020: R(180),
        2021: Y(200, 80, 30, 20, 25, 10, 30, 5, 25, 70, 40, 100, 200, 10 * M, "2", depreciation=8, interest_expense=3,
                pretax_income=27, tax_expense="5.4", inventory=20, receivables=25, payables=15),
        2022: Y(210, "73.5", 24, 15, 15, 10, 25, 5, 30, 72, 45, 100, 210, 10 * M, "1.5", depreciation=8, interest_expense=3,
                pretax_income=21, tax_expense="4.2", inventory=24, receivables=33, payables=15),
        2023: Y(200, 60, 12, 5, 3, 10, 15, 8, 40, 70, 55, 95, 220, 10 * M, "0.5", depreciation=8, interest_expense=4,
                pretax_income=8, tax_expense="1.6", inventory=30, receivables=40, payables=14),
    },
    hand={"revenue_yoy": "-0.04761904761904761905", "gross_margin": "0.3", "gross_margin_change": "-0.05",
          "fcf": "-7000000", "debt_change": "13000000", "cash_change": "-10000000",
          "receivables_yoy": None},
    hand_flags=["NEGATIVE_FCF", "FALLING_MARGINS", "REVENUE_DECELERATION"],
)
GOLDEN["J"]["hand"].pop("receivables_yoy")
# receivables +21.2% vs sales -4.76%: receivables_vs_sales > 0
GOLDEN["J"]["hand_flags"].append("RECEIVABLES_OUTPACE_SALES")

# ---------------------------------------------------------------- K: bank-like
GOLDEN["K"] = dict(
    note="bank-like: liquidity/leverage ratios and working-capital days are not meaningful",
    sector="banks", price="15", shares_out=10 * M, forward_eps="1.5",
    years={
        2018: R(300), 2020: R(320),
        2021: Y(340, 340, 120, 90, 100, 5, 800, 400, 600, 1500, 1400, 1000, 12000, 10 * M, "9", depreciation=5, interest_expense=60,
                pretax_income=118, tax_expense=28, receivables=500, payables=400),
        2022: Y(350, 350, 125, 95, 105, 5, 820, 420, 620, 1550, 1450, 1050, 12500, 10 * M, "9.5", depreciation=5, interest_expense=62,
                pretax_income=122, tax_expense=27, receivables=520, payables=410),
        2023: Y(360, 360, 130, 100, 110, 5, 850, 430, 650, 1600, 1500, 1100, 13000, 10 * M, "10", depreciation=5, interest_expense=65,
                pretax_income=128, tax_expense=28, receivables=540, payables=420),
    },
    hand={"current_ratio": None, "quick_ratio": None, "debt_equity": None, "net_debt_ebitda": None, "interest_coverage": None,
          "cash_debt": None, "receivable_days": None, "payable_days": None, "inventory_days": None,
          "cash_conversion_cycle": None, "revenue_yoy": "0.02857142857142857143", "roe": "0.09302325581395348837",
          "roa": "0.00784313725490196078", "net_margin": "0.27777777777777777778"},
    hand_flags=["REVENUE_DECELERATION", "RECEIVABLES_OUTPACE_SALES"],
)

# ---------------------------------------------------------------- L: zero denominators
GOLDEN["L"] = dict(
    note="zero denominators: revenue 0, equity 0, interest 0, debt 0, NI 0, OCF 0, cogs 0",
    sector=None, price="3", shares_out=5 * M, forward_eps="0",
    years={
        2018: R(0), 2020: R(0),
        2021: Y(0, 0, 0, 0, 0, 0, 5, 0, 0, 10, 5, 0, 20, 5 * M, "0", depreciation=0, interest_expense=0, pretax_income=0,
                tax_expense=0, inventory=0, receivables=0, payables=0),
        2022: Y(0, 0, 0, 0, 0, 0, 5, 0, 0, 10, 5, 0, 20, 5 * M, "0", depreciation=0, interest_expense=0, pretax_income=0,
                tax_expense=0, inventory=0, receivables=0, payables=0),
        2023: Y(0, 0, 0, 0, 0, 0, 5, 0, 0, 10, 5, 0, 20, 5 * M, "0", depreciation=0, interest_expense=0, pretax_income=0,
                tax_expense=0, inventory=0, receivables=0, payables=0),
    },
    hand={"gross_margin": None, "net_margin": None, "revenue_yoy": None, "debt_equity": None, "interest_coverage": None,
          "cash_debt": None, "roe": None, "roic": None, "ocf_net_income": None, "earnings_stability": None,
          "revenue_cagr_3y": None, "fcf": "0", "debt": "0", "net_debt": "-5000000", "current_ratio": "2"},
    hand_valuation={"pe": None, "price_fcf": None, "ps": None, "pb": None, "forward_pe": None, "fcf_yield": "0",
                    "earnings_yield": "0", "enterprise_value": "10000000"},
    hand_flags=[],
)

# ---------------------------------------------------------------- M: missing inputs
GOLDEN["M"] = dict(
    note="sparse filer: no short debt, no depreciation, no prepaids, no cogs => dependent ratios are None, not 0",
    sector=None, price="10", shares_out=10 * M, forward_eps="1",
    years={
        2018: R(60), 2020: R(70),
        2021: Y(80, 40, 10, 6, 9, 3, 10, None, 20, 30, 20, 60, 120, 10 * M, "0.6", cogs=None),
        2022: Y(90, 45, 12, 7, 10, 3, 12, None, 20, 32, 21, 64, 125, 10 * M, "0.7", cogs=None),
        2023: Y(100, 50, 14, 8, 11, 3, 14, None, 20, 34, 22, 68, 130, 10 * M, "0.8", cogs=None),
    },
    hand={"debt": None, "net_debt": None, "debt_equity": None, "net_debt_ebitda": None, "cash_debt": None, "ebitda_proxy": None,
          "ebitda_margin": None, "quick_ratio": None, "roic": None, "inventory_turnover": None, "revenue_yoy": "0.11111111111111111111",
          "fcf": "8000000", "gross_margin": "0.5", "current_ratio": "1.54545454545454545455"},
    hand_valuation={"enterprise_value": None, "ev_revenue": None, "ev_ebitda": None, "pe": "12.5", "pb": "1.4705882352941176471"},
    hand_flags=["REVENUE_DECELERATION"],
)
