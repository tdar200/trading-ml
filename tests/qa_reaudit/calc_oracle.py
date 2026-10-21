"""Track C (calc) independent Decimal oracle.

Written from the textbook definitions plus the documented conventions (positive
denominators, average balances for ROE/ROA/ROIC, ratios as fractions). It does NOT
import anything from fundamentals.* and shares no code with tests/fundamental_oracle.py.
All money amounts are in BASE currency units (not millions); the callers scale.
"""
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 80

ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)


def D(x):
    """Exact Decimal of an int/str/float/Decimal; None stays None."""
    if x is None:
        return None
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(x)  # exact binary expansion of the float that production sees
    return Decimal(x)


# ---------------------------------------------------------------- primitives
def pdiv(num, den):
    """num/den, defined only for a strictly positive denominator (documented convention)."""
    if num is None or den is None or den <= 0:
        return None
    return num / den


def gro(new, old):
    q = pdiv(new, old)
    return None if q is None else q - ONE


def plus(*xs):
    return None if any(x is None for x in xs) else sum(xs, ZERO)


def minus(a, b):
    return None if a is None or b is None else a - b


def avg2(a, b):
    return None if a is None or b is None else (a + b) / TWO


def sample_std(xs):
    n = len(xs)
    m = sum(xs, ZERO) / n
    return (sum(((x - m) ** 2 for x in xs), ZERO) / (n - 1)).sqrt()


FIN_SECTORS = ("banks", "financials", "insurance")
NO_DAYS_SECTORS = ("banks", "financials", "saas", "insurance")


# ---------------------------------------------------------------- accounting features
def accounting(cur, prior=None, old=None, sector=None):
    """Return (features, flags). cur/prior/old are dicts of Decimal|None in base units."""
    p = prior or {}
    o = old or {}
    c = cur.get
    pv = p.get
    ov = o.get

    debt = plus(c("short_debt"), c("long_debt"))
    pdebt = plus(pv("short_debt"), pv("long_debt"))
    fcf = minus(c("ocf"), c("capex"))
    pfcf = minus(pv("ocf"), pv("capex"))
    ofcf = minus(ov("ocf"), ov("capex"))
    net_debt = minus(debt, c("cash"))
    ebitda = plus(c("operating_income"), c("depreciation"))
    pebitda = plus(pv("operating_income"), pv("depreciation"))
    rev = c("revenue")

    f = {}
    f["fcf"] = fcf
    f["ebitda_proxy"] = ebitda
    f["debt"] = debt
    f["net_debt"] = net_debt
    f["gross_margin"] = pdiv(c("gross_profit"), rev)
    f["operating_margin"] = pdiv(c("operating_income"), rev)
    f["net_margin"] = pdiv(c("net_income"), rev)
    f["fcf_margin"] = pdiv(fcf, rev)
    f["ebitda_margin"] = pdiv(ebitda, rev)

    avg_assets = avg2(c("assets"), pv("assets"))
    avg_equity = avg2(c("equity"), pv("equity"))
    f["roe"] = pdiv(c("net_income"), avg_equity)
    f["roa"] = pdiv(c("net_income"), avg_assets)
    f["gross_profitability"] = pdiv(c("gross_profit"), c("assets"))

    def invested(debt_, equity, cash):
        return None if debt_ is None or equity is None or cash is None else debt_ + equity - cash

    inv_now = invested(debt, c("equity"), c("cash"))
    inv_prev = invested(pdebt, pv("equity"), pv("cash"))

    def nopat(d):
        pretax = d.get("pretax_income")
        tax = pdiv(d.get("tax_expense"), pretax)
        oi = d.get("operating_income")
        if tax is None or oi is None or not (0 <= tax <= 1):
            return None
        return oi * (ONE - tax)

    nop_now = nopat(cur)
    nop_prev = nopat(p)
    f["roic"] = pdiv(nop_now, avg2(inv_now, inv_prev))

    skip_fin = sector in FIN_SECTORS
    pairs = [
        ("current_ratio", c("current_assets"), c("current_liabilities")),
        ("quick_ratio", minus(minus(c("current_assets"), c("inventory")), c("prepaids")), c("current_liabilities")),
        ("debt_equity", debt, c("equity")),
        ("net_debt_ebitda", net_debt, ebitda),
        ("interest_coverage", c("operating_income"), c("interest_expense")),
        ("cash_debt", c("cash"), debt),
    ]
    for name, num, den in pairs:
        f[name] = None if skip_fin else pdiv(num, den)

    f["ocf_net_income"] = pdiv(c("ocf"), c("net_income"))
    f["fcf_net_income"] = pdiv(fcf, c("net_income"))
    f["accrual_ratio"] = pdiv(minus(c("net_income"), c("ocf")), avg_assets)

    for m in ("revenue", "eps", "operating_income", "receivables", "inventory", "diluted_shares", "cash", "assets"):
        f[m + "_yoy"] = gro(c(m), pv(m))
    f["fcf_yoy"] = gro(fcf, pfcf)
    f["ebitda_yoy"] = gro(ebitda, pebitda)
    f["revenue_acceleration"] = minus(f["revenue_yoy"], gro(pv("revenue"), ov("revenue")))
    f["receivables_vs_sales"] = minus(f["receivables_yoy"], f["revenue_yoy"])
    f["inventory_vs_sales"] = minus(f["inventory_yoy"], f["revenue_yoy"])
    f["gross_margin_change"] = minus(f["gross_margin"], pdiv(pv("gross_profit"), pv("revenue")))
    f["operating_margin_change"] = minus(f["operating_margin"], pdiv(pv("operating_income"), pv("revenue")))
    f["debt_change"] = minus(debt, pdebt)
    f["cash_change"] = minus(c("cash"), pv("cash"))

    f["asset_turnover"] = pdiv(rev, avg_assets)
    f["inventory_turnover"] = pdiv(c("cogs"), avg2(c("inventory"), pv("inventory")))
    f["rd_sales"] = pdiv(c("rd"), rev)
    f["capex_sales"] = pdiv(c("capex"), rev)
    f["buybacks_ocf"] = pdiv(c("buybacks"), c("ocf"))

    def days(bal, base):
        x = pdiv(avg2(c(bal), pv(bal)), base)
        return None if x is None or sector in NO_DAYS_SECTORS else x * 365

    f["receivable_days"] = days("receivables", rev)
    f["inventory_days"] = days("inventory", c("cogs"))
    f["payable_days"] = days("payables", c("cogs"))
    f["cash_conversion_cycle"] = minus(plus(f["receivable_days"], f["inventory_days"]), f["payable_days"])
    f["eps_acceleration"] = minus(f["eps_yoy"], gro(pv("eps"), ov("eps")))
    f["fcf_acceleration"] = minus(f["fcf_yoy"], gro(pfcf, ofcf))
    f["incremental_roic"] = pdiv(minus(nop_now, nop_prev), minus(inv_now, inv_prev))
    f["dividends_ocf"] = pdiv(c("dividends"), c("ocf"))
    f["acquisitions_assets"] = pdiv(c("acquisitions"), c("assets"))
    f["issuance_equity"] = pdiv(c("issuance"), c("equity"))

    margins = [pdiv(d.get("gross_profit"), d.get("revenue")) for d in (o, p, cur)]
    f["gross_margin_stability"] = sample_std(margins) if all(m is not None for m in margins) else None
    incomes = [d.get("net_income") for d in (o, p, cur)]
    if all(i is not None for i in incomes) and abs(sum(incomes, ZERO) / 3) > 0:
        f["earnings_stability"] = sample_std(incomes) / abs(sum(incomes, ZERO) / 3)
    else:
        f["earnings_stability"] = None

    flags = []
    if fcf is not None and fcf < 0:
        flags.append("NEGATIVE_FCF")
    if f["gross_margin_change"] is not None and f["gross_margin_change"] < 0:
        flags.append("FALLING_MARGINS")
    if f["revenue_acceleration"] is not None and f["revenue_acceleration"] < 0:
        flags.append("REVENUE_DECELERATION")
    if f["diluted_shares_yoy"] is not None and f["diluted_shares_yoy"] > 0:
        flags.append("DILUTION")
    if f["receivables_vs_sales"] is not None and f["receivables_vs_sales"] > 0:
        flags.append("RECEIVABLES_OUTPACE_SALES")
    return f, flags


def cagr(now_value, then_value, years):
    """(now/then)^(1/years)-1 for strictly positive endpoints."""
    if now_value is None or then_value is None or now_value <= 0 or then_value <= 0:
        return None
    return (now_value / then_value) ** (ONE / Decimal(years)) - ONE


# ---------------------------------------------------------------- valuation
def valuation(price, shares, v, forward_eps=None):
    price, shares = D(price), D(shares)
    mcap = price * shares
    debt = plus(v.get("short_debt"), v.get("long_debt"))
    ev = None
    parts = [mcap, debt, v.get("preferred_equity"), v.get("minority_interest"), v.get("cash")]
    if all(x is not None for x in parts):
        ev = mcap + debt + v["preferred_equity"] + v["minority_interest"] - v["cash"]
    fcf = minus(v.get("ocf"), v.get("capex"))
    ebitda = plus(v.get("operating_income"), v.get("depreciation"))
    ni = v.get("net_income")
    return {
        "market_cap": mcap,
        "enterprise_value": ev,
        "pe": pdiv(mcap, ni),
        "forward_pe": pdiv(price, D(forward_eps)),
        "ps": pdiv(mcap, v.get("revenue")),
        "pb": pdiv(mcap, v.get("equity")),
        "ev_revenue": pdiv(ev, v.get("revenue")),
        "ev_ebitda": pdiv(ev, ebitda),
        "price_fcf": pdiv(mcap, fcf),
        "fcf_yield": pdiv(fcf, mcap),
        "earnings_yield": pdiv(ni, mcap),
    }


# ---------------------------------------------------------------- cross-section
def quantile7(sorted_vals, q):
    """Linear-interpolation (R type 7) quantile, exact Decimal."""
    n = len(sorted_vals)
    pos = Decimal(q) * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])


def mid_percentiles(values):
    """(average_rank - 0.5)/n with tied values sharing the mean of their ranks."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [ZERO] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = Decimal(i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return [(r - Decimal("0.5")) / n for r in ranks]


def winsor_z(values, lo_q="0.01", hi_q="0.99"):
    s = sorted(values)
    lo, hi = quantile7(s, lo_q), quantile7(s, hi_q)
    # Lead re-audit (QA-CALC-01): for n <= 100 the 1%/99% quantiles barely clip, so also clip at 5 robust sigmas
    # (1.4826 * MAD) around the median whenever the MAD is positive.
    n = len(s)
    med = (s[n // 2] + s[(n - 1) // 2]) / 2
    dev = sorted(abs(v - med) for v in values)
    robust = Decimal("1.4826") * (dev[n // 2] + dev[(n - 1) // 2]) / 2
    if robust > 0:
        lo, hi = max(lo, med - 5 * robust), min(hi, med + 5 * robust)
    w = [min(max(v, lo), hi) for v in values]
    sd = sample_std(w)
    if sd == 0:
        return None
    m = sum(w, ZERO) / len(w)
    return [(x - m) / sd for x in w]


# ---------------------------------------------------------------- ridge (exact rational)
def solve_exact(matrix, rhs):
    """Gauss-Jordan over Fractions."""
    n = len(rhs)
    a = [[Fraction(x) for x in row] + [Fraction(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        a[col] = [x / pv for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                fac = a[r][col]
                a[r] = [x - fac * y for x, y in zip(a[r], a[col])]
    return [a[i][n] for i in range(n)]


def ridge_fit(rows, targets, alpha):
    """Standardise with TRAINING mean and population std (const column -> scale 1),
    centre y, solve (Z'Z + alpha I) w = Z'(y - ybar). Returns means, scales, coefs, intercept."""
    n, k = len(rows), len(rows[0])
    fr = [[Fraction(x) for x in r] for r in rows]
    fy = [Fraction(y) for y in targets]
    means = [sum(r[j] for r in fr) / n for j in range(k)]
    scales = []
    for j in range(k):
        var = sum((r[j] - means[j]) ** 2 for r in fr) / n
        sd = Decimal(var.numerator) / Decimal(var.denominator)
        sd = sd.sqrt()
        scales.append(Fraction(sd) if sd > Decimal("1e-12") else Fraction(1))
    z = [[(r[j] - means[j]) / scales[j] for j in range(k)] for r in fr]
    ybar = sum(fy) / n
    yc = [y - ybar for y in fy]
    gram = [[sum(z[i][a] * z[i][b] for i in range(n)) + (Fraction(alpha) if a == b else 0) for b in range(k)] for a in range(k)]
    rhs = [sum(z[i][a] * yc[i] for i in range(n)) for a in range(k)]
    w = solve_exact(gram, rhs)
    return means, scales, w, ybar


def linear_score(features, means, scales, coefs, intercept):
    comps = [(Decimal(x) - Decimal(m)) / Decimal(s) * Decimal(w) for x, m, s, w in zip(features, means, scales, coefs)]
    return Decimal(intercept) + sum(comps, ZERO), comps


# ---------------------------------------------------------------- fiscal-index TTM oracle
def ttm_by_index(visible, metric, additive=True):
    """Independent TTM from VISIBLE rows addressed by fiscal-quarter index.

    visible: iterable of dicts {fy, qa, qb, value} meaning the row covers fiscal quarters
    qa..qb (1-based, inclusive) of fiscal-year index fy. Returns a dict {(fy,k): Decimal}
    of discrete quarters derivable from visible rows only:
      D(k) = row(k..k)                      if visible
           = row(1..k) - row(1..k-1)        if both cumulative rows visible (k>=2; row(1..0)=0)
    EPS-like (non additive) metrics are direct-only: pass additive=False.
    """
    cover = {}
    for r in visible:
        cover[(r["fy"], r["qa"], r["qb"])] = D(r["value"])
    fys = sorted({r["fy"] for r in visible})
    out = {}
    for fy in fys:
        for k in range(1, 5):
            if (fy, k, k) in cover:
                out[(fy, k)] = cover[(fy, k, k)]
            elif additive and (fy, 1, k) in cover:
                if k == 1:
                    out[(fy, k)] = cover[(fy, 1, 1)]
                elif (fy, 1, k - 1) in cover:
                    out[(fy, k)] = cover[(fy, 1, k)] - cover[(fy, 1, k - 1)]
    return out


def ttm_reported_totals(rows, end_key):
    """Published form of TTM from VISIBLE cumulative rows addressed by fiscal index (lead re-audit, QA-REAL-02):
    TTM(fy,k) = FY(fy-1) + YTD(fy,k) - YTD(fy-1,k); at a fiscal year end (k=4) it IS the reported FY figure.
    Returns None when a needed total is not visible (callers then fall back to summing four quarters)."""
    cover = {(r["fy"], r["qa"], r["qb"]): D(r["value"]) for r in rows}
    fy, k = end_key
    cur = cover.get((fy, 1, k))
    if cur is None:
        return None
    if k == 4:
        return cur
    full, prior = cover.get((fy - 1, 1, 4)), cover.get((fy - 1, 1, k))
    if full is None or prior is None:
        return None
    return full + cur - prior


def last_four(discrete, end_key):
    """Sum of the four consecutive fiscal quarters ending at end_key=(fy,k), or None."""
    fy, k = end_key
    keys = []
    for _ in range(4):
        keys.append((fy, k))
        k -= 1
        if k == 0:
            fy, k = fy - 1, 4
    if all(key in discrete for key in keys):
        return sum((discrete[key] for key in keys), ZERO)
    return None


def rel_close(a, b, rel=1e-12, abs_tol=0.0):
    """True when float a matches Decimal/float b to relative rel (or abs_tol)."""
    if a is None or b is None:
        return a is None and b is None
    a_, b_ = Decimal(a), Decimal(b)
    diff = abs(a_ - b_)
    return diff <= Decimal(rel) * abs(b_) or diff <= Decimal(abs_tol)
