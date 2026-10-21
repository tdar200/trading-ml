"""Causal accounting transforms. Missing/invalid denominators stay missing.

Amounts are comparable only within identical currency, unit and fiscal duration.
No weights, recommendations or execution occur in this module.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from fundamentals.store import utc,consistent_read

VERSION='fundamentals-v4-audited'
FLAG_NOISE=1e-12   # below this a difference of ratios is floating-point noise, not a direction
EVIDENCE={
 'quality':'Novy-Marx 2013 gross profitability; Asness/Frazzini/Pedersen 2019 QMJ; local incremental evidence required',
 'growth':'Earnings/fundamental momentum hypothesis; growth alone is not a priced edge; local IC and ablation required',
 'value':'Fama/French value evidence; negative earnings and distress complicate multiples; local cost-aware ablation required',
 'health':'Balance-sheet distress hypothesis; sector-specific interpretation; no validated trading threshold',
 'earnings_quality':'Sloan accrual hypothesis; independent historical cash-flow definitions and costs required',
 'allocation':'Investment and issuance literature; signs may depend on profitability; local validation required',
}


def ratio(a,b):
    if a is None or b is None or b<=0:return None
    value=a/b
    return value if math.isfinite(value) else None

def growth(a,b):
    value=ratio(a,b)
    return None if value is None else value-1

def subtract(a,b):
    value=None if a is None or b is None else a-b
    return value if value is None or math.isfinite(value) else None

def add(a,b):
    value=None if a is None or b is None else a+b
    return value if value is None or math.isfinite(value) else None

def annual_cagr(a,b,years):
    if a is None or b is None or a<=0 or b<=0 or years<=0:return None
    multiple=ratio(a,b)
    if multiple is None:return None
    try:value=multiple**(1/years)-1
    except OverflowError:return None
    return value if math.isfinite(value) else None


def accounting_features(v,prior=None,older=None,sector=None):
    """All v flow values are ONE matching fiscal duration, not mixed quarters/YTD."""
    p=prior or {}; old=older or {}; g=v.get; prev=p.get
    # Sector gating is a literal label match; 'Financials' / ' Banks ' must gate exactly like 'financials' / 'banks'.
    sector=None if sector is None else str(sector).strip().lower()
    for values in (v,p,old):
        if any(x is not None and (isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)) for x in values.values()):raise ValueError('nonfinite or nonnumeric accounting input')
        if values.get('capex') is not None and values['capex']<0:raise ValueError('canonical CapEx must be nonnegative')
    f={}; dimensions={}
    def put(name,value,dimension):f[name]=value;dimensions[name]=dimension
    debt=add(g('short_debt'),g('long_debt'))
    fcf=subtract(g('ocf'),g('capex')); netdebt=subtract(debt,g('cash'))
    ebitda=add(g('operating_income'),g('depreciation'))
    put('fcf',fcf,'quality');put('ebitda_proxy',ebitda,'quality');put('debt',debt,'health');put('net_debt',netdebt,'health')
    for name,numerator in [('gross_margin',g('gross_profit')),('operating_margin',g('operating_income')),('net_margin',g('net_income')),('fcf_margin',fcf),('ebitda_margin',ebitda)]:
        put(name,ratio(numerator,g('revenue')),'quality')
    avgassets=ratio(add(g('assets'),prev('assets')),2); avgequity=ratio(add(g('equity'),prev('equity')),2)
    put('roe',ratio(g('net_income'),avgequity),'quality');put('roa',ratio(g('net_income'),avgassets),'quality')
    put('gross_profitability',ratio(g('gross_profit'),g('assets')),'quality')
    invested=subtract(add(debt,g('equity')),g('cash'))
    oldinvested=subtract(add(add(prev('short_debt'),prev('long_debt')),prev('equity')),prev('cash'))
    tax=ratio(g('tax_expense'),g('pretax_income'))
    nopat=g('operating_income')*(1-tax) if tax is not None and 0<=tax<=1 and g('operating_income') is not None else None
    put('roic',ratio(nopat,ratio(add(invested,oldinvested),2)),'quality')
    for name,num,den in [('current_ratio',g('current_assets'),g('current_liabilities')),
        ('quick_ratio',subtract(subtract(g('current_assets'),g('inventory')),g('prepaids')),g('current_liabilities')),
        ('debt_equity',debt,g('equity')),('net_debt_ebitda',netdebt,ebitda),
        ('interest_coverage',g('operating_income'),g('interest_expense')),('cash_debt',g('cash'),debt)]:
        put(name,ratio(num,den) if sector not in ('banks','financials','insurance') else None,'health')
    put('ocf_net_income',ratio(g('ocf'),g('net_income')),'earnings_quality')
    put('fcf_net_income',ratio(fcf,g('net_income')),'earnings_quality')
    put('accrual_ratio',ratio(subtract(g('net_income'),g('ocf')),avgassets),'earnings_quality')
    for metric in ('revenue','eps','operating_income','receivables','inventory','diluted_shares','cash','assets'):
        put(metric+'_yoy',growth(g(metric),prev(metric)),'growth' if metric not in ('diluted_shares','assets') else 'allocation')
    put('fcf_yoy',growth(fcf,subtract(prev('ocf'),prev('capex'))),'growth')
    put('ebitda_yoy',growth(ebitda,add(prev('operating_income'),prev('depreciation'))),'growth')
    put('revenue_acceleration',subtract(f['revenue_yoy'],growth(prev('revenue'),old.get('revenue'))),'growth')
    put('receivables_vs_sales',subtract(f['receivables_yoy'],f['revenue_yoy']),'earnings_quality')
    put('inventory_vs_sales',subtract(f['inventory_yoy'],f['revenue_yoy']),'earnings_quality')
    put('gross_margin_change',subtract(f['gross_margin'],ratio(prev('gross_profit'),prev('revenue'))),'quality')
    put('operating_margin_change',subtract(f['operating_margin'],ratio(prev('operating_income'),prev('revenue'))),'quality')
    put('debt_change',subtract(debt,add(prev('short_debt'),prev('long_debt'))),'health')
    put('cash_change',subtract(g('cash'),prev('cash')),'health')
    for name,num,den in [('asset_turnover',g('revenue'),avgassets),('inventory_turnover',g('cogs'),ratio(add(g('inventory'),prev('inventory')),2)),
                         ('rd_sales',g('rd'),g('revenue')),('capex_sales',g('capex'),g('revenue')),('buybacks_ocf',g('buybacks'),g('ocf'))]:
        put(name,ratio(num,den),'allocation')
    # Day-count scaling is valid only for annual flows; quarter caller does not use these fields.
    receivable_days=ratio(ratio(add(g('receivables'),prev('receivables')),2),g('revenue'))
    inventory_days=ratio(ratio(add(g('inventory'),prev('inventory')),2),g('cogs'))
    payable_days=ratio(ratio(add(g('payables'),prev('payables')),2),g('cogs'))
    for name,value in [('receivable_days',receivable_days),('inventory_days',inventory_days),('payable_days',payable_days)]:
        put(name,value*365 if value is not None and sector not in ('banks','financials','saas','insurance') else None,'allocation')
    put('cash_conversion_cycle',subtract(add(f['receivable_days'],f['inventory_days']),f['payable_days']),'allocation')
    put('eps_acceleration',subtract(f['eps_yoy'],growth(prev('eps'),old.get('eps'))),'growth')
    put('fcf_acceleration',subtract(f['fcf_yoy'],growth(subtract(prev('ocf'),prev('capex')),subtract(old.get('ocf'),old.get('capex')))),'growth')
    oldtax=ratio(prev('tax_expense'),prev('pretax_income'))
    oldnopat=prev('operating_income')*(1-oldtax) if oldtax is not None and 0<=oldtax<=1 and prev('operating_income') is not None else None
    put('incremental_roic',ratio(subtract(nopat,oldnopat),subtract(invested,oldinvested)),'allocation')
    for name,num,den in [('dividends_ocf',g('dividends'),g('ocf')),('acquisitions_assets',g('acquisitions'),g('assets')),('issuance_equity',g('issuance'),g('equity'))]:
        put(name,ratio(num,den),'allocation')
    margins=[ratio(x.get('gross_profit'),x.get('revenue')) for x in (old,p,v)]
    incomes=[x.get('net_income') for x in (old,p,v)]
    with np.errstate(all='ignore'):   # extreme magnitudes may overflow; the guard below turns that into "unknown"
        put('gross_margin_stability',float(np.std(margins,ddof=1)) if all(x is not None for x in margins) else None,'quality')
        put('earnings_stability',float(np.std(incomes,ddof=1)/abs(np.mean(incomes))) if all(x is not None for x in incomes) and abs(np.mean(incomes))>0 else None,'quality')
    # A non-finite result is "unknown", never a populated feature (NaN would also be counted as coverage
    # and make the JSON response unserialisable).
    for name,value in list(f.items()):
        if value is not None and not math.isfinite(value):f[name]=None
    flags=[]
    if fcf is not None and fcf<0:flags.append('NEGATIVE_FCF')
    # Differences of two ratios carry ~1e-16 floating-point noise: an exact tie must not raise a directional flag.
    if f['gross_margin_change'] is not None and f['gross_margin_change']<-FLAG_NOISE:flags.append('FALLING_MARGINS')
    if f['revenue_acceleration'] is not None and f['revenue_acceleration']<-FLAG_NOISE:flags.append('REVENUE_DECELERATION')
    if f['diluted_shares_yoy'] is not None and f['diluted_shares_yoy']>0:flags.append('DILUTION')
    # Threshold-free descriptive flags; they do not change risk allocation.
    if f['receivables_vs_sales'] is not None and f['receivables_vs_sales']>FLAG_NOISE:flags.append('RECEIVABLES_OUTPACE_SALES')
    return f,dimensions,flags


def valuation_features(price,shares,values,*,price_currency,statement_currency,price_basis,share_basis,forward_eps=None):
    # Same input contract as accounting_features: a bool/str/None/inf must not become a ratio of 0.0 or a TypeError.
    for name,x in (('price',price),('shares',shares)):
        if isinstance(x,bool) or not isinstance(x,(int,float)):raise ValueError(name+' must be a number')
    for name,x in (*values.items(),('forward_eps',forward_eps)):
        if x is not None and (isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)):
            raise ValueError('nonfinite or nonnumeric valuation input: '+name)
    if values.get('capex') is not None and values['capex']<0:raise ValueError('canonical CapEx must be nonnegative')
    if price_currency!=statement_currency:raise ValueError('historical currency conversion required')
    if not price_basis or price_basis!=share_basis:raise ValueError('consistent split basis required')
    if not all(math.isfinite(x) and x>0 for x in (price,shares)):raise ValueError('invalid market capitalization inputs')
    marketcap=price*shares
    if not math.isfinite(marketcap):raise ValueError('market cap overflow')
    debt=add(values.get('short_debt'),values.get('long_debt'))
    # Missing preferred/minority claims make EV unavailable, never implicitly zero.
    ev=subtract(add(add(add(marketcap,debt),values.get('preferred_equity')),values.get('minority_interest')),values.get('cash'))
    fcf=subtract(values.get('ocf'),values.get('capex')); ebitda=add(values.get('operating_income'),values.get('depreciation'))
    return {'market_cap':marketcap,'enterprise_value':ev,'pe':ratio(marketcap,values.get('net_income')),
            'forward_pe':ratio(price,forward_eps),'ps':ratio(marketcap,values.get('revenue')),'pb':ratio(marketcap,values.get('equity')),
            'ev_revenue':ratio(ev,values.get('revenue')),'ev_ebitda':ratio(ev,ebitda),'price_fcf':ratio(marketcap,fcf),
            'fcf_yield':ratio(fcf,marketcap),'earnings_yield':ratio(values.get('net_income'),marketcap)}


def normalize_cross_section(frame,metric,group='sector',minimum=5):
    """Only caller-supplied PIT universe/classifications. Ties get average rank."""
    out=frame.copy();out[metric+'_percentile']=np.nan;out[metric+'_zscore']=np.nan
    for _,rows in out.dropna(subset=[group,metric]).groupby(group):
        if len(rows)<minimum:continue
        x=rows[metric].astype(float);lo,hi=x.quantile([.01,.99])
        # With n <= 100 the 1%/99% quantiles interpolate toward the extremes, so one outlier would pass almost
        # untouched and flatten every other z-score. Also clip at 5 robust sigmas (1.4826*MAD) around the median.
        median=x.median();robust=1.4826*(x-median).abs().median()
        if robust>0:lo,hi=max(lo,median-5*robust),min(hi,median+5*robust)
        w=x.clip(lo,hi);sd=w.std(ddof=1)
        out.loc[rows.index,metric+'_percentile']=(x.rank(method='average')-.5)/len(x)
        if sd>0:out.loc[rows.index,metric+'_zscore']=(w-w.mean())/sd
    return out


def quarters(rows,metric):
    """Discrete quarters, including YTD differencing; no rolling-YTD summation."""
    candidates=[r for r in rows if r['metric']==metric and r['period_start'] and r['value'] is not None]
    out={}
    if len({(r['source'],r['currency'],r['units']) for r in candidates})>1:raise ValueError('ambiguous quarter source/currency/units')
    for r in candidates:
        days=(pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1
        if 60<=days<=120:out[r['period_end']]={'value':r['value'],'ids':[r['id']],'start':r['period_start'],'tags':{_tag(r)}}
    if metric in ('eps','diluted_shares','shares_outstanding'):return out
    for r in candidates:
        if r['period_end'] in out:continue
        earlier=[p for p in candidates if p['period_start']==r['period_start'] and p['period_end']<r['period_end'] and p['currency']==r['currency'] and p['units']==r['units'] and p['source']==r['source']]
        if earlier:
            p=max(earlier,key=lambda x:x['period_end']);days=(pd.Timestamp(r['period_end'])-pd.Timestamp(p['period_end'])).days
            if 60<=days<=120:out[r['period_end']]={'value':r['value']-p['value'],'ids':[r['id'],p['id']],'start':str((pd.Timestamp(p['period_end'])+pd.Timedelta(days=1)).date()),'tags':{_tag(r),_tag(p)}}
    return out


def _tag(row):
    """Source taxonomy tag of a stored fact (None for synthetic/JSON data that carries none)."""
    return (row.get('metadata') or {}).get('tag')


def _ttm_from_reported_totals(rows,metric,end):
    """TTM = FY + current YTD - prior-year YTD, the standard published form; at a fiscal year end it IS the
    reported annual figure. Summing separately-filed discrete quarters mixes vintages (originals for Q1-Q3, a
    recast YTD for Q4) and misses the 10-K total by 1-2% on real filers.

    Returns (value, ids, tags, start) or None when the needed reported totals are not all visible."""
    flow=[r for r in rows if r['metric']==metric and r['period_start'] and r['value'] is not None]
    if len({(r['source'],r['currency'],r['units']) for r in flow})>1:raise ValueError('ambiguous quarter source/currency/units')
    def days(r):return (pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1
    current=[r for r in flow if r['period_end']==end and 55<=days(r)<=380]
    if not current:return None
    cur=max(current,key=days)
    if days(cur)>=350:return cur['value'],[cur['id']],{_tag(cur)},cur['period_start']
    before=str((pd.Timestamp(cur['period_start'])-pd.Timedelta(days=1)).date())
    annual=[r for r in flow if r['period_end']==before and 350<=days(r)<=380]
    if len(annual)!=1:return None
    fy=annual[0];target=pd.Timestamp(end)-pd.DateOffset(years=1)
    prior=[r for r in flow if r['period_start']==fy['period_start'] and abs((pd.Timestamp(r['period_end'])-target).days)<=10 and abs(days(r)-days(cur))<=10]
    if not prior:return None
    pri=min(prior,key=lambda r:abs((pd.Timestamp(r['period_end'])-target).days))
    return fy['value']+cur['value']-pri['value'],[fy['id'],cur['id'],pri['id']],{_tag(fy),_tag(cur),_tag(pri)},str((pd.Timestamp(pri['period_end'])+pd.Timedelta(days=1)).date())


def comparable_accounting_rows(rows):
    revenues=[r for r in rows if r['metric']=='revenue' and r['value'] is not None]
    if not revenues:return rows
    latest=max(revenues,key=lambda r:(r['period_end'],r['available_at']))
    standard=latest['metadata'].get('accounting_standard')
    return [r for r in rows if r['metadata'].get('accounting_standard')==standard]


def trailing_twelve_months(rows,as_of=None,max_age_days=550):
    """Trailing twelve months of one fiscal calendar.

    Revenue, net income, operating cash flow and CapEx use the published form FY + current YTD - prior YTD
    (at a fiscal year end: the reported annual figure) and fall back to four contiguous discrete quarters.
    Annual/YTD totals are never double counted. A value whose inputs mix XBRL definitions (e.g. parent-only
    vs consolidated net income) is withheld. EPS is additive only with an explicitly shared split basis;
    weighted share counts and balance-sheet stocks are never summed. With `as_of`, the TTM carries its own
    age and becomes STALE past `max_age_days` (it can otherwise lag the annual block by years).
    """
    rows=comparable_accounting_rows(rows)
    metrics=('revenue','net_income','ocf','capex','eps')
    result={'values':{k:None for k in (*metrics,'fcf')},'lineage':[],
            'status':'INCOMPLETE','period_start':None,'period_end':None,'reasons':{}}
    revenue=quarters(rows,'revenue')
    if not revenue:return result
    end=max(revenue)
    result['period_end']=end
    for metric in metrics:
        value=ids=start=tags=None
        if metric!='eps':
            totals=_ttm_from_reported_totals(rows,metric,end)
            if totals is not None:value,ids,tags,start=totals
        if value is None:
            q=quarters(rows,metric)
            dates=sorted(d for d in q if d<=end)[-4:]
            if len(dates)!=4 or dates[-1]!=end:
                result['reasons'][metric]='MISSING_QUARTERS';continue
            if any(pd.Timestamp(q[b]['start'])!=pd.Timestamp(a)+pd.Timedelta(days=1) for a,b in zip(dates,dates[1:])):
                result['reasons'][metric]='NONCONTIGUOUS_QUARTERS';continue
            start=q[dates[0]]['start']
            if not 350<=(pd.Timestamp(end)-pd.Timestamp(start)).days+1<=380:
                result['reasons'][metric]='NON_ANNUAL_SPAN';continue
            ids=[i for d in dates for i in q[d]['ids']]
            tags={t for d in dates for t in q[d]['tags']}
            value=sum(q[d]['value'] for d in dates)
        if result['period_start'] is not None and start!=result['period_start']:
            result['reasons'][metric]='PERIOD_MISMATCH';continue
        if len(tags-{None})>1:
            result['reasons'][metric]='MIXED_TAG_DEFINITIONS';continue
        if metric=='eps':
            bases={r['metadata'].get('share_basis') for r in rows if r['id'] in ids}
            if len(bases)!=1 or None in bases:
                result['reasons'][metric]='UNVERIFIED_SPLIT_BASIS';continue
        if not math.isfinite(value):
            result['reasons'][metric]='OVERFLOW';continue
        result['values'][metric]=value;result['lineage']+=ids
        result['period_start']=start
    result['values']['fcf']=subtract(result['values']['ocf'],result['values']['capex'])
    result['lineage']=sorted(set(result['lineage']))
    result['status']='AVAILABLE' if all(v is not None for v in result['values'].values()) else 'INCOMPLETE'
    if as_of is not None:
        result['age_days']=(pd.Timestamp(utc(as_of)).normalize()-pd.Timestamp(end,tz='UTC')).days
        if result['age_days']>max_age_days:result['status']='STALE'
    return result


class FundamentalFeatureEngine:
    def __init__(self,store,max_period_age_days=550):
        # NaN would silently disable the staleness gate (every comparison is False); a non-positive value would mark all data stale.
        if isinstance(max_period_age_days,bool) or not isinstance(max_period_age_days,(int,float)) or not math.isfinite(max_period_age_days) or max_period_age_days<=0:
            raise ValueError('max_period_age_days must be a positive finite number')
        self.store=store;self.max_age=max_period_age_days

    @consistent_read
    def snapshot(self,symbol,as_of,*,asset_class='equity',persist=False):
        ts=pd.Timestamp(utc(as_of)); rows=comparable_accounting_rows(self.store.as_of(symbol,as_of,kind='financial'))
        result={'symbol':symbol.upper(),'as_of':utc(as_of),'version':VERSION,'asset_class':asset_class,'features':{},'raw':{},'lineage':[],
                'dimensions':{},'flags':[],'status':'MISSING','score':None,'confidence':None,'completeness':0.,'sector':None}
        if asset_class!='equity':
            result['status']='REQUIRES_'+{'etf':'DATED_HOLDINGS','forex':'MACRO_MODEL','crypto':'NATIVE_DATA'}.get(asset_class,'ASSET_ADAPTER')
            return result
        classifications=self.store.as_of(symbol,as_of,kind='classification')
        if classifications:result['sector']=classifications[-1]['metadata'].get('sector')
        annual=[r for r in rows if r['metric']=='revenue' and r['period_start'] and 300<=(pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1<=380 and r['value'] is not None]
        result['ttm']=trailing_twelve_months(rows,as_of,self.max_age)
        if not annual:return result
        latest=max(annual,key=lambda r:r['period_end']);end=latest['period_end'];start=latest['period_start'];currency=latest['currency']
        def period_values(target):
            matching=[r for r in annual if abs((pd.Timestamp(r['period_end'])-target).days)<=10]
            if not matching:return {},[]
            period=min(matching,key=lambda r:abs((pd.Timestamp(r['period_end'])-target).days)); selected={}; ids=[]
            for r in rows:
                if r['period_end']!=period['period_end'] or r['currency'] not in (currency,''):continue
                if r['period_start'] and r['period_start']!=period['period_start']:continue
                if r['metric'] in selected:raise ValueError('ambiguous currency/unit/source; select one provider')
                selected[r['metric']]=r['value'];ids.append(r['id'])
            return selected,ids
        current,ids=period_values(pd.Timestamp(end));prior,pids=period_values(pd.Timestamp(end)-pd.DateOffset(years=1));old,oids=period_values(pd.Timestamp(end)-pd.DateOffset(years=2))
        f,dimensions,flags=accounting_features(current,prior,old,result['sector'])
        for years in (3,5):
            earlier,eids=period_values(pd.Timestamp(end)-pd.DateOffset(years=years));ids+=eids
            f[f'revenue_cagr_{years}y']=annual_cagr(current.get('revenue'),earlier.get('revenue'),years);dimensions[f'revenue_cagr_{years}y']='growth'
        for metric in ('revenue','eps','ocf','operating_income'):
            f[metric+'_qoq']=None;dimensions[metric+'_qoq']='growth'
            q=quarters([r for r in rows if r['currency'] in (currency,'')],metric); dates=sorted(q)
            if len(dates)>=2:
                a,b=dates[-1],dates[-2]
                consecutive=60<=(pd.Timestamp(a)-pd.Timestamp(b)).days<=120
                f[metric+'_qoq']=growth(q[a]['value'],q[b]['value']) if consecutive else None
                dimensions[metric+'_qoq']='growth';ids+=q[a]['ids']+q[b]['ids']
        # Per-share growth needs documented common split basis. An accession's
        # 'as_reported' marker alone cannot distinguish splits from issuance.
        share_rows=[r for r in rows if r['metric'] in ('eps','diluted_shares') and r['id'] in ids+pids+oids]
        bases={r['metadata'].get('share_basis') for r in share_rows}
        if not share_rows or None in bases or len(bases)!=1:
            for name in ('eps_yoy','eps_acceleration','eps_qoq','diluted_shares_yoy'):f[name]=None
            flags=[x for x in flags if x!='DILUTION']
            result['per_share_comparability']='UNVERIFIED_SPLIT_BASIS'
        all_ids=sorted(set(ids+pids+oids));used=[r for r in rows if r['id'] in all_ids]
        age=(ts.normalize()-pd.Timestamp(end,tz='UTC')).days
        # Revenue anchors the period. If other annual flows are already public for a LATER year (e.g. an unmapped
        # revenue tag), the snapshot is a year behind the filing and must not be served as current.
        newer=sorted(r['period_end'] for r in rows if r['metric'] in ('net_income','ocf','operating_income') and r['period_start'] and r['value'] is not None
                     and 300<=(pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1<=380 and pd.Timestamp(r['period_end'])>pd.Timestamp(end)+pd.Timedelta(days=30))
        if newer:result['period_anchor_warning']={'anchored_period_end':end,'later_annual_period_end':newer[-1],'reason':'REVENUE_MISSING_FOR_LATER_PERIOD'}
        result.update(features=f,raw=current,dimensions=dimensions,flags=flags,period_end=end,period_start=start,currency=currency,
                      lineage=all_ids,reported_at=max(r['reported_at'] for r in used),available_at=max(r['available_at'] for r in used),
                      age_days=age,completeness=sum(v is not None for v in f.values())/len(f),
                      status='STALE' if age>self.max_age or newer else 'AVAILABLE',confidence=None,
                      confidence_note='Completeness and freshness are data quality, not calibrated predictive confidence',
                      evidence={d:EVIDENCE[d] for d in sorted(set(dimensions.values()))})
        if persist:self.store.save('feature_snapshots',result,VERSION)
        return result

    @consistent_read
    def etf_snapshot(self,symbol,as_of,metric):
        holdings=self.store.as_of(symbol,as_of,kind='holding')
        if not holdings:return {'symbol':symbol,'as_of':utc(as_of),'value':None,'coverage':0.,'status':'MISSING_DATED_HOLDINGS'}
        end=max(r['period_end'] for r in holdings); holdings=[r for r in holdings if r['period_end']==end]
        latest=max(holdings,key=lambda r:(r['available_at'],r['revision']))
        holdings=[r for r in holdings if r['revision']==latest['revision'] and r['source']==latest['source']]
        if (pd.Timestamp(utc(as_of))-pd.Timestamp(end,tz='UTC')).days>180:
            return {'symbol':symbol,'as_of':utc(as_of),'value':None,'coverage':0.,'status':'STALE_HOLDINGS'}
        if any(r['value'] is None for r in holdings):
            return {'symbol':symbol,'as_of':utc(as_of),'value':None,'coverage':0.,'status':'INCOMPLETE_HOLDING_WEIGHTS'}
        if any(r['value']<0 for r in holdings):raise ValueError('long-only ETF adapter rejects negative holding weights')
        constituents=[r['metadata'].get('constituent') for r in holdings]
        if None in constituents or len(set(constituents))!=len(constituents):raise ValueError('missing or duplicate constituent')
        total=sum(r['value'] for r in holdings)
        if total<=0 or total>1.001:raise ValueError('invalid disclosed portfolio weights')
        weighted=0.;covered=0.;lineage=[]
        for row in holdings:
            snap=self.snapshot(row['metadata']['constituent'],as_of);value=snap['features'].get(metric)
            if value is not None and snap['status']=='AVAILABLE':
                weighted+=row['value']*value;covered+=row['value'];lineage+=snap['lineage']+[row['id']]
        # No reweighting of covered survivors: missing constituents invalidate aggregate.
        return {'symbol':symbol,'as_of':utc(as_of),'value':weighted if covered>=.999 else None,'coverage':covered,
                'status':'AVAILABLE' if covered>=.999 else 'INCOMPLETE_HOLDINGS_COVERAGE','lineage':lineage,
                'holdings_period_end':end,'method':'disclosed-weight average of constituent ratios; not aggregate accounting ratio'}


FeatureEngine=FundamentalFeatureEngine
