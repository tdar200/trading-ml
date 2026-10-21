"""Instrument/industry-specific transforms; no universal equity score."""
from __future__ import annotations
import math
import pandas as pd
from fundamentals.features import ratio,growth,subtract,add
from fundamentals.store import utc

SECTOR_INPUTS={
 'banks':('net_interest_income','earning_assets','cet1_capital','risk_weighted_assets','loans','nonperforming_loans','provisions'),
 'saas':('arr','nrr','cac','ltv','revenue','gross_profit','ocf','capex'),
 'retail':('same_store_sales_growth','store_count','inventory','revenue'),
 'semiconductors':('inventory','gross_profit','revenue','utilization','bookings'),
 'energy':('production','reserves','realized_price','production_cost_per_unit'),
}


def sector_features(sector,current,prior=None):
    if sector not in SECTOR_INPUTS:raise ValueError('unknown industry adapter')
    p=prior or {}; g=current.get
    out={k:g(k) for k in SECTOR_INPUTS[sector]}
    if sector=='banks':out.update(net_interest_margin=ratio(g('net_interest_income'),g('earning_assets')),
                                  cet1_ratio=ratio(g('cet1_capital'),g('risk_weighted_assets')),
                                  npl_ratio=ratio(g('nonperforming_loans'),g('loans')),loan_growth=growth(g('loans'),p.get('loans')))
    if sector=='saas':out.update(arr_growth=growth(g('arr'),p.get('arr')),ltv_cac=ratio(g('ltv'),g('cac')),
                                 rule_of_40=add(growth(g('revenue'),p.get('revenue')),ratio(subtract(g('ocf'),g('capex')),g('revenue'))))
    if sector=='retail':out['store_growth']=growth(g('store_count'),p.get('store_count'))
    if sector=='semiconductors':out['inventory_growth']=growth(g('inventory'),p.get('inventory'))
    if sector=='energy':out.update(production_growth=growth(g('production'),p.get('production')),
                                    unit_margin=subtract(g('realized_price'),g('production_cost_per_unit')))
    return out


def crypto_features(current,prior=None):
    p=prior or {};out={}
    for k in ('active_addresses','transaction_volume','network_fees','tvl','circulating_supply','stablecoin_supply'):
        out[k]=current.get(k);out[k+'_growth']=growth(current.get(k),p.get(k))
    out['scheduled_unlock_fraction']=ratio(current.get('scheduled_unlocks'),current.get('circulating_supply'))
    out['staking_fraction']=ratio(current.get('staked_supply'),current.get('circulating_supply'))
    out['funding_rate']=current.get('funding_rate')
    return out


def estimate_revisions(store,symbol,as_of,metric,forecast_period):
    """Hold the forecast fiscal target constant when comparing estimate revisions."""
    def get(at):
        rows=[r for r in store.as_of(symbol,at,kind='estimate') if r['metric']==metric and r['period_end']==forecast_period]
        if len(rows)>1:raise ValueError('ambiguous estimate source/units')
        return rows[0] if rows else None
    current=get(as_of);out={}
    for days in (30,60,90):
        old=get(pd.Timestamp(utc(as_of))-pd.Timedelta(days=days)); value=None
        if current and old:
            if (current['currency'],current['units'])!=(old['currency'],old['units']):raise ValueError('estimate currency/unit change')
            a,b=current['value'],old['value'];value=(a-b)/abs(b) if a is not None and b else None
        out[f'{days}d']=value
    return out


def historical_fx(store,amount,source_currency,target_currency,as_of):
    if not math.isfinite(amount):raise ValueError('nonfinite amount')
    if source_currency==target_currency:return {'value':amount,'rate':1.,'lineage':[]}
    pair=source_currency+target_currency
    rows=[r for r in store.as_of('FX:'+pair,as_of,kind='fx') if r['metric']=='rate']
    if not rows:return {'value':None,'rate':None,'lineage':[]}
    row=max(rows,key=lambda r:r['period_end'])
    # Two sources quoting different rates for the same date is an unresolved disagreement: refuse to convert
    # rather than let max() silently pick whichever row happened to sort first.
    if len({r['value'] for r in rows if r['period_end']==row['period_end']})>1:raise ValueError('conflicting FX sources for one quote date')
    if (pd.Timestamp(utc(as_of)).date()-pd.Timestamp(row['period_end']).date()).days>7:
        return {'value':None,'rate':None,'lineage':[row['id']]}
    if row['currency']!=target_currency or row['units']!=target_currency+'/'+source_currency:
        raise ValueError('FX quote currency/units do not match requested conversion')
    if row['value'] is None or row['value']<=0:return {'value':None,'rate':None,'lineage':[row['id']]}
    if (pd.Timestamp(utc(as_of))-pd.Timestamp(row['available_at'])).total_seconds()>7*86400:return {'value':None,'rate':None,'lineage':[row['id']]}
    return {'value':amount*row['value'],'rate':row['value'],'lineage':[row['id']]}


def apply_known_splits(store,symbol,quantity,price,basis_at,as_of):
    q,p=float(quantity),float(price);start=pd.Timestamp(utc(basis_at));end=pd.Timestamp(utc(as_of));ids=[]
    effective={}
    for event in sorted(store.events(symbol,as_of),key=lambda e:e['event_at']):
        if event['event_type']!='split' or not start<pd.Timestamp(event['event_at'])<=end:continue
        ratio_=event['payload'].get('ratio')
        if isinstance(ratio_,bool) or not isinstance(ratio_,(int,float)) or not math.isfinite(ratio_) or ratio_<=0:raise ValueError('invalid split')
        # One real split reported by several vendors (distinct source/key) is still ONE corporate action.
        if event['event_at'] in effective:
            if not math.isclose(effective[event['event_at']][0],ratio_,rel_tol=1e-9):raise ValueError('conflicting split ratios for one effective date')
            effective[event['event_at']][1].append(event['id']);continue
        effective[event['event_at']]=(ratio_,[event['id']])
    for ratio_,event_ids in effective.values():
        q*=ratio_;p/=ratio_;ids+=event_ids
    return {'quantity':q,'price':p,'lineage':ids}


def own_history_percentile(store,symbol,metric,value,as_of,years=3):
    if years not in (3,5):raise ValueError('expected three or five years')
    ts=pd.Timestamp(utc(as_of));start=ts-pd.DateOffset(years=years)
    rows=[r for r in store.as_of(symbol,as_of) if r['metric']==metric and start.date()<=pd.Timestamp(r['period_end']).date()<=ts.date() and r['value'] is not None]
    # A flow must be ranked against comparable periods only: 3-month, year-to-date and annual rows of one
    # metric are different quantities. Compare like with like, anchored on the most recent period's duration.
    def duration_class(r):
        if not r['period_start']:return 'instant'
        days=(pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1
        return 'quarter' if days<=120 else 'ytd' if days<300 else 'annual'
    if rows:
        newest=max(rows,key=lambda r:(r['period_end'],{'instant':0,'quarter':1,'ytd':2,'annual':3}[duration_class(r)]*-1))
        rows=[r for r in rows if duration_class(r)==duration_class(newest)]
    if len(rows)<8 or value is None:return None
    if len({(r['currency'],r['units'],r['source']) for r in rows})!=1:
        raise ValueError('historical comparison mixes currency, units or sources')
    values=[r['value'] for r in rows]
    return (sum(v<value for v in values)+.5*sum(v==value for v in values))/len(values)


def altman_manufacturing_z(values,market_cap,sector):
    """Original public-manufacturing discriminant, descriptive only, not calibrated locally."""
    if sector!='manufacturing':return None
    a=values.get('assets');l=values.get('liabilities')
    terms=[ratio(subtract(values.get('current_assets'),values.get('current_liabilities')),a),
           ratio(values.get('retained_earnings'),a),ratio(values.get('operating_income'),a),
           ratio(market_cap,l),ratio(values.get('revenue'),a)]
    if any(x is None for x in terms):return None
    return sum(w*x for w,x in zip((1.2,1.4,3.3,.6,1.),terms))


def insider_activity(store,symbol,as_of,days=90):
    start=pd.Timestamp(utc(as_of))-pd.Timedelta(days=days);buys=0.;sales=0.;buyers=set();unknown=0
    if days<=0:raise ValueError('positive window required')
    records=[r for r in store.as_of(symbol,as_of,kind='ownership') if pd.Timestamp(r['available_at'])>=start and r['value'] is not None]
    if not records:return {'status':'MISSING','discretionary_purchase_value':None,'reported_sale_value':None,'distinct_buyers':None}
    currencies={(r['currency'],r['units']) for r in records}
    if len(currencies)!=1 or any(not c or c!=u for c,u in currencies):raise ValueError('insider transaction values require one explicit currency')
    for row in records:
        meta=row['metadata'];at=pd.Timestamp(row['available_at'])
        if at<start or row['value'] is None:continue
        # Publication age, not transaction date, controls what was observable.
        if meta.get('transaction_code')=='P' and meta.get('discretionary') is True:
            buys+=row['value'];buyers.add(meta.get('insider_id',row['id']))
        elif meta.get('transaction_code')=='S':sales+=row['value']
        else:unknown+=1
    return {'status':'AVAILABLE','currency':records[0]['currency'],'discretionary_purchase_value':buys,'reported_sale_value':sales,'distinct_buyers':len(buyers),
            'unclassified_transactions':unknown,'note':'Sales are not automatically bearish; currencies must be standardized before ingestion'}
