"""Independent audit oracle: Decimal/std-library only, no production imports."""
from decimal import Decimal,localcontext
from math import sqrt
from statistics import mean,stdev
D=lambda x:None if x is None else Decimal(str(x))

def divide(a,b):
    a,b=D(a),D(b)
    return None if a is None or b is None or b<=0 else float(a/b)

def calculate(v,p):
    v={k:D(x) for k,x in v.items()};p={k:D(x) for k,x in p.items()}
    debt=v['short_debt']+v['long_debt'];fcf=v['ocf']-v['capex'];ebitda=v['operating_income']+v['depreciation']
    avgassets=(v['assets']+p['assets'])/2;avgequity=(v['equity']+p['equity'])/2
    invested=debt+v['equity']-v['cash'];oldinvested=p['short_debt']+p['long_debt']+p['equity']-p['cash']
    nopat=v['operating_income']*(1-v['tax_expense']/v['pretax_income'])
    out={'fcf':float(fcf),'debt':float(debt),'net_debt':float(debt-v['cash']),
         'gross_margin':divide(v['gross_profit'],v['revenue']),'operating_margin':divide(v['operating_income'],v['revenue']),
         'net_margin':divide(v['net_income'],v['revenue']),'fcf_margin':divide(fcf,v['revenue']),'ebitda_margin':divide(ebitda,v['revenue']),
         'roe':divide(v['net_income'],avgequity),'roa':divide(v['net_income'],avgassets),'roic':divide(nopat,(invested+oldinvested)/2),
         'current_ratio':divide(v['current_assets'],v['current_liabilities']),
         'quick_ratio':divide(v['current_assets']-v['inventory']-v['prepaids'],v['current_liabilities']),
         'debt_equity':divide(debt,v['equity']),'net_debt_ebitda':divide(debt-v['cash'],ebitda),
         'interest_coverage':divide(v['operating_income'],v['interest_expense']),'cash_debt':divide(v['cash'],debt),
         'accrual_ratio':divide(v['net_income']-v['ocf'],avgassets),'ocf_net_income':divide(v['ocf'],v['net_income']),'fcf_net_income':divide(fcf,v['net_income'])}
    for k in ('revenue','eps','diluted_shares','receivables','inventory','cash','assets','operating_income'):
        out[k+'_yoy']=None if p[k]<=0 else float(v[k]/p[k]-1)
    out['revenue_acceleration']=None # supplied separately by timeline test
    return out

def valuation(v,price,shares,forward_eps):
    cap=D(price)*D(shares);debt=D(v['short_debt'])+D(v['long_debt'])
    ev=cap+debt-D(v['cash'])+D(v['preferred_equity'])+D(v['minority_interest'])
    fcf=D(v['ocf'])-D(v['capex']);ebitda=D(v['operating_income'])+D(v['depreciation'])
    return {'market_cap':float(cap),'enterprise_value':float(ev),'pe':divide(cap,v['net_income']),
        'forward_pe':divide(price,forward_eps),'ps':divide(cap,v['revenue']),'pb':divide(cap,v['equity']),
        'ev_revenue':divide(ev,v['revenue']),'ev_ebitda':divide(ev,ebitda),'price_fcf':divide(cap,fcf),
        'fcf_yield':divide(fcf,cap),'earnings_yield':divide(v['net_income'],cap)}

def ranks_z(values):
    ordered=sorted(values);n=len(values)
    def quantile(p):
        index=(n-1)*p;lo=int(index);hi=min(lo+1,n-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)
    lo,hi=quantile(.01),quantile(.99)
    # Lead re-audit: also clip at 5 robust sigmas (1.4826*MAD) so one outlier cannot flatten a small group.
    med=(ordered[n//2]+ordered[(n-1)//2])/2;dev=sorted(abs(x-med) for x in values);robust=1.4826*(dev[n//2]+dev[(n-1)//2])/2
    if robust>0:lo,hi=max(lo,med-5*robust),min(hi,med+5*robust)
    clipped=[min(hi,max(lo,x)) for x in values];mu=mean(clipped);sd=stdev(clipped)
    return [((sum(y<x for y in values)+.5*sum(y==x for y in values))/n,(w-mu)/sd) for x,w in zip(values,clipped)]

def linear_prediction(values,means,scales,coefficients,intercept):
    return float(D(intercept)+sum((D(x)-D(m))/D(s)*D(c) for x,m,s,c in zip(values,means,scales,coefficients)))

def performance(equity,journal,annual=252):
    eq=[100.,*map(float,equity)];returns=[b/a-1 for a,b in zip(eq,eq[1:])];peak=eq[0];dd=[]
    for x in eq:peak=max(peak,x);dd.append(1-x/peak)
    profits=[float(x['pnl']) for x in journal if x['event']=='EXIT'];wins=[x for x in profits if x>0];losses=[x for x in profits if x<0]
    downside=sqrt(sum(min(x,0)**2 for x in returns)/len(returns))
    return {'final_equity':eq[-1],'total_return':eq[-1]/100-1,'sharpe':mean(returns)/stdev(returns)*sqrt(annual) if stdev(returns)>0 else None,
        'sortino':mean(returns)/downside*sqrt(annual) if downside else None,'max_drawdown':max(dd),
        'profit_factor':sum(wins)/-sum(losses) if losses else None,'expectancy_gbp':mean(profits) if profits else None,
        'win_rate':len(wins)/len(profits) if profits else None,'trades':len(profits),'fees_gbp':sum(x['fee'] for x in journal if x.get('fee') is not None),
        'turnover':sum(x.get('price',0)*x.get('quantity',0) for x in journal if x['event'] in ('ENTRY','EXIT'))/100}
