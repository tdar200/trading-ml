"""Independent audit oracle: stdlib only, no application imports.

Decimal for economic arithmetic; sample standard deviation; zero risk-free rate;
Sortino downside RMS over ALL observations; break-even is neither win nor loss.
"""
from decimal import Decimal, ROUND_FLOOR
from math import sqrt, comb, log
from statistics import mean, stdev


def D(x):
    return Decimal(str(x))


def size(equity, cash, entry, stop, fee=0, friction=0, risk='.005', cap='.25', step='.000001'):
    e,c,p,s,f,a,r,k,t=map(D,(equity,cash,entry,stop,fee,friction,risk,cap,step))
    loss=p*(1+f)-s*(1-a)*(1-f)
    q=min(e*r/loss,c/(p*(1+f)),e*k/p)
    return (q/t).to_integral_value(rounding=ROUND_FLOOR)*t


class Ledger:
    def __init__(self, cash=100):
        self.initial=D(cash); self.cash=D(cash); self.quantity=D(0); self.basis=D(0)
        self.fees=D(0); self.realized=D(0); self.entries=[]
    def buy(self, quantity, price, fee=0):
        q,p,f=map(D,(quantity,price,fee)); cost=q*p+f
        assert q>0 and cost<=self.cash
        self.cash-=cost; self.quantity+=q; self.basis+=cost; self.fees+=f
        self.entries.extend([('BUY',-q*p),('ENTRY_FEE',-f)])
    def sell(self, quantity, price, fee=0):
        q,p,f=map(D,(quantity,price,fee)); assert 0<q<=self.quantity
        basis=self.basis*q/self.quantity; pnl=q*p-f-basis
        self.cash+=q*p-f; self.quantity-=q; self.basis-=basis
        self.fees+=f; self.realized+=pnl
        self.entries.extend([('SELL',q*p),('EXIT_FEE',-f)])
        return pnl
    def mark(self, price):
        return self.cash+self.quantity*D(price)
    def unrealized(self, price):
        return self.quantity*D(price)-self.basis


def statistics(equity, pnl, annual=252):
    e=list(map(float,equity)); trades=list(map(float,pnl))
    returns=[b/a-1 for a,b in zip(e,e[1:])]; peak=e[0]; dd=[]
    for x in e:
        peak=max(peak,x); dd.append(1-x/peak)
    wins=[x for x in trades if x>0]; losses=[x for x in trades if x<0]
    sigma=stdev(returns) if len(returns)>1 else 0
    down=sqrt(mean([min(x,0)**2 for x in returns]))
    cagr=(e[-1]/e[0])**(annual/len(returns))-1
    return dict(final_equity=e[-1],total_return=e[-1]/e[0]-1,cagr=cagr,
                daily_mean=mean(returns),monthly_geometric=(e[-1]/e[0])**(annual/12/len(returns))-1,
                sharpe=mean(returns)/sigma*sqrt(annual) if sigma else None,
                sortino=mean(returns)/down*sqrt(annual) if down else None,
                max_drawdown=max(dd),average_drawdown=mean(dd),
                calmar=cagr/max(dd) if max(dd)>0 else None,
                trades=len(trades),win_rate=len(wins)/len(trades) if trades else None,
                profit_factor=sum(wins)/-sum(losses) if losses else None,
                expectancy_gbp=mean(trades) if trades else None,
                average_win=mean(wins) if wins else None,average_loss=mean(losses) if losses else None,
                reward_risk_realized=mean(wins)/-mean(losses) if wins and losses else None)


def indicators(rows, n=20):
    """Exact simple rolling convention used by the six small-account rules."""
    out=[]; close=[D(r['close']) for r in rows]; tr=[]; rsis=[]
    for i,r in enumerate(rows):
        h,l=D(r['high']),D(r['low']); tr.append(max(h-l,abs(h-close[i-1]),abs(l-close[i-1])) if i else h-l)
        sma=sum(close[i-n+1:i+1])/n if i>=n-1 else None
        sd=stdev(map(float,close[i-n+1:i+1])) if sma is not None else None
        changes=[close[j]-close[j-1] for j in range(i-n+1,i+1)] if i>=n else []
        gain=sum(max(x,D(0)) for x in changes); loss=sum(max(-x,D(0)) for x in changes)
        rsi=100*gain/(gain+loss) if gain+loss else None; rsis.append(rsi)
        atr=sum(tr[-14:])/14 if i>=13 else None
        signals={k:False for k in ('donchian','ma','rsi','bollinger','volume','gap')}
        if atr is not None:
            if i>=n: signals['donchian']=close[i]>max(D(x['high']) for x in rows[i-n:i])
            if i>=n: signals['ma']=close[i]>sma and close[i-1]<=sum(close[i-n:i])/n
            signals['rsi']=rsi is not None and i>0 and rsis[i-1] is not None and rsi>30 and rsis[i-1]<=30
            if sma is not None:
                signals['bollinger']=float(close[i])<float(sma)-2*sd
                signals['gap']=i>0 and D(r['open'])>close[i-1]*D('1.01') and close[i]>D(r['open']) and close[i]>sma
            if i>=n:
                signals['volume']=close[i]>close[i-1] and D(r['volume'])>D('1.5')*sum(D(x['volume']) for x in rows[i-n:i])/n and close[i]>sma
        out.append(dict(atr=float(atr) if atr is not None else None,signals=signals))
    return out


def terminal_probability(p,reward,risk,n,cost=.1,target=10):
    up=1+risk*(reward-cost); down=max(0,1-risk*(1+cost))
    return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)
               if up**k*down**(n-k)>=target*(1-1e-13))


def hit_probability(p,reward,risk,n,target,cost=.1):
    # Absorbing DP over surviving (step,wins) states; independent of path sampling.
    up=1+risk*(reward-cost); down=max(0,1-risk*(1+cost)); states={0:1.}; hit=0.
    for t in range(1,n+1):
        nxt={}
        for wins,prob in states.items():
            for k,weight in ((wins,1-p),(wins+1,p)):
                equity=100*up**k*down**(t-k)
                if equity>=target*(1-1e-13): hit+=prob*weight
                else: nxt[k]=nxt.get(k,0)+prob*weight
        states=nxt
    return hit
