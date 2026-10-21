"""Small preregistered causal rule family. No confidence scores invented."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from analysis.rule_discovery import validate_bars


@dataclass(frozen=True)
class Setup:
    family: str = 'donchian'
    period: int = 20
    hold: int = 10
    stop_kind: str = 'atr'
    reward: float = 2.

    def __post_init__(self):
        if self.family not in ('donchian','ma','rsi','bollinger','volume','gap') or type(self.period) is not int or type(self.hold) is not int or self.period<2 or self.hold<1:
            raise ValueError('invalid setup')
        if self.stop_kind not in ('atr','fixed','trailing','structure') or self.reward not in (1.,1.5,2.,3.):
            raise ValueError('invalid exit')


def features(bars, setup=Setup(), *, annual=252):
    if annual not in (252,365): raise ValueError('annual must be 252 or 365')
    f=validate_bars(bars); c=f.close; n=setup.period
    delta=c.diff(); gains=delta.clip(lower=0).rolling(n).mean(); losses=(-delta.clip(upper=0)).rolling(n).mean()
    rsi=100*gains/(gains+losses).replace(0,np.nan)
    mean=c.rolling(n).mean(); sd=c.rolling(n).std()
    tr=pd.concat([f.high-f.low,(f.high-c.shift()).abs(),(f.low-c.shift()).abs()],axis=1).max(axis=1)
    f['atr']=tr.rolling(14).mean(); f['volatility']=c.pct_change().rolling(20).std()*np.sqrt(annual)
    f['regime']=np.where(c>c.rolling(50).mean(),'trend_up','range_or_down')
    signal={
        'donchian':c>f.high.shift().rolling(n).max(),
        'ma':(c>mean)&(c.shift()<=mean.shift()),
        'rsi':(rsi>30)&(rsi.shift()<=30),
        'bollinger':(c<mean-2*sd),
        'volume':(c>c.shift())&(f.volume>1.5*f.volume.shift().rolling(n).mean())&(c>mean),
        # Gap continuation proxy, not an earnings/news strategy.
        'gap':(f.open>c.shift()*1.01)&(c>f.open)&(c>mean),
    }[setup.family]
    f['signal']=(signal & f.atr.notna()).fillna(False)
    f['distance']=2*f.atr
    if setup.stop_kind=='fixed': f['distance']=c*.02
    if setup.stop_kind=='structure': f['distance']=c-f.low.rolling(n).min()
    f['confidence']=np.nan  # unavailable until calibration has independently passed OOS tests
    return f
