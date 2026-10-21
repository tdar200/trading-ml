"""Independent deterministic, long-only, cash-account risk boundary."""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = .005
    max_daily_loss: float = .02
    max_weekly_loss: float = .04
    max_drawdown: float = .10
    max_exposure: float = .25
    max_correlated_exposure: float = .25
    max_leverage: float = 1.
    max_positions: int = 1
    max_spread_bps: float = 20.
    max_slippage_bps: float = 20.
    max_data_age_seconds: float = 90.
    max_clock_skew_seconds: float = 5.
    min_notional: float = 1.
    quantity_step: float = .000001

    def __post_init__(self):
        if any(not math.isfinite(getattr(self,f.name)) or getattr(self,f.name)<=0 for f in fields(self)):
            raise ValueError('limits must be finite and positive')
        if any(getattr(self,name)>1 for name in ('max_daily_loss','max_weekly_loss','max_drawdown','max_correlated_exposure')):
            raise ValueError('fractional limits cannot exceed one')
        if self.risk_per_trade>.01 or self.max_exposure>1 or self.max_leverage>1 or self.max_positions!=1:
            raise ValueError('small-account implementation supports <=1% risk, no leverage, one position')


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    cash: float
    price: float
    stop: float
    fee_rate: float
    spread_bps: float
    slippage_bps: float
    daily_return: float = 0.
    weekly_return: float = 0.
    drawdown: float = 0.
    gross_exposure: float = 0.
    correlated_exposure: float = 0.
    positions: int = 0
    data_age_seconds: float = 0.
    clock_skew_seconds: float = 0.
    healthy: bool = True
    consistent: bool = True
    database_ok: bool = True
    order_status_known: bool = True
    qualified: bool = False
    research_shadow: bool = False
    edge_degraded: bool = False
    losing_streak: int = 0
    duplicate: bool = False
    kill: bool = False


class RiskEngine:
    def __init__(self, limits=RiskLimits()): self.limits=limits

    def approve_reduction(self, quantity, position_quantity, price):
        """Reduce-only exits cannot add exposure; entry halts must not trap a position.

        Feed/account/status health is validated by the execution event boundary.
        Unknown quantities or unavailable risk engine fail closed.
        """
        if not all(math.isfinite(x) and x > 0 for x in (quantity, position_quantity, price)):
            return False
        return quantity <= position_quantity

    def approve(self, s: RiskSnapshot):
        l=self.limits
        values=[getattr(s,f.name) for f in fields(s) if isinstance(getattr(s,f.name),(float,int))]
        if not all(math.isfinite(v) for v in values): return 0., 'NO_TRADE: nonfinite account/market state'
        if os.getenv('TRADING_MODE','PAPER').upper()!='PAPER':
            return 0., 'NO_TRADE: live execution unavailable; qualification required'
        kill_path=Path(os.getenv('STOP_TRADING_FILE','results/small_account/STOP_TRADING'))
        if s.kill or kill_path.exists() or os.getenv('STOP_TRADING','false').lower() in ('true','1','yes'):
            return 0., 'NO_TRADE: kill switch'
        for good,reason in ((s.healthy,'feed/broker unavailable'),(s.consistent,'account inconsistent'),
                            (s.database_ok,'database unavailable'),(s.order_status_known,'unknown order status'),
                            (s.qualified or s.research_shadow,'no validated edge')):
            if not good: return 0., 'NO_TRADE: '+reason
        if s.duplicate or s.edge_degraded: return 0., 'NO_TRADE: duplicate or edge degradation'
        if s.equity<=0 or s.cash<0 or s.cash>s.equity+1e-8 or s.price<=0 or not 0<s.stop<s.price:
            return 0., 'NO_TRADE: invalid balance/stop'
        if min(s.fee_rate,s.spread_bps,s.slippage_bps,s.gross_exposure,s.correlated_exposure,s.positions,s.drawdown)<0:
            return 0., 'NO_TRADE: negative risk state'
        if s.fee_rate>=1 or s.data_age_seconds< -l.max_clock_skew_seconds:
            return 0., 'NO_TRADE: invalid costs/time'
        if s.data_age_seconds>l.max_data_age_seconds or abs(s.clock_skew_seconds)>l.max_clock_skew_seconds:
            return 0., 'NO_TRADE: stale data or clock skew'
        if s.spread_bps>l.max_spread_bps or s.slippage_bps>l.max_slippage_bps:
            return 0., 'NO_TRADE: execution cost limit'
        if s.daily_return<=-l.max_daily_loss or s.weekly_return<=-l.max_weekly_loss or s.drawdown>=l.max_drawdown:
            return 0., 'NO_TRADE: loss limit'
        if s.positions>=l.max_positions: return 0., 'NO_TRADE: position limit'
        # Budget entry/exit fees and adverse exit spread/slippage as well as stop distance.
        exit_price=s.stop*(1-(s.spread_bps/2+s.slippage_bps)/10000)
        loss=s.price*(1+s.fee_rate)-exit_price*(1-s.fee_rate)
        scale=.5 if s.losing_streak>=3 or s.weekly_return<=-l.max_weekly_loss/2 else 1.
        cap=min(l.max_exposure-s.gross_exposure,l.max_correlated_exposure-s.correlated_exposure,l.max_leverage-s.gross_exposure)
        qty=min(s.equity*l.risk_per_trade*scale/loss,s.cash/(s.price*(1+s.fee_rate)),s.equity*max(0,cap)/s.price)
        qty=math.floor(qty/l.quantity_step)*l.quantity_step
        if qty*s.price<l.min_notional: return 0., 'NO_TRADE: below economical minimum'
        return qty, 'APPROVED_SHADOW' if s.research_shadow else 'APPROVED_PAPER'


def sizing_comparison(equity, price, stop_distance, volatility, p=.45, reward=2., cost_r=.1):
    """Analytical sizing only; actual execution always passes RiskEngine."""
    if min(equity,price,stop_distance,volatility)<=0: raise ValueError('invalid sizing inputs')
    a,b=reward-cost_r,1+cost_r
    k=max(0,(p*a-(1-p)*b)/(a*b)) if a>0 else 0
    return {'fixed_fractional_notional':equity*.005*price/stop_distance,
            'atr_notional':equity*.005*price/(2*stop_distance),
            'vol_target_notional':equity*min(.25,.05/volatility),
            'quarter_kelly_risk':min(.005,k/4), 'full_kelly_analytical_only':k,
            'inverse_vol_single_asset_weight':1., 'hard_notional_cap':equity*.25}
