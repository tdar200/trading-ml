"""Transparent small-account break-even calculations, not estimated alpha."""
import math


def minimum_economic_notional(gross_edge_bps, proportional_roundtrip_bps, fixed_roundtrip_fee, *, max_cost_fraction=.5):
    if not all(math.isfinite(x) for x in (gross_edge_bps,proportional_roundtrip_bps,fixed_roundtrip_fee,max_cost_fraction)):
        raise ValueError('finite inputs required')
    if gross_edge_bps<=0 or min(proportional_roundtrip_bps,fixed_roundtrip_fee)<0 or not 0<max_cost_fraction<=1:
        raise ValueError('invalid costs/edge budget')
    allowance=(gross_edge_bps*max_cost_fraction-proportional_roundtrip_bps)/10000
    return fixed_roundtrip_fee/allowance if allowance>0 else math.inf
