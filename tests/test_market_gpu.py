import numpy as np,pandas as pd,pytest
import torch
from analysis.market_gpu import screen_batch,signal_tensor,FAMILIES
from strategies.small_account import Setup,features
from analysis.small_account_research import replay,metrics


def fixture(n=400,seed=13):
    rng=np.random.default_rng(seed);close=100*np.exp(np.cumsum(rng.normal(0,.025,n)));opening=np.r_[100,close[:-1]]*np.exp(rng.normal(0,.006,n))
    return pd.DataFrame({'ts':pd.date_range('2021-01-01',periods=n,tz='UTC',freq='D'),'open':opening,'high':np.maximum(opening,close)*1.025,'low':np.minimum(opening,close)*.975,'close':close,'volume':rng.uniform(500,2500,n)})


def test_signal_tensor_matches_production_all_families():
    f=fixture();a=torch.tensor(f[['open','high','low','close','volume']].to_numpy()[None],dtype=torch.float64)
    signals,atr,specs=signal_tensor(a,(10,20,50))
    for i,(family,p) in enumerate(specs):
        oracle=features(f,Setup(family,p));np.testing.assert_array_equal(signals[0,i].numpy(),oracle.signal.to_numpy())
        np.testing.assert_allclose(atr[0].numpy(),oracle.atr.to_numpy(),equal_nan=True,rtol=1e-12)


def test_tensor_backtest_matches_canonical_accounting():
    f=fixture();r,_=screen_batch({'TEST':f},device='cpu',periods=(10,20),holds=(5,10))
    for row in r.to_dict('records'):
        e=replay(f,Setup(row['family'],row['period'],row['hold']),symbol='TEST',begin=240,end=320)
        want=metrics(e)
        assert row['final_equity']==pytest.approx(want['final_equity'],abs=1e-8)
        assert row['max_drawdown']==pytest.approx(want['max_drawdown'],abs=1e-10)
        assert row['trades']==want['trades']
        assert row['fees']==pytest.approx(want['fees_gbp'],abs=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device absent')
def test_cuda_cpu_parity_variable_calendars_and_holdout_canary():
    frames={'A':fixture(400),'B':fixture(355,15)}
    cpu,_=screen_batch(frames,device='cpu',periods=(10,20),holds=(5,10))
    gpu,meta=screen_batch(frames,device='cuda',periods=(10,20),holds=(5,10))
    np.testing.assert_allclose(cpu[['final_equity','max_drawdown','fees']],gpu[['final_equity','max_drawdown','fees']],rtol=1e-10,atol=1e-8)
    assert meta['device']=='cuda' and meta['cases']==48 and meta['peak_allocated_bytes']>0
    for f in frames.values():
        end=int(len(f)*.8);f.loc[end:,['open','high','low','close']]*=123456789
    canary,_=screen_batch(frames,device='cuda',periods=(10,20),holds=(5,10))
    pd.testing.assert_frame_equal(gpu,canary)


def test_bad_financial_input_rejected():
    f=fixture();f.loc[20,'close']=np.inf
    with pytest.raises(ValueError):screen_batch({'A':f},device='cpu')


def test_flat_price_regression_and_dst_elapsed_holding_time():
    from pathlib import Path
    f=pd.read_csv(Path(__file__).parent/'fixtures/market_gpu/abr_prd_daily.csv',parse_dates=['ts'],float_precision='round_trip')
    frames={'FLAT':f,'DST':fixture()}
    frames['DST']['ts']=pd.date_range('2020-06-01',periods=400,tz='America/New_York',freq='D').tz_convert('UTC')
    result,_=screen_batch(frames,device='cuda' if torch.cuda.is_available() else 'cpu',periods=(10,),holds=(5,))
    for row in result.to_dict('records'):
        bars=frames[row['instrument']]
        e=replay(bars,Setup(row['family'],10,5),symbol=row['instrument'],begin=int(len(bars)*.6),end=int(len(bars)*.8))
        want=metrics(e)
        assert row['final_equity']==pytest.approx(want['final_equity'],abs=1e-8)
        assert row['trades']==want['trades']
