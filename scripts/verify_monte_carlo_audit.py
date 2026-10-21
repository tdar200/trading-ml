"""Reproduce all archived scenario summaries and cross-check exact IID tails."""
import sys,json,math,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from analysis.small_account_mc import binary_paths
from tests.financial_oracle import hit_probability,terminal_probability
root=Path('results/qa_audit_20260930'); source=json.loads(Path('results/small_account/monte_carlo.json').read_text())
rows=[]; start=time.perf_counter(); paths=0
for group in ('scenarios','extreme_60trades','cost_sensitivity'):
    for i,s in enumerate(source[group]):
        labels={'30_days':21,'90_days':63,'6_months':126,'12_months':252,'30_days_60_trades':60}
        horizons={labels[label]:label for label in s['horizons']}
        rerun=binary_paths(s['p'],s['reward_risk'],s['risk'],cost_r=s['cost_r'],paths=100000,seed=source['seed'],horizons=horizons,stress=s['stress'])
        assert rerun['horizons']==s['horizons'],(group,i,'reproduction')
        paths+=100000
        for key in ('expectancy_r','variance_r','kelly_analytic','expected_log_growth'):
            assert rerun[key]==s[key]
        if not s['stress']:
            first=min(horizons); h=s['horizons'][horizons[first]]
            for target in (125,150,200,250,500,1000):
                expected=hit_probability(s['p'],s['reward_risk'],s['risk'],first,target,s['cost_r'])
                observed=h[f'hit_{target}']; bound=6*math.sqrt(expected*(1-expected)/100000)+1/100000
                rows.append(dict(group=group,scenario=i,p=s['p'],risk=s['risk'],target=target,expected=expected,observed=observed,difference=observed-expected,tolerance=bound,result='PASS' if abs(observed-expected)<=bound else 'FAIL'))
                exact=terminal_probability(s['p'],s['reward_risk'],s['risk'],first,s['cost_r'],target/100)
                assert abs(h[f'end_above_{target}']-exact)<=6*math.sqrt(exact*(1-exact)/100000)+1/100000
pd.DataFrame(rows).to_csv(root/'scenario_oracle.csv',index=False)
assert all(x['result']=='PASS' for x in rows)
summary=dict(reproduced_paths=paths,scenarios=paths//100000,independent_hit_comparisons=len(rows),seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(root/'mc_verification.json').write_text(json.dumps(summary,indent=2)); print(summary)
