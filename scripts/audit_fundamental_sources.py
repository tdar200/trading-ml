"""Reconcile cached official SEC data to independently transcribed issuer reports.

Audit-only quarantine is explicit; the production importer remains fail-closed.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.providers import sec_facts
ROOT=Path('results/fundamental_audit_20260930')
SOURCES={
'MSFT':'https://www.microsoft.com/en-us/Investor/earnings/FY-2024-Q4/income-statements',
'JPM':'https://jpmorganchaseco.gcs-web.com/node/733411/html',
'WMT':'https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-25-000010/earningsreleasefy25q4.htm',
'JNJ':'https://www.investor.jnj.com/investor-news/news-details/2025/Johnson--Johnson-Reports-Q4-and-Full-Year-2024-Results/default.aspx',
'CAT':'https://www.caterpillar.com/content/dam/caterpillarDotCom/releases/4q24/4q24-caterpillar-inc-financial-results.pdf',
'SONY':'https://www.sec.gov/Archives/edgar/data/313838/000119312521168464/d17497d6k.htm',
'XOM':'https://corporate.exxonmobil.com/news/news-releases/2024/0202_exxonmobil-announces-2023-results',
'BA':'https://investors.boeing.com/investors/news/press-release-details/2025/Boeing-Reports-Fourth-Quarter-Results/default.aspx'}
# Currency amounts are USD millions in official tables, EPS is USD/share.
EXPECTED={
'SONY':('2021-03-31',{'revenue':8999360,'net_income':1171776},'2020-04-01'),
'MSFT':('2024-06-30',{'revenue':245122,'net_income':88136,'eps':11.8},'2023-07-01'),
'JPM':('2024-12-31',{'revenue':177556,'net_income':58471,'eps':19.75},'2024-01-01'),
'WMT':('2025-01-31',{'revenue':680985,'net_income':19436,'eps':2.41,'diluted_shares':8081},'2024-02-01'),
'JNJ':('2024-12-29',{'revenue':88821,'net_income':14066,'eps':5.79,'diluted_shares':2429.4},'2024-01-01'),
'CAT':('2024-12-31',{'revenue':64809,'net_income':10788,'eps':22.05},'2024-01-01'),
'XOM':('2023-12-31',{'net_income':36010,'ocf':55369,'capex':21919,'eps':8.89,'assets':376317},'2023-01-01'),
'BA':('2024-12-31',{'revenue':66517,'net_income':-11817,'eps':-18.36,'ocf':-12080,'capex':2230},'2024-01-01')}
if __name__=='__main__':
    store=PointInTimeFundamentalStore(ROOT/'verified_companies.sqlite');rejected=[];checks=[];coverage=[]
    for symbol in EXPECTED:
        raw=ROOT/'raw';facts=sec_facts(json.loads((raw/f'{symbol}_facts.json').read_text()),json.loads((raw/f'{symbol}_submissions.json').read_text()),symbol);valid=[]
        for f in facts:
            try:f.normalized();valid.append(f)
            except ValueError as e:rejected.append({'symbol':symbol,'record':f.__dict__,'reason':str(e)})
        store.ingest(valid)
        for label in ('facts','submissions'):store.archive(f'SEC/{symbol}/{label}',(raw/f'{symbol}_{label}.json').read_bytes())
        rows=store.as_of(symbol,'2021-07-01T00:00:00Z' if symbol=='SONY' else '2025-06-01T00:00:00Z',kind='financial')
        coverage.append({'symbol':symbol,'records':len(rows),'last_period':max((r['period_end'] for r in rows),default=None),'currencies':sorted(set(r['currency'] for r in rows))})
        if symbol not in EXPECTED:continue
        end,expected,start=EXPECTED[symbol]
        for metric,value in expected.items():
            matches=[r for r in rows if r['period_end']==end and r['metric']==metric and r['period_start'] in ('',start)]
            actual=matches[0]['value'] if len(matches)==1 else None;want=value if metric=='eps' else value*1000000
            checks.append({'symbol':symbol,'metric':metric,'period_start':start,'period_end':end,'currency':('JPY' if symbol=='SONY' else 'USD') if metric!='diluted_shares' else '',
                'source_value':want,'stored':actual,'difference':actual-want if actual is not None else None,'status':'PASS' if actual==want else 'FAIL',
                'source':SOURCES[symbol],'available_at':matches[0]['available_at'] if matches else None,'revision':matches[0]['revision'] if matches else None})
    from fundamentals.features import quarters
    for symbol,end,values in [('MSFT','2024-06-30',{'revenue':64727,'net_income':22036}),('WMT','2025-01-31',{'revenue':180554,'net_income':5254}),('JNJ','2024-12-29',{'revenue':22520,'net_income':3431}),('BA','2024-12-31',{'revenue':15242,'net_income':-3865})]:
        for metric,value in values.items():
            q=quarters(store.as_of(symbol,'2025-06-01T00:00:00Z',kind='financial'),metric).get(end)
            actual=q['value'] if q else None;expected=value*1000000
            checks.append({'symbol':symbol,'metric':metric+'_quarter','period_end':end,'currency':'USD','source_value':expected,'stored':actual,'difference':actual-expected if actual is not None else None,'status':'PASS' if actual==expected else 'FAIL','source':SOURCES[symbol],'reconstruction':'Direct quarter or additive YTD difference'})
    pd.DataFrame(checks).to_csv(ROOT/'official_reconciliation.csv',index=False)
    (ROOT/'provider_quarantine.json').write_text(json.dumps(rejected,indent=2));(ROOT/'coverage.json').write_text(json.dumps(coverage,indent=2))
    print(pd.DataFrame(checks)[['symbol','metric','source_value','stored','difference','status']].to_string(index=False));store.close()
