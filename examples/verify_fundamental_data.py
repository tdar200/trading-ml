import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.api import FundamentalAPI
from fundamentals.dictionary import feature_dictionary,RAW_DICTIONARY
from decimal import Decimal
root=Path('results/fundamentals');s=PointInTimeFundamentalStore(root/'fundamentals.sqlite',readonly=True)
a=FundamentalAPI(s).get('/fundamentals/XOM',{'as_of':'2024-06-01T00:00:00Z'})
expected={'net_income':36010000000,'ocf':55369000000,'capex':21919000000,'assets':376317000000,'eps':8.89}
rows=[]
for k,v in expected.items():
    actual=a['raw'][k];rows.append({'metric':k,'official_report_value':v,'API_value':actual,'difference':actual-v,'pass':actual==v})
v=float(Decimal('55369000000')-Decimal('21919000000'))
rows.append({'metric':'fcf_ocf_less_ppe','independent_value':v,'API_value':a['features']['fcf'],'difference':a['features']['fcf']-v,'pass':a['features']['fcf']==v})
assert all(r['pass'] for r in rows)
source='https://corporate.exxonmobil.com/news/news-releases/2024/0202_exxonmobil-announces-2023-results'
(root/'official_filing_reconciliation.json').write_text(json.dumps({'symbol':'XOM','period':'2023','source':source,'as_of':a['as_of'],'currency':'USD','checks':rows,'note':'OCF-minus-PP&E definition; not company non-GAAP FCF'},indent=2))
(root/'xom_pit_snapshot.json').write_text(json.dumps(a,indent=2,allow_nan=False))
Path('docs/fundamental-feature-dictionary.json').write_text(json.dumps({'feature_version':'fundamentals-v1','raw':RAW_DICTIONARY,'derived':feature_dictionary()},indent=2))
print(json.dumps({'observations':s.db.execute('select count(*) from observations').fetchone()[0],'documents':s.db.execute('select count(*) from raw_documents').fetchone()[0],'events':s.db.execute('select count(*) from corporate_events').fetchone()[0],'features':len(feature_dictionary()),'raw_metrics':len(RAW_DICTIONARY),'xom_coverage':a['completeness'],'source_checks':len(rows)}));s.close()
