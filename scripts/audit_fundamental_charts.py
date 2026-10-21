"""Inspect the actual Arrow datasets sent by Streamlit to its financial chart."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from tests.test_fundamentals import populated_store
import pyarrow as pa
root=Path('results/fundamental_audit_20260930');db=root/'chart_fixture_v2.sqlite'
s=populated_store(db);s.close()
a=AppTest.from_file('dashboard/fundamentals_app.py',default_timeout=20).run()
a.text_input(key='fund_db').set_value(str(db));a.text_input(key='fund_symbol').set_value('TEST');a.run()
checks=[]
for at,expected in [('2024-06-01T00:00:00Z',[80.,90.,100.,120.]),('2024-04-15T00:00:00Z',[80.,90.,100.])]:
    a.text_input(key='fund_asof').set_value(at);a.run();assert not a.exception and not a.error
    charts=a.get('vega_lite_chart');assert len(charts)==1
    frames=[pa.ipc.open_stream(d.data.data).read_all().to_pandas() for d in charts[0].proto.datasets]
    frame=frames[0];assert frame['value'].tolist()==expected
    frame.to_csv(root/('chart_'+at[:10]+'.csv'),index=False)
    checks.append({'as_of':at,'expected':expected,'actual':frame['value'].tolist(),'PASS':True})
(root/'chart_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
