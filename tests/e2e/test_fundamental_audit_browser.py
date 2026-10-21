"""Independent golden company reconciled through actual HTTP and Chrome UI."""
import json,socket,subprocess,sys,time,threading
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright,expect
from fundamentals.store import Fact,PointInTimeFundamentalStore
from fundamentals.api import serve
from tests.fundamental_oracle import calculate


def test_golden_company_database_http_browser_oracle(tmp_path):
    golden=json.loads(Path('tests/fixtures/fundamental_companies.json').read_text())['companies']['golden']
    db=tmp_path/'golden.sqlite';store=PointInTimeFundamentalStore(db)
    flows={'revenue','gross_profit','operating_income','net_income','ocf','capex','eps','depreciation','interest_expense','pretax_income','tax_expense','diluted_shares'}
    for year,values in [(2022,golden['prior']),(2023,golden['current'])]:
        for metric,value in values.items():
            currency='GBP';units='GBP millions'
            if metric=='eps':units='GBP/shares'
            if metric=='diluted_shares':currency='';units='shares';value*=1000000
            store.ingest([Fact('GOLD',metric,value,f'{year}-12-31',f'{year+1}-05-10T07:00:00Z',f'{year+1}-05-10T07:00:00Z','golden',str(year),currency,units,
                period_start=f'{year}-01-01' if metric in flows else '',metadata={'share_basis':'constant-no-splits'})])
    assert store.db.execute("SELECT value FROM observations WHERE metric='revenue' AND period_end='2023-12-31'").fetchone()[0]==120000000
    store.close();api=serve(db,port=0);thread=threading.Thread(target=api.serve_forever,daemon=True);thread.start()
    out=Path('results/fundamental_audit_20260930/browser');out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    log=(out/'server.log').open('w');process=subprocess.Popen([sys.executable,'-m','streamlit','run','dashboard/fundamentals_app.py','--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}','--browser.gatherUsageStats=false'],stdout=log,stderr=subprocess.STDOUT)
    try:
        url=f'http://127.0.0.1:{api.server_port}/fundamentals/GOLD?as_of=2024-06-01T00:00:00Z'
        data=json.load(urlopen(url));oracle=calculate(golden['current'],golden['prior'])
        for metric in ('revenue_yoy','gross_margin','roe','roic','debt_equity'):
            assert abs(data['features'][metric]-oracle[metric])<1e-12
        assert data['features']['fcf']==oracle['fcf']*1000000
        for _ in range(200):
            try:
                if urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1).status==200:break
            except OSError:time.sleep(.1)
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True);context=browser.new_context(accept_downloads=True)
            context.tracing.start(screenshots=True,snapshots=True,sources=True);page=context.new_page();page.goto(f'http://127.0.0.1:{port}')
            field=page.get_by_label('Fundamental database');field.fill(str(db));field.press('Enter')
            field=page.get_by_label('Fundamental symbol');field.fill('GOLD');field.press('Enter')
            expect(page.get_by_test_id('stMetricValue').nth(6)).to_have_text('20,000,000.00',timeout=30000)
            expect(page.get_by_test_id('stMetricValue').nth(4)).to_have_text('20.00%')
            expect(page.get_by_test_id('stMetricValue').nth(5)).to_have_text('50.00%')
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            with page.expect_download() as d:page.get_by_role('button',name='Download PIT snapshot').click()
            d.value.save_as(out/'golden_export.json');export=json.loads((out/'golden_export.json').read_text());assert export==data
            page.screenshot(path=str(out/'golden_company.png'),full_page=True)
            field=page.get_by_label('Information available as of (UTC)');field.fill('2024-05-10T06:59:00Z');field.press('Enter')
            expect(page.get_by_test_id('stMetricValue').nth(4)).to_have_text('Missing') # prior year has no prior revenue fixture
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            context.tracing.stop(path=str(out/'trace.zip'));browser.close()
        rows=[{'metric':k,'oracle':oracle[k]*(1000000 if k=='fcf' else 1),'API':data['features'][k],'UI_export':export['features'][k],'PASS':True} for k in ('revenue_yoy','fcf','gross_margin','roe','roic','debt_equity')]
        (out/'reconciliation.json').write_text(json.dumps(rows,indent=2))
    finally:
        process.terminate();process.wait(timeout=20);log.close();api.shutdown();api.server_close();thread.join(timeout=5)
