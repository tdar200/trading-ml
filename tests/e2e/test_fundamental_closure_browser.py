"""Exercise new TTM, quarantine and evidence-bound news through Chrome."""
import json,socket,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
from dataclasses import replace,asdict
from playwright.sync_api import sync_playwright,expect
from fundamentals.store import PointInTimeFundamentalStore,Fact
from fundamentals.catalysts import ingest_disclosures
from tests.test_fundamental_closure import quarter_facts,model_and_panel
from fundamentals.confidence import fit_confidence


def test_closure_browser_ttm_quarantine_and_news(tmp_path):
    db=tmp_path/'closure.sqlite';s=PointInTimeFundamentalStore(db);facts=quarter_facts()
    s.ingest_validated(facts+[replace(facts[0],period_end='2030-01-01')])
    for year,value in [(2023,90),(2024,100)]:
        s.ingest([Fact('GOLD','revenue',value,f'{year}-12-31',f'{year+1}-02-10T07:00:00Z',f'{year+1}-02-10T07:00:00Z','fixture',str(year),'GBP','GBP',f'{year}-01-01')])
    a,panel=model_and_panel();a=replace(a,features=('revenue_yoy',));panel=panel.rename(columns={'x':'revenue_yoy'})
    calibration=fit_confidence(a,panel,calibrated_at='2024-03-01T00:00:00Z')
    bundle=tmp_path/'model.json';bundle.write_text(json.dumps({'model':asdict(a),'calibration':asdict(calibration)}))
    ingest_disclosures(s,symbol='GOLD',document='We raised guidance.',source='fixture://news',published_at='2025-02-10T07:00:00Z',observed_at='2025-02-10T07:01:00Z')
    s.close()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    out=Path('results/fundamental_closure_20260930/browser');out.mkdir(parents=True,exist_ok=True)
    with (out/'server.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','streamlit','run','dashboard/fundamentals_app.py','--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}','--browser.gatherUsageStats=false'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                try:
                    if urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1).status==200:break
                except OSError:time.sleep(.1)
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
                context=browser.new_context(accept_downloads=True);page=context.new_page();page.goto(f'http://127.0.0.1:{port}')
                for label,value in [('Fundamental database',str(db)),('Fundamental symbol','GOLD'),('Information available as of (UTC)','2025-02-11T00:00:00Z'),('Research model bundle (optional)',str(bundle))]:
                    field=page.get_by_label(label);field.fill(value);field.press('Enter')
                expect(page.get_by_text('Trailing twelve months',exact=True)).to_be_visible()
                expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
                text=page.get_by_test_id('stJson').all_text_contents()
                assert any('100' in t and '15' in t and 'AVAILABLE' in t for t in text)
                assert any('QUARANTINED_RECORDS' in t for t in text)
                assert any('0.75' in t and 'calibration_id' in t for t in text)
                assert any('guidance_up' in t and 'We raised guidance' in t for t in text)
                with page.expect_download() as download:page.get_by_role('button',name='Download PIT snapshot').click()
                download.value.save_as(out/'ttm_snapshot.json')
                snapshot=json.loads((out/'ttm_snapshot.json').read_text())
                assert snapshot['ttm']['values']['revenue']==100 and snapshot['ttm']['values']['fcf']==15
                page.screenshot(path=str(out/'closure.png'),full_page=True)
                field=page.get_by_label('Information available as of (UTC)');field.fill('2025-02-10T06:59:59Z');field.press('Enter')
                expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
                expect(page.get_by_test_id('stJson').filter(has_text='guidance_up')).to_have_count(0)
                with page.expect_download() as download:page.get_by_role('button',name='Download PIT snapshot').click()
                download.value.save_as(out/'ttm_before_release.json')
                assert json.loads((out/'ttm_before_release.json').read_text())['ttm']['values']['fcf'] is None
                browser.close()
        finally:proc.terminate();proc.wait(timeout=20)
