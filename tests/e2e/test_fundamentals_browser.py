"""Real browser verifies source-grounded financial numbers and PIT date changes."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright,expect
from tests.test_fundamentals import populated_store


def test_fundamental_browser_pit_numerics_and_export(tmp_path):
    db=tmp_path/'f.sqlite';store=populated_store(db);store.close()
    out=Path('results/fundamentals/browser');out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    log=(out/'server.log').open('w');process=subprocess.Popen([sys.executable,'-m','streamlit','run','dashboard/fundamentals_app.py','--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}','--browser.gatherUsageStats=false'],stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(200):
            try:
                with urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1) as r:
                    if r.status==200:break
            except OSError:time.sleep(.1)
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
            context=browser.new_context(accept_downloads=True,viewport={'width':1440,'height':1100});context.tracing.start(screenshots=True,snapshots=True,sources=True)
            page=context.new_page();page.goto(f'http://127.0.0.1:{port}')
            field=page.get_by_label('Fundamental database');field.fill(str(db));field.press('Enter')
            field=page.get_by_label('Fundamental symbol');field.fill('TEST');field.press('Enter')
            expect(page.get_by_test_id('stMetricValue').nth(4)).to_have_text('20.00%',timeout=30000)
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            values=page.get_by_test_id('stMetricValue').all_text_contents()
            assert values[0]=='Unvalidated' and values[5]=='40.00%' and values[6]=='12.00'
            with page.expect_download() as d:page.get_by_role('button',name='Download PIT snapshot').click()
            d.value.save_as(out/'snapshot.json');snapshot=json.loads((out/'snapshot.json').read_text())
            assert snapshot['features']['fcf']==12 and abs(snapshot['features']['revenue_yoy']-.2)<1e-12
            assert snapshot['period_end']=='2023-12-31'
            page.screenshot(path=str(out/'after_release.png'),full_page=True)
            field=page.get_by_label('Information available as of (UTC)');field.fill('2024-04-15T00:00:00Z');field.press('Enter')
            expect(page.get_by_test_id('stMetricValue').nth(4)).to_have_text('11.11%',timeout=30000)
            expect(page.get_by_test_id('stMetricValue').nth(6)).to_have_text('10.00')
            page.screenshot(path=str(out/'before_release.png'),full_page=True)
            context.tracing.stop(path=str(out/'trace.zip'));browser.close()
    finally:process.terminate();process.wait(timeout=10);log.close()
