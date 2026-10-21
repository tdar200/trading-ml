"""Real isolated Chrome: assert financial values, download and reconcile exports."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import pytest
from playwright.sync_api import sync_playwright,expect
from tests.test_financial_audit import golden_run,bar,signal
from tests.financial_oracle import D,Ledger,size


def test_golden_database_dashboard_exports_and_open_position(tmp_path):
    artifacts=Path('results/qa_audit_20260930/browser'); artifacts.mkdir(parents=True,exist_ok=True)
    engine,golden=golden_run(tmp_path/'paper.sqlite')
    with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    log=(artifacts/'server.log').open('w')
    process=subprocess.Popen([sys.executable,'-m','streamlit','run','dashboard/small_account_app.py',
        '--server.address=127.0.0.1',f'--server.port={port}','--server.headless=true','--browser.gatherUsageStats=false'],stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'TRADING_MODE':'PAPER'})
    try:
        for _ in range(150):
            try:
                with urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1) as r:
                    if r.status==200: break
            except OSError: time.sleep(.1)
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
            context=browser.new_context(accept_downloads=True,viewport={'width':1440,'height':1100})
            context.tracing.start(screenshots=True,snapshots=True,sources=True)
            page=context.new_page(); page.goto(f'http://127.0.0.1:{port}')
            root=page.get_by_label('Results directory'); root.fill(str(tmp_path)); root.press('Enter')
            expect(page.get_by_test_id('stMetricValue').first).to_have_text('£112.41',timeout=30000)
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            expected=['£112.41','£112.41','£12.41','0.53%','£112.41','£12.41','£0.00','£1.43']
            assert page.get_by_test_id('stMetricValue').all_text_contents()==expected
            def download(label,name):
                with page.expect_download() as pending: page.get_by_role('button',name=label,exact=True).click()
                path=artifacts/name; pending.value.save_as(path); return path
            account=json.loads(download('Download account JSON','account_closed.json').read_text())
            assert abs(account['equity']-golden['expected']['final_equity'])<1e-8
            journal=pd.read_csv(download('Download journal CSV','journal.csv'))
            assert abs(journal[journal.event=='EXIT'].pnl.sum()-golden['expected']['net_pnl'])<1e-8
            equity=pd.read_csv(download('Download equity CSV','equity.csv'))
            assert equity.equity.tolist()==pytest.approx(golden['equity'][1:],abs=1e-8)
            page.screenshot(path=str(artifacts/'closed.png'),full_page=True)
            reconciliation=[]
            for key,oracle in [('cash',golden['expected']['final_equity']),('equity',golden['expected']['final_equity']),
                               ('realized_pnl',golden['expected']['net_pnl']),('unrealized_pnl',0.),('fees',golden['expected']['fees_gbp'])]:
                reconciliation.append(dict(metric=key,oracle=oracle,read_model=account[key],export=account[key],ui=f'£{account[key]:.2f}',result='PASS'))
            # Submit a new simulated entry through the actual risk/execution path while browser stays open.
            cash=D(golden['trades'][-1]['final_cash']); q=size(cash,cash,'100.04','98.04','.001','.0004')
            ledger=Ledger(cash); ledger.buy(q,'100.04',q*D('100.04')*D('.001'))
            engine.step(bar(365,100,101,99,101),signal(364))
            page.reload(); root=page.get_by_label('Results directory'); root.fill(str(tmp_path)); root.press('Enter')
            expect(page.get_by_test_id('stMetricValue').first).to_have_text(f'£{ledger.mark(101):.2f}',timeout=30000)
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            values=page.get_by_test_id('stMetricValue').all_text_contents()
            assert values[1]==f'£{ledger.cash:.2f}' and values[6]==f'£{ledger.unrealized(101):.2f}'
            account=json.loads(download('Download account JSON','account_open.json').read_text())
            assert account['unrealized_pnl']==pytest.approx(float(ledger.unrealized(101)),abs=1e-8)
            page.screenshot(path=str(artifacts/'open.png'),full_page=True)
            # Exit at a known next mark, fees both sides; reconcile again.
            engine.step(bar(366,102,102,102,102),close_final=True)
            price=D(102)*D('.9996'); ledger.sell(q,price,q*price*D('.001'))
            page.reload(); root=page.get_by_label('Results directory'); root.fill(str(tmp_path)); root.press('Enter')
            expect(page.get_by_test_id('stMetricValue').first).to_have_text(f'£{ledger.cash:.2f}',timeout=30000)
            expect(page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
            assert page.get_by_test_id('stMetricValue').nth(6).inner_text()=='£0.00'
            pd.DataFrame(reconciliation).to_csv(artifacts/'reconciliation.csv',index=False)
            context.tracing.stop(path=str(artifacts/'trace.zip')); browser.close()
    finally:
        process.terminate(); process.wait(timeout=10); log.close(); engine.db.close()
