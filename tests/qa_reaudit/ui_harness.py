"""TRACK F (ui) harness: loopback API server in a thread, strict HTTP client, Streamlit subprocess, Playwright helpers
and a websocket/protobuf tap that exposes the exact chart/dataframe payloads Streamlit sends to the browser.
Ports are confined to 8800-8899 (TRACK F prefers 8850-8899)."""
from __future__ import annotations
import http.client as _hc
import io
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

REPO = Path('/home/tahseen-dar/Projects/trading-ml')
OUT = REPO / 'results' / 'qa_reaudit_20261001' / 'ui'
OUT.mkdir(parents=True, exist_ok=True)
PY = str(REPO / '.venv' / 'bin' / 'python')
_PORT_LOCK = threading.Lock()
_USED = []


def free_port(lo=8850, hi=8899):
    """A currently-free loopback port in lo..hi (bind-probed). Random start so parallel pytest processes rarely collide."""
    with _PORT_LOCK:
        ports = list(range(lo, hi + 1))
        random.Random(os.getpid() ^ int(time.time() * 1000) % 100000).shuffle(ports)
        recent = list(_USED)[-6:]
        for p in ports:
            if p in recent:
                continue
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(('127.0.0.1', p))
                except OSError:
                    continue
            _USED.append(p)
            return p
    raise RuntimeError('no free port in 8850-8899')


def strict_loads(body):
    def bad(token):
        raise ValueError('non-standard JSON constant ' + token)
    return json.loads(body, parse_constant=bad)


@contextmanager
def api_server(db, score=None):
    from fundamentals.api import serve
    for attempt in range(8):
        port = free_port()
        try:
            server = serve(str(db), port=port, score=score)
            break
        except OSError:
            if attempt == 7:
                raise
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


def http(port, path, method='GET', headers=None, body=None, timeout=15):
    """Raw HTTP/1.1 request (no client-side normalisation of the path). Returns (status, headers dict, bytes)."""
    c = _hc.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        c.putrequest(method, path, skip_host=bool(headers and 'Host' in headers), skip_accept_encoding=True)
        for k, v in (headers or {}).items():
            c.putheader(k, v)
        c.endheaders(body)
        r = c.getresponse()
        data = r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data
    finally:
        c.close()


def get_json(port, path, expect=None):
    st, h, b = http(port, path)
    if expect is not None:
        assert st == expect, (path, st, b[:300])
    return st, h, strict_loads(b)


# ------------------------------------------------------------------ Streamlit subprocess
@contextmanager
def streamlit_app(tag='ui', cwd=None):
    """Launch dashboard/fundamentals_app.py. cwd defaults to an empty temp dir so the page's relative default DB path
    (results/fundamentals/fundamentals.sqlite) does NOT exist and the real DB is never opened."""
    port = free_port()
    work = cwd or str((OUT / 'work').resolve())
    Path(work).mkdir(parents=True, exist_ok=True)
    log = (OUT / f'streamlit_{tag}_{port}.log').open('w')
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': ''}
    proc = subprocess.Popen([PY, '-m', 'streamlit', 'run', str(REPO / 'dashboard' / 'fundamentals_app.py'), '--server.headless=true',
                             '--server.address=127.0.0.1', f'--server.port={port}', '--browser.gatherUsageStats=false',
                             '--server.fileWatcherType=none'], stdout=log, stderr=subprocess.STDOUT, cwd=work, env=env)
    try:
        for _ in range(300):
            try:
                with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError('streamlit did not start')
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()


# ------------------------------------------------------------------ websocket tap
class Tap:
    """Collects every ForwardMsg Streamlit sends and groups them per script run (a run starts at new_session)."""

    def __init__(self, page):
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
        self.F = ForwardMsg
        self.runs = []
        self.cache = {}
        self._lock = threading.Lock()
        page.on('websocket', self._on_ws)

    def _on_ws(self, ws):
        ws.on('framereceived', self._on_frame)

    def _on_frame(self, payload):
        if isinstance(payload, str):
            return
        try:
            m = self.F.FromString(payload)
        except Exception:
            return
        with self._lock:
            kind = m.WhichOneof('type')
            # Streamlit's ForwardMsg cache: a big element already sent in an earlier run is replaced by a 'ref_hash'
            # placeholder (the browser keeps the content). Resolve it exactly like the frontend: cached content + new metadata.
            if kind == 'ref_hash':
                orig = self.cache.get(m.ref_hash)
                if orig is not None:
                    resolved = self.F()
                    resolved.CopyFrom(orig)
                    resolved.metadata.CopyFrom(m.metadata)
                    m = resolved
                    kind = m.WhichOneof('type')
            elif m.hash:
                self.cache[m.hash] = m
            if kind == 'new_session' or not self.runs:
                self.runs.append([])
            self.runs[-1].append(m)

    def last_run(self):
        with self._lock:
            return list(self.runs[-1]) if self.runs else []

    def finished(self):
        return any(m.WhichOneof('type') == 'script_finished' for m in self.last_run())

    def sync(self, page, timeout_ms=30000):
        """Pump Playwright events until the latest script run has fully arrived (DOM idle can precede tap delivery under load)."""
        waited = 0
        while not self.finished() and waited < timeout_ms:
            page.wait_for_timeout(100)
            waited += 100
        page.wait_for_timeout(50)

    def elements(self, which=None):
        """(delta_path, element_kind, element) of the latest run, in emission order, last write per path wins."""
        latest = {}
        order = []
        for m in self.last_run():
            if m.WhichOneof('type') != 'delta':
                continue
            d = m.delta
            if d.WhichOneof('type') != 'new_element':
                continue
            path = tuple(m.metadata.delta_path)
            kind = d.new_element.WhichOneof('type')
            if path not in latest:
                order.append(path)
            latest[path] = (kind, d.new_element)
        res = [(p, *latest[p]) for p in order]
        return [r for r in res if which is None or r[1] == which]


def arrow_to_frame(arrow_bytes):
    import pyarrow as pa
    reader = pa.ipc.open_stream(pa.BufferReader(arrow_bytes))
    return reader.read_all().to_pandas()


def vega_chart_data(element):
    """Return (spec dict, DataFrame list) for a vega_lite_chart / arrow_vega_lite_chart element."""
    vl = element.arrow_vega_lite_chart if element.WhichOneof('type') == 'arrow_vega_lite_chart' else element.vega_lite_chart
    spec = json.loads(vl.spec)
    frames = []
    if vl.HasField('data') and vl.data.data:
        frames.append(arrow_to_frame(vl.data.data))
    for ds in vl.datasets:
        frames.append(arrow_to_frame(ds.data.data))
    # inline datasets inside the spec
    inline = spec.get('datasets', {})
    return spec, frames, inline


# ------------------------------------------------------------------ Playwright helpers
def wait_idle(page, timeout=45000):
    page.wait_for_function("""() => { const a = document.querySelector('[data-testid="stApp"]');
        return a && a.getAttribute('data-test-script-state') === 'notRunning'; }""", timeout=timeout)
    page.wait_for_timeout(250)
    # a rerun may have been queued by the widget we just changed: confirm idle twice
    page.wait_for_function("""() => { const a = document.querySelector('[data-testid="stApp"]');
        return a && a.getAttribute('data-test-script-state') === 'notRunning'; }""", timeout=timeout)


def set_text(page, label, value):
    field = page.get_by_label(label, exact=True)
    field.fill(value)
    field.press('Enter')
    page.wait_for_timeout(300)
    wait_idle(page)


def metrics(page):
    labels = page.get_by_test_id('stMetricLabel').all_text_contents()
    values = page.get_by_test_id('stMetricValue').all_text_contents()
    return dict(zip(labels, values))


def alerts(page):
    return [t.strip() for t in page.get_by_test_id('stAlert').all_text_contents()]


def body_text(page):
    return page.locator('[data-testid="stApp"]').inner_text()


# ------------------------------------------------------------------ high level UI session
def _json_of(el):
    return json.loads(el.json.body)


class Ui:
    """Drives dashboard/fundamentals_app.py; reads the DOM (what the user sees) and the transport (what Streamlit sent)."""

    def __init__(self, browser, port, viewport=(1440, 1500)):
        self.port = port
        self.ctx = browser.new_context(viewport={'width': viewport[0], 'height': viewport[1]}, accept_downloads=True)
        self.page = self.ctx.new_page()
        self.dialogs = []
        self.page.on('dialog', lambda d: (self.dialogs.append(d.message), d.dismiss()))
        self.console = []
        self.page.on('console', lambda m: self.console.append((m.type, m.text)))
        self.tap = Tap(self.page)
        self.page.goto(f'http://127.0.0.1:{port}')
        wait_idle(self.page)

    def close(self):
        self.ctx.close()

    def text(self, label, value):
        set_text(self.page, label, value)
        return self

    def set(self, db=None, symbol=None, asof=None, bundle=None, asset=None):
        if db is not None:
            self.text('Fundamental database', str(db))
        if bundle is not None:
            self.text('Research model bundle (optional)', bundle)
        if asset is not None:
            self.select('Instrument class', asset)
        if symbol is not None:
            self.text('Fundamental symbol', symbol)
        if asof is not None:
            self.text('Information available as of (UTC)', asof)
        return self

    def select(self, label, option):
        box = self.page.get_by_test_id('stSelectbox').filter(has_text=label)
        box.locator('input').first.click()
        self.page.get_by_role('option', name=option, exact=True).click()
        self.page.wait_for_timeout(300)
        wait_idle(self.page)
        return self

    # ---- reading
    def metrics(self):
        return metrics(self.page)

    def alerts(self):
        return alerts(self.page)

    def body(self):
        return body_text(self.page)

    def jsons(self):
        self.tap.sync(self.page)
        out = {}
        for path, kind, el in self.tap.elements('json'):
            d = _json_of(el)
            if isinstance(d, list):
                out['flags'] = d
            elif 'values' in d and 'lineage' in d:
                out['ttm'] = d
            elif 'rejections' in d:
                out['quality'] = d
            elif 'regime' in d:
                out['macro'] = d
            elif 'candidates' in d:
                out['watchlist'] = d
            elif 'events' in d:
                out['events'] = d
            else:
                out.setdefault('other', []).append(d)
        return out

    def frames(self):
        self.tap.sync(self.page)
        return [arrow_to_frame(el.dataframe.arrow_data.data) for _, kind, el in self.tap.elements('dataframe')]

    def features_table(self):
        for f in self.frames():
            if 'Feature' in f.columns:
                return f
        return None

    def history_table(self):
        for f in self.frames():
            if 'period_end' in f.columns or any(str(c).startswith('period_end') for c in f.columns):
                return f
        return None

    def chart(self):
        self.tap.sync(self.page)
        for _, kind, el in self.tap.elements('vega_lite_chart'):
            spec, frames, inline = vega_chart_data(el)
            return spec, (frames[0] if frames else None)
        return None, None

    def download(self, path):
        with self.page.expect_download() as d:
            self.page.get_by_role('button', name='Download PIT snapshot').click()
        d.value.save_as(str(path))
        return json.loads(Path(path).read_text())

    def exception_shown(self):
        return self.page.get_by_test_id('stException').count() > 0
