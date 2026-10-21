"""Fixture builders and instant arithmetic for QA track A (prefix pit). Stdlib oracle only."""
from __future__ import annotations
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
OBS = '2030-01-01T00:00:00Z'
M = 1_000_000


def ns_instant(text: str) -> int:
    """Integer nanoseconds since epoch for an ISO text with offset and 0-9 fractional digits (independent parser)."""
    body = text.replace('Z', '+00:00')
    sign_pos = max(body.rfind('+'), body.rfind('-'))
    main, off = body[:sign_pos], body[sign_pos:]
    frac = ''
    if '.' in main:
        main, frac = main.split('.')
    dt = datetime.strptime(main, '%Y-%m-%dT%H:%M:%S')
    sgn = 1 if off[0] == '+' else -1
    hh, mm = off[1:].split(':')
    offset = timedelta(hours=int(hh), minutes=int(mm)) * sgn
    base = (dt - offset).replace(tzinfo=UTC)
    secs = int((base - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds())
    return secs * 10**9 + int((frac + '000000000')[:9])


def fact(metric, value, end, start, pub, *, symbol='ACME', rev='r', currency='USD', units=None, basis=None,
         avail=None, observed=OBS, kind='financial', source='SRC', metadata=None):
    from fundamentals.store import Fact
    if units is None:
        units = 'shares' if metric in ('diluted_shares', 'shares_outstanding') else (currency + '/shares' if metric == 'eps' else currency)
    meta = dict(metadata or {})
    if basis:
        meta['share_basis'] = basis
    return Fact(symbol, metric, value, end, pub, avail or pub, source, rev, currency, units, start, observed, kind, meta)


def fy_facts(label, start, end, pub, R, *, symbol='ACME', rev=None, avail=None, extra=None):
    """One fiscal year (flows + balance sheet) with simple proportional values (USD, plain units)."""
    rev = rev or label
    vals = dict(revenue=R, gross_profit=.4 * R, operating_income=.2 * R, net_income=.1 * R, ocf=.15 * R, capex=.05 * R)
    out = [fact(m, v, end, start, pub, symbol=symbol, rev=rev, avail=avail) for m, v in vals.items()]
    inst = dict(assets=2 * R, equity=R, cash=.1 * R, current_assets=.3 * R, current_liabilities=.15 * R, short_debt=.02 * R, long_debt=.1 * R)
    out += [fact(m, v, end, '', pub, symbol=symbol, rev=rev, avail=avail) for m, v in inst.items()]
    out += list(extra or [])
    return out


def acme_nov_fy(p25='2025-05-10T07:00:00Z', symbol='ACME'):
    """FY ending 31 March (non-calendar): FY2023 R=700m, FY2024 R=800m, FY2025 R=1000m published at p25."""
    facts = []
    facts += fy_facts('fy23', '2022-04-01', '2023-03-31', '2023-05-13T07:00:00Z', 700 * M, symbol=symbol)
    facts += fy_facts('fy24', '2023-04-01', '2024-03-31', '2024-05-12T07:00:00Z', 800 * M, symbol=symbol)
    facts += fy_facts('fy25', '2024-04-01', '2025-03-31', p25, 1000 * M, symbol=symbol)
    return facts


@contextmanager
def http_server(db_path):
    """Real loopback HTTP server (ephemeral port) over a read-only API; yields a GET helper."""
    from fundamentals.api import serve
    httpd = serve(str(db_path), port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    def get(path):
        try:
            r = urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=30)
            raw = r.read()
            return r.status, json.loads(raw), dict(r.headers), raw
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, json.loads(raw), dict(e.headers), raw

    try:
        yield get
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
