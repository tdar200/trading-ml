"""One-off fetch of fresh public SEC EDGAR companyfacts + submissions (read-only GETs).

Run manually:  .venv/bin/python tests/qa_reaudit/real_fetch.py
Responses are cached under results/qa_reaudit_20261001/real/raw; cached files are never re-fetched.
Spacing between uncached SEC calls is >= 0.3 s.
"""
from __future__ import annotations
import sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fundamentals.providers import fetch  # noqa: E402  (public provider entry point, used as the SUT's own fetcher)

NEW={'AAPL':320193,'PFE':78003,'F':37996,'KO':21344,'NEE':753308,'AMZN':1018724,'PLUG':1093691,'NKE':320187,'GSK':1131399}
RAW=ROOT/'results'/'qa_reaudit_20261001'/'real'/'raw'


def main(symbols=None):
    for symbol,cik in NEW.items():
        if symbols and symbol not in symbols:continue
        cik10=f'{cik:010d}'
        for label,url in (('facts',f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json'),
                          ('submissions',f'https://data.sec.gov/submissions/CIK{cik10}.json')):
            path=RAW/f'{symbol}_{label}.json'
            cached=path.exists()
            try:
                body=fetch(url,path)
            except Exception as error:
                print(symbol,label,'FETCH FAILED',type(error).__name__,error);continue
            print(symbol,label,len(body),'cached' if cached else 'fetched')
            if not cached:time.sleep(0.35)


if __name__=='__main__':
    main(sys.argv[1:] or None)
