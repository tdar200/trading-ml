"""Official-document spot-check helper (QA track B, prefix real).

Fetches the PRIMARY 10-K/20-F document of a filing from SEC Archives (public read-only GET) and extracts
the inline-XBRL numbers EXACTLY as the filer printed them (displayed text, scale, sign) together with the
printed row label.  This is a different path from the companyfacts API that production ingests: it is the
text of the filing itself.  Stdlib only (no fundamentals imports).

Manual run:  .venv/bin/python tests/qa_reaudit/real_official.py   (uses cache under results/qa_reaudit_20261001/real/official)
"""
from __future__ import annotations
import html,json,re,sys,time
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'results'/'qa_reaudit_20261001'/'real'/'raw'
OFF=ROOT/'results'/'qa_reaudit_20261001'/'real'/'official'
UA='trading-ml research tradingml-qa@example.com'  # www.sec.gov Archives rejects the bare provider UA with 403; data.sec.gov accepts it

CONCEPTS={  # concept local name -> oracle metric (what a human reads as the headline figure)
 'Revenues':'revenue','RevenueFromContractWithCustomerExcludingAssessedTax':'revenue','RegulatedAndUnregulatedOperatingRevenue':'revenue',
 'SalesRevenueNet':'revenue','Revenue':'revenue',
 'NetIncomeLoss':'net_income','ProfitLossAttributableToOwnersOfParent':'net_income',
 'EarningsPerShareDiluted':'eps','DilutedEarningsLossPerShare':'eps',
 'NetCashProvidedByUsedInOperatingActivities':'ocf','CashFlowsFromUsedInOperatingActivities':'ocf',
 'PaymentsToAcquirePropertyPlantAndEquipment':'capex','PaymentsToAcquireProductiveAssets':'capex',
 'Assets':'assets','StockholdersEquity':'equity','EquityAttributableToOwnersOfParent':'equity',
 'CashAndCashEquivalentsAtCarryingValue':'cash','CashAndCashEquivalents':'cash',
 'GrossProfit':'gross_profit','OperatingIncomeLoss':'operating_income',
 'WeightedAverageNumberOfDilutedSharesOutstanding':'diluted_shares',
}


def get(url,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return path.read_bytes()
    req=Request(url,headers={'User-Agent':UA,'Accept-Encoding':'identity'})
    with urlopen(req,timeout=60) as r:body=r.read()
    path.write_bytes(body);time.sleep(0.35)
    return body


def contexts(doc:str):
    out={}
    for m in re.finditer(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>',doc,re.S):
        cid,body=m.group(1),m.group(2)
        if 'segment' in body or 'explicitMember' in body or 'typedMember' in body:
            out[cid]=None;continue
        s=re.search(r'<xbrli:startDate>([^<]+)</xbrli:startDate>',body);e=re.search(r'<xbrli:endDate>([^<]+)</xbrli:endDate>',body)
        i=re.search(r'<xbrli:instant>([^<]+)</xbrli:instant>',body)
        out[cid]=(s.group(1).strip() if s else '',e.group(1).strip() if e else i.group(1).strip()) if (e or i) else None
    return out


def strip(tagged:str)->str:
    text=re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',tagged))).strip()
    return text.replace('\u2014','-').replace('\u2013','-').replace('\u2212','-')   # ASCII hyphens only (repo rule)


def extract(doc:str):
    ctx=contexts(doc);rows=[]
    for m in re.finditer(r'<ix:nonFraction\b([^>]*)>(.*?)</ix:nonFraction>',doc,re.S):
        attrs=dict(re.findall(r'(\w[\w:.-]*)="([^"]*)"',m.group(1)))
        name=attrs.get('name','');local=name.split(':')[-1]
        if local not in CONCEPTS or ctx.get(attrs.get('contextRef'),None) is None:continue
        shown=strip(m.group(2));scale=int(attrs.get('scale','0') or 0)
        fmt=attrs.get('format','')
        if 'zerodash' in fmt or shown in ('-',''):num=0.0
        else:
            try:num=float(shown.replace(',','').replace('$','').replace('(','').replace(')',''))
            except ValueError:continue
        value=num*(10**scale)*(-1 if attrs.get('sign')=='-' else 1)
        tr=doc.rfind('<tr',0,m.start());label=strip(doc[tr:m.start()]) if tr!=-1 and m.start()-tr<4000 else ''
        start,end=ctx[attrs['contextRef']]
        rows.append(dict(concept=name,metric=CONCEPTS[local],start=start,end=end,shown=shown,scale=scale,sign=attrs.get('sign',''),
                         value=value,unit=attrs.get('unitRef',''),label=label[:90]))
    return rows


def main(symbols=None):
    for symbol in ['AAPL','PFE','F','KO','NEE','AMZN','PLUG','NKE','GSK']:
        if symbols and symbol not in symbols:continue
        sub=json.loads((RAW/f'{symbol}_submissions.json').read_text());rc=sub['filings']['recent'];cik=int(sub['cik'])
        picks=[]
        for i,form in enumerate(rc['form']):
            if form in ('10-K','20-F'):picks.append(i)
        chosen=[picks[0]]+([picks[min(len(picks)-1,4)]] if len(picks)>1 else [])
        for i in chosen:
            accn=rc['accessionNumber'][i];doc=rc['primaryDocument'][i]
            if not doc or not rc['isInlineXBRL'][i]:continue
            url=f'https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace("-","")}/{doc}'
            body=get(url,OFF/f'{symbol}_{accn}.htm').decode('utf-8','replace')
            rows=extract(body)
            (OFF/f'{symbol}_{accn}.extract.json').write_text(json.dumps(dict(symbol=symbol,accn=accn,url=url,form=rc['form'][i],filed=rc['filingDate'][i],report_date=rc['reportDate'][i],rows=rows),indent=1))
            print(symbol,accn,len(body),len(rows))


if __name__=='__main__':
    main(sys.argv[1:] or None)


def fetch_filing(symbol:str,accn:str):
    """Fetch+extract any filing listed in the cached submissions JSON (adds a 10-Q for the official TTM check)."""
    sub=json.loads((RAW/f'{symbol}_submissions.json').read_text());rc=sub['filings']['recent'];cik=int(sub['cik'])
    i=rc['accessionNumber'].index(accn);doc=rc['primaryDocument'][i]
    url=f'https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace("-","")}/{doc}'
    body=get(url,OFF/f'{symbol}_{accn}.htm').decode('utf-8','replace');rows=extract(body)
    (OFF/f'{symbol}_{accn}.extract.json').write_text(json.dumps(dict(symbol=symbol,accn=accn,url=url,form=rc['form'][i],filed=rc['filingDate'][i],report_date=rc['reportDate'][i],rows=rows),indent=1))
    return len(rows)


QUARTERLY={'AAPL':'0000320193-26-000020','PFE':'0000078003-26-000095','KO':'0001628280-26-028802','NKE':'0000320187-26-000037','AMZN':'0001018724-26-000026','PLUG':'0001104659-26-093454'}
