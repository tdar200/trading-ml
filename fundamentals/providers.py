"""Explicit source adapters. No latest-value endpoint is backdated.

SEC companyfacts retains accession-tagged facts; restatements become new versions.
ALFRED graph snapshots are available only at the END of their requested vintage
calendar day. That conservative delay is not an inferred release timestamp.
"""
from __future__ import annotations
import io,json,os,re,time,hashlib
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
import pandas as pd
from fundamentals.store import Fact,utc,now

SEC_TAGS={
 'revenue':('Revenues','RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','SalesRevenueGoodsNet','RegulatedAndUnregulatedOperatingRevenue'),
 'gross_profit':('GrossProfit',),'cogs':('CostOfRevenue','CostOfGoodsAndServicesSold','CostOfGoodsSold'),
 'operating_income':('OperatingIncomeLoss',),'net_income':('NetIncomeLoss','ProfitLoss'),
 'eps':('EarningsPerShareDiluted',),
 'cash':('CashAndCashEquivalentsAtCarryingValue',),'current_assets':('AssetsCurrent',),
 'assets':('Assets',),'current_liabilities':('LiabilitiesCurrent',),'liabilities':('Liabilities',),
 # Short-term borrowings alone omit current maturities; never substitute them for total current debt.
 'short_debt':('DebtCurrent',),'long_debt':('LongTermDebtNoncurrent',),
 'equity':('StockholdersEquity','StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
 'inventory':('InventoryNet',),'receivables':('AccountsReceivableNetCurrent',),'payables':('AccountsPayableCurrent',),
 'prepaids':('PrepaidExpenseCurrent',),'goodwill':('Goodwill',),'ocf':('NetCashProvidedByUsedInOperatingActivities',),
 'capex':('PaymentsToAcquirePropertyPlantAndEquipment',),
 'investing_cf':('NetCashProvidedByUsedInInvestingActivities',),'financing_cf':('NetCashProvidedByUsedInFinancingActivities',),
 'interest_expense':('InterestExpense','InterestAndDebtExpense'),
 'pretax_income':('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',),
 'tax_expense':('IncomeTaxExpenseBenefit',),'depreciation':('DepreciationDepletionAndAmortization','DepreciationDepletionAndAmortizationPropertyPlantAndEquipment'),
 'diluted_shares':('WeightedAverageNumberOfDilutedSharesOutstanding',),
 'shares_outstanding':('CommonStockSharesOutstanding',),'retained_earnings':('RetainedEarningsAccumulatedDeficit',),
 'rd':('ResearchAndDevelopmentExpense',),'buybacks':('PaymentsForRepurchaseOfCommonStock',),
 'dividends':('PaymentsOfDividendsCommonStock','PaymentsOfDividends'),
 'issuance':('ProceedsFromStockOptionsExercised','ProceedsFromIssuanceOfCommonStock'),
 'acquisitions':('PaymentsToAcquireBusinessesNetOfCashAcquired',),
}


def fetch(url,cache:Path,*,user_agent=None,validate=None):
    """Cached GET. `validate(body)` must raise ValueError for a body that is not the expected document:
    a transient 200 error page is then never cached or archived, so a retry really refetches."""
    cache=Path(cache); cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists(): return cache.read_bytes()
    req=Request(url,headers={'User-Agent':user_agent or os.getenv('SEC_USER_AGENT','trading-ml point-in-time research'),'Accept-Encoding':'identity'})
    with urlopen(req,timeout=45) as response: body=response.read()
    if validate is not None:validate(body)
    tmp=cache.with_suffix(cache.suffix+'.tmp');tmp.write_bytes(body);tmp.replace(cache)
    return body


def _json_document(body):
    try:document=json.loads(body)
    except (ValueError,UnicodeDecodeError) as e:raise ValueError('provider returned a non-JSON body; not cached') from e
    if not isinstance(document,dict):raise ValueError('provider returned an unexpected JSON shape; not cached')


def sec_facts(payload,submissions,symbol,*,observed_at=None,_taxonomy="us-gaap",_tags=None,rejects=None):
    """Facts from SEC companyfacts. A malformed source row raises ValueError, or, when a `rejects` list is
    supplied, is appended there verbatim and skipped so one bad row cannot poison a whole company."""
    if not isinstance(payload,dict) or not isinstance(payload.get('facts'),dict) or not isinstance(submissions,dict):
        raise ValueError('malformed companyfacts/submissions payload')
    facts=[]; selected={}; raw=payload['facts'].get(_taxonomy,{})
    filings=submissions.get('filings')
    recent=filings.get('recent') if isinstance(filings,dict) else None
    if not isinstance(recent,dict):recent=submissions
    acceptance=dict(zip(recent.get('accessionNumber',[]),recent.get('acceptanceDateTime',[])))
    for metric,tags in (_tags or SEC_TAGS).items():
        for priority,tag in enumerate(tags):
            for unit,rows in raw.get(tag,{}).get('units',{}).items():
                if unit not in ('USD','GBP','EUR','JPY','USD/shares','GBP/shares','EUR/shares','JPY/shares','shares'):continue
                for row in rows:
                    try:
                        if row.get('form') not in ('10-K','10-Q','10-K/A','10-Q/A','20-F','20-F/A'):continue
                        acc=row['accn']; filed=pd.Timestamp(row['filed'],tz='America/New_York')
                        timestamp=acceptance.get(acc)
                        reported=pd.Timestamp(timestamp) if timestamp else filed
                        if reported.tzinfo is None: reported=reported.tz_localize('America/New_York')
                        # Do not trade at an unobserved intraday acceptance when only a date is known.
                        available=reported+pd.Timedelta(minutes=5) if timestamp else filed+pd.DateOffset(days=1)
                        currency=unit.split('/')[0] if unit!='shares' else ''
                        value=row['val']
                        # float(True) is 1.0: a JSON boolean (or string) is a corrupt number, not a value.
                        if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError('non-numeric source value')
                        f=Fact(symbol,metric,float(value),row['end'],utc(reported),utc(available),'SEC/companyfacts' if _taxonomy=='us-gaap' else 'SEC/companyfacts-ifrs',acc,currency,unit,
                               row.get('start',''),observed_at or now(),metadata={'tag':tag,'accounting_standard':_taxonomy,'cik':payload['cik'],
                               'form':row.get('form'),'filing_date':row['filed'],'timing_precision':'acceptance+5min' if timestamp else 'following_local_day',
                               'historical_mode':'accession-tagged reconstruction','split_basis':'as_reported',
                               'source_url':f'https://www.sec.gov/Archives/edgar/data/{payload["cik"]}/{acc.replace("-","")}/'})
                        key=(metric,row.get('start',''),row['end'],unit,acc)
                    except (KeyError,TypeError,ValueError,AttributeError,ArithmeticError) as error:
                        reason=f'{type(error).__name__}: {error}'
                        if rejects is None:raise ValueError(f'malformed SEC row for {tag}: {reason}') from error
                        rejects.append({'tag':tag,'unit':unit,'reason':reason,'row':row});continue
                    if key not in selected or priority<selected[key][0]:selected[key]=(priority,f)
                    elif priority==selected[key][0] and selected[key][1].value!=f.value:
                        # Ambiguous contexts must not silently choose the last value.
                        selected[key]=(priority,Fact(**{**f.__dict__,'value':None,'metadata':{**f.metadata,'quality_flag':'CONFLICTING_CONTEXTS'}}))
    result=[v[1] for v in selected.values()]
    if _taxonomy=='us-gaap' and 'ifrs-full' in payload['facts']:
        result+=sec_facts(payload,submissions,symbol,observed_at=observed_at,_taxonomy='ifrs-full',rejects=rejects,_tags={
            'revenue':('Revenue',),'net_income':('ProfitLossAttributableToOwnersOfParent',),
            'eps':('DilutedEarningsLossPerShare',),'cash':('CashAndCashEquivalents',),
            'ocf':('CashFlowsFromUsedInOperatingActivities',),'assets':('Assets',),
            'equity':('EquityAttributableToOwnersOfParent',)})
    return result


def ingest_sec(store,symbol,cik,cache):
    root=Path(cache); cik=f'{int(cik):010d}'
    urls=[f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',f'https://data.sec.gov/submissions/CIK{cik}.json']
    payload=[]
    for label,url in zip(('facts','submissions'),urls):
        body=fetch(url,root/f'{symbol}_{label}.json',validate=_json_document); store.archive(url,body);payload.append(json.loads(body));time.sleep(.15)
    rejects=[]
    facts=sec_facts(*payload,symbol,rejects=rejects)
    for reject in rejects:store.quarantine(symbol,'malformed source row: '+reject['reason'],reject)
    validation=store.ingest_validated(facts)
    if rejects:validation={**validation,'rejected':validation['rejected']+len(rejects),'status':'QUARANTINED'}
    count=validation['accepted']
    recent=payload[1]['filings']['recent']
    for i,form in enumerate(recent['form']):
        if form not in ('8-K','10-K','10-Q','10-K/A','10-Q/A'):continue
        accepted=recent['acceptanceDateTime'][i]; acc=recent['accessionNumber'][i]
        store.add_event(symbol=symbol,event_type='filing',event_at=accepted,reported_at=accepted,
            available_at=utc(pd.Timestamp(accepted)+pd.Timedelta(minutes=5)),source='SEC/submissions',revision=acc,
            payload={'form':form,'accession':acc,'items':recent.get('items',['']*len(recent['form']))[i],
                     'event_key':acc,'direction':None,'note':'Filing existence is not sentiment or earnings surprise'})
    return count


def alfred_snapshot(body,series,vintage,*,units='index',currency='',observed_at=None):
    if not isinstance(vintage,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',vintage):raise ValueError('vintage must be a YYYY-MM-DD date string')
    try:frame=pd.read_csv(io.BytesIO(body))
    except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as e:raise ValueError('ALFRED body is not a readable CSV') from e
    expected=series+'_'+pd.Timestamp(vintage).strftime('%Y%m%d')
    if expected not in frame or 'observation_date' not in frame:raise ValueError('wrong/missing ALFRED vintage column')
    # Daily snapshots carry no release time. Use following UTC day, never period_end.
    reported=pd.Timestamp(vintage,tz='UTC');available=reported+pd.Timedelta(days=1); rows=[]
    for r in frame.to_dict('records'):
        value=pd.to_numeric(r[expected],errors='coerce')
        if pd.isna(value):continue
        if pd.Timestamp(r['observation_date'])>pd.Timestamp(vintage):raise ValueError('future macro observation')
        rows.append(Fact('MACRO:'+series,series,float(value),r['observation_date'],utc(reported),utc(available),
            'ALFRED/graph',str(vintage),currency,units,observed_at=observed_at or now(),kind='macro',
            metadata={'vintage':str(vintage),'timing_precision':'vintage_day+1UTC','not_exact_release_time':True}))
    return rows


def ingest_alfred(store,series,vintage,cache,units=None):
    units=units or {'CPIAUCSL':'index','UNRATE':'percent','FEDFUNDS':'percent'}.get(series)
    if units is None:raise ValueError('explicit units required for an unmapped macro series')
    start=(pd.Timestamp(vintage)-pd.DateOffset(years=3)).date().isoformat()
    params=dict(id=series,cosd=start,coed=vintage,vintage_date=vintage)
    url='https://alfred.stlouisfed.org/graph/alfredgraph.csv?'+urlencode(params)
    def _vintage_csv(text):
        # An HTML error page returned with HTTP 200 must not be cached or archived as if it were data.
        alfred_snapshot(text,series,vintage,units=units)
    body=fetch(url,Path(cache)/f'{series}_{vintage}.csv',validate=_vintage_csv);store.archive(url,body)
    return store.ingest(alfred_snapshot(body,series,vintage,units=units))


def ingest_json_records(store,path):
    """Provider-neutral authenticated/export adapter; timestamps are mandatory."""
    body=Path(path).read_bytes();digest=store.archive(str(path),body)
    records=json.loads(body)
    if not isinstance(records,list) or not all(isinstance(r,dict) for r in records):raise ValueError('import file must be a JSON list of record objects')
    facts=[]
    for r in records:
        try:facts.append(Fact(**{**r,'metadata':{**(r.get('metadata') or {}),'raw_sha256':digest}}))
        except (TypeError,AttributeError) as e:raise ValueError(f'malformed import record: {e}') from e
    return store.ingest(facts)
