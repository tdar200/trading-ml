"""Append-only point-in-time observations and raw-source archive.

Publication time and local observation time are separate. Research may reconstruct
public knowledge; operational queries may additionally require observed_at <= T.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,field,replace
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,re,sqlite3
from contextlib import contextmanager
from functools import wraps
import pandas as pd


_ISO_STAMP=re.compile(r'^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?\s?(?P<offset>[Zz]|[+-]\d{2}(?::?\d{2})?)?$')
_ISO_DATE=re.compile(r'^\d{4}-\d{2}-\d{2}(?:[T ].*)?$')


def utc(value):
    """Strict instant parser: ISO-8601 with 'Z' or a numeric UTC offset, or an aware datetime.

    Free-form strings are rejected on purpose. pandas/dateutil would read '04/05/2025'
    month-first, expand two-digit years, and resolve zone abbreviations ('GMT', 'BST',
    'EST') through the HOST machine's local zone, so one record could become a different
    (possibly EARLIER, i.e. look-ahead) instant on another computer.
    """
    if isinstance(value,str):
        m=_ISO_STAMP.match(value.strip())
        if not m:raise ValueError('unsupported timestamp format; explicit timezone required (ISO-8601 with Z or a numeric UTC offset)')
        if m.group('offset') is None:raise ValueError('explicit timezone required')
    t=pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None: raise ValueError('explicit timezone required')
    return t.tz_convert('UTC').isoformat()


def iso_date(value,name='date'):
    """Strict calendar-date parser (YYYY-MM-DD, optionally followed by a time that is ignored)."""
    if value is None or (isinstance(value,float) and math.isnan(value)):raise ValueError('missing '+name)
    if isinstance(value,str) and not _ISO_DATE.match(value.strip()):raise ValueError('unsupported '+name+' format; YYYY-MM-DD required')
    try:t=pd.Timestamp(value)
    except (TypeError,ValueError,OverflowError) as e:raise ValueError('invalid '+name) from e
    if pd.isna(t):raise ValueError('invalid '+name)
    return t.date().isoformat()


def now(): return datetime.now(timezone.utc).isoformat()


def _natural(text):
    """Sort key where digit runs compare as numbers: 'r2' < 'r10'."""
    return tuple((0,int(part),'') if part.isdigit() else (1,0,part) for part in re.split(r'(\d+)',str(text)) if part!='')


def consistent_read(method):
    @wraps(method)
    def wrapped(self,*args,**kwargs):
        store=getattr(self,'store',None) or self.features.store
        with store.read_snapshot():return method(self,*args,**kwargs)
    return wrapped


@dataclass(frozen=True)
class Fact:
    symbol:str
    metric:str
    value:float|None
    period_end:str
    reported_at:str
    available_at:str
    source:str
    revision:str
    currency:str
    units:str
    period_start:str=''
    observed_at:str=field(default_factory=now)
    kind:str='financial'
    metadata:dict=field(default_factory=dict)

    def normalized(self):
        """Validated canonical row. Malformed input raises ValueError, never TypeError/OverflowError."""
        fact=self
        # JSON null means "not applicable": a balance-sheet item has no period_start, a macro series no currency.
        if fact.period_start is None or fact.currency is None:
            fact=replace(fact,period_start='' if fact.period_start is None else fact.period_start,
                         currency='' if fact.currency is None else fact.currency)
        try:return fact._normalized()
        except (TypeError,KeyError,AttributeError,ArithmeticError) as e:
            raise ValueError(f'malformed fact: {type(e).__name__}: {e}') from e

    def _normalized(self):
        d=asdict(self)
        for key in ('symbol','metric','source','revision','units'):
            if not isinstance(d[key],str) or not d[key].strip(): raise ValueError('missing '+key)
        if not isinstance(self.currency,str) or not isinstance(self.period_start,str):raise ValueError('currency and period_start must be text')
        if not isinstance(d['metadata'],dict):raise ValueError('metadata must be a mapping')
        json.dumps(d['metadata'],sort_keys=True,allow_nan=False)
        if self.value is not None and (isinstance(self.value,bool) or not isinstance(self.value,(int,float)) or not math.isfinite(self.value)):
            raise ValueError('value must be finite or explicitly missing')
        if self.kind=='financial':
            if self.currency not in ('USD','GBP','EUR','JPY','CHF','CAD','AUD','CNY','HKD',''):
                raise ValueError('unsupported financial currency')
            scales={'thousands':1000,'millions':1000000,'billions':1000000000}
            parts=self.units.split()
            if len(parts)==2 and parts[0]==self.currency and parts[1] in scales:
                d['metadata']={**d['metadata'],'original_units':self.units,'original_value':self.value,'unit_multiplier':scales[parts[1]]}
                d['value']=None if self.value is None else self.value*scales[parts[1]]
                d['units']=self.currency
            if d['units'] not in (self.currency,self.currency+'/shares','shares') or not d['units']:
                raise ValueError('unsupported financial units; explicit normalization required')
            expected='shares' if self.metric in ('diluted_shares','shares_outstanding') else self.currency+'/shares' if self.metric=='eps' else self.currency
            if d['units']!=expected or (expected=='shares' and self.currency):raise ValueError('financial metric unit/currency contract mismatch')
            if d['value'] is not None and not math.isfinite(d['value']):raise ValueError('scaled value overflow')
            if self.metric=='capex' and d['value'] is not None and d['value']<0:
                raise ValueError('canonical capex must be positive cash expenditure; normalize signed provider cash flows explicitly')
        if d['value'] is not None:d['value']=float(d['value'])
        for key in ('reported_at','available_at','observed_at'): d[key]=utc(d[key])
        if d['available_at']<d['reported_at']: raise ValueError('available_at precedes publication')
        d['period_end']=iso_date(self.period_end,'period_end')
        if self.period_start:
            d['period_start']=iso_date(self.period_start,'period_start')
            if d['period_start']>d['period_end']: raise ValueError('negative reporting duration')
        if self.kind not in ('financial','macro','estimate','ownership','crypto','classification','price','fx','holding'):
            raise ValueError('unsupported observation kind')
        if self.kind not in ('estimate','classification','holding') and pd.Timestamp(d['period_end']).date()>pd.Timestamp(d['available_at']).date():
            raise ValueError('observation precedes measured period')
        d['symbol']=d['symbol'].upper(); return d


class PointInTimeFundamentalStore:
    def __init__(self,path=':memory:',readonly=False):
        if readonly:
            self.db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
        else:
            if str(path)!=':memory:': Path(path).parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(str(path))
        self.db.row_factory=sqlite3.Row
        if readonly: return
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS raw_documents (
          sha256 TEXT PRIMARY KEY,source TEXT NOT NULL,fetched_at TEXT NOT NULL,body BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS ingestion_rejections (
          id TEXT PRIMARY KEY,symbol TEXT NOT NULL,detected_at TEXT NOT NULL,
          reason TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations (
          id TEXT PRIMARY KEY,symbol TEXT NOT NULL,metric TEXT NOT NULL,value REAL,
          period_start TEXT NOT NULL,period_end TEXT NOT NULL,reported_at TEXT NOT NULL,
          available_at TEXT NOT NULL,observed_at TEXT NOT NULL,source TEXT NOT NULL,
          revision TEXT NOT NULL,currency TEXT NOT NULL,units TEXT NOT NULL,kind TEXT NOT NULL,
          metadata TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS pit_lookup ON observations(symbol,available_at,metric,period_end);
        CREATE TABLE IF NOT EXISTS corporate_events (
          id TEXT PRIMARY KEY,symbol TEXT NOT NULL,event_type TEXT NOT NULL,event_at TEXT NOT NULL,
          reported_at TEXT NOT NULL,available_at TEXT NOT NULL,source TEXT NOT NULL,
          revision TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS feature_snapshots (
          id TEXT PRIMARY KEY,symbol TEXT NOT NULL,as_of TEXT NOT NULL,version TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model_artifacts (
          id TEXT PRIMARY KEY,created_at TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions (
          id TEXT PRIMARY KEY,symbol TEXT NOT NULL,as_of TEXT NOT NULL,payload TEXT NOT NULL);
        ''')
        self.db.executescript("""
        CREATE VIEW IF NOT EXISTS FinancialStatements AS SELECT * FROM observations WHERE kind='financial';
        CREATE VIEW IF NOT EXISTS MacroSeries AS SELECT * FROM observations WHERE kind='macro';
        CREATE VIEW IF NOT EXISTS EstimateRevisions AS SELECT * FROM observations WHERE kind='estimate';
        CREATE VIEW IF NOT EXISTS InsiderTransactions AS SELECT * FROM observations WHERE kind='ownership';
        CREATE VIEW IF NOT EXISTS EarningsReports AS SELECT * FROM corporate_events WHERE event_type='earnings';
        CREATE VIEW IF NOT EXISTS NewsEvents AS SELECT * FROM corporate_events WHERE event_type='document_annotation';
        CREATE VIEW IF NOT EXISTS FundamentalFeatures AS SELECT * FROM feature_snapshots;
        CREATE VIEW IF NOT EXISTS FundamentalScores AS SELECT * FROM decisions;
        PRAGMA user_version=1;
        """)
        for table in ('ingestion_rejections','observations','raw_documents','corporate_events','feature_snapshots','model_artifacts','decisions'):
            for operation in ('UPDATE','DELETE'):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only provenance'); END")
        self.db.commit()

    def archive(self,source,body,fetched_at=None):
        if isinstance(body,str): body=body.encode()
        digest=hashlib.sha256(body).hexdigest()
        with self.db: self.db.execute('INSERT OR IGNORE INTO raw_documents VALUES (?,?,?,?)',(digest,source,utc(fetched_at or now()),body))
        return digest

    def ingest(self,facts):
        rows=[]
        for fact in facts:
            d=fact.normalized(); meta=json.dumps(d.pop('metadata'),sort_keys=True,allow_nan=False)
            # observed_at does not create a spurious economic revision on re-download.
            identity={k:v for k,v in d.items() if k!='observed_at'}
            # The raw file hash records where a row came from; it is not part of the economic fact, so
            # re-importing identical records from a re-formatted file must not create a new version.
            identity['metadata']={k:v for k,v in json.loads(meta).items() if k!='raw_sha256'}
            identifier=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
            rows.append((identifier,d['symbol'],d['metric'],d['value'],d['period_start'],d['period_end'],d['reported_at'],d['available_at'],d['observed_at'],d['source'],d['revision'],d['currency'],d['units'],d['kind'],meta))
        # ON CONFLICT(id) skips only a true duplicate. INSERT OR IGNORE would also silently swallow
        # NOT NULL / CHECK violations while the caller is told the batch succeeded.
        try:
            with self.db:
                self.db.executemany('INSERT INTO observations VALUES ('+','.join('?'*15)+') ON CONFLICT(id) DO NOTHING',rows)
        except sqlite3.IntegrityError as e:
            raise ValueError('observation violates a storage constraint: '+str(e)) from e
        return len(rows)

    def ingest_validated(self,facts):
        """Retain invalid source records verbatim; never invent repaired values."""
        valid=[];rejected=[]
        for fact in facts:
            try:fact.normalized();valid.append(fact)
            except ValueError as error:
                payload=json.dumps(asdict(fact),sort_keys=True,default=str)
                identity=asdict(fact);identity.pop('observed_at',None)
                identifier=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
                rejected.append((identifier,fact.symbol,now(),str(error),payload))
        # Atomic accepted and rejected record transaction, including ingest's
        # transaction context. This method must own the transaction.
        if self.db.in_transaction:raise ValueError('ingestion requires its own transaction')
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO ingestion_rejections VALUES (?,?,?,?,?)',rejected)
            # ingest validates again before entering its transaction; any failure
            # rolls back the surrounding transaction as well.
            count=self.ingest(valid)
        return {'accepted':count,'rejected':len(rejected),'status':'QUARANTINED' if rejected else 'VALIDATED'}

    def quarantine(self,symbol,reason,payload):
        """Record a source record that could not even be turned into a Fact (verbatim, never repaired)."""
        body=json.dumps(payload,sort_keys=True,default=str)
        identifier=hashlib.sha256((str(symbol)+'|'+str(reason)+'|'+body).encode()).hexdigest()
        with self.db:
            self.db.execute('INSERT INTO ingestion_rejections VALUES (?,?,?,?,?) ON CONFLICT(id) DO NOTHING',
                            (identifier,str(symbol).upper(),now(),str(reason),body))
        return identifier

    def quality(self,symbol):
        rows=self.db.execute('SELECT id,detected_at,reason FROM ingestion_rejections WHERE symbol=? ORDER BY id',(symbol.upper(),)).fetchall()
        return {'symbol':symbol.upper(),'rejections':[dict(r) for r in rows],
                'status':'QUARANTINED_RECORDS' if rows else 'NO_RECORDED_REJECTIONS',
                'accuracy_certified':False,'time_scope':'current ingestion audit, not historical signal input','note':'Validation is not independent verification of source truth'}

    def as_of(self,symbol,timestamp,*,observed_only=False,kind=None):
        ts=utc(timestamp); args=[symbol.upper(),ts]
        where="symbol=? AND available_at<=? AND (kind NOT IN ('classification','holding') OR period_end<=?)"
        args.append(pd.Timestamp(ts).date().isoformat())
        if observed_only: where+=' AND observed_at<=?'; args.append(ts)
        if kind: where+=' AND kind=?'; args.append(kind)
        rows=self.db.execute(f'''SELECT * FROM observations WHERE {where}
            ORDER BY available_at,reported_at,id''',args).fetchall()
        selected={}
        for raw in rows:
            d=dict(raw);d['metadata']=json.loads(d['metadata'])
            key=tuple(d[k] for k in ('metric','period_start','period_end','currency','units','source'))
            # Distinct insider transactions share metric/period/source. A provider that supplies
            # metadata.transaction_id keeps them apart (an amendment re-uses the id and supersedes). Without an id the
            # legacy rule applies: same metric/period/source supersedes, so ids are REQUIRED for multi-trade coverage.
            if d['kind']=='ownership' and 'transaction_id' in d['metadata']:key+=(str(d['metadata']['transaction_id']),)
            old=selected.get(key)
            if old and old['available_at']==d['available_at'] and old['reported_at']==d['reported_at'] and (old['value']!=d['value'] or (d['kind'] in ('classification','holding','ownership') and {k:v for k,v in old['metadata'].items() if k!='raw_sha256'}!={k:v for k,v in d['metadata'].items() if k!='raw_sha256'}) or old['metadata'].get('share_basis')!=d['metadata'].get('share_basis')):
                raise ValueError('ambiguous conflicting source facts')
            selected[key]=d
        return sorted(selected.values(),key=lambda d:(d['period_end'],d['metric'],d['available_at']))

    def history(self,symbol,metric=None,as_of=None):
        where='symbol=?'; args=[symbol.upper()]
        if metric: where+=' AND metric=?';args.append(metric)
        if as_of:where+=' AND available_at<=?';args.append(utc(as_of))
        return [dict(r) for r in self.db.execute('SELECT * FROM observations WHERE '+where+' ORDER BY available_at,period_end,id',args)]

    def add_event(self,*,symbol,event_type,event_at,reported_at,available_at,source,revision,payload):
        reported,available=utc(reported_at),utc(available_at)
        if available<reported:raise ValueError('event available before publication')
        # The log is append-only: a malformed payload could never be removed and would break every later read.
        if not isinstance(payload,dict):raise ValueError('event payload must be a mapping')
        if not all(isinstance(x,str) and x.strip() for x in (symbol,event_type,source)):raise ValueError('event symbol, type and source required')
        event=dict(symbol=symbol.upper(),event_type=event_type,event_at=utc(event_at),reported_at=reported,
                   available_at=available,source=source,revision=str(revision),payload=payload)
        body=json.dumps(event,sort_keys=True,allow_nan=False); ident=hashlib.sha256(body.encode()).hexdigest()
        with self.db:self.db.execute('INSERT OR IGNORE INTO corporate_events VALUES (?,?,?,?,?,?,?,?,?)',
            (ident,event['symbol'],event_type,event['event_at'],reported,available,source,str(revision),json.dumps(payload,allow_nan=False)))
        return ident

    def events(self,symbol,as_of):
        # Latest version of each source event; event_at may legitimately be future
        # for a calendar event announced before the decision time.
        rows=[dict(r) for r in self.db.execute('SELECT * FROM corporate_events WHERE symbol IN (?,\'*\') AND available_at<=?',(symbol.upper(),utc(as_of)))]
        # Total order: publication time, then local reported time, then NUMERIC-aware revision ("2" < "10"),
        # never the sha256 id, which would let an older revision win an exact timestamp tie.
        rows.sort(key=lambda d:(d['available_at'],d['reported_at'],_natural(d['revision']),d['id']))
        latest={};tied=set()
        for d in rows:
            try:payload=json.loads(d['payload'])
            except ValueError:payload=None
            # Rows written before payload validation can hold a non-mapping; they stay readable and inert.
            d['payload']=payload if isinstance(payload,dict) else {'malformed_payload':True}
            key=(d['symbol'],d['event_type'],d['payload'].get('event_key',d['event_at']),d['source'])
            old=latest.get(key)
            if old is not None:
                # Re-observing an unchanged document later must not restart its age: keep the first public copy.
                if old['revision']==d['revision'] and old['payload']==d['payload']:continue
                if (old['available_at'],old['reported_at'])==(d['available_at'],d['reported_at']):tied.add(key)
                else:tied.discard(key)
            latest[key]=d
        for key in tied:latest[key]={**latest[key],'conflicting_versions':True}
        return list(latest.values())

    def save(self,table,payload,version='v1'):
        body=json.dumps(payload,sort_keys=True,allow_nan=False); ident=hashlib.sha256(body.encode()).hexdigest()
        with self.db:
            if table=='feature_snapshots':self.db.execute('INSERT OR IGNORE INTO feature_snapshots VALUES (?,?,?,?,?)',(ident,payload['symbol'],utc(payload['as_of']),version,body))
            elif table=='model_artifacts':self.db.execute('INSERT OR IGNORE INTO model_artifacts VALUES (?,?,?)',(ident,now(),body))
            elif table=='decisions':self.db.execute('INSERT OR IGNORE INTO decisions VALUES (?,?,?,?)',(ident,payload['symbol'],utc(payload['timestamp']),body))
            else:raise ValueError('unsupported artifact table')
        return ident

    @contextmanager
    def read_snapshot(self):
        owner=not self.db.in_transaction
        if owner:self.db.execute('BEGIN')
        try:
            yield
            if owner and self.db.in_transaction:self.db.commit()
        except BaseException:
            if owner and self.db.in_transaction:self.db.rollback()
            raise

    def symbols(self,as_of=None):
        if as_of is None:return [r[0] for r in self.db.execute('SELECT DISTINCT symbol FROM observations ORDER BY symbol')]
        return [r[0] for r in self.db.execute("SELECT DISTINCT symbol FROM observations WHERE available_at<=? AND kind='financial' ORDER BY symbol",(utc(as_of),))]
    def close(self):self.db.close()


FundamentalDataService=PointInTimeFundamentalStore
