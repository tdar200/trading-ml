"""Public Binance spot archives; immutable raw cache and checksum validation."""
import hashlib
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests

BUCKET='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
BASE='https://data.binance.vision/'
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}

def listing(prefix, delimiter=None):
    params={'list-type':'2','prefix':prefix,'max-keys':1000}; seen=set()
    if delimiter:params['delimiter']=delimiter
    while True:
        r=requests.get(BUCKET,params=params,timeout=30);r.raise_for_status();tree=ET.fromstring(r.content)
        path='s:CommonPrefixes/s:Prefix' if delimiter else 's:Contents/s:Key'
        yield from (n.text for n in tree.findall(path,NS))
        if tree.findtext('s:IsTruncated',namespaces=NS)!='true':break
        token=tree.findtext('s:NextContinuationToken',namespaces=NS)
        if not token or token in seen:raise ValueError('invalid pagination')
        seen.add(token);params['continuation-token']=token

def symbols():
    return sorted({p.rstrip('/').split('/')[-1] for p in listing('data/spot/monthly/klines/','/')})

def parse_archive(payload, expected_sha, month):
    if hashlib.sha256(payload).hexdigest()!=expected_sha:raise ValueError('checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        members=z.infolist()
        if len(members)!=1 or members[0].file_size>2_000_000:raise ValueError('unexpected daily archive size/members')
        f=pd.read_csv(z.open(members[0]),header=None)
    if f.shape[1]!=12:raise ValueError('expected 12 kline columns')
    unit='us' if month>='2025-01' else 'ms'
    out=pd.DataFrame({'ts':pd.to_datetime(f[0],unit=unit,utc=True)})
    for col,index in zip(('open','high','low','close','volume'),range(1,6)):
        out[col]=pd.to_numeric(f[index],errors='raise')
    if not out.ts.dt.strftime('%Y-%m').eq(month).all():raise ValueError('archive month mismatch')
    from analysis.rule_discovery import validate_bars
    return validate_bars(out)

def fetch_month(key, cache):
    if not re.fullmatch(r'data/spot/monthly/klines/[A-Z0-9_]+/1d/[A-Z0-9_]+-1d-\d{4}-\d{2}\.zip',key):raise ValueError('invalid archive key')
    path=Path(cache)/key;checksum=path.with_suffix('.zip.CHECKSUM')
    if path.exists() and checksum.exists():payload=path.read_bytes();sha=checksum.read_text().split()[0]
    else:
        r=requests.get(BASE+key+'.CHECKSUM',timeout=30);r.raise_for_status();sha=r.text.split()[0]
        r=requests.get(BASE+key,timeout=30);r.raise_for_status();payload=r.content
    frame=parse_archive(payload,sha,key[-11:-4])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(payload);checksum.write_text(sha+'  '+path.name+'\n')
    return frame
