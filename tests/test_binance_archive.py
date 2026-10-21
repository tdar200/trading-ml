import hashlib,io,zipfile
import pytest
from data.binance_archive import parse_archive

def archive(ts):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('sample.csv',f'{ts},100,110,90,105,20,{ts},0,1,0,0,0\n')
    b=out.getvalue();return b,hashlib.sha256(b).hexdigest()

@pytest.mark.parametrize('ts,month',[(1704067200000,'2024-01'),(1735689600000000,'2025-01')])
def test_timestamp_units(ts,month):
    b,h=archive(ts);f=parse_archive(b,h,month)
    assert str(f.ts.iloc[0])==month+'-01 00:00:00+00:00'
    assert f.close.iloc[0]==105

def test_checksum_and_wrong_month_fail():
    b,h=archive(1704067200000)
    with pytest.raises(ValueError,match='checksum'):parse_archive(b,'0'*64,'2024-01')
    with pytest.raises(ValueError,match='month'):parse_archive(b,h,'2024-02')
