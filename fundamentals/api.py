"""Read-only loopback HTTP API and shared dashboard service boundary."""
from __future__ import annotations
import json,re,socket,sqlite3
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs,unquote
from fundamentals.store import PointInTimeFundamentalStore,utc,now,consistent_read
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import FundamentalScoreEngine,MacroRegimeService,EventCatalystEngine,CombinedSignalEngine


class FundamentalAPI:
    def __init__(self,store,score=None):
        self.store=store;self.features=FundamentalFeatureEngine(store);self.score=score or FundamentalScoreEngine()

    def _score(self,snapshot,timestamp):
        # CombinedSignalEngine refuses stale fundamentals; the API/watchlist must not rank them with a number.
        if snapshot['status']=='STALE':return {'score':None,'confidence':None,'status':'STALE_FUNDAMENTALS','components':{},'deployment_approved':False}
        return self.score.score(snapshot['features'],timestamp)

    @consistent_read
    def get(self,path,query=None):
        # 'as_of' present but blank is a client error, never "now" (a bug upstream must not silently read the latest data).
        q=query or {};timestamp=utc(q['as_of'] if 'as_of' in q else now());parts=[unquote(p) for p in path.split('/') if p]
        if parts==['health']:
            self.store.db.execute('SELECT 1 FROM observations LIMIT 1').fetchone()   # a corrupt/unreadable database must not report OK
            return {'status':'OK','read_only':True,'live_trading':False}
        if parts==['macro','regime']:return MacroRegimeService(self.store).state(timestamp)
        if parts==['watchlist','fundamental']:
            rows=[]
            for symbol in self.store.symbols(timestamp):
                if symbol.startswith('MACRO:'):continue
                try:snap=self.features.snapshot(symbol,timestamp)
                except ValueError as error:
                    # One symbol with conflicting vendors/currencies must not hide every healthy candidate.
                    rows.append({'symbol':symbol,'score':None,'coverage':0.,'data_status':'AMBIGUOUS_SOURCE','flags':[],'reason':str(error),'dimensions':{}})
                    continue
                score=self._score(snap,timestamp)
                rows.append({'symbol':symbol,'score':score['score'],'coverage':snap['completeness'],'data_status':snap['status'],
                             'flags':snap['flags'],'reason':score['status'],'dimensions':snap['dimensions']})
            rows.sort(key=lambda x:(x['score'] is None,-(x['score'] or 0),x['symbol']))
            return {'as_of':timestamp,'candidates':rows,'ranked':any(r['score'] is not None for r in rows),'deployment_approved':False}
        if len(parts)>=2 and parts[0]=='fundamentals':
            symbol=parts[1]
            if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,32}',symbol):raise ValueError('invalid symbol')
            if len(parts)==2:return self.features.snapshot(symbol,timestamp,asset_class=q.get('asset_class','equity'))
            if parts[2]=='quality':return self.store.quality(symbol)
            if parts[2]=='ttm':
                from fundamentals.features import trailing_twelve_months
                return {'symbol':symbol,'as_of':timestamp,**trailing_twelve_months(self.store.as_of(symbol,timestamp,kind='financial'),timestamp,self.features.max_age)}
            if parts[2]=='history':return {'symbol':symbol,'as_of':timestamp,'records':self.store.history(symbol,q.get('metric'),timestamp)}
            if parts[2]=='score':
                snap=self.features.snapshot(symbol,timestamp,asset_class=q.get('asset_class','equity'))
                return {'symbol':symbol,'as_of':timestamp,**self._score(snap,timestamp)}
            if parts[2]=='events':return EventCatalystEngine(self.store).state(symbol,timestamp)
        if len(parts)==3 and parts[0]=='strategies' and parts[2]=='explanation':
            symbol=q.get('symbol','XLE')
            if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,32}',symbol):raise ValueError('invalid symbol')
            if not re.fullmatch(r'[a-z]{1,16}',q.get('asset_class','etf')):raise ValueError('invalid asset_class')
            engine=CombinedSignalEngine(self.features,self.score,MacroRegimeService(self.store),EventCatalystEngine(self.store))
            return engine.decide(symbol,timestamp,{'signal':q.get('technical_signal','false').lower()=='true'},asset_class=q.get('asset_class','etf'))
        raise KeyError('unknown endpoint')


class _LoopbackIPv6Server(ThreadingHTTPServer):
    address_family=socket.AF_INET6


def serve(database,host='127.0.0.1',port=8766,score=None):
    if host not in ('127.0.0.1','localhost','::1'):raise ValueError('read API must bind to loopback')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            store=None
            try:
                parsed=urlparse(self.path);query={k:v[-1] for k,v in parse_qs(parsed.query,keep_blank_values=True).items()}
                store=PointInTimeFundamentalStore(database,readonly=True)
                payload=FundamentalAPI(store,score).get(parsed.path,query);status=200
                json.dumps(payload,allow_nan=False)   # an unserialisable payload must become a JSON error, not a dropped connection
            except ValueError as e:payload={'error':str(e)};status=400
            except KeyError:payload={'error':'not found'};status=404
            except (sqlite3.Error,OSError):payload={'error':'fundamental data unavailable; no trade'};status=503
            except Exception:payload={'error':'internal error; no trade'};status=500
            finally:
                if store:store.close()
            data=json.dumps(payload,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    return (_LoopbackIPv6Server if ':' in host else ThreadingHTTPServer)((host,port),Handler)
