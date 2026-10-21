import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fundamentals.api import serve
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--database',default='results/fundamentals/fundamentals.sqlite');p.add_argument('--port',type=int,default=8766)
 p.add_argument('--model-bundle')
 a=p.parse_args()
 from fundamentals.confidence import load_score_bundle
 server=serve(a.database,port=a.port,score=load_score_bundle(a.model_bundle) if a.model_bundle else None)
 try:server.serve_forever()
 finally:server.server_close()
