"""Held-out sign calibration, separate from return predictions and data coverage.

Disjoint, matured labels only. A probability is a historical empirical frequency,
not a guarantee or authorization to trade. No artifact means unknown confidence.
"""
from dataclasses import dataclass,asdict
import hashlib,json,math
import pandas as pd
from fundamentals.store import utc

@dataclass(frozen=True)
class ConfidenceCalibration:
    model_id:str
    calibrated_at:str
    bins:tuple
    sample_sha256:str
    observations:int
    brier:float
    version:str='heldout-sign-frequency-v1'

    def __post_init__(self):
        utc(self.calibrated_at)
        if len(self.bins)!=2 or any(len(row)!=2 or any(type(x) is not int for x in row) or row[0]<0 or not 0<=row[1]<=row[0] for row in self.bins):raise ValueError('invalid confidence counts')
        if self.observations!=sum(row[0] for row in self.bins) or self.observations<2 or not math.isfinite(self.brier) or not 0<=self.brier<=1:raise ValueError('invalid confidence statistics')
        if not self.model_id or not self.sample_sha256:raise ValueError('missing calibration provenance')

    @property
    def id(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def probability(self,score,as_of):
        if utc(as_of)<=self.calibrated_at:raise ValueError('confidence calibration unavailable at decision time')
        if not math.isfinite(score):raise ValueError('invalid score')
        row=self.bins[int(score>0)]
        n,k=row
        if n==0:return {'probability':None,'status':'NO_CALIBRATION_SUPPORT'}
        p=k/n;z=1.959963984540054;den=1+z*z/n
        centre=(p+z*z/(2*n))/den
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return {'probability':p,'interval_95':[centre-half,centre+half],
                'observations':n,'status':'HELDOUT_EMPIRICAL','calibration_id':self.id,
                'event':'net forward return > 0','calibrated_at':self.calibrated_at}


def fit_confidence(model,panel,*,calibrated_at):
    # Caller supplies net-of-cost targets and the same horizon used by the model.
    from fundamentals.models import FundamentalScoreEngine
    p=panel.sort_values('as_of').copy();at=utc(calibrated_at)
    if len(p)<2:raise ValueError('insufficient calibration observations')
    starts=[utc(x) for x in p.as_of];ends=[utc(x) for x in p.label_end]
    if any(a<=utc(model.trained_through) or a>=b or b>=at for a,b in zip(starts,ends)):
        raise ValueError('calibration must follow training and use matured labels')
    if any(a<=b for a,b in zip(starts[1:],ends[:-1])):raise ValueError('overlapping calibration labels')
    if 'horizon' not in p or not (p.horizon==model.horizon).all():raise ValueError('calibration horizon mismatch')
    if 'net_of_costs' not in p or not p.net_of_costs.map(lambda x:x is True).all():raise ValueError('explicit net-of-cost targets required')
    bins=[[0,0],[0,0]];pairs=[];engine=FundamentalScoreEngine(model)
    for row in p.to_dict('records'):
        target=row['target']
        if isinstance(target,bool) or not math.isfinite(target):raise ValueError('invalid target')
        score=engine.score(row,row['as_of'])['score']
        if score is None:raise ValueError('incomplete calibration features')
        group=int(score>0);outcome=int(target>0);bins[group][0]+=1;bins[group][1]+=outcome;pairs.append((group,outcome))
    # Diagnostic apparent calibration Brier; explicitly not an OOS evaluation.
    brier=sum((bins[g][1]/bins[g][0]-y)**2 for g,y in pairs)/len(pairs)
    digest=hashlib.sha256(p.to_csv(index=False).encode()).hexdigest()
    return ConfidenceCalibration(model.id,at,tuple(tuple(x) for x in bins),digest,len(p),brier)


def load_score_bundle(path):
    """Explicit research configuration; loading never approves deployment."""
    from pathlib import Path
    from fundamentals.models import ModelArtifact,FundamentalScoreEngine
    bundle=json.loads(Path(path).read_text())
    a=ModelArtifact(**bundle['model'])
    c=ConfidenceCalibration(**bundle['calibration']) if bundle.get('calibration') else None
    return FundamentalScoreEngine(a,c)
