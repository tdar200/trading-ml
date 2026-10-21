"""Fitted models and descriptive macro/catalyst state, separate from execution."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from fundamentals.store import utc,consistent_read


@dataclass(frozen=True)
class ModelArtifact:
    features:tuple[str,...]
    means:tuple[float,...]
    scales:tuple[float,...]
    coefficients:tuple[float,...]
    intercept:float
    trained_through:str
    alpha:float
    training_rows:int
    training_sha256:str
    horizon:int
    deployment_approved:bool=False

    def __post_init__(self):
        n=len(self.features)
        if len(set(self.features))!=n or self.alpha<0 or self.horizon<=0 or self.training_rows<=0:raise ValueError('invalid model specification')
        if not n or any(len(x)!=n for x in (self.means,self.scales,self.coefficients)):raise ValueError('invalid model shape')
        if not all(math.isfinite(x) for x in (*self.means,*self.scales,*self.coefficients,self.intercept,self.alpha)) or min(self.scales)<=0:raise ValueError('invalid model values')
        utc(self.trained_through)
        if self.deployment_approved:raise ValueError('research artifacts cannot self-approve deployment')

    @property
    def id(self):return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()


def fit_model(panel,features,*,fit_at,alpha=1.,horizon=5):
    """fit_at is the simulated training clock; labels must already have matured."""
    ts=pd.Timestamp(utc(fit_at)); p=panel.copy()
    if not {'as_of','label_end','target'}.issubset(p):raise ValueError('timestamped labels required')
    if (pd.to_datetime(p.label_end,utc=True)>=ts).any():raise ValueError('unmatured labels at model fit time')
    if (pd.to_datetime(p.as_of,utc=True)>=pd.to_datetime(p.label_end,utc=True)).any():raise ValueError('invalid label intervals')
    x=p[list(features)].to_numpy(float);y=p.target.to_numpy(float)
    if len(p)<max(20,5*len(features)) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('insufficient complete finite training data')
    means=x.mean(axis=0); scales=x.std(axis=0);scales=np.where(scales>1e-12,scales,1.)
    z=(x-means)/scales; model=Ridge(alpha=alpha).fit(z,y)
    digest=hashlib.sha256(p[['as_of','label_end',*features,'target']].to_csv(index=False).encode()).hexdigest()
    return ModelArtifact(tuple(features),tuple(map(float,means)),tuple(map(float,scales)),tuple(map(float,model.coef_)),
                         float(model.intercept_),utc(ts),float(alpha),len(p),digest,horizon)


class FundamentalScoreEngine:
    def __init__(self,artifact=None,calibration=None):
        self.artifact=artifact;self.calibration=calibration
        if calibration is not None and (artifact is None or calibration.model_id!=artifact.id):raise ValueError('calibration model mismatch')
    def score(self,features,as_of):
        a=self.artifact
        if a is None:return {'score':None,'confidence':None,'status':'UNVALIDATED','components':{},'deployment_approved':False}
        if pd.Timestamp(utc(as_of))<=pd.Timestamp(a.trained_through):raise ValueError('model unavailable at historical timestamp')
        missing=[k for k in a.features if features.get(k) is None or not math.isfinite(features[k])]
        if missing:return {'score':None,'confidence':None,'status':'MISSING_FEATURES','missing':missing,'components':{},'deployment_approved':False}
        contributions={name:(features[name]-mean)/scale*weight for name,mean,scale,weight in zip(a.features,a.means,a.scales,a.coefficients)}
        result=float(a.intercept+sum(contributions.values()))
        if not math.isfinite(result) or not all(math.isfinite(v) for v in contributions.values()):
            return {'score':None,'confidence':None,'status':'INVALID_MODEL_OUTPUT','components':{},'deployment_approved':False}
        confidence=self.calibration.probability(result,as_of) if self.calibration else None
        return {'score':result,'score_units':f'predicted {a.horizon}-session return, not probability',
                'confidence':confidence['probability'] if confidence else None,'confidence_details':confidence,'status':'RESEARCH_MODEL','components':contributions,'intercept':a.intercept,'model_id':a.id,
                'trained_through':a.trained_through,'deployment_approved':False}


class MacroRegimeService:
    """Continuous states; zero-change thresholds have literal economic meaning.

    No composite regime score or risk multiplier is invented. Vintage data only.
    """
    def __init__(self,store,max_vintage_age=100):
        # NaN would silently disable the STALE gate (every comparison is False); non-positive would mark everything stale.
        if isinstance(max_vintage_age,bool) or not isinstance(max_vintage_age,(int,float)) or not math.isfinite(max_vintage_age) or max_vintage_age<=0:
            raise ValueError('max_vintage_age must be a positive finite number')
        self.store=store;self.max_age=max_vintage_age
    @consistent_read
    def state(self,as_of):
        ts=pd.Timestamp(utc(as_of)); output={'as_of':utc(as_of),'features':{},'lineage':[],'status':'AVAILABLE','score':None,'confidence':None}
        for series in ('CPIAUCSL','UNRATE','FEDFUNDS'):
            rows=self.store.as_of('MACRO:'+series,as_of,kind='macro')
            rows=[r for r in rows if r['value'] is not None]
            if not rows:
                if output['status']!='STALE':output['status']='INCOMPLETE'
                continue
            expected_units='index' if series=='CPIAUCSL' else 'percent'
            if any(r['units']!=expected_units or r['currency'] for r in rows) or len({r['source'] for r in rows})>1:
                raise ValueError('ambiguous macro source or incorrect units')
            latest=max(rows,key=lambda r:r['period_end']); age=(ts-pd.Timestamp(latest['available_at'])).days
            if age>self.max_age:output['status']='STALE';continue
            def lag(months):
                target=pd.Timestamp(latest['period_end'])-pd.DateOffset(months=months)
                match=[r for r in rows if pd.Timestamp(r['period_end']).year==target.year and pd.Timestamp(r['period_end']).month==target.month]
                if not match:return None
                item=max(match,key=lambda r:r['available_at']);output['lineage'].append(item['id']);return item['value']
            current=latest['value'];output['lineage'].append(latest['id'])
            if series=='CPIAUCSL':
                prior=lag(12);three=lag(3);fifteen=lag(15)
                inflation=current/prior-1 if prior and prior>0 else None
                previous=three/fifteen-1 if three is not None and fifteen and fifteen>0 else None
                output['features']['inflation_yoy']=inflation
                output['features']['inflation_change_3m']=inflation-previous if inflation is not None and previous is not None else None
            else:
                previous=lag(3);name='unemployment' if series=='UNRATE' else 'policy_rate'
                output['features'][name]=current
                output['features'][name+'_change_3m']=current-previous if previous is not None else None
        f=output['features']
        def direction(key,up,down):
            # A difference of two ratios carries ~1e-16 float noise (equal CPI ratios must be UNCHANGED, not RISING).
            x=f.get(key);return 'UNKNOWN' if x is None else up if x>1e-12 else down if x<-1e-12 else 'UNCHANGED'
        output['regime']={'inflation':direction('inflation_change_3m','RISING','FALLING'),
                          'labour_market':direction('unemployment_change_3m','WEAKENING','STRENGTHENING'),
                          'policy':direction('policy_rate_change_3m','TIGHTENING','EASING')}
        if output['status']!='STALE' and (len(f)!=6 or any(v is None for v in f.values())):output['status']='INCOMPLETE'
        return output


MacroRegimeEngine=MacroRegimeService
MacroDataService=MacroRegimeService


def _finite_number(x):
    return float(x) if isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) else None


class EventCatalystEngine:
    def __init__(self,store,half_life_days=None):
        # Decay is optional research configuration, not a default fitted edge.
        if half_life_days is not None and (not math.isfinite(half_life_days) or half_life_days<=0):raise ValueError('invalid decay')
        self.store=store;self.half_life=half_life_days
    @consistent_read
    def state(self,symbol,as_of):
        ts=pd.Timestamp(utc(as_of));events=self.store.events(symbol,as_of); structured=[]
        for e in events:
            age=max(0.,(ts-pd.Timestamp(e['available_at'])).total_seconds()/86400)
            e={**e,'age_days':age,'decay':math.exp(-math.log(2)*age/self.half_life) if self.half_life else None}
            payload=e['payload'];actual=payload.get('reported_eps');consensus=payload.get('consensus_eps')
            if actual is not None and consensus is not None:
                # One malformed descriptive record must flag itself, never abort every later decision.
                a,c=_finite_number(actual),_finite_number(consensus);estimate_at=payload.get('consensus_available_at')
                try:known_before=bool(estimate_at) and pd.Timestamp(utc(estimate_at))<pd.Timestamp(e['event_at'])
                except (ValueError,TypeError):known_before=None
                if a is None or c is None:e['eps_surprise']=None;e['quality_flag']='INVALID_EARNINGS_PAYLOAD'
                elif known_before is None:e['eps_surprise']=None;e['quality_flag']='INVALID_CONSENSUS_TIMESTAMP'
                elif not known_before:e['eps_surprise']=None;e['quality_flag']='NO_PRE_EVENT_CONSENSUS'
                else:e['eps_surprise']=(a-c)/abs(c) if c else None
            structured.append(e)
        return {'events':structured,'score':None,'confidence':None,'status':'DESCRIPTIVE_ONLY' if events else 'MISSING',
                'note':'No sentiment model or decay is approved for execution'}


CatalystEngine=EventCatalystEngine


class CombinedSignalEngine:
    """Explicit fusion policies; a decision is never an order or risk approval."""
    def __init__(self,features,score=None,macro=None,catalysts=None,*,policy='observe',threshold=0.,research_shadow=False):
        if policy not in ('observe','filter','selection','weighted','regime'):raise ValueError('invalid fusion policy')
        if not math.isfinite(threshold):raise ValueError('invalid threshold')
        self.features=features;self.score=score or FundamentalScoreEngine();self.macro=macro;self.catalysts=catalysts
        if any(x is not None and x.store is not features.store for x in (macro,catalysts)):raise ValueError('combined decision requires one consistent database snapshot')
        self.policy=policy;self.threshold=threshold;self.shadow=research_shadow
    @consistent_read
    def decide(self,symbol,as_of,technical,*,asset_class='equity'):
        fundamental=self.features.snapshot(symbol,as_of,asset_class=asset_class)
        macro=self.macro.state(as_of) if self.macro else {'features':{},'regime':'UNKNOWN','status':'MISSING'}
        catalyst=self.catalysts.state(symbol,as_of) if self.catalysts else {'events':[],'score':None,'status':'MISSING'}
        inputs={**fundamental['features'],**{'macro_'+k:v for k,v in macro.get('features',{}).items()},
                'technical_signal':float(bool(technical.get('signal')))}
        scored=self.score.score(inputs,as_of)
        permit=False;why=[]
        if self.policy=='observe':
            permit=bool(technical.get('signal'));why.append('Fundamental observation only; technical decision unchanged')
        elif not self.shadow:why.append('No fundamental model approved for deployment')
        elif scored['score'] is None:why.append('Missing or unvalidated model/feature coverage')
        elif fundamental['status']!='AVAILABLE' and any(not k.startswith(('macro_','technical_')) for k in self.score.artifact.features):why.append('Fundamentals missing/stale or wrong instrument type')
        elif self.policy=='regime' and macro['status']!='AVAILABLE':why.append('Macro vintage coverage missing/stale')
        else:
            # A: fundamentals filter technical entries. B: ranked selection still
            # waits for technical confirmation. C/D require a fitted joint model.
            permit=bool(technical.get('signal')) and scored['score']>self.threshold
            joint_vetoed=self.policy in ('weighted','regime') and self.score.artifact is not None and 'technical_signal' not in self.score.artifact.features
            # The pass/fail sentence must describe the FINAL outcome, so it is skipped when the joint-model veto applies.
            if not joint_vetoed:why.append('Fitted research prediction exceeds frozen threshold' if permit else 'Technical/fundamental contradiction or prediction below threshold')
        if self.policy in ('weighted','regime') and self.score.artifact and 'technical_signal' not in self.score.artifact.features:
            permit=False;why.append('Joint model must contain technical input; no arbitrary fusion weights')
        snapshot_id=hashlib.sha256(json.dumps({'fundamental':fundamental,'macro':macro,'catalysts':catalyst},sort_keys=True,allow_nan=False).encode()).hexdigest()
        configuration={'policy':self.policy,'threshold':self.threshold,'research_shadow':self.shadow,
            'feature_version':fundamental['version'],'max_period_age_days':self.features.max_age,
            'macro_max_age':self.macro.max_age if self.macro else None,'catalyst_half_life':self.catalysts.half_life if self.catalysts else None,
            'model':self.score.artifact.__dict__ if self.score.artifact else None,
            'calibration':self.score.calibration.__dict__ if self.score.calibration else None}
        return {'symbol':symbol,'timestamp':utc(as_of),'data_snapshot_id':snapshot_id,'configuration':configuration,'technical':{'signal':bool(technical.get('signal'))},
                'fundamental':{**scored,'features':fundamental['features'],'flags':fundamental['flags'],'coverage':fundamental['completeness'],
                               'data_status':fundamental['status'],'lineage':fundamental['lineage'],'raw':fundamental['raw'],
                               'feature_version':fundamental['version'],'available_at':fundamental.get('available_at'),'period_end':fundamental.get('period_end')},
                'macro':macro,'catalysts':catalyst,'risk':{'status':'NOT_EVALUATED','flags':fundamental['flags']},
                'decision':'RESEARCH_CANDIDATE' if permit and self.policy!='observe' else 'TECHNICAL_CANDIDATE' if permit else 'NO_TRADE',
                'permit_signal':permit,'positionSize':None,'policy':self.policy,'deployment_approved':False,'explanation':why}


FundamentalStrategyFilter=CombinedSignalEngine
