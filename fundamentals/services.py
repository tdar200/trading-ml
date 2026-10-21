"""Application services keep ingestion, feature construction and risk separate."""
from fundamentals.providers import ingest_sec,ingest_alfred,ingest_json_records
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import MacroRegimeService


class FundamentalDataService:
    def __init__(self,store,cache='results/fundamentals/raw'):
        self.store=store;self.cache=cache
    def ingest_company(self,symbol,cik):return ingest_sec(self.store,symbol,cik,self.cache)
    def ingest_export(self,path):return ingest_json_records(self.store,path)
    def observations(self,symbol,as_of,observed_only=False):
        return self.store.as_of(symbol,as_of,observed_only=observed_only)
    def features(self,symbol,as_of,asset_class='equity'):
        return FundamentalFeatureEngine(self.store).snapshot(symbol,as_of,asset_class=asset_class)


class MacroDataService:
    def __init__(self,store,cache='results/fundamentals/raw'):
        self.store=store;self.cache=cache
    def ingest_vintage(self,series,vintage,units):return ingest_alfred(self.store,series,vintage,self.cache,units)
    def state(self,as_of):return MacroRegimeService(self.store).state(as_of)
