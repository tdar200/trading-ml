"""Track G (prefix bt) helpers: run the ETF research systems through PUBLIC entry points only.

Only the systems under test are imported (PointInTimeFundamentalStore, CombinedSignalEngine, PaperEngine via
replay_with_fundamentals, strategies.small_account.features). No calculation helper is used as an oracle.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import json
import subprocess
import sys
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'qa_reaudit_20261001' / 'bt'
REPRO = OUT / 'repro'
DB_COPY = OUT / 'fundamentals_copy.sqlite'
SRC_DB = ROOT / 'results' / 'fundamentals' / 'fundamentals.sqlite'
PRICES = ROOT / 'results' / 'small_account'
SYMBOLS = ('SPY', 'XLE', 'XLF', 'XLK', 'XLP', 'XLV')
SYSTEMS = ('technical', 'fundamental_observe', 'macro_filter_A_B', 'joint_C', 'joint_regime_D')
STORED = {
    'audit': ROOT / 'results' / 'fundamental_audit_20260930' / 'reproduction',
    'closure': ROOT / 'results' / 'fundamental_closure_20260930' / 'reproduction',
    'original': ROOT / 'results' / 'fundamentals' / 'experiment',
}


def ensure_db_copy():
    """Never touch the real DB: a plain file copy (the -wal is empty, no -shm copied)."""
    OUT.mkdir(parents=True, exist_ok=True)
    if not DB_COPY.exists():
        shutil.copyfile(SRC_DB, DB_COPY)
    return DB_COPY


def ensure_repro():
    """Run the real CLI (examples/research_fundamentals.py) into results/qa_reaudit_20261001/bt/repro if absent."""
    if (REPRO / 'research.json').exists():
        return REPRO
    ensure_db_copy()
    REPRO.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in __import__('os').environ.items()}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    subprocess.run([sys.executable, 'examples/research_fundamentals.py', '--database', str(DB_COPY), '--output', str(REPRO)],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    return REPRO


def load_prices(symbol):
    return pd.read_parquet(PRICES / (symbol + '.parquet'))


def open_store():
    from fundamentals.store import PointInTimeFundamentalStore
    return PointInTimeFundamentalStore(str(ensure_db_copy()), readonly=True)


def frozen_models(directory=REPRO):
    from fundamentals.models import ModelArtifact
    arts = json.loads((Path(directory) / 'frozen_models.json').read_text())
    out = {}
    for a in arts:
        out[(a['symbol'], a['model'])] = ModelArtifact(**{f.name: a[f.name] for f in fields(ModelArtifact)})
    return out


def build_combo(system, symbol, store, models, threshold=0.0):
    from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, MacroRegimeService
    from fundamentals.features import FundamentalFeatureEngine
    fe = FundamentalFeatureEngine(store)
    macro = MacroRegimeService(store)
    if system == 'technical':
        return None
    if system == 'fundamental_observe':
        return CombinedSignalEngine(fe, policy='observe')
    key = 'joint_C' if system == 'joint_regime_D' else system
    return CombinedSignalEngine(fe, FundamentalScoreEngine(models[(symbol, key)]), macro,
                                policy={'macro_filter_A_B': 'filter', 'joint_C': 'weighted', 'joint_regime_D': 'regime'}[system],
                                research_shadow=True, threshold=threshold)


def run_system(system, symbol, store, models, *, cost_multiple=1, begin=None, end=None, threshold=0.0, frame=None):
    from fundamentals.research import replay_with_fundamentals
    from strategies.small_account import Setup
    frame = load_prices(symbol) if frame is None else frame
    n = len(frame)
    begin = int(n * .75) if begin is None else begin
    end = n if end is None else end
    combo = build_combo(system, symbol, store, models, threshold=threshold)
    return replay_with_fundamentals(frame, Setup('bollinger', 20), symbol, begin, end, combo, cost_multiple=cost_multiple)


def curve_values(engine):
    return [c['equity'] for c in engine.curve]
