# Stage 0 + 1: Skeleton & Alpaca → Nautilus Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Python/NautilusTrader project that fetches daily bars for 12 US ETFs from Alpaca and persists them into a Nautilus ParquetDataCatalog, idempotently, with tests that verify the catalog is replayable through a BacktestEngine and matches an independent data source within 0.5%.

**Architecture:** Single project, uv-managed Python 3.12. Alpaca's free market-data API is the source; data flows through a thin pure-function converter into Nautilus's local Parquet catalog. The catalog is the system's source of truth — Alpaca is hit once per ticker-date. The ingest pipeline is split into pure-function pieces (testable without network) and a thin orchestrator (integration-tested).

**Tech Stack:**
- Python 3.12 (`uv`-managed)
- `nautilus_trader==1.227.0` — engine, data catalog
- `alpaca-py==0.43.4` — market data client
- `pandas`, `polars` — DataFrame wrangling
- `yfinance` — independent data source for the cross-check test only
- `pytest`, `pytest-mock` — testing

---

## File Structure

Files created in this plan (all under `/home/tahseen-dar/Projects/trading-ml/`):

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv project config, pinned deps, Python 3.12 |
| `.env.example` | Documents required env vars (Alpaca keys) |
| `README.md` | Run instructions + project orientation |
| `docs/nautilus_api_notes.md` | Recon notes locking in exact Nautilus v1.227 API symbols |
| `examples/smoke.py` | Stage-0 smoke backtest — proves Nautilus works |
| `data/__init__.py` | Package marker |
| `data/tickers.py` | `UNIVERSE` constant (12-ticker list) |
| `data/instruments.py` | Pure: builds Nautilus `Equity` instruments for the universe |
| `data/bar_converter.py` | Pure: Alpaca bars DataFrame → list of Nautilus `Bar` objects |
| `data/alpaca_client.py` | Thin Alpaca wrapper (fetch_bars function) |
| `data/ingest_alpaca.py` | Orchestrator: fetch → convert → write to catalog (idempotent) |
| `data/catalog/` | Nautilus ParquetDataCatalog output (gitignored) |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Shared fixtures (sample Alpaca DataFrame, tmp catalog) |
| `tests/test_tickers.py` | Universe sanity |
| `tests/test_instruments.py` | Instrument builder |
| `tests/test_bar_converter.py` | DataFrame → Bar conversion |
| `tests/test_ingest.py` | Ingest orchestrator with mocked Alpaca client; idempotency |
| `tests/test_catalog_integration.py` | End-to-end: ingest small slice → load catalog → replay through BacktestEngine |
| `scripts/cross_check_yfinance.py` | Runnable script: Alpaca close vs yfinance close < 0.5% drift |

**Boundaries — why this decomposition:**
- `alpaca_client` isolates all network IO into one place. Everywhere else is mockable.
- `bar_converter` is a pure function — no IO, fully unit-testable.
- `instruments` is pure — Nautilus `Equity` construction is fiddly enough to deserve its own file with its own test.
- `ingest_alpaca` is a thin orchestrator: client + converter + catalog writer + idempotency check. The pieces it composes are each independently tested.

---

## Task 1: Project skeleton with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `data/__init__.py`, `tests/__init__.py`, `examples/`, `docs/`, `scripts/`
- Modify: `.gitignore` (already exists — verify entries present)

- [ ] **Step 1: Initialize uv project and create directory skeleton**

Run from `/home/tahseen-dar/Projects/trading-ml/`:
```bash
uv init --python 3.12 --no-readme --no-workspace
mkdir -p data tests examples scripts docs/nautilus
touch data/__init__.py tests/__init__.py
```

- [ ] **Step 2: Replace generated `pyproject.toml` with the project spec**

Overwrite `pyproject.toml` with:
```toml
[project]
name = "trading-ml"
version = "0.1.0"
description = "ML + trading research lab on NautilusTrader with Alpaca data"
requires-python = ">=3.12,<3.15"
dependencies = [
    "nautilus_trader==1.227.0",
    "alpaca-py==0.43.4",
    "pandas>=2.2",
    "polars>=1.0",
    "python-dotenv>=1.0",
    "yfinance>=0.2",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 3: Sync dependencies**

```bash
uv sync
```

Expected: lockfile created, `.venv/` populated, no resolution errors. If `nautilus_trader==1.227.0` fails to resolve on this platform, note the failure and consult the Nautilus README's platform table before proceeding.

- [ ] **Step 4: Verify versions**

```bash
uv run python -c "import nautilus_trader, alpaca; print('nautilus_trader', nautilus_trader.__version__); print('alpaca-py OK')"
```

Expected output contains `nautilus_trader 1.227.0`.

- [ ] **Step 5: Create `.env.example`**

```bash
cat > .env.example <<'EOF'
# Alpaca paper-trading API keys (free; sign up at https://alpaca.markets/)
# Copy this file to .env and fill in real values. .env is gitignored.
ALPACA_API_KEY=
ALPACA_API_SECRET=
EOF
```

- [ ] **Step 6: Verify `.gitignore` has the needed entries**

Check `.gitignore` contains at minimum: `.venv/`, `.env`, `__pycache__/`, `data/catalog/`, `.pytest_cache/`. Add any missing entries.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .env.example data/__init__.py tests/__init__.py .gitignore
git commit -m "chore: uv project skeleton with pinned deps

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Nautilus API reconnaissance

The exact module paths and constructor signatures for `Bar`, `BarType`, `Equity`, and `ParquetDataCatalog` in v1.227 must be verified before code is written against them. This task locks them in.

**Files:**
- Create: `docs/nautilus_api_notes.md`

- [ ] **Step 1: Fetch the current Nautilus quick-start docs**

```bash
curl -sL -H "User-Agent: trading-ml" "https://nautilustrader.io/docs/latest/getting_started/quickstart" -o /tmp/quickstart.html
curl -sL -H "User-Agent: trading-ml" "https://nautilustrader.io/docs/latest/concepts/data" -o /tmp/data_concepts.html
curl -sL -H "User-Agent: trading-ml" "https://nautilustrader.io/docs/latest/api_reference/persistence" -o /tmp/persistence_api.html
```

Read the three files and extract: (a) how to construct a `BacktestEngine` with a minimal config, (b) the constructor signature of `Bar`, `BarType`, `InstrumentId`, `Symbol`, `Equity`, and (c) how to instantiate `ParquetDataCatalog` and call its write/query methods.

- [ ] **Step 2: Verify symbols are importable in this env**

```bash
uv run python -c "
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
print('all imports OK')
"
```

Expected: `all imports OK`. If any import fails, that's the v1.227 API drift — locate the new path via `python -c "import nautilus_trader; help(nautilus_trader)"` exploration or the docs and record the corrected path.

- [ ] **Step 3: Write `docs/nautilus_api_notes.md`**

The file MUST capture, with code examples copied from the verification above:
- Exact import paths for `Bar`, `BarType`, `InstrumentId`, `Symbol`, `Venue`, `Equity`, `Price`, `Quantity`, `ParquetDataCatalog`, `BacktestEngine`.
- The `BarType.from_str()` string format used for daily US-equity bars (e.g. `"SPY.ARCX-1-DAY-LAST-EXTERNAL"`). Confirm the aggregation source token for ingested-from-external-source data.
- The `Equity(...)` constructor signature, all required args, and their types.
- The `ParquetDataCatalog(...)` constructor and the method names for writing data and querying bars.
- The expected venue MIC code for US ETFs (XLE/XLF/etc. trade on NYSE Arca → `ARCX`; SPY also Arca). If the docs are ambiguous, use `ARCX` for all 12 tickers in this universe — they are all NYSE-Arca-listed ETFs.

Template the file as a reference document the later tasks read from. Example skeleton:
```markdown
# Nautilus v1.227 API notes

Verified 2026-05-26 against pip-installed nautilus_trader 1.227.0.

## Imports
```python
from nautilus_trader.model.data import Bar, BarType
# ... etc
```

## Daily equity BarType
Format: `"<SYMBOL>.<VENUE>-1-DAY-LAST-EXTERNAL"` — `EXTERNAL` aggregation source
because we ingest pre-aggregated daily bars from Alpaca.

## Bar constructor
```python
Bar(bar_type=..., open=Price(...), high=Price(...), low=Price(...),
    close=Price(...), volume=Quantity(...), ts_event=int_ns, ts_init=int_ns)
```

## Equity constructor
[Fill in with actual signature from docs/source]

## ParquetDataCatalog
```python
catalog = ParquetDataCatalog(path="data/catalog")
catalog.write_data([bar1, bar2, ...])
bars = catalog.bars(instrument_ids=["SPY.ARCX"])  # confirm exact method name
```
```

- [ ] **Step 4: Commit**

```bash
git add docs/nautilus_api_notes.md
git commit -m "docs: lock in Nautilus v1.227 API symbols for ingestion

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Smoke backtest (Stage 0 milestone)

Proves the engine works end-to-end before any project-specific code.

**Files:**
- Create: `examples/smoke.py`

- [ ] **Step 1: Write a minimal Nautilus backtest example**

Use the import paths verified in `docs/nautilus_api_notes.md`. The simplest viable smoke test is a `BacktestEngine` that runs against a small in-memory dataset (or uses Nautilus's built-in test data if available); a strategy that subscribes to bars and logs every bar is sufficient — no trading needed.

Create `examples/smoke.py`:
```python
"""
Stage-0 smoke test for trading-ml.

Builds a BacktestEngine, feeds it a handful of synthetic daily bars for SPY,
runs a do-nothing Strategy that logs each bar received, and prints the final
account balance. If this script exits 0 and prints bars, the Nautilus install
and the API paths in docs/nautilus_api_notes.md are correct.
"""
from decimal import Decimal

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class LogBarsConfig(StrategyConfig, frozen=True):
    bar_type: BarType


class LogBars(Strategy):
    def __init__(self, config: LogBarsConfig) -> None:
        super().__init__(config)

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.log.info(f"bar: {bar}")


def build_spy_instrument() -> Equity:
    # Constructor args confirmed against docs/nautilus_api_notes.md
    return Equity(
        instrument_id=InstrumentId.from_str("SPY.ARCX"),
        raw_symbol=Symbol("SPY"),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        isin=None,
        margin_init=Decimal("0"),
        margin_maint=Decimal("0"),
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
        ts_event=0,
        ts_init=0,
    )


def build_synthetic_bars(bar_type: BarType, n: int = 5) -> list[Bar]:
    bars: list[Bar] = []
    base_ts = 1_700_000_000_000_000_000  # arbitrary nanos
    for i in range(n):
        ts = base_ts + i * 86_400_000_000_000  # +1 day per bar
        bars.append(Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{500 + i}.00"),
            high=Price.from_str(f"{500 + i}.50"),
            low=Price.from_str(f"{499 + i}.50"),
            close=Price.from_str(f"{500 + i}.25"),
            volume=Quantity.from_int(1_000_000),
            ts_event=ts,
            ts_init=ts,
        ))
    return bars


def main() -> None:
    instrument = build_spy_instrument()
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")

    config = BacktestEngineConfig(
        trader_id=TraderId("SMOKE-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    engine = BacktestEngine(config=config)

    venue = Venue("ARCX")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )
    engine.add_instrument(instrument)
    engine.add_data(build_synthetic_bars(bar_type))
    engine.add_strategy(LogBars(config=LogBarsConfig(bar_type=bar_type)))
    engine.run()

    account = engine.cache.account_for_venue(venue)
    print(f"final balance: {account.balance_total(USD)}")


if __name__ == "__main__":
    main()
```

**IF** the verification in Task 2 revealed different signatures (e.g. `Equity` no longer takes `margin_init` / `maker_fee`, or `BacktestEngineConfig` has renamed fields), **adapt this file to match `docs/nautilus_api_notes.md`** — do not blindly use these lines if they conflict with the verified API.

- [ ] **Step 2: Run smoke**

```bash
uv run python examples/smoke.py
```

Expected: prints log lines for each synthetic bar, then `final balance: 100000.00 USD` (or similar — exact format depends on Money's repr).

If it crashes, the failure is almost certainly an API-drift mismatch with v1.227 — fix the file using `docs/nautilus_api_notes.md` and re-run until it exits 0.

- [ ] **Step 3: Commit**

```bash
git add examples/smoke.py
git commit -m "feat(examples): Stage-0 smoke backtest proves engine works

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Universe constant + test

**Files:**
- Create: `tests/test_tickers.py`
- Create: `data/tickers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tickers.py`:
```python
from data.tickers import UNIVERSE, BENCHMARK, ALL_TICKERS


def test_universe_has_eleven_sector_etfs():
    assert len(UNIVERSE) == 11


def test_benchmark_is_spy():
    assert BENCHMARK == "SPY"


def test_all_tickers_includes_benchmark():
    assert BENCHMARK in ALL_TICKERS
    assert set(UNIVERSE).issubset(ALL_TICKERS)
    assert len(ALL_TICKERS) == 12


def test_tickers_uppercase_no_duplicates():
    for t in ALL_TICKERS:
        assert t == t.upper()
        assert " " not in t
    assert len(ALL_TICKERS) == len(set(ALL_TICKERS))


def test_universe_covers_all_sectors():
    # The 11 SPDR sector ETFs as of 2026 (XLC added in 2018, XLRE in 2015)
    expected = {"XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"}
    assert set(UNIVERSE) == expected
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_tickers.py -v
```

Expected: ImportError / ModuleNotFoundError for `data.tickers`.

- [ ] **Step 3: Implement**

Create `data/tickers.py`:
```python
"""
The 12-ticker universe for trading-ml.

UNIVERSE: the 11 S&P sector SPDR ETFs (long/trade candidates).
BENCHMARK: SPY (used for performance comparison, not trading).
ALL_TICKERS: union of the above — what the data pipeline ingests.

All 12 tickers are NYSE-Arca-listed and existed for the entire 2016→ window
chosen in the design spec, so survivorship bias is sidestepped at this stage.
"""

UNIVERSE: tuple[str, ...] = (
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
)

BENCHMARK: str = "SPY"

ALL_TICKERS: tuple[str, ...] = UNIVERSE + (BENCHMARK,)
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_tickers.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add data/tickers.py tests/test_tickers.py
git commit -m "feat(data): define 12-ticker universe (11 SPDRs + SPY)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Instrument builder

**Files:**
- Create: `tests/test_instruments.py`
- Create: `data/instruments.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_instruments.py`:
```python
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Equity

from data.instruments import build_equity, build_universe_instruments
from data.tickers import ALL_TICKERS


def test_build_equity_returns_equity_with_correct_id():
    eq = build_equity("SPY")
    assert isinstance(eq, Equity)
    assert eq.id == InstrumentId.from_str("SPY.ARCX")
    assert str(eq.raw_symbol) == "SPY"


def test_build_equity_price_precision_is_two():
    eq = build_equity("XLF")
    assert eq.price_precision == 2


def test_build_universe_instruments_returns_twelve():
    instruments = build_universe_instruments()
    assert len(instruments) == len(ALL_TICKERS)
    ids = {str(eq.id) for eq in instruments}
    assert ids == {f"{t}.ARCX" for t in ALL_TICKERS}
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_instruments.py -v
```

Expected: ImportError for `data.instruments`.

- [ ] **Step 3: Implement**

Create `data/instruments.py`. **Use the exact `Equity` constructor signature recorded in `docs/nautilus_api_notes.md`**. The example below uses the constructor I expect; correct it if v1.227 differs:
```python
"""
Nautilus Equity instrument builders for the trading-ml universe.

All 12 ETFs are constructed identically (USD, 2dp prices, $0.01 increment).
`build_universe_instruments()` returns one Equity per ticker in ALL_TICKERS.
"""
from decimal import Decimal

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from data.tickers import ALL_TICKERS

VENUE_MIC: str = "ARCX"  # NYSE Arca — all 12 ETFs are Arca-listed


def build_equity(ticker: str) -> Equity:
    """Build a Nautilus Equity instrument for a single ticker."""
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        isin=None,
        margin_init=Decimal("0"),
        margin_maint=Decimal("0"),
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
        ts_event=0,
        ts_init=0,
    )


def build_universe_instruments() -> list[Equity]:
    """Build Equity instruments for every ticker in ALL_TICKERS."""
    return [build_equity(t) for t in ALL_TICKERS]
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_instruments.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add data/instruments.py tests/test_instruments.py
git commit -m "feat(data): Equity instrument builders for universe

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Alpaca bars → Nautilus Bar converter (pure function)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_bar_converter.py`
- Create: `data/bar_converter.py`

- [ ] **Step 1: Create shared test fixtures**

Create `tests/conftest.py`:
```python
"""Shared pytest fixtures for trading-ml tests."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_alpaca_bars_df() -> pd.DataFrame:
    """
    Three days of daily bars for SPY in the shape alpaca-py returns when
    .df is called on a StockBarsRequest result (MultiIndex on symbol+timestamp).
    """
    idx = pd.MultiIndex.from_tuples(
        [
            ("SPY", pd.Timestamp("2024-01-02 05:00:00+00:00")),
            ("SPY", pd.Timestamp("2024-01-03 05:00:00+00:00")),
            ("SPY", pd.Timestamp("2024-01-04 05:00:00+00:00")),
        ],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open":   [470.50, 471.20, 469.10],
            "high":   [472.00, 472.10, 470.80],
            "low":    [469.80, 470.30, 467.50],
            "close":  [471.10, 470.40, 468.90],
            "volume": [50_000_000, 48_000_000, 52_000_000],
            "trade_count": [400_000, 380_000, 410_000],
            "vwap":   [471.00, 471.10, 469.20],
        },
        index=idx,
    )
```

- [ ] **Step 2: Write the failing converter test**

Create `tests/test_bar_converter.py`:
```python
from nautilus_trader.model.data import Bar, BarType

from data.bar_converter import alpaca_df_to_bars


def test_converter_returns_one_bar_per_row(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    assert len(bars) == 3
    assert all(isinstance(b, Bar) for b in bars)


def test_converter_preserves_ohlcv_values(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    first = bars[0]
    assert str(first.open) == "470.50"
    assert str(first.high) == "472.00"
    assert str(first.low) == "469.80"
    assert str(first.close) == "471.10"
    assert int(first.volume) == 50_000_000


def test_converter_sets_correct_bar_type(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    expected_str = "SPY.ARCX-1-DAY-LAST-EXTERNAL"
    assert str(bars[0].bar_type) == expected_str


def test_converter_timestamps_monotonic_and_nanos(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    ts = [b.ts_event for b in bars]
    assert ts == sorted(ts)
    # ts_event must be int nanoseconds (Unix epoch)
    assert all(isinstance(t, int) for t in ts)
    assert all(t > 10**18 for t in ts)  # 2001-09-09 onwards in ns


def test_converter_ts_init_equals_ts_event(sample_alpaca_bars_df):
    bars = alpaca_df_to_bars(sample_alpaca_bars_df, "SPY")
    for b in bars:
        assert b.ts_init == b.ts_event


def test_converter_filters_by_symbol_in_multi_symbol_df(sample_alpaca_bars_df):
    # Build a frame with two symbols, ask for just SPY
    other = sample_alpaca_bars_df.copy()
    other.index = other.index.set_levels(["XLF"], level=0)
    import pandas as pd
    combined = pd.concat([sample_alpaca_bars_df, other])
    bars = alpaca_df_to_bars(combined, "SPY")
    assert len(bars) == 3
    assert all(str(b.bar_type).startswith("SPY") for b in bars)


def test_converter_empty_df_returns_empty_list():
    import pandas as pd
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
    )
    assert alpaca_df_to_bars(empty, "SPY") == []
```

- [ ] **Step 3: Run to confirm failure**

```bash
uv run pytest tests/test_bar_converter.py -v
```

Expected: collection error (no `data.bar_converter`).

- [ ] **Step 4: Implement the converter**

Create `data/bar_converter.py`:
```python
"""
Pure conversion from an Alpaca bars DataFrame to a list of Nautilus Bar objects.

Alpaca returns daily bars indexed on (symbol, timestamp) with the columns
open/high/low/close/volume (and trade_count, vwap). We slice by symbol, then
map each row to a Nautilus Bar with a fixed BarType of
"<TICKER>.ARCX-1-DAY-LAST-EXTERNAL".

No IO, no Alpaca SDK dependency — this stays unit-testable.
"""
from __future__ import annotations

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from data.instruments import VENUE_MIC


def _bar_type_for(ticker: str) -> BarType:
    return BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")


def _to_nanos(ts: pd.Timestamp) -> int:
    # Alpaca timestamps are tz-aware UTC; pandas .value gives ns since epoch.
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value)


def alpaca_df_to_bars(df: pd.DataFrame, ticker: str) -> list[Bar]:
    """Convert the (symbol, timestamp)-indexed Alpaca bars DataFrame for one
    ticker into a list of Nautilus Bar objects. Other symbols in the frame
    are ignored."""
    if df.empty:
        return []
    if ticker not in df.index.get_level_values(0):
        return []

    rows = df.xs(ticker, level=0)
    bar_type = _bar_type_for(ticker)
    bars: list[Bar] = []
    for ts, row in rows.iterrows():
        ts_ns = _to_nanos(ts)
        bars.append(Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{float(row['open']):.2f}"),
            high=Price.from_str(f"{float(row['high']):.2f}"),
            low=Price.from_str(f"{float(row['low']):.2f}"),
            close=Price.from_str(f"{float(row['close']):.2f}"),
            volume=Quantity.from_int(int(row["volume"])),
            ts_event=ts_ns,
            ts_init=ts_ns,
        ))
    return bars
```

- [ ] **Step 5: Run to confirm pass**

```bash
uv run pytest tests/test_bar_converter.py -v
```

Expected: 7 passed. If the `Price.from_str` precision check fails, the converter's `f"{...:.2f}"` formatting may need to align with what Nautilus's Price requires for the equity (2 dp). Adjust as needed.

- [ ] **Step 6: Commit**

```bash
git add data/bar_converter.py tests/conftest.py tests/test_bar_converter.py
git commit -m "feat(data): pure converter from Alpaca DataFrame to Nautilus Bars

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Alpaca client wrapper

**Files:**
- Create: `data/alpaca_client.py`

(No dedicated test file — this is a thin wrapper. It's exercised in Task 8's mocked ingest test and Task 11's live integration test.)

- [ ] **Step 1: Implement the wrapper**

Create `data/alpaca_client.py`:
```python
"""
Thin wrapper around alpaca-py's StockHistoricalDataClient for daily bars.

Isolates all network IO into one place so the rest of the pipeline is
mockable. Reads credentials from environment variables; raises a clear
error if they are missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaCredentialsError(RuntimeError):
    """Raised when ALPACA_API_KEY / ALPACA_API_SECRET are not in the env."""


@dataclass(frozen=True)
class AlpacaClient:
    """Wraps StockHistoricalDataClient. Use AlpacaClient.from_env() in app code."""
    client: StockHistoricalDataClient

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise AlpacaCredentialsError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set "
                "(copy .env.example to .env and fill in values)"
            )
        return cls(client=StockHistoricalDataClient(key, secret))

    def fetch_daily_bars(
        self,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch adjusted (splits+dividends) daily bars for the given tickers
        in [start, end]. Returns the Alpaca .df result (MultiIndex on
        symbol+timestamp). Empty DataFrame if no bars are returned.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        request = StockBarsRequest(
            symbol_or_symbols=list(tickers),
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",  # apply splits AND dividends
            feed="iex",        # free-tier compatible
        )
        bars = self.client.get_stock_bars(request)
        if bars.df is None or bars.df.empty:
            return pd.DataFrame()
        return bars.df
```

- [ ] **Step 2: Verify the import path resolves**

```bash
uv run python -c "from data.alpaca_client import AlpacaClient, AlpacaCredentialsError; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add data/alpaca_client.py
git commit -m "feat(data): thin Alpaca client wrapper for daily bars

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Ingest orchestrator + idempotency (mocked)

**Files:**
- Create: `tests/test_ingest.py`
- Create: `data/ingest_alpaca.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:
```python
"""
Tests for the ingest_alpaca orchestrator using a mocked AlpacaClient.

Live Alpaca calls are covered by the integration test (test_catalog_integration.py)
and the cross-check script (scripts/cross_check_yfinance.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from data.ingest_alpaca import ingest, latest_ingested_date


@pytest.fixture
def fake_client(sample_alpaca_bars_df):
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], datetime, datetime]] = []

        def fetch_daily_bars(self, tickers, start, end):
            self.calls.append((tuple(tickers), start, end))
            # Replay the fixture's bars for every requested ticker.
            frames = []
            for t in tickers:
                df = sample_alpaca_bars_df.copy()
                df.index = df.index.set_levels([t], level=0)
                frames.append(df)
            return pd.concat(frames) if frames else pd.DataFrame()

    return FakeClient()


def test_ingest_writes_bars_to_fresh_catalog(tmp_path: Path, fake_client) -> None:
    catalog_path = tmp_path / "catalog"
    ingest(
        client=fake_client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )
    assert catalog_path.exists()
    assert any(catalog_path.rglob("*.parquet")), "no parquet files written"
    assert len(fake_client.calls) == 1


def test_ingest_is_idempotent_skips_already_ingested(tmp_path: Path, fake_client) -> None:
    catalog_path = tmp_path / "catalog"
    args = dict(
        client=fake_client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )
    ingest(**args)
    ingest(**args)  # second call covers the same range
    # The second call must short-circuit — either no second client call,
    # or a call whose effective start > end (nothing to fetch).
    assert len(fake_client.calls) == 1 or fake_client.calls[1][1] > fake_client.calls[1][2]


def test_latest_ingested_date_none_when_empty(tmp_path: Path) -> None:
    assert latest_ingested_date(tmp_path / "catalog", "SPY") is None


def test_latest_ingested_date_after_ingest(tmp_path: Path, fake_client) -> None:
    catalog_path = tmp_path / "catalog"
    ingest(
        client=fake_client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )
    latest = latest_ingested_date(catalog_path, "SPY")
    assert latest is not None
    assert latest.date() == datetime(2024, 1, 4).date()  # last fixture bar
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: ImportError for `data.ingest_alpaca`.

- [ ] **Step 3: Implement the orchestrator**

Create `data/ingest_alpaca.py`:
```python
"""
Orchestrate ingestion: fetch from Alpaca → convert → write to Parquet catalog.

Idempotency strategy: before fetching, ask the catalog for the latest bar
timestamp we already have for each ticker. Only fetch (latest + 1 day) → end.
If the catalog already has data through `end`, the ticker is skipped entirely.

CLI entry point fetches the full universe from 2016-01-01 → today and writes to
`data/catalog/`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from data.bar_converter import alpaca_df_to_bars
from data.instruments import build_universe_instruments
from data.tickers import ALL_TICKERS

DEFAULT_CATALOG_PATH = Path("data/catalog")
DEFAULT_START = datetime(2016, 1, 1, tzinfo=timezone.utc)


class AlpacaLike(Protocol):
    """Minimal interface the ingester needs — keeps tests mock-friendly."""
    def fetch_daily_bars(
        self, tickers: Iterable[str], start: datetime, end: datetime
    ) -> pd.DataFrame: ...


def _catalog(path: Path) -> ParquetDataCatalog:
    path.mkdir(parents=True, exist_ok=True)
    return ParquetDataCatalog(path=str(path))


def latest_ingested_date(catalog_path: Path, ticker: str) -> datetime | None:
    """Return the timestamp of the last bar in the catalog for `ticker`,
    or None if no data exists for that ticker yet."""
    if not catalog_path.exists():
        return None
    catalog = _catalog(catalog_path)
    instrument_id = f"{ticker}.ARCX"
    try:
        bars = catalog.bars(instrument_ids=[instrument_id])
    except Exception:
        return None
    if not bars:
        return None
    last_ts_ns = max(b.ts_event for b in bars)
    return datetime.fromtimestamp(last_ts_ns / 1e9, tz=timezone.utc)


def ingest(
    client: AlpacaLike,
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    catalog_path: Path,
) -> None:
    """Fetch and write bars for `tickers` over [start, end]; idempotent."""
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = _catalog(catalog_path)

    # Compute the effective start for each ticker (skip already-covered range).
    effective_start = start
    for t in tickers:
        latest = latest_ingested_date(catalog_path, t)
        if latest is not None and latest + timedelta(days=1) > effective_start:
            effective_start = max(effective_start, latest + timedelta(days=1))

    if effective_start > end:
        return  # everything already covered

    df = client.fetch_daily_bars(tuple(tickers), effective_start, end)
    if df.empty:
        return

    all_bars = []
    for t in tickers:
        all_bars.extend(alpaca_df_to_bars(df, t))
    if all_bars:
        catalog.write_data(all_bars)


def main() -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Ingest Alpaca daily bars into Nautilus catalog")
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    args = parser.parse_args()

    load_dotenv()
    from data.alpaca_client import AlpacaClient

    client = AlpacaClient.from_env()
    catalog_path = Path(args.catalog)

    # Ensure instruments are persisted so BacktestEngine can resolve them later.
    _catalog(catalog_path).write_data(build_universe_instruments())

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    ingest(
        client=client,
        tickers=ALL_TICKERS,
        start=start,
        end=end,
        catalog_path=catalog_path,
    )
    print(f"ingest complete → {catalog_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to confirm pass**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: 4 passed. If `latest_ingested_date` returns None unexpectedly after ingest, the issue is almost certainly that `ParquetDataCatalog.bars()` requires `instrument_ids` to be a different type or shape in v1.227 — verify against `docs/nautilus_api_notes.md` and adjust.

- [ ] **Step 5: Commit**

```bash
git add data/ingest_alpaca.py tests/test_ingest.py
git commit -m "feat(data): idempotent Alpaca→Nautilus ingest orchestrator

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Catalog replay integration test

This is the Stage-1 milestone check: a fresh ingest produces a catalog that the BacktestEngine can replay without errors, with the expected bar counts.

**Files:**
- Create: `tests/test_catalog_integration.py`

- [ ] **Step 1: Write the integration test**

This test makes a real Alpaca call when `ALPACA_API_KEY` is set, otherwise it's skipped. The slice is intentionally tiny (one ticker, 60 days) so the test runs in seconds.

Create `tests/test_catalog_integration.py`:
```python
"""
Integration test: ingest a small real slice from Alpaca, replay through
BacktestEngine. Skipped automatically when credentials are absent so CI on
machines without keys still passes.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

from data.alpaca_client import AlpacaClient
from data.ingest_alpaca import ingest, latest_ingested_date

load_dotenv()
needs_keys = pytest.mark.skipif(
    not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET")),
    reason="Alpaca credentials not in environment (.env)",
)


@needs_keys
def test_real_ingest_one_ticker_60_days(tmp_path: Path) -> None:
    client = AlpacaClient.from_env()
    catalog_path = tmp_path / "catalog"
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)

    ingest(
        client=client,
        tickers=("SPY",),
        start=start,
        end=end,
        catalog_path=catalog_path,
    )

    latest = latest_ingested_date(catalog_path, "SPY")
    assert latest is not None
    assert start <= latest <= end


@needs_keys
def test_real_ingest_then_backtest_engine_replays(tmp_path: Path) -> None:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    from data.instruments import build_equity

    client = AlpacaClient.from_env()
    catalog_path = tmp_path / "catalog"
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    ingest(
        client=client,
        tickers=("SPY",),
        start=start,
        end=end,
        catalog_path=catalog_path,
    )

    catalog = ParquetDataCatalog(path=str(catalog_path))
    bars = catalog.bars(instrument_ids=["SPY.ARCX"])
    assert len(bars) > 35  # ≈42 trading days in Apr–May 2024

    # All OHLCV non-null and monotonic timestamps
    ts_list = [b.ts_event for b in bars]
    assert ts_list == sorted(ts_list)
    for b in bars:
        assert b.open > 0
        assert b.high >= b.open
        assert b.low <= b.open
        assert b.close > 0
        assert int(b.volume) > 0

    # Replay through a BacktestEngine — proves the catalog is engine-compatible.
    config = BacktestEngineConfig(
        trader_id=TraderId("IT-001"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    engine.add_venue(
        venue=Venue("ARCX"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )
    engine.add_instrument(build_equity("SPY"))
    engine.add_data(bars)
    engine.run()
    # If we reach here, the catalog replayed without exception.
```

- [ ] **Step 2: Decide whether to run this now**

If `.env` is populated with real Alpaca paper-account keys, run:
```bash
uv run pytest tests/test_catalog_integration.py -v
```
Expected: 2 passed (one real Alpaca call, ~1–2 seconds each).

If `.env` is not yet populated, run the same command — both tests should be `SKIPPED`. Document in the README that credentials are required to run the integration tests; populate `.env` and re-run before declaring Stage 1 done.

- [ ] **Step 3: Commit**

```bash
git add tests/test_catalog_integration.py
git commit -m "test: integration test for real Alpaca ingest + Nautilus replay

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: yfinance cross-check script

The spec requires close-price drift vs an independent source < 0.5%. This is the catch-net for adjustment bugs.

**Files:**
- Create: `scripts/cross_check_yfinance.py`

- [ ] **Step 1: Implement the cross-check**

Create `scripts/cross_check_yfinance.py`:
```python
"""
Validate Alpaca-ingested daily closes against yfinance for one ticker.

Asserts max relative drift on adjusted close < 0.5% over the overlap window.
Run after a successful ingest. Exits non-zero on failure for easy CI wiring.

Usage:
    uv run python scripts/cross_check_yfinance.py --ticker SPY --start 2024-01-01 --end 2024-06-30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def alpaca_closes_from_catalog(catalog_path: Path, ticker: str) -> pd.Series:
    catalog = ParquetDataCatalog(path=str(catalog_path))
    bars = catalog.bars(instrument_ids=[f"{ticker}.ARCX"])
    rows = [
        (pd.Timestamp(b.ts_event, unit="ns", tz="UTC").normalize(), float(b.close))
        for b in bars
    ]
    if not rows:
        return pd.Series(dtype=float, name="alpaca_close")
    s = pd.Series(dict(rows), name="alpaca_close").sort_index()
    s.index = s.index.tz_convert(None)
    return s


def yfinance_closes(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float, name="yfinance_close")
    s = df["Close"].rename("yfinance_close")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def cross_check(catalog_path: Path, ticker: str, start: str, end: str, tolerance: float = 0.005) -> int:
    a = alpaca_closes_from_catalog(catalog_path, ticker)
    y = yfinance_closes(ticker, start, end)
    if a.empty or y.empty:
        print(f"FAIL: no overlap (alpaca={len(a)}, yfinance={len(y)})")
        return 2

    joined = pd.concat([a, y], axis=1, join="inner").dropna()
    if joined.empty:
        print("FAIL: no overlapping dates between Alpaca catalog and yfinance")
        return 2

    rel_drift = (joined["alpaca_close"] - joined["yfinance_close"]).abs() / joined["yfinance_close"]
    max_drift = float(rel_drift.max())
    print(f"{ticker} compared days={len(joined)} max_rel_drift={max_drift:.5f}")
    if max_drift >= tolerance:
        print(f"FAIL: max drift {max_drift:.5%} >= tolerance {tolerance:.2%}")
        worst = rel_drift.sort_values(ascending=False).head(5)
        print("worst rows:\n", joined.loc[worst.index].assign(rel=worst))
        return 1
    print("OK")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--catalog", default="data/catalog")
    args = parser.parse_args()
    return cross_check(Path(args.catalog), args.ticker, args.start, args.end)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the cross-check (gated on having ingested data)**

If `data/catalog/` has been populated by a real `ingest_alpaca.py` run, execute:
```bash
uv run python scripts/cross_check_yfinance.py --ticker SPY --start 2024-01-01 --end 2024-06-30
```
Expected: prints `OK` and exits 0. If drift > 0.5%, investigate Alpaca's `adjustment="all"` flag vs yfinance's `auto_adjust=True` semantics before declaring Stage 1 done — this is the canary for the adjustment-convention risk called out in the spec.

If the catalog is empty (no real ingest yet), skip running and note that this script must pass before Stage 1 is considered done.

- [ ] **Step 3: Commit**

```bash
git add scripts/cross_check_yfinance.py
git commit -m "feat(scripts): yfinance cross-check for Alpaca adjustment drift

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Full-universe live ingest

The actual Stage-1 deliverable: populate `data/catalog/` with the full 12-ticker, 2016→today catalog.

**Files:** none new — runs `data/ingest_alpaca.py` and verifies.

- [ ] **Step 1: Populate `.env` with paper-account credentials**

If you haven't already:
1. Sign up at https://alpaca.markets/ (free, email only — paper account).
2. Generate API keys for paper trading.
3. `cp .env.example .env` and fill in `ALPACA_API_KEY` / `ALPACA_API_SECRET`.

- [ ] **Step 2: Run the full ingest**

```bash
uv run python data/ingest_alpaca.py
```
Expected: completes in 1–5 minutes; prints `ingest complete → data/catalog`. The catalog directory should now contain Parquet files for all 12 tickers.

- [ ] **Step 3: Sanity-check the catalog**

```bash
uv run python -c "
from pathlib import Path
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from data.tickers import ALL_TICKERS

catalog = ParquetDataCatalog(path='data/catalog')
for t in ALL_TICKERS:
    bars = catalog.bars(instrument_ids=[f'{t}.ARCX'])
    print(f'{t}: {len(bars)} bars')
    assert len(bars) > 240 * 5, f'{t} has too few bars: {len(bars)}'
print('all 12 tickers have >1200 daily bars — sanity OK')
"
```
Expected: prints per-ticker bar counts (all should be ≥ ~2500 for a 2016→2026 window) and the success line.

- [ ] **Step 4: Run the cross-check on a representative ticker**

```bash
uv run python scripts/cross_check_yfinance.py --ticker SPY --start 2020-01-01 --end 2024-12-31
uv run python scripts/cross_check_yfinance.py --ticker XLF --start 2020-01-01 --end 2024-12-31
```
Expected: both print `OK` and exit 0.

- [ ] **Step 5: Re-run ingest to verify idempotency**

```bash
uv run python data/ingest_alpaca.py
```
Expected: completes in seconds (no significant Alpaca traffic), confirming the idempotency path works on the real catalog.

- [ ] **Step 6: Commit any incidental fixes**

If steps 2–5 surfaced bugs you patched (likely candidates: adjustment convention, BarType format, `bars()` query shape), commit the fixes now with a focused message:
```bash
git add -A
git commit -m "fix(ingest): align with verified v1.227 API + adjustment convention

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: README and Stage 1 wrap-up

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:
```markdown
# trading-ml

ML + trading research lab on NautilusTrader using free Alpaca daily-bar data.
Long-running project — see `docs/superpowers/specs/2026-05-26-trading-ml-design.md`
for the full design, philosophy, and staged roadmap.

## Status

- [x] Stage 0 — Project skeleton + Nautilus smoke backtest
- [x] Stage 1 — Alpaca → Nautilus ParquetDataCatalog ingestion (12 tickers, daily, since 2016)
- [ ] Stage 2 — First strategy (momentum top-N)
- [ ] Stages 3–8 — see design spec

## Setup

```bash
# 1. Install dependencies (Python 3.12)
uv sync

# 2. Get free Alpaca paper-account API keys at https://alpaca.markets/
cp .env.example .env
# edit .env and fill in ALPACA_API_KEY / ALPACA_API_SECRET

# 3. Smoke-test the engine (no data needed)
uv run python examples/smoke.py

# 4. Populate the data catalog (1–5 min the first time, seconds on re-runs)
uv run python data/ingest_alpaca.py

# 5. Validate the catalog against an independent source
uv run python scripts/cross_check_yfinance.py --ticker SPY
```

## Tests

```bash
# Unit tests (no network)
uv run pytest tests -v --ignore=tests/test_catalog_integration.py

# Integration tests (real Alpaca calls — needs .env populated)
uv run pytest tests/test_catalog_integration.py -v
```

## Layout

```
data/         pipeline modules + the ParquetDataCatalog (catalog gitignored)
strategies/   Nautilus Strategy implementations (populated from Stage 2)
examples/     runnable demos
scripts/      one-off scripts (cross-checks, sanity tools)
tests/        pytest suite
docs/         design spec, plans, API notes
```
```

- [ ] **Step 2: Run the full test suite as the Stage-1 acceptance gate**

```bash
uv run pytest -v
```
Expected: all unit tests pass; integration tests pass *if* `.env` is populated, else skipped. Zero failures.

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: README with setup, tests, and project layout

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Stage 1 done**

The Stage-1 milestones from the design spec are now satisfied:
- `data/catalog/` populated with daily bars for 12 tickers, 2016 → today
- Ingest is idempotent (verified in Task 11 Step 5)
- Catalog round-trips through `BacktestEngine` (verified in Task 9 integration test)
- yfinance cross-check < 0.5% drift on at least one ticker (verified in Task 11 Step 4)

Stage 2 (momentum top-N strategy) is the next plan.

---

## Self-review

**Spec coverage:**
- §2 Architecture commitments (1 single-source-of-truth catalog, 2 universe scope) → Tasks 4, 8, 11.
- §3 Stack (Python 3.12, uv, nautilus_trader 1.227, alpaca-py 0.43.4, pandas/polars, lightgbm/streamlit out-of-scope) → Task 1's pyproject pins exactly the in-scope deps; ML/Streamlit correctly absent.
- §4 Universe + adjustment convention → Tasks 4 (constants), 7 (`adjustment="all"`), 10 (cross-check catches drift).
- §5 Stage 0 milestone (smoke runs, produces a tearsheet-style output) → Task 3.
- §5 Stage 1 milestone (catalog loads + replays + cross-checks) → Tasks 8, 9, 10, 11.
- §6 Repo layout → Task 1 + all subsequent file creations match the layout in the spec, with one *addition* (`data/instruments.py`, `data/bar_converter.py`, `data/alpaca_client.py`) — a decomposition refinement noted in this plan's File Structure section. No spec contradiction; this is the "design for isolation" guidance from the brainstorming skill applied.
- §7 Risk 1 (adjustment convention) → Task 10 cross-check is the catch-net.
- §7 Risk 2 (Nautilus API drift) → Task 2 is the explicit recon, called out in every later task.
- §7 Risk 5 (Stage 2 may not beat SPY) → not relevant in this plan (Stage 2 is a future plan).

**Placeholder scan:** every code-bearing step has full code. No "TBD", no "implement later", no "similar to Task N".

**Type consistency:** `AlpacaLike` Protocol in Task 8 matches the public method `fetch_daily_bars(tickers, start, end) -> pd.DataFrame` in Task 7. `alpaca_df_to_bars` signature in Task 6 (`df, ticker -> list[Bar]`) is what Task 8 calls. `latest_ingested_date(catalog_path, ticker)` signature in Task 8 implementation matches Task 8 tests. `VENUE_MIC="ARCX"` is defined in `data/instruments.py` (Task 5) and reused in `data/bar_converter.py` (Task 6).

**One residual risk:** several Tasks (3, 5, 6, 8) hardcode the Nautilus v1.227 API surface based on the best information available at planning time. Task 2 is the verification gate; Tasks 3+ explicitly direct the executing agent to *adapt code to the verified API* if drift is found. This is intentional — the alternative (deferring all code until after recon) would balloon the plan size and doesn't change the engineering work.
