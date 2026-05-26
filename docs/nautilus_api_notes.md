# NautilusTrader v1.227.0 — Verified API Notes

> **Source of truth:** every signature and snippet below was confirmed by introspecting and
> instantiating the **installed** `nautilus_trader==1.227.0` package (not from memory or docs).
> Verified 2026-05-26. Run Python via `uv run python ...`.
>
> The core model classes (`Bar`, `Equity`, `Price`, etc.) are **Cython** types, so
> `inspect.signature()` only returns `(*args, **kwargs)`. The real signatures shown here come
> from each class's Cython-embedded docstring (`Cls.__doc__`) and were validated by actual
> construction. The config classes (`BacktestEngineConfig`, `LoggingConfig`) are msgspec Structs.
>
> The official docs (quickstart / concepts/data / api_reference/persistence) were checked and
> agree with the installed package on all import paths and method names — no doc-vs-package drift.

## 1. Import paths (all verified importable in this env)

```python
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, TraderId
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, AggregationSource
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.trading.strategy import Strategy
```

Notes on paths that sometimes differ in older tutorials:
- Config classes live in `nautilus_trader.config` (re-exported): `BacktestEngineConfig`,
  `LoggingConfig`, `StrategyConfig`. Their real modules are `nautilus_trader.backtest.config`
  and `nautilus_trader.common.config`, but **import from `nautilus_trader.config`** (canonical,
  matches docs).
- `TraderId` is in `nautilus_trader.model.identifiers` (alongside `InstrumentId`/`Symbol`/`Venue`).
- `ParquetDataCatalog` is at `nautilus_trader.persistence.catalog.parquet` (confirmed; not
  `nautilus_trader.persistence.catalog` directly, though it may also be re-exported there).

## 2. Venue / MIC code for US ETFs

**Use `ARCX`** for all 12 NYSE-Arca-listed ETFs (SPY, XLE, XLF, etc.).

- `Venue("ARCX")` is a valid venue string (verified).
- `InstrumentId.from_str("SPY.ARCX")` parses to `symbol=SPY, venue=ARCX` (verified).
- Format of an `InstrumentId` string is `"{SYMBOL}.{VENUE}"`.

## 3. BarType string format (Task 6 critical)

```python
bt = BarType.from_str("SPY.ARCX-1-DAY-LAST-EXTERNAL")
# -> SPY.ARCX-1-DAY-LAST-EXTERNAL
# bt.instrument_id        -> SPY.ARCX
# bt.aggregation_source   -> AggregationSource.EXTERNAL (int value 1)
```

`BarType.from_str(value: str) -> BarType` — **verified parses without error.**

Format: `"{INSTRUMENT_ID}-{STEP}-{AGGREGATION}-{PRICE_TYPE}-{AGGREGATION_SOURCE}"`
i.e. `"SPY.ARCX" + "-1" + "-DAY" + "-LAST" + "-EXTERNAL"`.

- **Aggregation source token for ingested-from-external-source data = `EXTERNAL`.** Confirmed:
  `AggregationSource.EXTERNAL` (value `1`) vs `AggregationSource.INTERNAL` (value `2`). Bars that
  come from an external data provider (Alpaca → catalog) use `EXTERNAL`. `INTERNAL` is for bars
  the `DataEngine` aggregates itself from ticks — not our case.
- Daily aggregation token = `DAY`. Step = `1`. Price type for OHLCV closes = `LAST`.

So **Task 6 uses `BarType.from_str("<TICKER>.ARCX-1-DAY-LAST-EXTERNAL")` for every ETF.**

## 4. `Bar(...)` — full constructor (verified by construction)

```
Bar(BarType bar_type, Price open, Price high, Price low, Price close,
    Quantity volume, uint64_t ts_event, uint64_t ts_init, bool is_revision=False) -> None
```

| arg          | type        | required | notes |
|--------------|-------------|----------|-------|
| `bar_type`   | `BarType`   | yes      | |
| `open`       | `Price`     | yes      | |
| `high`       | `Price`     | yes      | must be `>= low`, `>= close`, `>= open` |
| `low`        | `Price`     | yes      | must be `<= close` |
| `close`      | `Price`     | yes      | |
| `volume`     | `Quantity`  | yes      | |
| `ts_event`   | `uint64_t`  | yes      | UNIX **nanoseconds** when the event occurred |
| `ts_init`    | `uint64_t`  | yes      | UNIX **nanoseconds** when the object was created |
| `is_revision`| `bool`      | no (False) | |

`Bar` raises `ValueError` if `high < low`, `high < close`, or `low > close`.

Working snippet (ran successfully):

```python
ts = 1_700_000_000_000_000_000  # nanoseconds
bar = Bar(
    bar_type=BarType.from_str("SPY.ARCX-1-DAY-LAST-EXTERNAL"),
    open=Price.from_str("400.00"),
    high=Price.from_str("405.50"),
    low=Price.from_str("399.25"),
    close=Price.from_str("404.75"),
    volume=Quantity.from_int(1_000_000),
    ts_event=ts,
    ts_init=ts,
)
# repr: SPY.ARCX-1-DAY-LAST-EXTERNAL,400.00,405.50,399.25,404.75,1000000,1700000000000000000
```

## 5. `Equity(...)` — full constructor (the fiddly one; verified by construction)

```
Equity(InstrumentId instrument_id, Symbol raw_symbol, Currency currency, int price_precision,
       Price price_increment, Quantity lot_size, uint64_t ts_event, uint64_t ts_init,
       Quantity max_quantity=None, Quantity min_quantity=None,
       margin_init: Decimal | None = None, margin_maint: Decimal | None = None,
       maker_fee: Decimal | None = None, taker_fee: Decimal | None = None,
       str isin=None, str tick_scheme_name=None, dict info=None) -> None
```

**Required** (the first 8 positionals — all must be supplied):

| arg              | type           | notes |
|------------------|----------------|-------|
| `instrument_id`  | `InstrumentId` | |
| `raw_symbol`     | `Symbol`       | the venue-native symbol, e.g. `Symbol("SPY")` |
| `currency`       | `Currency`     | use `USD` |
| `price_precision`| `int`          | e.g. `2` for US equities |
| `price_increment`| `Price`        | tick size, e.g. `Price.from_str("0.01")` |
| `lot_size`       | `Quantity`     | e.g. `Quantity.from_int(1)` |
| `ts_event`       | `uint64_t`     | UNIX nanoseconds |
| `ts_init`        | `uint64_t`     | UNIX nanoseconds |

**Optional** (all default to `None` and are NOT required — confirmed by successfully
constructing an `Equity` passing only the 8 required args):

`max_quantity`, `min_quantity`, `margin_init`, `margin_maint`, `maker_fee`, `taker_fee`,
`isin`, `tick_scheme_name`, `info`.

- When `margin_init` / `margin_maint` / `maker_fee` / `taker_fee` are omitted they default to `0`
  in the resulting instrument (repr shows `margin_init=0, ..., maker_fee=0, taker_fee=0`).
- `isin` is **optional** — not required.
- Docstring quirk: it labels `currency` as "The futures contract currency" — ignore, it's the
  instrument's quote/denomination currency (use `USD`).
- Validation: raises `ValueError` if `price_precision < 0`, `price_increment <= 0`,
  `lot_size <= 0`, `margin_init < 0`, or `margin_maint < 0`.

Working snippet (ran successfully — for Task 5):

```python
ts = 1_700_000_000_000_000_000
equity = Equity(
    instrument_id=InstrumentId.from_str("SPY.ARCX"),
    raw_symbol=Symbol("SPY"),
    currency=USD,
    price_precision=2,
    price_increment=Price.from_str("0.01"),
    lot_size=Quantity.from_int(1),
    ts_event=ts,
    ts_init=ts,
)
# equity.id -> SPY.ARCX
```

## 6. Identifiers & objects (verified)

```python
Symbol(str value)                 # Symbol("SPY")
Venue(str name)                   # Venue("ARCX")
InstrumentId(Symbol symbol, Venue venue)
InstrumentId.from_str(value: str) -> InstrumentId          # "SPY.ARCX"
TraderId(str value)               # TraderId("BACKTESTER-001")  format "NAME-NNN"

Price(value, uint8_t precision)   # Price(400.0, 2) or Price.from_str("400.00")
Quantity(value, uint8_t precision)# Quantity.from_int(1_000_000) or Quantity(1.5, 1)
Money(value, Currency currency)   # Money(100_000, USD)
```

Helper constructors confirmed working: `Price.from_str("0.01")`, `Quantity.from_int(1)`.

## 7. `ParquetDataCatalog` (Task 8 critical)

### Constructor (verified)

```python
ParquetDataCatalog(
    path: PathLike[str] | str,            # must be an ABSOLUTE path that exists
    fs_protocol: str | None = "file",
    fs_storage_options: dict | None = None,
    fs_rust_storage_options: dict | None = None,
    max_rows_per_group: int = 5000,
    show_query_paths: bool = False,
) -> None
```

Also available: `ParquetDataCatalog.from_env()`, `ParquetDataCatalog.from_uri(...)`.

### Write method — `write_data` (verified)

```
write_data(self, data: list, start=None, end=None, data_cls: type | None = None,
           identifier: str | None = None, **kwargs) -> None
```

Pass a `list` of objects. **Instruments and bars are written via the same `write_data` method.**
Write instruments first (or separately) then bars:

```python
catalog.write_data([equity])          # instruments
catalog.write_data(list_of_bars)      # bars
```

(There is **no** `write_chunk` method; `write_custom_data` exists for custom data types but is
not needed here.)

### Query bars — `bars` (verified — this is what Task 8 uses)

```
bars(self, bar_types: list[str] | None = None,
     instrument_ids: list[str] | None = None, **kwargs) -> list[Bar]
```

**`instrument_ids` takes a `list[str]` (strings like `"SPY.ARCX"`), NOT `InstrumentId` objects.**
Likewise `bar_types` takes a `list[str]`. Both verified to return the written bars:

```python
catalog.bars(instrument_ids=["SPY.ARCX"])                       # -> list[Bar]
catalog.bars(bar_types=["SPY.ARCX-1-DAY-LAST-EXTERNAL"])        # -> list[Bar]  (equivalent here)
```

### Query instruments — `instruments` (verified)

```
instruments(self, instrument_type: type | None = None,
            instrument_ids: list[str] | None = None, **kwargs) -> list[Instrument]
```

```python
catalog.instruments()                              # -> list[Equity], all instruments
catalog.instruments(instrument_ids=["SPY.ARCX"])   # filtered (strings, not objects)
```

### Generic `query` (lower-level; note the param name)

```
query(self, data_cls: type, identifiers: list[str] | None = None,
      start=None, end=None, where: str | None = None,
      files: list[str] | None = None, **kwargs) -> list[Data | CustomData]
```

> **Heads-up for downstream:** the low-level `query()` uses **`identifiers`** (list of strings),
> while the convenience `bars()` / `instruments()` use **`instrument_ids`**. Prefer
> `catalog.bars(instrument_ids=[...])` for bars (matches Task 8's stated interface).

Full other public methods (for reference, not needed now): `quote_ticks`, `trade_ticks`,
`order_book_deltas`, `custom_data`, `consolidate_data`, `delete_data_range`,
`query_first_timestamp`, `query_last_timestamp`, `get_intervals`, `list_data_types`, etc.

### End-to-end round-trip (ran successfully in a tmp dir)

```python
import tempfile
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

with tempfile.TemporaryDirectory() as d:        # d is an absolute path
    catalog = ParquetDataCatalog(d)
    catalog.write_data([equity])
    catalog.write_data(bars)                    # list of Bar
    got = catalog.bars(instrument_ids=["SPY.ARCX"])   # -> list[Bar], same count back
    insts = catalog.instruments()                     # -> [Equity(SPY.ARCX)]
```

## 8. `BacktestEngine` minimal setup (Task 3 — ran full end-to-end)

### Config classes

```python
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.identifiers import TraderId

config = BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-001"),
    logging=LoggingConfig(bypass_logging=True),   # silence logs in tests
)
engine = BacktestEngine(config=config)
```

`LoggingConfig` useful fields: `log_level` (default `"INFO"`), `bypass_logging` (bool),
`log_level_file`, `log_directory`, `print_config`.

### `add_venue` (verified signature — many optional kwargs; only these needed)

```python
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.objects import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import Venue

engine.add_venue(
    venue=Venue("ARCX"),
    oms_type=OmsType.NETTING,        # OmsType: UNSPECIFIED | NETTING | HEDGING
    account_type=AccountType.CASH,   # AccountType: CASH | MARGIN | BETTING | WALLET
    base_currency=USD,
    starting_balances=[Money(100_000, USD)],   # list[Money]
)
```

### Add instrument, data, strategy, run

```python
engine.add_instrument(equity)        # add_instrument(self, instrument)
engine.add_data(bars)                # add_data(self, data, client_id=None, validate=True, sort=True)
engine.add_strategy(Strategy())      # add_strategy(self, strategy: Strategy) — base Strategy is a no-op
engine.run()                         # run(self, start=None, end=None, run_config_id=None, streaming=False)
```

### Read the resulting account balance (verified)

```python
account = engine.portfolio.account(Venue("ARCX"))
balance = account.balance_total(USD)      # -> Money, e.g. 100000.00 USD
print(balance)

result = engine.get_result()              # -> BacktestResult (summary object)

engine.reset()      # reset state to re-run
engine.dispose()    # release resources when done
```

Full minimal backtest (engine config -> add_venue -> add_instrument -> add_data ->
add_strategy(Strategy()) -> run -> read `portfolio.account(venue).balance_total(USD)`) was
**run end-to-end and produced `100000.00 USD`** with a no-op strategy.

## 9. Enums (verified members)

```python
from nautilus_trader.model.enums import AccountType, OmsType, AggregationSource
# AccountType:        CASH, MARGIN, BETTING, WALLET
# OmsType:            UNSPECIFIED, NETTING, HEDGING
# AggregationSource:  EXTERNAL (=1), INTERNAL (=2)
```

## 10. Summary for downstream tasks

- **Task 3 (smoke backtest):** use the §8 skeleton; balance read via
  `engine.portfolio.account(venue).balance_total(USD)`.
- **Task 5 (Equity builder):** §5 — only 8 required args; `isin`/fees/margins all optional.
- **Task 6 (DataFrame → Bar):** §4 + §3 — `BarType.from_str("<TICKER>.ARCX-1-DAY-LAST-EXTERNAL")`,
  timestamps in **UNIX nanoseconds**.
- **Task 8 (catalog write/query):** §7 — `catalog.write_data([...])` for both instruments and
  bars; query bars via `catalog.bars(instrument_ids=["SPY.ARCX"])` (strings, not objects).
