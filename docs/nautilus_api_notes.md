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
    starting_balances=[Money(100_000, USD)],   # list[Money]  — positional order is (starting_balances, base_currency); pass as keyword args
    base_currency=USD,
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

## 9. Strategy + Order + Portfolio + FillModel (Stage 2)

> Every signature and result below was confirmed by `inspect.signature`, the Cython-embedded
> docstring, **and** an end-to-end backtest in this env that placed a market buy, filled it, and
> read back the position quantity and the changed account balance (snippet at end of §9).
> Verified 2026-05-26 against `nautilus_trader==1.227.0`.

### 9.1 Imports (all verified)

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.orders import MarketOrder            # the returned type
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.model.objects import Quantity, Price, Money
from nautilus_trader.model.currencies import USD
```

### 9.2 Strategy lifecycle hooks (verified members on the `Strategy` class)

All discovered `on_*` hooks in v1.227:

```
on_bar                  on_data                 on_degrade
on_dispose              on_event                on_fault
on_funding_rate         on_historical_data      on_index_price
on_instrument           on_instrument_close     on_instrument_status
on_load                 on_mark_price           on_market_exit
on_option_chain         on_option_greeks
on_order_accepted       on_order_book           on_order_book_deltas
on_order_book_depth     on_order_cancel_rejected  on_order_canceled
on_order_denied         on_order_emulated       on_order_event
on_order_expired        on_order_filled         on_order_initialized
on_order_modify_rejected  on_order_pending_cancel  on_order_pending_update
on_order_rejected       on_order_released       on_order_submitted
on_order_triggered      on_order_updated
on_position_changed     on_position_closed      on_position_event
on_position_opened      on_quote_tick           on_reset
on_resume               on_save                 on_signal
on_start                on_stop                 on_trade_tick
```

Hooks Stage 2 actually uses (verified by end-to-end run):

| Hook        | When it fires                                              |
|-------------|------------------------------------------------------------|
| `on_start`  | Once at strategy startup — **subscribe bars here**.        |
| `on_bar`    | Each new `Bar` for any subscribed `BarType`.               |
| `on_stop`   | Once on graceful shutdown — for cleanup / unsubscribes.    |
| `on_reset`  | When engine state is reset between runs.                    |

Bar subscription idiom (called from `on_start`):

```python
def on_start(self) -> None:
    self.subscribe_bars(self.config.bar_type)   # bar_type: BarType
```

### 9.3 `self` attributes available inside a Strategy (verified by `dir(Strategy)`)

| Attribute / method     | Type / use                                                   |
|------------------------|--------------------------------------------------------------|
| `self.config`          | The `StrategyConfig` instance you constructed the strategy with. |
| `self.log`             | Logger with methods: `info`, `debug`, `warning`, `error`, `exception`. Use `self.log.info("msg")`. |
| `self.cache`           | The shared `Cache` (positions, orders, instruments, accounts, bars). |
| `self.portfolio`       | `Portfolio` — read `account(venue)`, `net_position(...)`, P&L. |
| `self.clock`           | `Clock`; `self.clock.timestamp_ns()` → `uint64` UNIX nanoseconds (also `timestamp`, `timestamp_ms`, `timestamp_us`, `utc_now()`). |
| `self.order_factory`   | `OrderFactory` — has `.market(...)`, `.limit(...)`, etc.     |
| `self.submit_order(o)` | Submits a single `Order` to the engine.                      |
| `self.submit_order_list(ol)` | Submits an `OrderList` (multiple orders).              |
| `self.cancel_order`, `self.cancel_all_orders`, `self.close_position`, `self.close_all_positions` | Standard order/position management. |
| `self.subscribe_bars(bar_type)` | Subscribe to a `BarType`.                           |

### 9.4 `order_factory.market(...)` — exact verified signature

From `inspect.signature(OrderFactory.market)`:

```
market(self,
       instrument_id: InstrumentId,
       order_side: OrderSide,
       quantity: Quantity,
       time_in_force: TimeInForce = TimeInForce.GTC,
       reduce_only: bool = False,
       quote_quantity: bool = False,
       exec_algorithm_id: ExecAlgorithmId | None = None,
       exec_algorithm_params: dict | None = None,
       tags: list[str] | None = None,
       client_order_id: ClientOrderId | None = None) -> MarketOrder
```

Stage-2 idiom (only the three required args; `time_in_force` defaults to `GTC`):

```python
order = self.order_factory.market(
    instrument_id=self.config.instrument_id,
    order_side=OrderSide.BUY,         # or OrderSide.SELL
    quantity=Quantity.from_int(100),
)
self.submit_order(order)              # the actual submission call
```

Raises `ValueError` if `quantity <= 0` or `time_in_force == GTD`.

### 9.5 Querying the current position

**Key finding:** there is **no `position_for_instrument_id` / `position_for_instrument`** in v1.227.
The verified API is `Cache.positions_open(...)` (or `Cache.positions(...)`), which returns a
**list** — filter by `instrument_id`:

```
positions_open(self,
               venue: Venue | None = None,
               instrument_id: InstrumentId | None = None,
               strategy_id: StrategyId | None = None,
               side: PositionSide = PositionSide.NO_POSITION_SIDE,  # i.e. no filter
               account_id: AccountId | None = None) -> list[Position]
```

Idiom inside a Strategy:

```python
positions = self.cache.positions_open(instrument_id=self.config.instrument_id)
position = positions[0] if positions else None   # None when flat
```

Other useful methods on `Cache` related to positions (verified to exist):
`positions(...)`, `positions_closed(...)`, `position(position_id)`, `position_for_order(client_order_id)`,
`positions_open_count(...)`, `is_position_open(position_id)`, `position_exists(position_id)`.

Verified `Position` fields (from the end-to-end run, long 100 @ 50.21):

| field            | example value           | notes                                            |
|------------------|-------------------------|--------------------------------------------------|
| `quantity`       | `Quantity(100)`         | **unsigned** absolute size.                      |
| `signed_qty`     | `100.0` (float)         | **signed** — positive long, negative short.      |
| `side`           | `PositionSide.LONG`     | enum: `LONG`, `SHORT`, `FLAT`, `NO_POSITION_SIDE`. |
| `avg_px_open`    | `50.21`                 | average entry price (float).                     |
| `is_long`        | `True`                  |                                                  |
| `is_short`       | `False`                 |                                                  |
| `instrument_id`  | `TEST.ARCX`             |                                                  |

For "am I flat?" use `positions_open(instrument_id=...)` returning `[]` (preferred), or
`self.portfolio.is_flat(instrument_id)` / `self.portfolio.net_position(instrument_id)`.

### 9.6 Querying account equity

```python
venue = self.config.instrument_id.venue       # or a stored Venue
account = self.portfolio.account(venue)       # -> CashAccount (or MarginAccount)
base_ccy = account.base_currency              # Currency object, e.g. USD
balance: Money = account.balance_total(base_ccy)   # 94979.00 USD
equity_decimal = balance.as_decimal()         # decimal.Decimal('94979.00')   for math
```

Also verified to exist on the account object:

- `account.balance_free(currency)` → `Money` — cash available (not locked by open orders).
- `account.balance_locked(currency)` → `Money` — cash currently held against open orders.
- `account.base_currency` → `Currency` (the venue's base currency; pass this into `balance_total`).
- `account.id` → `AccountId` (e.g. `ARCX-001`).
- `account.events` → `list[AccountState]`.  **Two snapshots after one fill** in the test run:
  pre-fill `total=100_000.00 USD`, post-fill `total=94_979.00 USD`.

`balance_total(...)` returns a `Money`; extract its magnitude via either `.as_decimal()` (returns
`decimal.Decimal`) or `.as_double()` (returns `float`). `.as_decimal()` is preferred for any P&L
math to preserve precision.

### 9.6.1 Portfolio equity (cash + mark value)

For total portfolio value INCLUSIVE of open positions' mark value, use
`portfolio.equity(venue)` which returns `dict[Currency, Money]`. This is what
strategies should use for equal-weight position sizing across multiple
rebalances — `account.balance_total(...)` returns CASH ONLY, which drops
sharply after the first rebalance deploys equity into positions.

```python
account = strategy.portfolio.account(venue)
equity_by_ccy = strategy.portfolio.equity(venue)            # dict[Currency, Money]
equity_usd = float(equity_by_ccy[account.base_currency].as_decimal())
```

Verified signature (from `inspect.signature(Portfolio.equity)`):

```
Portfolio.equity(self, Venue venue=None, AccountId account_id=None) -> dict
```

Verified docstring excerpt:

> Return the per-currency total equity for the given venue or account (if found).
>
> For cash and betting accounts: `balance.total + Σ mark_value(open positions)`.
> For margin accounts: `balance.total + Σ unrealized_pnl(open positions)`.
>
> Instruments that cannot be priced are tracked via `missing_price_instruments`,
> so equity understatement surfaces via a warn-once log.

For a CashAccount, `portfolio.equity(venue)[ccy]` ≈ `balance.total + Σ mark_value(open_positions[ccy])`.
Verified by behavior: in a backtest where the strategy deploys ~$99,790 of $100k into stock on
the first rebalance, `account.balance_total` reports ~$200 (the residual cash) while
`portfolio.equity` reports ~$100,000 (cash + mark value of holdings). The Strategy in
`strategies/momentum_topn.py` uses `portfolio.equity` for this reason.

### 9.7 Account-state time-series (for tearsheet equity curve)

**There is no built-in per-bar equity time-series API on `Cache` in v1.227.** What exists:

- `engine.cache.accounts()` → `list[Account]` (one per venue; current state only, no history).
- `account.events` → `list[AccountState]` where each `AccountState` has `ts_event` and
  `balances` (a list of `AccountBalance(total, locked, free)` per currency). **This is the
  closest thing to a time-series.** In the recon run it contained 2 entries: the initial
  funding and the post-fill state. Each fill (or other account change) appends one event.

Verified `AccountState` public attrs: `account_id`, `account_type`, `balances`, `base_currency`,
`info`, `is_reported`, `margins`, `ts_event`, `ts_init`, plus `to_dict()` / `from_dict()`.

**For the Stage 2 tearsheet (Task 11)** the practical equity-curve reconstruction is:

Option A (preferred — event-driven, exact):
```python
account = engine.portfolio.account(venue)
rows = []
for ev in account.events:                    # already in chronological order
    bal = next(b for b in ev.balances if b.currency == account.base_currency)
    rows.append((ev.ts_event, float(bal.total.as_decimal())))
# rows is [(ts_ns, equity_usd), ...]
```
This yields equity at every account state change (i.e. every fill / funding event), not at every
bar — but for daily-bar Stage 2 with one rebalance per month it is sufficient.

Option B (per-bar mark-to-market): reconstruct by combining `engine.cache.bars(bar_type)` with
`engine.cache.orders()` (which exposes each order's `events` list, including `OrderFilled`
events with `last_qty`, `last_px`, `ts_event`). This is needed only if the tearsheet wants
daily resolution; otherwise stick with Option A.

Note: `engine.cache.position_snapshots()` returns `[]` unless `ExecEngine` is configured with
`snapshot_positions=True` (default `False`). Same for `snapshot_orders=False`. If we ever turn
these on, snapshots become persisted and queryable.

### 9.8 `FillModel` for slippage

```python
from nautilus_trader.backtest.models import FillModel
fill_model = FillModel(
    prob_fill_on_limit=1.0,   # P(limit order fills when market rests on its price)
    prob_slippage=1.0,        # P(market/aggressor fill slips by ONE TICK)
    random_seed=42,           # int | None
)
```

Verified Cython docstring:

```
FillModel(double prob_fill_on_limit=1.0,
          double prob_slippage=0.0,
          int random_seed: int | None = None,
          config=None) -> None
```

**Critical: `prob_slippage` is a Bernoulli probability of slipping, and the slip magnitude is
exactly one tick — NOT configurable.** Confirmed empirically:

| `prob_slippage` | fill price | balance after 100-share buy |
|-----------------|------------|-----------------------------|
| `0.0`           | `50.20`    | `94980.00 USD`              |
| `1.0`           | `50.21`    | `94979.00 USD`              |

Difference: exactly `$0.01` per share = one tick (`price_increment=0.01`) × 100 shares = `$1.00`.

So for a Stage 2 ETF (~$50 share price, $0.01 tick):
- one tick = $0.01 ≈ **2.0 bps** of $50
- nominal Stage 2 target was 0.5 bps ≈ $0.0025 per share — **below one tick**.
- v1.227's `FillModel` cannot directly express 0.5 bps; the closest available models are:
  - `prob_slippage=0.0` (no slippage, 0 bps)
  - `prob_slippage=0.25` (one tick on ~25 % of fills ≈ 0.5 bps average) — **stochastic, only an average match**
  - `prob_slippage=1.0` (always one tick, ≈ 2 bps)

`nautilus_trader.backtest.models` also exposes a dedicated `OneTickSlippageFillModel` that forces
exactly one tick of slippage deterministically:

```python
from nautilus_trader.backtest.models import OneTickSlippageFillModel
fill_model = OneTickSlippageFillModel()
```

**Wiring into the engine** — `BacktestEngine.add_venue(..., fill_model=...)` (verified kwarg name):

```python
engine.add_venue(
    venue=Venue("ARCX"),
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    starting_balances=[Money(100_000, USD)],
    base_currency=USD,
    fill_model=fill_model,                # <-- here
)
```

### 9.9 Complete end-to-end snippet (this ran; balance changed; position confirmed)

```python
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy


class DemoConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: int = 100


class DemoStrategy(Strategy):
    def __init__(self, config: DemoConfig) -> None:
        super().__init__(config=config)
        self._placed = False

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self._placed:
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(self.config.trade_size),
            )
            self.submit_order(order)
            self._placed = True


engine = BacktestEngine(config=BacktestEngineConfig(
    trader_id=TraderId("BACKTESTER-001"),
    logging=LoggingConfig(bypass_logging=True),
))
venue = Venue("ARCX")
engine.add_venue(
    venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH,
    starting_balances=[Money(100_000, USD)], base_currency=USD,
    fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=1.0, random_seed=42),
)
ts = 1_700_000_000_000_000_000
iid = InstrumentId.from_str("TEST.ARCX")
engine.add_instrument(Equity(
    instrument_id=iid, raw_symbol=Symbol("TEST"), currency=USD,
    price_precision=2, price_increment=Price.from_str("0.01"),
    lot_size=Quantity.from_int(1), ts_event=ts, ts_init=ts,
))
bt = BarType.from_str("TEST.ARCX-1-DAY-LAST-EXTERNAL")
ONE_DAY_NS = 24 * 60 * 60 * 1_000_000_000
bars = []
for i in range(10):
    o = 50.00 + 0.10 * i
    bars.append(Bar(
        bar_type=bt,
        open=Price.from_str(f"{o:.2f}"), high=Price.from_str(f"{o + 0.50:.2f}"),
        low=Price.from_str(f"{o - 0.30:.2f}"), close=Price.from_str(f"{o + 0.20:.2f}"),
        volume=Quantity.from_int(1_000_000),
        ts_event=ts + i * ONE_DAY_NS, ts_init=ts + i * ONE_DAY_NS,
    ))
engine.add_data(bars)
engine.add_strategy(DemoStrategy(config=DemoConfig(instrument_id=iid, bar_type=bt)))
engine.run()

# Post-run reads (verified output values shown in comments):
account = engine.portfolio.account(venue)
print(account.balance_total(account.base_currency))            # -> 94979.00 USD
print(account.balance_total(account.base_currency).as_decimal())  # -> Decimal('94979.00')

positions = engine.cache.positions_open(instrument_id=iid)
p = positions[0]
print(p.quantity, p.signed_qty, p.side, p.avg_px_open)         # -> 100 100.0 2 50.21
# p.side is PositionSide.LONG (=2); use `p.side == PositionSide.LONG`, not the repr.

# Equity time series (Option A — per account-state event):
for ev in account.events:
    bal = next(b for b in ev.balances if b.currency == account.base_currency)
    print(ev.ts_event, float(bal.total.as_decimal()))
# -> 1700000000000000000 100000.0
# -> 1700000000000000000 94979.0

engine.dispose()
```

---

## 10. Enums (verified members)

```python
from nautilus_trader.model.enums import AccountType, OmsType, AggregationSource
# AccountType:        CASH, MARGIN, BETTING, WALLET
# OmsType:            UNSPECIFIED, NETTING, HEDGING
# AggregationSource:  EXTERNAL (=1), INTERNAL (=2)
```

Stage 2 additions:

```python
from nautilus_trader.model.enums import OrderSide, TimeInForce, PositionSide
# OrderSide:     NO_ORDER_SIDE, BUY, SELL
# TimeInForce:   GTC (default), IOC, FOK, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE
# PositionSide:  NO_POSITION_SIDE (filter sentinel), FLAT, LONG, SHORT
```

## 11. Summary for downstream tasks

Stage 1:
- **Task 3 (smoke backtest):** use the §8 skeleton; balance read via
  `engine.portfolio.account(venue).balance_total(USD)`.
- **Task 5 (Equity builder):** §5 — only 8 required args; `isin`/fees/margins all optional.
- **Task 6 (DataFrame → Bar):** §4 + §3 — `BarType.from_str("<TICKER>.ARCX-1-DAY-LAST-EXTERNAL")`,
  timestamps in **UNIX nanoseconds**.
- **Task 8 (catalog write/query):** §7 — `catalog.write_data([...])` for both instruments and
  bars; query bars via `catalog.bars(instrument_ids=["SPY.ARCX"])` (strings, not objects).

Stage 2:
- **Task 4 (Strategy skeleton):** §9.2–9.4 — subscribe in `on_start`, trade in `on_bar`,
  build orders via `self.order_factory.market(instrument_id, order_side, quantity)`,
  submit via `self.submit_order(order)`. Config via `class FooConfig(StrategyConfig, frozen=True)`.
- **Task 7 (Position reconciliation):** §9.5 — query with
  `self.cache.positions_open(instrument_id=iid)`; flat ⇔ list is empty.
  Use `position.signed_qty` (positive long, negative short) when computing target-vs-actual deltas.
  For equal-weight sizing across rebalances, use `portfolio.equity(venue)` (§9.6.1) — not
  `account.balance_total` which is cash-only and collapses after the first rebalance.
- **Task 9 (FillModel slippage helper):** §9.8 — `FillModel(prob_fill_on_limit=1.0,
  prob_slippage=p, random_seed=seed)` wired via `add_venue(fill_model=...)`. Slip magnitude is
  **fixed at one tick**; only the probability is tunable. For 0.5 bps nominal on ~$50 ETFs use
  `prob_slippage ≈ 0.25` (stochastic ~0.5 bps average), or `OneTickSlippageFillModel()` for
  deterministic ~2 bps. **Document the chosen value in code comments.**
- **Task 11 (Live backtest entry point / tearsheet):** §9.6–9.7 — read final equity via
  `engine.portfolio.account(venue).balance_total(account.base_currency).as_decimal()`. For the
  equity curve, prefer `account.events` (`AccountState.ts_event` + `balances`) which gives a
  fill-resolution time series; supplement with `engine.cache.orders()[*].events` for per-fill
  detail if needed. **No `account_states_for_venue` / built-in equity time-series API exists.**

## 10. LiveNode (Stage 4)

> Stage 4 ground truth. Verified 2026-05-26 against `nautilus_trader==1.227.0` by
> introspection (`inspect.signature` / `dir()`), reading the package's own `_template`
> adapter, and running the end-to-end snippet at §10.9 to completion. Anything I could
> NOT verify is called out inline.

### 10.1 Imports (all verified importable in this env)

```python
import asyncio

from nautilus_trader.cache.cache import Cache
from nautilus_trader.cache.config import CacheConfig             # also at nautilus_trader.config
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.common.config import DatabaseConfig
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.config import LoggingConfig
from nautilus_trader.live.config import (
    LiveDataClientConfig,
    LiveDataEngineConfig,
    LiveExecClientConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    RoutingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.data_client import LiveDataClient, LiveMarketDataClient
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import AccountId, ClientId, Venue
from nautilus_trader.model.enums import AccountType, OmsType
```

Key naming differences from older tutorials:

- **`LiveExecClientFactory`** (not `LiveExecutionClientFactory`). The factory class name uses
  the short form `Exec`, even though the **client base class** is `LiveExecutionClient` (full
  form). This is verified — be careful, mistype = `ImportError`.
- For market data feeds (bars / ticks / order books) use **`LiveMarketDataClient`**, not
  `LiveDataClient`. `LiveDataClient` is the generic base — it only exposes `_subscribe` /
  `_request` for **custom** data types. The bar/tick/quote-specific hooks like
  `_subscribe_bars` are defined on `LiveMarketDataClient`.
- `CacheConfig` is canonically imported from `nautilus_trader.config` (matches §1 pattern) —
  same object as `nautilus_trader.cache.config.CacheConfig`.

### 10.2 TradingNode lifecycle

Verified `TradingNode` class is from `nautilus_trader.live.node`. Signatures:

```python
TradingNode.__init__(self, config: TradingNodeConfig | None = None,
                     loop: asyncio.AbstractEventLoop | None = None) -> None

# Setup (must call before run)
TradingNode.add_data_client_factory(self, name: str,
                                    factory: type[LiveDataClientFactory]) -> None
TradingNode.add_exec_client_factory(self, name: str,
                                    factory: type[LiveExecClientFactory]) -> None
TradingNode.build(self) -> None                  # builds clients from config.data_clients/exec_clients

# Lifecycle — both sync wrappers and async coroutines exist
TradingNode.run(self, raise_exception: bool = False) -> None   # sync — calls run_async via loop
TradingNode.run_async(self) -> None              # async — coroutine
TradingNode.stop(self) -> None                   # sync — calls stop_async via loop
TradingNode.stop_async(self) -> None             # async — coroutine
TradingNode.dispose(self) -> None                # sync only — closes the event loop
```

**Lifecycle order:**

1. `node = TradingNode(config=...)` — builds the `NautilusKernel` (data/risk/exec engines,
   message bus, cache, portfolio) but does **not** build the clients yet.
2. `node.add_data_client_factory(name, factory)` / `add_exec_client_factory(name, factory)`
   — register the factory classes that `build()` will look up by name.
3. `node.build()` — iterates `config.data_clients` / `config.exec_clients` dicts, dispatches
   each key to its registered factory's `create()` classmethod, and registers the resulting
   client with the data/exec engine.
4. Add strategies. **Caveat:** `TradingNodeConfig.strategies: list[ImportableStrategyConfig]`
   only accepts importable config dataclasses (factory-built via `StrategyFactory.create`).
   To attach a pre-built Strategy *instance* (the common case in tests), call
   `node.kernel.trader.add_strategy(strategy)` after `node.build()`. Both work — verified.
5. `node.run()` (sync) **or** `await node.run_async()` (async). `run()` calls
   `loop.run_until_complete(run_async())` — so it blocks until the engines stop. Inside the
   loop the kernel goes through: register executor → start engines → connect clients →
   await connections → reconcile (optional) → initialize portfolio → start trader (which
   starts the strategies → `on_start()` fires here).
6. `node.stop()` (or `stop_async()`) — gracefully stops strategies, disconnects clients,
   stops engines, optionally saves state via cache.
7. `node.dispose()` — final cleanup, closes the event loop. **Not async-safe**; call it
   only after the loop has stopped (after `run()` returns).

**For tests / programmatic shutdown:** schedule `node.stop_async()` from inside the loop
via `node.kernel.loop.create_task(...)` before calling `node.run()`. See §10.9.

**Properties / accessors:**

- `node.cache` → the shared `Cache`
- `node.portfolio` → the `Portfolio` facade (same surface as `engine.portfolio` in §9.6)
- `node.kernel` → `NautilusKernel` (exposes `.trader`, `.loop`, `.cache`, `.msgbus`,
  `.data_engine`, `.exec_engine`, …). Used to attach strategies and tasks programmatically.
- `node.trader_id` → `TraderId`
- `node.is_running` → `bool`
- `node.is_built` → `bool`

### 10.3 `TradingNodeConfig` — fields

`TradingNodeConfig` is a **msgspec Struct** (subclass of `NautilusKernelConfig`) — pass
fields as kwargs. Full field list (default values from `msgspec.structs.fields(...)`):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `environment` | `Environment` | `Environment.LIVE` | enum from `nautilus_trader.common.enums` |
| `trader_id` | `TraderId` (str-coercible) | `'TRADER-001'` | format `'NAME-NNN'` |
| `instance_id` | `UUID4 \| None` | `None` | |
| `cache` | `CacheConfig \| None` | `None` | see §10.7 |
| `message_bus` | `MessageBusConfig \| None` | `None` | |
| `data_engine` | `LiveDataEngineConfig` | default instance | qsize, bar build flags |
| `risk_engine` | `LiveRiskEngineConfig` | default instance | rate limits etc. |
| `exec_engine` | `LiveExecEngineConfig` | default instance | **see reconciliation note** |
| `portfolio` | `PortfolioConfig \| None` | `None` | |
| `emulator` | `OrderEmulatorConfig \| None` | `None` | |
| `streaming` | `StreamingConfig \| None` | `None` | Parquet streaming-out of bus events |
| `catalogs` | `list[DataCatalogConfig]` | `[]` (factory) | read-side Parquet catalogs |
| `actors` | `list[ImportableActorConfig]` | `[]` | |
| `strategies` | `list[ImportableStrategyConfig]` | `[]` | importable form only |
| `exec_algorithms` | `list[ImportableExecAlgorithmConfig]` | `[]` | |
| `controller` | `ImportableControllerConfig \| None` | `None` | |
| `load_state` | `bool` | `False` | calls `Trader.load()` at startup (cache-backed) |
| `save_state` | `bool` | `False` | calls `Trader.save()` at shutdown (cache-backed) |
| `loop_debug` | `bool` | `False` | |
| `logging` | `LoggingConfig \| None` | `None` | |
| `timeout_connection` | `float > 0` | `120.0` | seconds |
| `timeout_reconciliation` | `float > 0` | `30.0` | |
| `timeout_portfolio` | `float > 0` | `10.0` | |
| `timeout_disconnection` | `float > 0` | `10.0` | |
| `timeout_post_stop` | `float > 0` | `10.0` | |
| `timeout_shutdown` | `float > 0` | `5.0` | |
| `data_clients` | `dict[str, Any]` | `{}` | key = client name; value = `LiveDataClientConfig` or `ImportableConfig` |
| `exec_clients` | `dict[str, Any]` | `{}` | key = client name; value = `LiveExecClientConfig` or `ImportableConfig` |

**Critical gotcha — reconciliation:** `LiveExecEngineConfig.reconciliation` defaults to
`True` and `reconciliation_startup_delay_secs` defaults to **10.0**. With a stub exec
client that returns no mass-status, the node still gates startup on a 10-second delay
before `Trader.start()` is called → strategies never go RUNNING in short tests. **For
simulated/stub exec clients, set `exec_engine=LiveExecEngineConfig(reconciliation=False)`.**
Verified empirically: with `reconciliation=True` the snippet at §10.9 records 0 bars in a
4-second window; with `reconciliation=False` it records 1 bar as expected.

**`data_clients` / `exec_clients` dict shape:** key is the **client name** (used as the
factory lookup key registered via `add_data_client_factory(name, factory)`), value is a
`LiveDataClientConfig` (resp. `LiveExecClientConfig`) instance. The builder also accepts
`ImportableConfig` values when you want everything driven from disk/YAML; that path is not
needed for in-process Stage 4 wiring.

`LiveDataClientConfig` fields: `handle_revised_bars: bool = False`,
`instrument_provider: InstrumentProviderConfig`, `routing: RoutingConfig`.
`LiveExecClientConfig` fields: `instrument_provider: InstrumentProviderConfig`,
`routing: RoutingConfig`. `RoutingConfig(default: bool = False, venues: frozenset[str] | None = None)`
— set `default=True` to make the client the default for its data/exec type, or list specific
venues to route by venue.

### 10.4 LiveDataClient / LiveMarketDataClient — abstract surface

Two base classes; pick based on what you emit:

- **`LiveDataClient`** (`nautilus_trader.live.data_client`): for **custom data types only**.
  Required overrides: `_connect`, `_disconnect`. Optional: `_subscribe`, `_unsubscribe`,
  `_request` (all take `*Data` command messages).
- **`LiveMarketDataClient`** (same module): for bars / ticks / quotes / order books.
  Required overrides: `_connect`, `_disconnect`. Optional but typically needed for what
  you want to emit: `_subscribe_bars`, `_unsubscribe_bars`, `_subscribe_quote_ticks`,
  `_subscribe_trade_ticks`, `_subscribe_instruments`, … and the matching `_request_*`
  methods if you want to support replay/backfill queries.

**Verified constructor signature** (`LiveDataClient`):

```python
LiveDataClient.__init__(
    self,
    loop: asyncio.AbstractEventLoop,
    client_id: ClientId,
    venue: Venue | None,
    msgbus: MessageBus,
    cache: Cache,
    clock: LiveClock,
    config: NautilusConfig | None = None,
) -> None
```

**Verified constructor signature** (`LiveMarketDataClient` — adds `instrument_provider`):

```python
LiveMarketDataClient.__init__(
    self,
    loop: asyncio.AbstractEventLoop,
    client_id: ClientId,
    venue: Venue | None,
    msgbus: MessageBus,
    cache: Cache,
    clock: LiveClock,
    instrument_provider: InstrumentProvider,
    config: NautilusConfig | None = None,
    is_sync: bool = False,
) -> None
```

**Verified abstract method signatures** (all async, all take a typed message):

```python
async def _connect(self) -> None: ...                    # required
async def _disconnect(self) -> None: ...                 # required
async def _subscribe_bars(self, command: SubscribeBars) -> None: ...
async def _unsubscribe_bars(self, command: UnsubscribeBars) -> None: ...
async def _request_bars(self, request: RequestBars) -> None: ...
async def _subscribe_quote_ticks(self, command: SubscribeQuoteTicks) -> None: ...
# … same pattern for trade_ticks, instruments, order_book_deltas, etc.
```

Per the `nautilus_trader.adapters._template.data.TemplateLiveMarketDataClient` reference,
**only `_connect` and `_disconnect` are *required*** — every `_subscribe_*` /
`_unsubscribe_*` / `_request_*` is *optional*: the base raises `NotImplementedError` if
called, but if no strategy subscribes to that type the method is never invoked. **However,
if you don't override `_subscribe_bars` and a strategy calls `self.subscribe_bars(...)`,
the engine logs an error and the upstream subscription state is inconsistent** — even
though `self._handle_data(bar)` calls from elsewhere still flow through to subscribed
strategies. Best practice: override `_subscribe_bars`/`_unsubscribe_bars` as no-ops if you
won't honour the subscription externally. Verified — see §10.9.

**Emitting bars into the engine: `self._handle_data(bar)`** — a Cython instance method
inherited from `DataClient`. Confirmed by grepping the binance/deribit/dydx adapters:
`self._handle_data(bar)`, `self._handle_data(quote_tick)`, `self._handle_data(trade_tick)`
are the universal patterns. There is no separate `emit_bar` / `publish_bar` method.
`_handle_data` does NOT take a correlation id — it just routes the data through the
data engine to all subscribers. For request/response (backfill) flows use
`self._handle_data_response(data_type, data, correlation_id, start, end, params)`.

**Calling `_handle_data` from `_connect`:** the message-bus subscriptions for strategies
are set up later in the lifecycle (`Trader.start()` runs after the connection step). If
you emit a bar synchronously inside `_connect()`, the strategy hasn't yet been started and
its `on_bar` won't fire. Schedule emission via `self.create_task(coro)` with an
`asyncio.sleep` so the bar lands after `Trader.start()`. Verified — see §10.9.

**Connection state:** the base class' `connect()` does NOT auto-set `is_connected = True`
— `_connect()` must call `self._set_connected(True)` (and `_disconnect` must call
`self._set_connected(False)`). Without this, the node hangs in
`Awaiting engine connections and initializations` until `timeout_connection` fires.
Verified.

### 10.5 LiveExecutionClient — abstract surface

**Verified constructor signature:**

```python
LiveExecutionClient.__init__(
    self,
    loop: asyncio.AbstractEventLoop,
    client_id: ClientId,
    venue: Venue | None,
    oms_type: OmsType,
    account_type: AccountType,
    base_currency: Currency | None,
    instrument_provider: InstrumentProvider,
    msgbus: MessageBus,
    cache: Cache,
    clock: LiveClock,
    config: NautilusConfig | None = None,
) -> None
```

Per `nautilus_trader.adapters._template.execution.TemplateLiveExecutionClient`, the
**required overrides** are:

| Method                            | Async | Notes |
|-----------------------------------|-------|-------|
| `_connect`                        | yes   | must call `self._set_connected(True)` |
| `_disconnect`                     | yes   | must call `self._set_connected(False)` |
| `_submit_order`                   | yes   | takes `SubmitOrder` command |
| `_cancel_order`                   | yes   | takes `CancelOrder` command |
| `_cancel_all_orders`              | yes   | takes `CancelAllOrders` command |
| `generate_order_status_report`    | yes   | takes `GenerateOrderStatusReport` → `OrderStatusReport \| None` |
| `generate_order_status_reports`   | yes   | takes `GenerateOrderStatusReports` → `list[OrderStatusReport]` |
| `generate_fill_reports`           | yes   | takes `GenerateFillReports` → `list[FillReport]` |
| `generate_position_status_reports`| yes   | takes `GeneratePositionStatusReports` → `list[PositionStatusReport]` |
| `generate_mass_status`            | yes   | `(lookback_mins: int \| None = None) -> ExecutionMassStatus \| None` |

Optional: `_submit_order_list`, `_modify_order`, `_batch_cancel_orders`, `_query_account`,
`_query_order`.

**Account ID:** the constructor does not set an account id. Call
`self._set_account_id(AccountId(f"{client_id.value}-001"))` (or any string of the form
`"<ISSUER>-<NUMBER>"`) in `__init__`. **The issuer (the part before the `-`) MUST equal
`client_id.value`** — `Condition.equal` enforces this at runtime; mismatch raises
`ValueError: 'id.value' was not equal to 'account_id.get_issuer()'`. Verified.

**Emitting fill events: `self.generate_order_filled(...)`** — a base-class method (NOT
abstract) that constructs an `OrderFilled` event and publishes it via msgbus. Verified
signature:

```python
generate_order_filled(
    self,
    strategy_id: StrategyId,
    instrument_id: InstrumentId,
    client_order_id: ClientOrderId,
    venue_order_id: VenueOrderId,
    venue_position_id: PositionId | None,
    trade_id: TradeId,
    order_side: OrderSide,
    order_type: OrderType,
    last_qty: Quantity,
    last_px: Price,
    quote_currency: Currency,
    commission: Money,
    liquidity_side: LiquiditySide,
    ts_event: int,
    info: dict | None = None,
)
```

Companion generators exist for the full order lifecycle:
`generate_order_submitted`, `generate_order_accepted`, `generate_order_rejected`,
`generate_order_canceled`, `generate_order_cancel_rejected`, `generate_order_expired`,
`generate_order_triggered`, `generate_order_updated`, `generate_order_modify_rejected`,
`generate_order_denied`, `generate_account_state(balances, margins, reported, ts_event, info=None)`.

For more complex reconciliation flows, emit reports via `_send_order_status_report`,
`_send_fill_report`, `_send_position_status_report`, `_send_mass_status_report`,
`_send_order_event` — these route through the exec engine's reconciliation path.

### 10.6 Factory protocol

Both factory base classes are plain classes (NOT abstract / NOT msgspec) with a single
**`staticmethod` `create()`**. Verified by `inspect.getattr_static(LiveDataClientFactory, 'create')`
→ `staticmethod` (not `classmethod`).

```python
class LiveDataClientFactory:
    @staticmethod
    def create(
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: LiveDataClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> LiveDataClient: ...

class LiveExecClientFactory:
    @staticmethod
    def create(
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: LiveExecClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> LiveExecutionClient: ...
```

**Wiring into TradingNode (verified pattern):**

```python
config = TradingNodeConfig(
    ...,
    data_clients={"MY_DATA": LiveDataClientConfig()},
    exec_clients={"MY_EXEC": LiveExecClientConfig()},
)
node = TradingNode(config=config)
node.add_data_client_factory("MY_DATA", MyDataClientFactory)   # class, not instance
node.add_exec_client_factory("MY_EXEC", MyExecClientFactory)
node.build()
```

The key in `data_clients={"MY_DATA": ...}` must match the `name` passed to
`add_data_client_factory("MY_DATA", ...)`. The builder splits the key on `'-'` and uses
the prefix as the factory lookup (e.g. `"BINANCE-BTC"` → factory `"BINANCE"`).

**Extra factory args (advanced):** the `TradingNodeBuilder.build_exec_clients` method has
a hardcoded special-case for `SandboxLiveExecClientFactory.__name__` — if your factory
class is named exactly `"SandboxLiveExecClientFactory"`, the builder also passes
`portfolio=self._portfolio` to `create()`. For any other factory name only the 6 args
above are passed. Verified in `nautilus_trader/live/node_builder.py:240-249`.

**ImportableConfig path:** if a `data_clients` / `exec_clients` value is an
`ImportableConfig` instance (factory_path + config_path strings), the builder will import
and instantiate the factory class for you — useful for config-from-disk flows but not
required for in-process wiring.

### 10.7 CacheConfig — Parquet support? **No (for state); yes (for read-side data).**

`CacheConfig` fields:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `database` | `DatabaseConfig \| None` | `None` | only `'redis'` supported (see below) |
| `encoding` | `str` | `'msgpack'` | serialization for cached state |
| `timestamps_as_iso8601` | `bool` | `False` | |
| `persist_account_events` | `bool` | `True` | |
| `buffer_interval_ms` | `int \| None` | `None` | DB write buffer |
| `bulk_read_batch_size` | `int \| None` | `None` | |
| `use_trader_prefix` | `bool` | `True` | DB key prefix |
| `use_instance_id` | `bool` | `False` | |
| `flush_on_start` | `bool` | `False` | |
| `drop_instruments_on_reset` | `bool` | `True` | |
| `tick_capacity` | `int > 0` | `10000` | in-memory ring buffer |
| `bar_capacity` | `int > 0` | `10000` | in-memory ring buffer |

`DatabaseConfig.type: str` — **only `'redis'`** is supported. From the package's own docstring
(`nautilus_trader/common/config.py:309`): `type : str, {'redis'}, default 'redis'`. There is
**no Parquet/local-disk backend for the Cache** in v1.227. Verified.

**Consequences for Stage 4:**

- **Strategy / actor state persistence (`Trader.save()` / `load()` → `Cache.update_strategy()` /
  `load_strategy()`):** only works when `cache=CacheConfig(database=DatabaseConfig(type='redis', host='...'))`.
  With the default `cache=None` (or `cache=CacheConfig(database=None)`), the cache is purely
  in-memory and `save_state=True`/`load_state=True` are effectively no-ops across restarts.
- **Read-side Parquet catalog (for historical bars / instruments):** fully supported via
  `TradingNodeConfig.catalogs: list[DataCatalogConfig]`. `DataCatalogConfig(path: str,
  fs_protocol: str | None = None, ...)` mounts a `ParquetDataCatalog` for the data engine.
  This is the right surface for Stage 4's "replay from catalog" requirement.
- **Write-side Parquet streaming-out (for events / bars after the fact):**
  `TradingNodeConfig.streaming = StreamingConfig(catalog_path: str, flush_interval_ms,
  rotation_mode, include_types, ...)`. Pipes message-bus events to Parquet on the fly.

**Recommendation:** for a single-process replay-style live node (Stage 4), use
`cache=None` and rely on **re-replay on startup** as the persistence fallback. Document
this explicitly in the strategy: any in-memory state is rebuilt by re-feeding bars from
the catalog at startup. No Redis required.

### 10.8 Strategy persistence — `on_save` / `on_load` exist, but are cache-backed

**Yes, both hooks exist in v1.227** (Strategy inherits them from Actor):

```python
class Strategy:
    def on_save(self) -> dict[str, bytes]: ...        # override to return state to persist
    def on_load(self, state: dict[str, bytes]) -> None: ...  # override to restore state
    # Public wrappers (do NOT override these — they call on_save/on_load):
    def save(self) -> dict[str, bytes]: ...
    def load(self, state: dict[str, bytes]) -> None: ...
```

**Storage path:** `Trader.save()` calls each strategy's `save()` and writes the returned
dict to `Cache.update_strategy(strategy)`. `Trader.load()` calls
`Cache.load_strategy(strategy)` and passes the dict to `strategy.load(state)`. **The Cache
must be Redis-backed for these calls to actually persist across restarts** (see §10.7).

**Triggering:** `Trader.save()` / `Trader.load()` run automatically on
`TradingNode.run()` / `stop()` if `TradingNodeConfig.save_state=True` /
`load_state=True` (default both `False`).

**Caveat verified empirically:** calling `strategy.save()` on a strategy that has NOT been
registered with a trader (no clock/cache wired up) returns `None`. The hook works only
within an active TradingNode/BacktestEngine lifecycle.

**Fallback if Redis isn't acceptable for the project:** leave `save_state` /
`load_state` at their defaults (`False`), accept that strategy state is lost on restart,
and **re-replay enough historical bars on startup to rebuild the strategy's indicator /
position state**. This is the pattern Stage 4 should adopt.

### 10.9 End-to-end working snippet (verified runs to `bars_received == 1`)

This snippet is run verbatim from `/tmp/nautilus_recon/snippet_v1.py` on
`nautilus_trader==1.227.0`, Python 3.12, and produces:

```
TESTER-001._CountingStrategy: BAR RECEIVED: close=404.75, total=1
FINAL bars_received = 1
SUCCESS: end-to-end bar flowed through TradingNode -> Strategy.on_bar
```

```python
"""
End-to-end NautilusTrader v1.227 live runtime smoke test.
TradingNode + stub LiveMarketDataClient + stub LiveExecutionClient + counting Strategy.
"""

from __future__ import annotations

import asyncio
import time

from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.config import LoggingConfig
from nautilus_trader.live.config import (
    LiveDataClientConfig,
    LiveExecClientConfig,
    LiveExecEngineConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import AccountId, ClientId, InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

VENUE = Venue("ARCX")
INSTRUMENT_ID = InstrumentId(Symbol("SPY"), VENUE)
BAR_TYPE = BarType.from_str("SPY.ARCX-1-DAY-LAST-EXTERNAL")


def _make_spy_instrument() -> Equity:
    return Equity(
        instrument_id=INSTRUMENT_ID,
        raw_symbol=Symbol("SPY"),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


class _StubDataClient(LiveMarketDataClient):
    def __init__(self, loop, client_id, venue, msgbus, cache, clock, instrument_provider):
        super().__init__(
            loop=loop, client_id=client_id, venue=venue,
            msgbus=msgbus, cache=cache, clock=clock,
            instrument_provider=instrument_provider, config=None,
        )

    async def _connect(self) -> None:
        self._cache.add_instrument(_make_spy_instrument())
        self._set_connected(True)
        # Emit bar AFTER Trader.start() has fired (which subscribes the strategy).
        self.create_task(self._emit_after_delay(1.0), log_msg="emit_synthetic_bar")

    async def _disconnect(self) -> None:
        self._set_connected(False)

    async def _subscribe_bars(self, command) -> None:
        pass  # no-op; the stub emits internally via _emit_after_delay

    async def _unsubscribe_bars(self, command) -> None:
        pass

    async def _emit_after_delay(self, delay: float) -> None:
        await asyncio.sleep(delay)
        ts_ns = int(time.time() * 1e9)
        bar = Bar(
            bar_type=BAR_TYPE,
            open=Price.from_str("400.00"),
            high=Price.from_str("405.50"),
            low=Price.from_str("399.25"),
            close=Price.from_str("404.75"),
            volume=Quantity.from_int(1_000_000),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        self._handle_data(bar)


class _StubDataClientFactory(LiveDataClientFactory):
    @staticmethod
    def create(loop, name, config, msgbus, cache, clock):
        return _StubDataClient(
            loop=loop, client_id=ClientId(name), venue=VENUE,
            msgbus=msgbus, cache=cache, clock=clock,
            instrument_provider=InstrumentProvider(),
        )


class _StubExecClient(LiveExecutionClient):
    def __init__(self, loop, client_id, venue, msgbus, cache, clock):
        super().__init__(
            loop=loop, client_id=client_id, venue=venue,
            oms_type=OmsType.NETTING, account_type=AccountType.CASH,
            base_currency=USD, instrument_provider=InstrumentProvider(),
            msgbus=msgbus, cache=cache, clock=clock, config=None,
        )
        # Issuer (prefix) MUST equal client_id.value.
        self._set_account_id(AccountId(f"{client_id.value}-001"))

    async def _connect(self) -> None:    self._set_connected(True)
    async def _disconnect(self) -> None: self._set_connected(False)
    async def _submit_order(self, command) -> None: pass
    async def _cancel_order(self, command) -> None: pass
    async def _cancel_all_orders(self, command) -> None: pass
    async def generate_order_status_report(self, command):     return None
    async def generate_order_status_reports(self, command):    return []
    async def generate_fill_reports(self, command):            return []
    async def generate_position_status_reports(self, command): return []
    async def generate_mass_status(self, lookback_mins=None):  return None


class _StubExecClientFactory(LiveExecClientFactory):
    @staticmethod
    def create(loop, name, config, msgbus, cache, clock):
        return _StubExecClient(
            loop=loop, client_id=ClientId(name), venue=VENUE,
            msgbus=msgbus, cache=cache, clock=clock,
        )


class _CountingStrategyConfig(StrategyConfig, frozen=True):
    bar_type: str = str(BAR_TYPE)


class _CountingStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.bars_received = 0

    def on_start(self) -> None:
        self.subscribe_bars(BarType.from_str(self.config.bar_type))

    def on_bar(self, bar: Bar) -> None:
        self.bars_received += 1
        self.log.info(f"BAR RECEIVED: close={bar.close}, total={self.bars_received}")


def main() -> int:
    config = TradingNodeConfig(
        trader_id="TESTER-001",
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=False),  # critical for stub
        data_clients={"STUB": LiveDataClientConfig()},
        exec_clients={"STUB": LiveExecClientConfig()},
        timeout_connection=5.0, timeout_reconciliation=2.0,
        timeout_portfolio=2.0, timeout_disconnection=5.0,
        timeout_post_stop=2.0, timeout_shutdown=2.0,
    )
    node = TradingNode(config=config)
    node.add_data_client_factory("STUB", _StubDataClientFactory)
    node.add_exec_client_factory("STUB", _StubExecClientFactory)
    node.build()

    strategy = _CountingStrategy(_CountingStrategyConfig())
    node.kernel.trader.add_strategy(strategy)

    async def shutdown_after(delay: float) -> None:
        await asyncio.sleep(delay)
        await node.stop_async()

    node.kernel.loop.create_task(shutdown_after(4.0))
    node.run()                              # blocks; runs the loop
    print(f"FINAL bars_received = {strategy.bars_received}")
    node.dispose()
    return 0 if strategy.bars_received > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### 10.10 Summary for downstream Stage 4 tasks

- **Task 3 (CatalogLiveDataClient — fast-replay):** subclass `LiveMarketDataClient`,
  override `_connect` (load the catalog + schedule `_handle_data(bar)` calls for each row),
  `_disconnect`, `_subscribe_bars` (no-op or just register the bar_type), `_request_bars`
  (for any backfill queries). Use `self.create_task(self._replay_loop())` from `_connect`
  to drive emission asynchronously.
- **Task 4 (polling delta tests):** the bar-flow contract is `self._handle_data(bar)` →
  `Strategy.on_bar`. Test by registering the strategy via
  `node.kernel.trader.add_strategy(...)` and asserting the counter increments after
  `node.run()` completes.
- **Task 5 (SimulatedLiveExecutionClient):** subclass `LiveExecutionClient`. Override the
  10 methods listed in §10.5. Emit fills via `self.generate_order_filled(...)`. Set
  `client_id.value`-matching `AccountId` in `__init__`. Either follow the sandbox adapter's
  pattern of wrapping a `SimulatedExchange` + `BacktestExecClient`, or implement bar-close
  fills inline.
- **Task 6 (factories):** plain classes inheriting `LiveDataClientFactory` /
  `LiveExecClientFactory` with a `@staticmethod create(loop, name, config, msgbus, cache, clock)`.
  Avoid naming the exec factory `SandboxLiveExecClientFactory` unless you actually want the
  builder to inject `portfolio` (§10.6).
- **Task 7 (build_trading_node_config helper):** produce a `TradingNodeConfig` with
  `exec_engine=LiveExecEngineConfig(reconciliation=False)` (no live venue to reconcile
  against), `cache=None` (no Redis), `data_clients={"CATALOG": LiveDataClientConfig()}`,
  `exec_clients={"SIM": LiveExecClientConfig()}`, `logging=LoggingConfig(log_level=...)`.
- **Task 8 (parity test):** start node with `node.run()` then assert `node.portfolio` /
  `node.cache` state matches the equivalent `BacktestEngine` snapshot (§9.6). Stop via
  `loop.create_task(node.stop_async())` scheduled before `run()`.
- **Task 9 (runnable example):** sync `main()` builds + adds factories + calls
  `node.run()`. For long-running operation pair with an external signal handler or a
  scheduled `stop_async()` task.
- **No Parquet-backed strategy state persistence:** Stage 4 needs to fall back to
  in-memory + re-replay on startup. If Redis becomes acceptable later, wire it via
  `cache=CacheConfig(database=DatabaseConfig(type='redis', host=..., port=...))`,
  `load_state=True`, `save_state=True`, and implement `on_save`/`on_load` on each
  strategy.
