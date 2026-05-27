# Stage 9 Research — Interactive Dashboard for NautilusTrader

Research synthesis from crawl4ai-fetched docs + community projects (2026-05-27).

## 1. Official Nautilus position

**Nautilus does NOT ship a long-running web dashboard.** Per multiple sources (the official site, GitHub discussions, and the search-results summaries):

> "UI dashboards and distributed orchestration are out of scope for the official NautilusTrader project to maintain focus on the core engine."

What it DOES ship (as of 1.222+, currently installed 1.227):

- `nautilus_trader.analysis.tearsheet` — **Plotly-based interactive HTML tearsheets**. Single function `create_tearsheet(engine, output_path, config=...)` produces a self-contained `.html` file you open in a browser. Interactive (Plotly widgets) but static (no live state, no server).
- 9 built-in chart functions: `equity`, `drawdown`, `monthly_returns`, `distribution`, `rolling_sharpe`, `yearly_returns`, `bars_with_fills`, `run_info`, `stats_table`.
- Themes: `plotly_white`, `plotly_dark`, `nautilus`, `nautilus_dark`; custom theme registry.
- Chart Registry (`register_chart` decorator) for plugging in custom charts.
- Requires `plotly>=6.3.1` (install via `uv pip install "nautilus_trader[visualization]"`).
- Per-currency filtering for multi-currency backtests.

So Nautilus's official offer is exactly what `quantstats` gives us in Stage 2 — a one-shot HTML report — just better-integrated with Nautilus's internal stats.

## 2. Community state — what people have actually built

| Project | Stack | Stars | What it covers |
|---|---|---|---|
| **[Black101081/Nautilus-Web-Interface](https://github.com/Black101081/Nautilus-Web-Interface)** | **FastAPI + React/Vite + Recharts + WebSocket + SQLite + Tailwind/Radix** | 24 ⭐ | Full-stack admin: strategies CRUD, orders/positions, risk metrics, **live Binance market data** (5s TTL cache, no key needed), persistent alerts (SQLite), backtesting with parameter sweep, system monitoring (psutil), JWT auth + API-key middleware, Cloudflare Pages deploy, 192 backend tests. v2.0.0, Mar 2026. **The closest existing reference for "production-grade Nautilus dashboard."** |
| [Patrick-code-Bot/nautilus_AItrader](https://github.com/Patrick-code-Bot/nautilus_AItrader) | NautilusTrader + Telegram Bot | — | Telegram-as-UI for live notifications. Not a dashboard but a useful reference for the "live ops surface" question. |
| [dang-trung/base-trading](https://github.com/dang-trung/base-trading) | Plotly + Dash | — | Standalone Dash app for backtest visualization. Not Nautilus-specific but a clean Dash-only reference architecture. |
| [jesse-ai/jesse](https://github.com/jesse-ai/jesse) (different framework, comparable design) | FastAPI + Redis pub/sub + WebSocket | many | Battle-tested live-trading UI pattern: isolated worker processes publish to Redis channels, WebSocket manager forwards to connected clients. This is the canonical architecture for "watch a live strategy run from a browser." |

### Black101081/Nautilus-Web-Interface — architecture detail

```
React (TS, Vite)    ──HTTP/WS──▶    nautilus_fastapi.py    ──Python──▶ NautilusTradingSystem
+ Recharts                          (production server)                nautilus_core.py
+ TanStack Query                              │                        nautilus_integration.py
+ Wouter (router)                             │
+ Tailwind + Radix                  market_data_service ──▶ Binance API
                                    alerts_db           ──▶ SQLite
                                    auth                ──▶ JWT + API-key
```

**Pages they ship** (Trader + Admin):
- Trader: Dashboard, Strategies, Orders, Positions, Risk, Market Data, Performance, Alerts, **Backtesting** (incl. parameter sweep)
- Admin: Components, Adapters, Monitoring (psutil), Settings, Database (PostgreSQL/Parquet/Redis ops)

**API surface** (REST + WS):
- `/api/engine/*` — initialize/info/shutdown
- `/api/strategies` — CRUD + start/stop
- `/api/orders`, `/api/positions`, `/api/trades`, `/api/account`
- `/api/risk/metrics`, `/api/risk/limits`
- `/api/market-data/*` — live Binance proxy
- `/api/nautilus/backtest`, `/api/nautilus/demo-backtest`, **`/api/nautilus/parameter-sweep`** (grid over SMA fast/slow)
- `WS /ws` — engine/position/risk push at 2s interval (JWT-authed via `?token=`)

This is the most complete reference. The trade-off is **complexity**: ~6 backend services, ~25 React pages, JWT auth, two FastAPI processes (main + admin DB), CI/CD to Cloudflare Pages — much more than our current Streamlit dashboard.

## 3. The three plausible Stage 9 paths

| | A. Extend Stage 7 Streamlit | B. Plotly Dash rebuild | C. FastAPI + React (Black101081 pattern) |
|---|---|---|---|
| **Effort** | ~1 day | ~3-5 days | ~2-3 weeks |
| **Charts** | Swap `st.line_chart` → `st.plotly_chart(create_equity_curve(...))` etc | Full Plotly Dash callback graph | Recharts (or Plotly via react-plotly) |
| **Live strategy runner** | ❌ Streamlit can't poll a live TradingNode well | ⚠️ Possible via `dcc.Interval` + Redis | ✅ WebSocket-native |
| **Backtest runner UI** | ✅ `st.form` + subprocess call to `examples/run_harness.py` | ✅ Dash callback | ✅ POST + WebSocket progress events |
| **Strategy CRUD** | ⚠️ Forms work but state mgmt awkward | ✅ Stateful via dcc.Store | ✅ TanStack Query + React state |
| **Auth** | Streamlit's basic auth (community plugin) | None built-in (use proxy) | ✅ JWT in repo already |
| **Mobile-friendly** | ⚠️ Streamlit's mobile is functional but cramped | ✅ Plotly Dash responsive | ✅ React-native ready |
| **Deploy** | `streamlit run`; single port | Plotly Dash `app.run_server` | Two-process + reverse proxy |
| **Risk if you walk away** | Low — minimal new code | Medium — new framework dependency | High — full webapp to maintain |

## 4. Recommendation

**Tiered approach. Pick A, optionally upgrade to C later.**

### Stage 9.A — "Best of both" (1-day effort, recommended)

Keep our Stage-7 Streamlit dashboard's layout (5-tab navigation, journal, multi-stage views) but swap the chart-rendering layer to Nautilus's Plotly tearsheet primitives:

- `dashboard/views/equity.py` — replace `st.line_chart(curves)` with:
  ```python
  from nautilus_trader.analysis.tearsheet import create_equity_curve, create_drawdown_chart, create_rolling_sharpe
  st.plotly_chart(create_equity_curve(returns, ...), use_container_width=True)
  st.plotly_chart(create_drawdown_chart(returns), use_container_width=True)
  st.plotly_chart(create_rolling_sharpe(returns), use_container_width=True)
  ```
- Add a sixth tab: **"Backtest Runner"** — `st.form` with strategy + params + window selectors, on submit shell out to `examples/run_harness.py` and stream output via `st.write_stream`. Cheap; uses what we already have.
- Add a seventh tab: **"Strategy Runner"** — same pattern around `examples/run_trading_node.py --exec-mode {simulated, alpaca-paper}` with `st.code` showing the launch command and a button that runs it via `subprocess.Popen` (or just prints the command for the user to run manually if local-process spawning feels iffy).
- Drop `quantstats` from Stage 2 (replace with `create_tearsheet`), shrinking deps.

**What you get:** Plotly-quality interactive charts everywhere, backtest + strategy-runner UI from the same browser tab as the verdicts, no new framework. No live-WebSocket state — that's the trade-off — but daily-bar strategies don't need it.

### Stage 9.B — Plotly Dash if Streamlit feels limiting (3-5 days)

Only if the user wants smoother interactivity (drag-to-zoom across linked charts, live-updating widgets without page rerun). Same rough scope as 9.A but as a Dash app — uses `dcc.Graph` + callbacks. Lose Streamlit's hot-reload simplicity, gain Plotly Dash's cleaner reactivity.

### Stage 9.C — Full webapp (2-3 weeks)

Replicate Black101081's pattern: FastAPI backend + React frontend + WebSocket live updates. Justified ONLY if:
- The 3-month paper observation reveals you actually need real-time push (not poll) for live monitoring.
- You're going multi-user.
- You're deploying to a remote VPS and need real auth.

Until any of those is true, Stage 9.A gives 80% of the value for 5% of the effort.

## 5. What I would actually build next

If you said "go" right now, I'd land Stage 9.A in ~1 day:

1. Add `nautilus_trader[visualization]` extra to `pyproject.toml` (pulls in `plotly>=6.3.1`).
2. Rewrite `dashboard/views/equity.py` to use Nautilus's `create_equity_curve` / `create_drawdown_chart` / `create_rolling_sharpe`. ~30 lines.
3. Add `dashboard/views/backtest_runner.py` — st.form → subprocess `examples/run_harness.py` → stream output. ~60 lines.
4. Add `dashboard/views/strategy_runner.py` — st.form → subprocess `examples/run_trading_node.py --exec-mode alpaca-paper --poll-seconds 60` → show stdout. ~60 lines.
5. Wire 2 new tabs into `dashboard/app.py`.
6. Update README + 5-line spec note.

Estimated test impact: +0 tests (subprocess views aren't unit-tested by convention; data loaders already covered).

## 6. Wider library survey (added 2026-05-27, second crawl pass)

The first cut covered Streamlit / Plotly Dash / FastAPI+React. There's a whole space we missed — especially **purpose-built financial charting**, which is the most leveraged choice for a trading dashboard.

### 6.1 The killer find — `lightweight-charts-python`

| Library | Stars | What it is |
|---|---|---|
| **[louisnw01/lightweight-charts-python](https://github.com/louisnw01/lightweight-charts-python)** | **2.0k** ⭐ | Pythonic wrapper around [TradingView's open-source Lightweight Charts JS library](https://github.com/tradingview/lightweight-charts). `pip install lightweight-charts`. |

**Why it matters for us**: this is what serious retail trading UIs render charts with. It gives us:

- `chart.set(df)` from a pandas OHLCV DataFrame — done.
- **`chart.update(series)` / `chart.update_from_tick(tick)` — live updates per bar / per tick.** This is the live-charts story Streamlit's `st.line_chart` can't give us.
- **Multi-pane subcharts** (price + volume + indicators stacked) via `chart.create_subchart(...)`.
- **Drawing toolbox**: trendlines, rectangles, rays, horizontal lines drawn directly on the chart by the user.
- **Topbar widgets**: textbox (symbol search), switcher (timeframe), with callback hooks.
- **Tables**: built-in watchlist / order-entry / trade-management widgets next to the chart.
- **Event callbacks**: `events.search`, `on_timeframe_selection`, `on_horizontal_line_move` etc.
- **Supports Streamlit, Jupyter, PyQt5/6, PySide6, wxPython, asyncio** — embeds anywhere.
- Direct Polygon.io market-data integration if we want it.

For our Stage-7 dashboard, this swaps `st.line_chart(equity_df)` for an actual trading-platform-quality candlestick + indicator chart in the Equity tab, and unlocks a Live tab where ticks from a running TradingNode update the chart in real time.

**Trade-offs**: it's TypeScript+CSS heavy (~50% of the repo is JS). The candlestick chart is *the* product — it doesn't give you the verdict cards, journal, model coverage tabs that our Streamlit dashboard does. Use it as a chart engine inside Streamlit, not as a dashboard framework.

### 6.2 Python-native dashboard frameworks (broader than the first cut)

| Framework | Style | When to pick |
|---|---|---|
| **Streamlit** (we use) | Script-rerun on interaction | Fastest "from zero to dashboard." Rigid layouts, hard to customize beyond components, but unmatched DX for research dashboards. |
| **[Plotly Dash](https://dash.plotly.com/)** | Callback-graph, React under the hood | Best plot interactivity. Enterprise option. Steeper curve — components, views, callbacks, sometimes HTML. |
| **[Panel](https://panel.holoviz.org/)** (HoloViz) | High-level reactive + low-level callback | Most flexible PyData ecosystem dashboard. Multi-page apps, grid layouts. Steeper curve than Streamlit. |
| **[NiceGUI](https://nicegui.io/)** | FastAPI-based, Vue/Quasar widgets | Better for "application-style" UIs — forms, admin panels, live monitoring. More control over events + state than Streamlit. 2.0 ships with Tailwind + better TS. |
| **[Reflex](https://reflex.dev/)** (formerly Pynecone) | React-style in Python (compiles to Next.js) | Full webapp without writing JS, but you need to think in React idioms. |
| **[Solara](https://solara.dev/)** | Built on Reacton (React + ipywidgets) | "Great promise" per surveys. Sits between Streamlit (easy) and Reflex (powerful). |
| **[Shiny for Python](https://shiny.posit.co/py/)** | Reactive (R Shiny port) | Doesn't re-render on every interaction — minimal re-render of affected components. Closest thing to "Streamlit done right" if reactivity bites. |
| **[Gradio](https://gradio.app/)** | Form-builder | ML-demo focused; embeds in Jupyter; HuggingFace Spaces native. Wrong fit for a trading dashboard. |
| **[Taipy](https://taipy.io/)** | Notebook-style + scenarios | Newer; good data-viz primitives; pipeline-aware. |
| **[Mercury](https://github.com/mljar/mercury)** / **[Voila](https://voila.readthedocs.io/)** | Notebook → web app | If our research lives in `.ipynb` already (it doesn't). |
| **[Marimo](https://marimo.io/)** | Reactive notebooks (deterministic execution) | New, gaining traction. Cleaner than Jupyter but new ecosystem. |
| **[Anvil](https://anvil.works/)** | Drag-and-drop + Python | Has a paywall for serious features. Cool for non-coders. |

### 6.3 Financial-charting libraries (the chart layer)

| Library | Style | Live updates | Trading-aware? |
|---|---|---|---|
| **`lightweight-charts-python`** | TradingView JS | ✅ tick / bar | ✅ purpose-built |
| **[Plotly](https://plotly.com/python/)** (Nautilus uses this) | JS via React/Plotly.py | ⚠️ via streaming traces, awkward | ⚠️ generic |
| **[Bokeh](https://bokeh.org/)** | JS via Bokeh server | ✅ via Bokeh server | ⚠️ generic |
| **[mplfinance](https://github.com/matplotlib/mplfinance)** | matplotlib | ❌ static | ✅ candle-aware |
| **[finplot](https://pypi.org/project/finplot/)** | pyqtgraph (desktop) | ✅ very fast | ✅ candle-aware |
| **[mplchart](https://github.com/furechan/mplchart)** | matplotlib | ❌ static | ✅ |
| **[Altair](https://altair-viz.github.io/)** | Vega-Lite | ❌ | ❌ generic |

**Takeaway**: for browser dashboards there are two serious choices — TradingView LWC (purpose-built) and Plotly (general-purpose, integrated into Nautilus). For desktop apps **finplot** is the fast one.

### 6.4 Backtesting libraries that ship their own visualization

These compete with the "use Nautilus + a separate dashboard" approach by giving you charts out of the box. Even if we stay on Nautilus, they're worth mentioning as reference UIs:

- **[VectorBT](https://vectorbt.dev/)** / **[VectorBT Pro](https://vectorbt.pro/)** — integrates Plotly + Jupyter widgets directly. Backtest a strategy, get an interactive dashboard in the notebook. The community has built [Streamlit apps around VectorBT](https://www.marketcalls.in/python/vectorbt-streamlit-backtesting-app-python-tutorial.html) (input EMA periods → run → see Plotly equity / drawdown / trade list). Closest competitor in spirit to what we built.
- **[backtrader](https://www.backtrader.com/)** — built-in matplotlib charts.
- **[zipline-reloaded](https://zipline.ml4trading.io/)** + **[pyfolio](https://github.com/quantopian/pyfolio)** — pyfolio's tearsheet (legacy but still used).
- **[QuantStats](https://github.com/ranaroussi/quantstats)** (we use in Stage 2) — HTML tearsheet, no live.

### 6.5 Live-trading UI architectures worth borrowing from

- **Jesse AI** — FastAPI server + Redis pub/sub + WebSocket forwarder. Battle-tested pattern.
- **vnpy** (Chinese open-source platform) — PyQt-based desktop UI with full live trading.
- **Freqtrade** — ships its own React dashboard for crypto strategies.

## 7. Revised recommendation (with `lightweight-charts-python` in the mix)

The earlier "Stage 9.A in ~1 day" plan stands, but with a different chart layer:

| | A. Streamlit + Plotly (was first rec) | **A'. Streamlit + `lightweight-charts-python`** | B. Plotly Dash | C. NiceGUI | D. FastAPI + React |
|---|---|---|---|---|---|
| Effort (new code) | ~1 day | **~1.5 days** | 3-5 days | ~3 days | ~2-3 weeks |
| Equity tab chart | st.plotly_chart(Nautilus chart) | **TradingView LWC candle + volume subchart** | dcc.Graph | ui.plotly | Recharts/react-plotly |
| **Live tick updates** | ❌ | **✅ via chart.update_from_tick** | ✅ via Interval+Redis | ✅ async + ui.timer | ✅ WebSocket |
| Backtest runner UI | st.form + subprocess | st.form + subprocess | dcc.Form + callback | ui.input + button | React form + POST |
| Strategy CRUD | OK | OK | Better | **Best (form-native)** | Best |
| Drawing tools on chart | ❌ | **✅ trendlines, rays, rectangles** | ⚠️ via Plotly editable | ❌ | only if you build them |
| Mobile-friendly | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |

**My updated recommendation: Stage 9.A' = Streamlit + lightweight-charts-python.**

- Adds ~1 dep (`pip install lightweight-charts`)
- Replaces our Equity tab's `st.line_chart` with `lightweight_charts.widgets.streamlit.StreamlitChart`
- Gives us drawing tools + candlesticks + live-tick updates "for free"
- Same backtest/strategy runner tabs as the original Stage 9.A
- Keeps all 5 of our existing tabs intact — only swaps the chart engine inside Equity (and optionally a new Live tab)
- ~1.5 days instead of 1 because we need to add a candle-data loader to `dashboard/data_sources.py` (the existing one only returns close-price series for equity comparison)

Want me to land it?

## 8. Backtesting / strategy tester / strategy selector — deep dive

The previous sections covered the dashboard shell + chart engine. The "Backtest Lab" tab — strategy selector + parameter form + tester + result comparison — is its own design question with its own established patterns. Five reference implementations:

### 8.1 Pattern: built-in optimizer + plot ([backtesting.py](https://github.com/kernc/backtesting.py), 8.4k ⭐)

The simplest end-to-end. Define a `Strategy` subclass, run `bt.optimize(n1=range(5,50,5), n2=range(10,100,10))`, get a `stats` DataFrame + interactive Bokeh plot via `bt.plot()`. Single-strategy, single-asset, no multi-config comparison built in. Good for a "one strategy at a time" UI; weak for the cross-strategy selector our Stage-3 winner search needs.

### 8.2 Pattern: vectorized parameter sweep ([VectorBT](https://github.com/polakowo/vectorbt), 7.7k ⭐)

The gold standard for "what does this parameter do?" visualization. `vbt.MA.run_combs(price, window=np.arange(2,101), r=2)` evaluates **10,000 dual-SMA combinations in seconds** via NumPy + Numba vectorization. Then:

```python
pf.total_return().vbt.heatmap(
    x_level="fast_window", y_level="slow_window", slider_level="symbol",
    symmetric=True,
)
```

→ Plotly heatmap of return-by-parameter with a per-symbol slider. Drill into any cell: `pf[(10, 20, "ETH-USD")].stats()` returns the full per-config stat block.

**For our project**: VectorBT competes with Nautilus on the "engine" axis, so adopting it for parameter-sweep would require routing one strategy through two engines (Nautilus for the deployed path, VectorBT for the explorer). Architectural cost ≠ trivial — better to write a *Nautilus-native* parameter sweep that **mimics** VectorBT's UX (heatmap + drill-down) without forking the engine choice.

### 8.3 Pattern: real-time hyperparameter optimization UI ([optuna-dashboard](https://github.com/optuna/optuna-dashboard), 775 ⭐)

**The killer find for hyperparameter search.** Standalone web app (`pip install optuna-dashboard`, then `optuna-dashboard sqlite:///db.sqlite3`) that visualizes any Optuna study in real time:

- **Optimization history** (objective value vs trial number, with best-so-far line)
- **Hyperparameter importances** (via random-forest feature importance over completed trials)
- **Parallel coordinates** plot — see how all hyperparams co-vary
- **Slice plots** — marginal effect of each param
- **Contour plots** — pair-wise interactions

Pluggable: SQLite, MySQL, or PostgreSQL backend. Has an **experimental browser-only version** (drag-drop SQLite file at `https://optuna.github.io/optuna-dashboard/`), VS Code extension, and JupyterLab extension.

**Integration with our project**: trivially adaptable. Wrap our existing Stage-3 walk-forward as an Optuna objective:

```python
def objective(trial):
    lookback = trial.suggest_int("lookback_days", 30, 200)
    skip = trial.suggest_int("skip_days", 0, 60)
    top_n = trial.suggest_int("top_n", 1, 5)
    cfg = MomentumTopNConfig(universe=UNIVERSE, benchmark=BENCHMARK,
                             top_n=top_n, lookback_days=lookback, skip_days=skip)
    return run_walk_forward(cfg).median_window_sharpe  # higher = better

study = optuna.create_study(storage="sqlite:///optuna.db", direction="maximize")
study.optimize(objective, n_trials=200, n_jobs=4)
```

Then point optuna-dashboard at `optuna.db` and explore. **This is dramatically richer than our current 5×6×3 = 90-cell grid search**, and the dashboard is built — we just integrate.

### 8.4 Pattern: full-stack admin UI ([FreqUI](https://github.com/freqtrade/frequi), 995 ⭐ Vue.js)

Freqtrade's official frontend. Production-grade reference for "trader-facing strategy tester":

- **Strategy selector** — dropdown auto-populated from the bot's `/strategies/` directory; can view source code in-browser
- **Pair selector** — multi-select of trading pairs
- **Timeframe selector** — `1m / 5m / 15m / 1h / 4h / 1d`
- **Backtest panel** — run a backtest with all the above + date range, watch progress, view results
- **Hyperopt panel** — define parameter space, run optimization, browse results
- **Trade list table** — sortable, filterable, with per-trade entry/exit chart overlays
- **Performance breakdown** — by pair, by month, by strategy, by exit signal

Built with Vue 3 + PrimeVue + TypeScript. Worth examining if our Stage 9.C path materializes — FreqUI is more polished and more battle-tested than Black101081's React project. Crypto-focused so the "pairs" abstraction doesn't 1:1 map to our ETF universe, but the **selector + form + results** UX is the canonical reference.

### 8.5 Pattern: parameter-sweep REST endpoint (Black101081's `/api/nautilus/parameter-sweep`)

Simplest webapp pattern. `POST /api/nautilus/parameter-sweep` accepts:
```json
{"fast_range": [5, 50, 5], "slow_range": [10, 100, 10]}
```
returns a ranked array of `{fast, slow, pnl}` capped at 25 combinations. Frontend renders the results as a sortable table with the best config highlighted. No real visualization, but cheap to build, easy to wrap in a Streamlit form.

---

## 9. Concrete design for a "Backtest Lab" tab in our dashboard

Synthesizing what works:

### 9.1 Layout (single Streamlit tab, 4 sections)

```
┌─ Strategy selector ────────────────────────┐
│ [Strategy ▼] MomentumTopN | MomentumMeta │
│ [View source ↗]                           │
└────────────────────────────────────────────┘
┌─ Parameter form (auto-rendered from        │
│   Pydantic StrategyConfig schema) ─────────┐
│ top_n:         [2     ▼]  (1-5)            │
│ lookback_days: [63    ▼]  (20-252)         │
│ skip_days:     [0     ▼]  (0-60)           │
│ universe:      [✓ XLK ✓ XLF ... ]          │
│ filter_threshold: [0.55] (for meta)        │
└────────────────────────────────────────────┘
┌─ Test mode ────────────────────────────────┐
│ ○ Single backtest                          │
│ ○ Grid sweep (define ranges)               │
│ ● Optuna optimization (n_trials, objective)│
│   Window: [2020-07-27] → [2026-05-22]      │
│   [Run]                                     │
└────────────────────────────────────────────┘
┌─ Results ──────────────────────────────────┐
│  [Equity tab] [Trades] [Stats] [Heatmap]  │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│   <chart / table for selected sub-tab>     │
│   [Save experiment: my-sma-sweep] [Save]   │
│   [Compare to Stage-3 winner: +X% Sharpe]  │
└────────────────────────────────────────────┘
```

### 9.2 Mode-specific behaviors

**Single backtest** — just calls `analysis.runner.run(spec)` with the form values, displays the same per-window summary card the current Harness Verdicts tab uses + an equity overlay vs the deployed strategy. <5 sec for our catalog.

**Grid sweep** — user specifies range per parameter; we call `analysis.walk_forward.walk_forward(...)` (same engine as Stage-3 harness). Results: sortable DataFrame (Sharpe, CAGR, max DD) + 2D heatmap if exactly 2 dimensions varied. Same engine = same numbers as the canonical harness; no engine fork. Limit: ≤200 cells (~1 min compute) before suggesting Optuna mode.

**Optuna optimization** — add Optuna as a dep, wire `objective(trial)` over our walk-forward harness, launch with `n_trials=N`. Pop open optuna-dashboard in a separate browser tab for the rich visualization (parallel coordinates, hyperparam importance) — don't reimplement those in-Streamlit. The dashboard auto-reads from the SQLite file the runner writes to.

### 9.3 Strategy selector — auto-discovery

```python
# dashboard/data_sources.py
import importlib, pkgutil, inspect
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

def discover_strategies() -> dict[str, tuple[type[Strategy], type[StrategyConfig]]]:
    """Walk strategies/ for (Strategy, StrategyConfig) class pairs."""
    out = {}
    for _, name, _ in pkgutil.iter_modules(["strategies"]):
        mod = importlib.import_module(f"strategies.{name}")
        strategies = [obj for _, obj in inspect.getmembers(mod, inspect.isclass)
                      if issubclass(obj, Strategy) and obj is not Strategy]
        configs = [obj for _, obj in inspect.getmembers(mod, inspect.isclass)
                   if issubclass(obj, StrategyConfig) and obj is not StrategyConfig]
        if strategies and configs:
            out[name] = (strategies[0], configs[0])
    return out
```

→ dropdown auto-populates with `momentum_topn`, `momentum_meta`. Adding a new strategy = drop a file in `strategies/`, dashboard picks it up.

### 9.4 Parameter form auto-rendering

Pydantic introspection on the selected StrategyConfig:

```python
from pydantic.fields import FieldInfo

def render_param_form(config_cls: type[StrategyConfig]) -> dict:
    """Auto-render st.input widgets per field in the StrategyConfig."""
    values = {}
    for name, field in config_cls.model_fields.items():
        if field.annotation is int:
            values[name] = st.number_input(name, value=field.default or 0, step=1)
        elif field.annotation is float:
            values[name] = st.number_input(name, value=field.default or 0.0, format="%.4f")
        elif field.annotation is bool:
            values[name] = st.checkbox(name, value=field.default)
        elif "tuple" in str(field.annotation).lower():
            values[name] = st.multiselect(name, options=ALL_TICKERS,
                                          default=list(field.default or []))
        # ... more types
    return values
```

Zero hand-coded forms, zero drift when a new field is added to a config.

### 9.5 Effort estimate

Stage 9.A' was 1.5 days. Adding the Backtest Lab tab as described:
- Strategy discovery: ~30 lines
- Param form auto-render: ~50 lines
- Single + grid mode wiring to `analysis.walk_forward.walk_forward`: ~80 lines
- Optuna integration + dashboard launch helper: ~60 lines + 1 new dep (`optuna>=4.0`, `optuna-dashboard>=0.20`)
- Results display (4 sub-tabs): ~120 lines
- "Compare to Stage-3 winner" diff: ~30 lines
- Tests: ~5 unit tests on the discovery + form-render helpers

**Total: +1 day** on top of Stage 9.A'. Combined Stage 9.A' + Lab = ~2.5 days for a dashboard that has:
- TradingView-quality charts (Equity, future Live)
- Cross-stage research navigation (Harness Verdicts, Sentiment, Models, Journal)
- **Self-service backtest + grid sweep + Optuna optimization with auto-discovered strategies and auto-rendered parameter forms**

That's the full vision. The pieces all exist; we're gluing.

## Sources

Crawled via crawl4ai at `http://localhost:11235/md`:
- [NautilusTrader Visualization (nightly docs)](https://nautilustrader.io/docs/nightly/concepts/visualization/) — official tearsheet API surface
- [Backtest visualisation discussion #245](https://github.com/nautechsystems/nautilus_trader/discussions/245) — community feature requests + maintainer responses
- [Black101081/Nautilus-Web-Interface](https://github.com/Black101081/Nautilus-Web-Interface) — most complete community webapp
- [Live Trading concepts](https://nautilustrader.io/docs/latest/concepts/live/) — reconciliation primitives a live dashboard would use

Backtest / optimizer / strategy-tester deep-dive crawls:
- [optuna/optuna-dashboard](https://github.com/optuna/optuna-dashboard) — real-time hyperparameter dashboard (775⭐, MIT, Docker-packaged)
- [polakowo/vectorbt](https://github.com/polakowo/vectorbt) — vectorized backtester with heatmap UI (7.7k⭐)
- [kernc/backtesting.py](https://github.com/kernc/backtesting.py) — single-strategy backtester with built-in optimizer + Bokeh plot (8.4k⭐)
- [freqtrade/frequi](https://github.com/freqtrade/frequi) — production Vue.js frontend for Freqtrade with strategy selector / backtest / hyperopt panels (995⭐)
- [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) — `reports.compare(...)` for multi-strategy tearsheet comparison

Plus WebSearch returns linking to:
- [nautechsystems/nautilus_trader main repo](https://github.com/nautechsystems/nautilus_trader)
- [jesse-ai/jesse architecture](https://deepwiki.com/jesse-ai/jesse) — FastAPI + Redis + WebSocket reference
- [dang-trung/base-trading](https://github.com/dang-trung/base-trading) — Plotly Dash reference for backtest UI
- [Nautilus 1.222.0 release](https://github.com/nautechsystems/nautilus_trader/releases) — when Plotly tearsheets first shipped
