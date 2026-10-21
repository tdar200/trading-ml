"""Keyless best-effort intraday ingest into an isolated research-only store."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.multitf import storage_symbol
from data.tickers import BENCHMARK, SCALPING_CRYPTO, SCALPING_SECURITIES

FREE_INTRADAY_ROOT = ROOT / "data" / "free_intraday"
YFINANCE_INTERVALS = ("1m", "5m", "15m", "30m", "1h")


def provider_symbol(symbol: str) -> str:
    """Map the project's slash crypto notation to Yahoo's dash notation."""
    return symbol.replace("/", "-")


def normalize_yfinance_frame(
    frame: pd.DataFrame, symbol: str, *, asset_class: str,
) -> pd.DataFrame:
    """Normalize one yfinance response without using future data or adjustments."""
    columns = ["ticker", "ts", "open", "high", "low", "close", "volume", "source"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        yahoo_symbol = provider_symbol(symbol)
        if yahoo_symbol in data.columns.get_level_values(-1):
            data = data.xs(yahoo_symbol, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)
    data.columns = [str(column).lower().replace(" ", "_") for column in data.columns]
    missing = {"open", "high", "low", "close", "volume"} - set(data.columns)
    if missing:
        raise ValueError(f"Yahoo response for {symbol} missing columns: {sorted(missing)}")
    timestamps = pd.DatetimeIndex(data.index)
    if timestamps.tz is None:
        timezone = "UTC" if asset_class == "crypto" else "America/New_York"
        timestamps = timestamps.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    timestamps = timestamps.tz_convert("UTC")
    result = data[["open", "high", "low", "close", "volume"]].copy()
    result.insert(0, "ts", timestamps)
    result.insert(0, "ticker", symbol)
    result["source"] = "yfinance_research_only"
    result = result.dropna(subset=["open", "high", "low", "close"])
    result = result[result[["open", "high", "low", "close"]].gt(0).all(axis=1)]
    return result[columns].reset_index(drop=True)


def write_free_shards(frame: pd.DataFrame, timeframe: str, root: Path) -> int:
    if frame.empty:
        return 0
    written = 0
    data = frame.copy()
    data["month"] = data["ts"].dt.strftime("%Y-%m")
    for (ticker, month), group in data.groupby(["ticker", "month"]):
        path = root / storage_symbol(str(ticker)) / timeframe / f"{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, group], ignore_index=True)
        else:
            combined = group
        columns = ["ticker", "ts", "open", "high", "low", "close", "volume", "source"]
        combined[columns].drop_duplicates("ts", keep="last").sort_values("ts").to_parquet(
            path, index=False,
        )
        written += len(group)
    return written


def fetch_yfinance(symbol: str, *, period: str, interval: str, asset_class: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        provider_symbol(symbol), period=period, interval=interval,
        auto_adjust=True, prepost=False, progress=False, threads=False,
        actions=False, repair=True,
    )
    return normalize_yfinance_frame(frame, symbol, asset_class=asset_class)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-class", choices=("stock", "crypto"), default="stock")
    parser.add_argument("--timeframe", choices=YFINANCE_INTERVALS, default="5m")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--tickers")
    parser.add_argument("--output-root", type=Path, default=FREE_INTRADAY_ROOT)
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="Rebuild provenance/count metadata from existing shards without downloading.",
    )
    args = parser.parse_args()
    if args.tickers:
        tickers = tuple(dict.fromkeys(x.strip().upper() for x in args.tickers.split(",") if x.strip()))
    else:
        tickers = SCALPING_CRYPTO if args.asset_class == "crypto" else (*SCALPING_SECURITIES, BENCHMARK)
    failures = []
    total = 0
    for index, ticker in enumerate(tickers, start=1):
        if args.manifest_only:
            directory = args.output_root / storage_symbol(ticker) / args.timeframe
            count = sum(len(pd.read_parquet(path)) for path in directory.glob("*.parquet"))
            total += count
            if count == 0:
                failures.append(ticker)
            print(f"[{index}/{len(tickers)}] {ticker}: {count} stored bars")
            continue
        try:
            frame = fetch_yfinance(
                ticker, period=args.period, interval=args.timeframe,
                asset_class=args.asset_class,
            )
            total += write_free_shards(frame, args.timeframe, args.output_root)
            print(f"[{index}/{len(tickers)}] {ticker}: {len(frame)} bars")
            if frame.empty:
                failures.append(ticker)
        except Exception as exc:  # a single Yahoo symbol must not corrupt other shards
            failures.append(ticker)
            print(f"[{index}/{len(tickers)}] {ticker}: FAILED ({exc})")
    manifest = {
        "source": "yfinance_research_only", "asset_class": args.asset_class,
        "timeframe": args.timeframe,
        "period": args.period, "symbols_requested": list(tickers),
        "failed_symbols": failures, "rows_received": total,
        "sip_nbbo": False, "broker_verified": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"manifest_{args.asset_class}_{args.timeframe}.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"wrote {total} research-only bars; failures={len(failures)}; root={args.output_root}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
