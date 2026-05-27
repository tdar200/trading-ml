"""
Stage-8 preflight: smoke-test connectivity to the Alpaca paper account
WITHOUT committing the whole TradingNode.

Prints account balance + open positions and exits 0 on success.
Exits 1 on missing credentials or auth failure with a clear error.

Usage:
    uv run python examples/preflight_alpaca.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")


def main() -> int:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    if not api_key or not secret_key:
        print(
            "ERROR: ALPACA_API_KEY / ALPACA_API_SECRET missing from .env. "
            "Get free paper-account keys at https://alpaca.markets/ and "
            "populate .env (see .env.example)."
        )
        return 1

    from alpaca.trading.client import TradingClient

    try:
        client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        account = client.get_account()
    except Exception as exc:
        print(f"ERROR: failed to authenticate against Alpaca paper API: {exc}")
        return 1

    print("=== Alpaca paper account ===")
    print(f"  account id:      {account.id}")
    print(f"  status:          {account.status}")
    print(f"  equity:          ${float(account.equity):,.2f}")
    print(f"  cash:            ${float(account.cash):,.2f}")
    print(f"  buying power:    ${float(account.buying_power):,.2f}")
    print(f"  pattern day:     {account.pattern_day_trader}")

    try:
        positions = client.get_all_positions()
    except Exception as exc:
        print(f"warning: get_all_positions failed: {exc}")
        positions = []

    print(f"\n=== Open positions ({len(positions)}) ===")
    if not positions:
        print("  (none)")
    else:
        for p in positions:
            print(
                f"  {p.symbol:6s}  qty={p.qty:>8s}  avg=${float(p.avg_entry_price):>8.2f}  "
                f"mkt=${float(p.market_value):>10,.2f}  pl=${float(p.unrealized_pl):>+10,.2f}"
            )

    # Recent (last 10) orders, for situational awareness.
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        recent = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=10,
        ))
    except Exception as exc:
        print(f"warning: get_orders failed: {exc}")
        recent = []

    print(f"\n=== Recent orders (last {len(recent)}) ===")
    if not recent:
        print("  (none)")
    else:
        for o in recent:
            print(
                f"  {str(o.created_at)[:19]}  {o.symbol:6s}  {o.side.value:4s}  "
                f"qty={o.qty:>6s}  status={o.status.value}"
            )

    print("\nOK: Alpaca paper connectivity verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
