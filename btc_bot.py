# Coinbase bot
import asyncio
import os
import time
import json
import math
from decimal import Decimal
from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Coinbase Advanced official SDK
from coinbase.rest import RESTClient
from coinbase.websocket import WSClient, WSMessage

# ---------- Config ----------
load_dotenv()

API_KEY = os.getenv("COINBASE_API_KEY")
PRIVATE_KEY = os.getenv("COINBASE_PRIVATE_KEY")
PRODUCT_ID = os.getenv("PRODUCT_ID", "BTC-AUD")   # use BTC-USD if preferred
BASE_CCY = PRODUCT_ID.split("-")[1]               # AUD or USD, etc.

# Strategy params (tune carefully)
FAST = 21        # fast EMA length (on 1-min candles)
SLOW = 55        # slow EMA length
RISK_PCT = 0.01  # risk per trade = 1% of bankroll
TAKE_PROFIT = 0.006  # +0.6%
STOP_LOSS  = 0.004   # -0.4%
MAX_OPEN_ORDERS = 1  # keep it simple at the start

# ---------- Helpers ----------
@dataclass
class Position:
    size: Decimal = Decimal("0")
    entry: Optional[Decimal] = None

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()

def clamp_qty(qty: Decimal, min_inc: Decimal, min_qty: Decimal) -> Decimal:
    # round down to increment and ensure >= min order size
    steps = (qty / min_inc).to_integral_value(rounding="ROUND_DOWN")
    qty_adj = steps * min_inc
    return qty_adj if qty_adj >= min_qty else Decimal("0")

# ---------- REST setup ----------
def rest() -> RESTClient:
    return RESTClient(api_key=API_KEY, private_key=PRIVATE_KEY)

# Fetch product increments & minimums so orders don’t get rejected
def get_product_rules(r: RESTClient, product_id: str):
    prod = r.get_products(product_id=product_id)["products"][0]
    price_inc = Decimal(prod["quote_increment"])
    base_inc  = Decimal(prod["base_increment"])
    min_qty   = Decimal(prod["base_min_size"])
    min_not   = Decimal(prod.get("min_market_funds", "0"))  # sometimes present
    return price_inc, base_inc, min_qty, min_not

def get_buying_power(r: RESTClient, ccy: str) -> Decimal:
    # Sum available balance in quote currency
    accs = r.get_accounts()["accounts"]
    for a in accs:
        if a["currency"] == ccy:
            return Decimal(a["available_balance"]["value"])
    return Decimal("0")

def existing_open_orders(r: RESTClient, product_id: str) -> int:
    resp = r.list_orders(product_id=product_id, order_status=["OPEN", "PENDING"])
    return len(resp.get("orders", []))

def place_limit_maker_buy(r: RESTClient, product_id: str, price: Decimal, size: Decimal):
    cfg = {
        "limit_limit_gtc": {
            "base_size": str(size),
            "limit_price": str(price),
            "post_only": True
        }
    }
    return r.create_order(product_id=product_id, side="BUY", order_configuration=cfg)

def place_limit_maker_sell(r: RESTClient, product_id: str, price: Decimal, size: Decimal):
    cfg = {
        "limit_limit_gtc": {
            "base_size": str(size),
            "limit_price": str(price),
            "post_only": True
        }
    }
    return r.create_order(product_id=product_id, side="SELL", order_configuration=cfg)

# ---------- Strategy state ----------
prices: Deque[float] = deque(maxlen=5000)
position = Position()

def generate_signal() -> str:
    if len(prices) < max(FAST, SLOW) + 2:
        return "HOLD"
    arr = np.array(prices, dtype=float)
    efast = ema(arr, FAST)
    eslow = ema(arr, SLOW)
    # Simple crossover
    if efast[-2] <= eslow[-2] and efast[-1] > eslow[-1]:
        return "LONG"
    if efast[-2] >= eslow[-2] and efast[-1] < eslow[-1]:
        return "FLAT"
    return "HOLD"

async def main():
    if not API_KEY or not PRIVATE_KEY:
        raise SystemExit("Set COINBASE_API_KEY and COINBASE_PRIVATE_KEY in .env")

    r = rest()
    price_inc, base_inc, min_qty, _ = get_product_rules(r, PRODUCT_ID)

    # WebSocket ticker stream (no auth needed for public ticker) — Advanced Trade WS.  [oai_citation:3‡docs.cdp.coinbase.com](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels?utm_source=chatgpt.com)
    async def on_msg(msg: WSMessage):
        if msg.channel == "ticker":
            try:
                p = float(msg.events[0]["tickers"][0]["price"])
                prices.append(p)
            except Exception:
                return

            # Every new tick, maybe act
            signal = generate_signal()
            last = Decimal(str(p))

            try:
                # Keep only one working order to stay small/risk-aware
                if existing_open_orders(r, PRODUCT_ID) >= MAX_OPEN_ORDERS:
                    return

                # If not in position and LONG signal → place maker limit buy slightly below last
                if position.size == 0 and signal == "LONG":
                    quote_bal = get_buying_power(r, BASE_CCY)
                    # Risk 1% per attempt; with $300 → ~$3 risk. Position size ≈ risk / stop distance
                    stop = last * Decimal(str(1 - STOP_LOSS))
                    stop_dist = (last - stop).max(Decimal("0.01"))
                    # naive size calc:
                    cash_risk = (quote_bal * Decimal(str(RISK_PCT))).max(Decimal("1"))
                    size_est = (cash_risk / stop_dist).quantize(base_inc)
                    size = clamp_qty(size_est, base_inc, min_qty)
                    if size == 0:
                        return

                    entry_price = (last * Decimal("0.999")).quantize(price_inc)  # a hair below to stay maker
                    resp = place_limit_maker_buy(r, PRODUCT_ID, entry_price, size)
                    print("BUY LMT posted:", resp.get("success", False), resp.get("order_id"))

                    # Remember planned bracket (we’ll place TP/SL after filled)
                    position.entry = entry_price
                    position.size = size

                # If in position and either FLAT signal or TP/SL levels reached → post maker sell
                elif position.size > 0:
                    tp = (position.entry * Decimal(str(1 + TAKE_PROFIT))).quantize(price_inc)
                    sl = (position.entry * Decimal(str(1 - STOP_LOSS))).quantize(price_inc)

                    target = None
                    if last >= tp or signal == "FLAT":
                        target = tp
                    elif last <= sl:
                        target = sl

                    if target:
                        resp = place_limit_maker_sell(r, PRODUCT_ID, target, position.size)
                        print("SELL LMT posted:", resp.get("success", False), resp.get("order_id"))
                        # flatten our local state (real fill confirmation is via fills API or user WS channel)
                        position.size = Decimal("0")
                        position.entry = None

            except Exception as e:
                print("trade loop error:", e)

    # Build and start the WS client
    ws = WSClient(product_ids=[PRODUCT_ID], channels=["ticker"], on_message=on_msg)
    await ws.connect()
    try:
        while True:
            await asyncio.sleep(1.0)  # keep the loop alive
    finally:
        await ws.close()

if __name__ == "__main__":
    asyncio.run(main())
