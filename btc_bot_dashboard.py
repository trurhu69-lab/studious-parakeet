import asyncio
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Deque, Optional, Dict, Any
from collections import deque

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from coinbase.rest import RESTClient
from coinbase.websocket import WSClient, WSMessage

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ================== Config & Globals ==================
load_dotenv()

API_KEY = os.getenv("COINBASE_API_KEY")
PRIVATE_KEY = os.getenv("COINBASE_PRIVATE_KEY")
PRODUCT_ID = os.getenv("PRODUCT_ID", "BTC-AUD")
BASE_CCY = PRODUCT_ID.split("-")[1]  # AUD / USD, etc.

# Strategy params (tune)
FAST = 21
SLOW = 55
RISK_PCT = 0.01
TAKE_PROFIT = 0.006
STOP_LOSS  = 0.004
MAX_OPEN_ORDERS = 1
ENTRY_OFFSET = Decimal("0.001")  # 0.1% below last for maker entry

# State containers
prices: Deque[float] = deque(maxlen=5000)

class Position:
    size: Decimal
    entry: Optional[Decimal]
    def __init__(self):
        self.size = Decimal("0")
        self.entry = None

position = Position()

# dashboard state
STATE_LOCK = asyncio.Lock()
STATE: Dict[str, Any] = {
    "last_price": None,
    "ema_fast": None,
    "ema_slow": None,
    "signal": "HOLD",
    "position_size": "0",
    "position_entry": None,
    "quote_balance": None,
    "base_balance": None,
    "pnl_unrealized": None,
    "last_update": None,
    "logs": deque(maxlen=50),  # strings
    "running": True
}

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    STATE["logs"].appendleft(f"[{ts}] {msg}")

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()

def rest() -> RESTClient:
    return RESTClient(api_key=API_KEY, private_key=PRIVATE_KEY)

def clamp_qty(qty: Decimal, inc: Decimal, min_qty: Decimal) -> Decimal:
    # round down to increment and ensure >= min order size
    steps = (qty / inc).to_integral_value(rounding="ROUND_DOWN")
    q = steps * inc
    return q if q >= min_qty else Decimal("0")

def generate_signal() -> str:
    if len(prices) < max(FAST, SLOW) + 2:
        return "HOLD"
    arr = np.array(prices, dtype=float)
    efast = ema(arr, FAST)
    eslow = ema(arr, SLOW)
    if efast[-2] <= eslow[-2] and efast[-1] > eslow[-1]:
        return "LONG"
    if efast[-2] >= eslow[-2] and efast[-1] < eslow[-1]:
        return "FLAT"
    return "HOLD"

# ================== REST helpers ==================
async def get_product_rules(r: RESTClient, product_id: str):
    prod = r.get_products(product_id=product_id)["products"][0]
    price_inc = Decimal(prod["quote_increment"])
    base_inc  = Decimal(prod["base_increment"])
    min_qty   = Decimal(prod["base_min_size"])
    return price_inc, base_inc, min_qty

def list_open_orders(r: RESTClient):
    return r.list_orders(product_id=PRODUCT_ID, order_status=["OPEN", "PENDING"]).get("orders", [])

def get_balances(r: RESTClient):
    q_bal = Decimal("0")
    b_bal = Decimal("0")
    for a in r.get_accounts()["accounts"]:
        if a["currency"] == BASE_CCY:
            q_bal = Decimal(a["available_balance"]["value"])
        if a["currency"] == PRODUCT_ID.split("-")[0]:
            b_bal = Decimal(a["available_balance"]["value"])
    return q_bal, b_bal

def place_limit_maker(r: RESTClient, side: str, price: Decimal, size: Decimal):
    cfg = {
        "limit_limit_gtc": {
            "base_size": str(size),
            "limit_price": str(price),
            "post_only": True
        }
    }
    return r.create_order(product_id=PRODUCT_ID, side=side, order_configuration=cfg)

# ================== Trading Task ==================
async def trading_loop():
    if not API_KEY or not PRIVATE_KEY:
        raise SystemExit("Set COINBASE_API_KEY and COINBASE_PRIVATE_KEY in .env")

    r = rest()
    price_inc, base_inc, min_qty = await get_product_rules(r, PRODUCT_ID)

    async def on_msg(msg: WSMessage):
        if msg.channel != "ticker":
            return
        try:
            p = float(msg.events[0]["tickers"][0]["price"])
        except Exception:
            return

        prices.append(p)
        last = Decimal(str(p))

        # compute EMAs and signal
        sig = generate_signal()
        efast = None
        eslow = None
        if len(prices) >= max(FAST, SLOW):
            arr = np.array(prices, dtype=float)
            efast = float(ema(arr, FAST)[-1])
            eslow = float(ema(arr, SLOW)[-1])

        # Update balances & unrealized PnL (cheaply, not every tick in production)
        try:
            q_bal, b_bal = get_balances(r)
        except Exception:
            q_bal, b_bal = (None, None)

        # Update shared state
        async with STATE_LOCK:
            STATE["last_price"] = float(last)
            STATE["ema_fast"] = efast
            STATE["ema_slow"] = eslow
            STATE["signal"] = sig
            STATE["position_size"] = str(position.size)
            STATE["position_entry"] = float(position.entry) if position.entry else None
            STATE["quote_balance"] = float(q_bal) if q_bal is not None else None
            STATE["base_balance"] = float(b_bal) if b_bal is not None else None
            STATE["pnl_unrealized"] = (float((last - position.entry) * position.size)
                                       if position.entry and position.size > 0 else 0.0)
            STATE["last_update"] = datetime.now(timezone.utc).isoformat()

        # If dashboard toggle paused, skip trading logic
        if not STATE["running"]:
            return

        # Trading logic
        try:
            # Keep order count small
            if len(list_open_orders(r)) >= MAX_OPEN_ORDERS:
                return

            # Entry
            if position.size == 0 and sig == "LONG":
                q_bal, _ = get_balances(r)
                stop = last * Decimal(str(1 - STOP_LOSS))
                stop_dist = (last - stop).max(Decimal("0.01"))
                cash_risk = (q_bal * Decimal(str(RISK_PCT))).max(Decimal("1"))
                size_est = (cash_risk / stop_dist)
                size = clamp_qty(size_est, base_inc, min_qty)
                if size == 0:
                    return
                entry_price = (last * (Decimal("1") - ENTRY_OFFSET)).quantize(price_inc)
                resp = place_limit_maker(r, "BUY", entry_price, size)
                ok = resp.get("success", False)
                log(f"BUY LMT {size} @ {entry_price} posted -> {ok}")
                position.entry = entry_price
                position.size = size

            # Exit
            elif position.size > 0:
                tp = (position.entry * Decimal(str(1 + TAKE_PROFIT))).quantize(price_inc)
                sl = (position.entry * Decimal(str(1 - STOP_LOSS))).quantize(price_inc)

                target = None
                if last >= tp or sig == "FLAT":
                    target = tp
                elif last <= sl:
                    target = sl

                if target:
                    resp = place_limit_maker(r, "SELL", target, position.size)
                    ok = resp.get("success", False)
                    log(f"SELL LMT {position.size} @ {target} posted -> {ok}")
                    position.size = Decimal("0")
                    position.entry = None

        except Exception as e:
            log(f"trade error: {e}")

    # WS ticker
    ws = WSClient(product_ids=[PRODUCT_ID], channels=["ticker"], on_message=on_msg)
    await ws.connect()
    log(f"Connected WS for {PRODUCT_ID}. Trading started.")

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await ws.close()
        log("WebSocket closed.")

# ================== Web App ==================
app = FastAPI(title="BTC Bot Dashboard")

DASH_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BTC Bot Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .muted { color: #666; }
    .big { font-size: 1.6rem; font-weight: 700; }
    .ok { color: #0a7; }
    .warn { color: #c70; }
    .bad { color: #c33; }
    .row { display:flex; gap:8px; align-items:center; }
    button { padding: 8px 12px; border-radius: 8px; border: 1px solid #ccc; cursor: pointer; background: #fafafa; }
    ul { margin: 0; padding-left: 18px; }
  </style>
</head>
<body>
  <h2>BTC Bot Dashboard <span class="muted" id="pair"></span></h2>
  <div class="row" style="gap:12px;margin-bottom:10px;">
    <button onclick="toggleRun()" id="runbtn">Pause</button>
    <button onclick="refreshNow()">Refresh</button>
  </div>

  <div class="grid">
    <div class="card">
      <div class="muted">Last Price</div>
      <div id="price" class="big">—</div>
      <div class="muted">Updated: <span id="updated">—</span></div>
    </div>

    <div class="card">
      <div class="muted">EMA(21) / EMA(55)</div>
      <div><span id="ema_fast">—</span> / <span id="ema_slow">—</span></div>
      <div class="muted">Signal: <b id="signal">—</b></div>
    </div>

    <div class="card">
      <div class="muted">Position</div>
      <div>Size: <b id="pos_size">—</b></div>
      <div>Entry: <b id="pos_entry">—</b></div>
      <div>Unrealized PnL: <b id="pnl">—</b></div>
    </div>

    <div class="card">
      <div class="muted">Balances</div>
      <div>Quote (<span id="quote_ccy"></span>): <b id="qbal">—</b></div>
      <div>Base (BTC): <b id="bbal">—</b></div>
      <div>Status: <b id="status" class="ok">Running</b></div>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <div class="muted">Recent events</div>
    <ul id="logs"></ul>
  </div>

<script>
const pair = "%PAIR%";
document.getElementById("pair").textContent = "(" + pair + ")";
document.getElementById("quote_ccy").textContent = pair.split("-")[1];

let running = true;

async function fetchState() {
  const r = await fetch("/state");
  const s = await r.json();

  function fmt(x, d=2) { return (x===null || x===undefined) ? "—" : Number(x).toFixed(d); }

  document.getElementById("price").textContent = fmt(s.last_price, 2);
  document.getElementById("updated").textContent = s.last_update ? new Date(s.last_update).toLocaleTimeString() : "—";
  document.getElementById("ema_fast").textContent = fmt(s.ema_fast, 2);
  document.getElementById("ema_slow").textContent = fmt(s.ema_slow, 2);
  document.getElementById("signal").textContent = s.signal || "—";
  document.getElementById("pos_size").textContent = s.position_size || "0";
  document.getElementById("pos_entry").textContent = s.position_entry ? s.position_entry.toFixed(2) : "—";
  document.getElementById("pnl").textContent = (s.pnl_unrealized!==null) ? s.pnl_unrealized.toFixed(2) : "—";
  document.getElementById("qbal").textContent = (s.quote_balance!==null) ? s.quote_balance.toFixed(2) : "—";
  document.getElementById("bbal").textContent = (s.base_balance!==null) ? s.base_balance.toFixed(8) : "—";
  running = !!s.running;
  document.getElementById("status").textContent = running ? "Running" : "Paused";
  document.getElementById("status").className = running ? "ok" : "warn";
  document.getElementById("runbtn").textContent = running ? "Pause" : "Resume";

  const logs = document.getElementById("logs");
  logs.innerHTML = "";
  (s.logs || []).forEach(line => {
    const li = document.createElement("li");
    li.textContent = line;
    logs.appendChild(li);
  });
}

async function toggleRun() {
  const r = await fetch("/toggle", { method: "POST" });
  const s = await r.json();
  await fetchState();
}

async function refreshNow() {
  await fetchState();
}

setInterval(fetchState, 1000);
fetchState();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    html = DASH_HTML.replace("%PAIR%", PRODUCT_ID)
    return HTMLResponse(html)

@app.get("/state")
async def get_state():
    async with STATE_LOCK:
        # Convert logs deque to list
        payload = dict(STATE)
        payload["logs"] = list(STATE["logs"])
        return JSONResponse(payload)

@app.post("/toggle")
async def toggle():
    async with STATE_LOCK:
        STATE["running"] = not STATE["running"]
        log(f"Bot {'resumed' if STATE['running'] else 'paused'}.")
        return JSONResponse({"running": STATE["running"]})

# Kick off the trading task when server starts
@app.on_event("startup")
async def on_start():
    asyncio.create_task(trading_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
