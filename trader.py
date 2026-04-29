import anthropic

import json

import os

import pandas as pd

import numpy as np

import base64

import requests as req

from datetime import datetime, timedelta

api_key = os.environ.get("ANTHROPIC_API_KEY")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

GITHUB_REPO = "thelyricsman01/Trade-agent-2.1-Multitrader-externalhosting"

GITHUB_BRANCH = "main"

client = anthropic.Anthropic(api_key=api_key)

TRADES_FILE = "trades.json"

PORTFOLIO_FILE = "portfolio.json"

START_BALANCE = 2000.0

# -- RISK MANAGEMENT (enforced in code) -------------------------------------
STOP_LOSS_PCT = 0.07
TAKE_PROFIT_PCT = 0.15
TRAILING_STOP_PCT = 0.05
MIN_HOLD_HOURS = 24
MAX_POSITIONS = 3
MAX_POSITION_PCT = 0.40
CASH_RESERVE_PCT = 0.25
STOP_LOSS_COOLDOWN_HOURS = 72
CLOSE_COOLDOWN_HOURS = 24     # min hours before re-entering any recently closed symbol
MIN_CONFIDENCE = 70           # Claude must be >= 70% confident to enter
MAX_SINGLE_RUN_DROP = 0.20

# -- ASSET UNIVERSE ---------------------------------------------------------
UNIVERSE_MAX_ASSETS = 60
MIN_AVG_VOLUME_USD = 10_000_000

STABLECOIN_PREFIXES = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "GUSD",
    "PYUSD", "USDD", "FRAX", "LUSD", "SUSD", "CUSD", "EURC", "EURT",
    "PAXG", "XAUT",
}

BINANCE_BASE = "https://api.binance.com"

def fetch_dynamic_universe(top_n=UNIVERSE_MAX_ASSETS):
    """Fetch top N USDT pairs from Binance by 24h volume, excluding stablecoins and leveraged tokens."""
    try:
        r = req.get(f"{BINANCE_BASE}/api/v3/ticker/24hr", timeout=15)
        if r.status_code != 200:
            return None
        tickers = r.json()
        usdt_pairs = []
        for t in tickers:
            sym = t["symbol"]
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            if base in STABLECOIN_PREFIXES:
                continue
            if any(x in base for x in ("UP", "DOWN", "BULL", "BEAR", "3L", "3S")):
                continue
            try:
                vol = float(t["quoteVolume"])
            except Exception:
                continue
            usdt_pairs.append((base, vol))

        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        result = {}
        for base, _ in usdt_pairs[:top_n]:
            result[base] = f"{base}-USD"
        return result
    except Exception as e:
        print(f"  fetch_dynamic_universe failed: {e}")
        return None

# -- GITHUB HELPERS ---------------------------------------------------------
def push_to_github(filename):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        with open(filename, "r") as f:
            content = f.read()
        encoded = base64.b64encode(content.encode()).decode()
        get_resp = req.get(url, headers=headers)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
        data = {"message": f"Update {filename}", "content": encoded, "branch": GITHUB_BRANCH}
        if sha:
            data["sha"] = sha
        put_resp = req.put(url, headers=headers, json=data)
        if put_resp.status_code in (200, 201):
            print(f"  Pushed {filename} to GitHub (HTTP {put_resp.status_code})")
        else:
            print(f"  Push FAILED for {filename}: HTTP {put_resp.status_code} - {put_resp.text[:200]}")
    except Exception as e:
        print(f"  Error pushing {filename}: {e}")

# -- BINANCE HELPERS --------------------------------------------------------
def to_binance_symbol(ticker):
    return ticker.replace("-USD", "USDT")

def binance_klines(binance_symbol, interval="1d", limit=200):
    try:
        url = f"{BINANCE_BASE}/api/v3/klines"
        r = req.get(url, params={"symbol": binance_symbol, "interval": interval, "limit": limit}, timeout=15)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_vol", "Trades", "Taker_base", "Taker_quote", "Ignore"
        ])
        df.index = pd.to_datetime(df["Open_time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None

# -- TECHNICAL INDICATORS ---------------------------------------------------
def get_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

def get_macd(series):
    exp12 = series.ewm(span=12).mean()
    exp26 = series.ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return round(float(macd.iloc[-1]), 6), round(float(signal.iloc[-1]), 6), round(float(hist.iloc[-1]), 6)

def get_bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = float(series.iloc[-1])
    denom = float(upper.iloc[-1] - lower.iloc[-1]) + 1e-12
    pct_b = (price - float(lower.iloc[-1])) / denom
    width = float((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1] * 100)
    return round(pct_b, 3), round(width, 2)

def get_atr_pct(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr / close.iloc[-1] * 100), 2)

def get_volume_ratio(data):
    avg = data["Volume"].rolling(20).mean().iloc[-1]
    rec = data["Volume"].iloc[-5:].mean()
    return round(float(rec / avg), 2)

def get_weekly_trend(ticker):
    try:
        bs = to_binance_symbol(ticker)
        w = binance_klines(bs, "1w", 30)
        if w is None or len(w) < 8:
            return "unknown"
        close = w["Close"]
        ma8 = close.rolling(8).mean().iloc[-1]
        ma21 = close.rolling(min(21, len(close))).mean().iloc[-1]
        price = float(close.iloc[-1])
        if price > ma8 > ma21: return "uptrend"
        elif price < ma8 < ma21: return "downtrend"
        else: return "sideways"
    except Exception:
        return "unknown"

def get_market_data(symbol, ticker):
    bs = to_binance_symbol(ticker)
    data = None
    for attempt in range(3):
        try:
            data = binance_klines(bs, "1d", 200)
            if data is not None and len(data) >= 50:
                break
            data = None
        except Exception:
            if attempt < 2:
                import time; time.sleep(3)
    try:
        if data is None or len(data) < 50:
            return None

        close = data["Close"]
        price = round(float(close.iloc[-1]), 8)
        avg_vol_usd = float(data["Volume"].rolling(20).mean().iloc[-1]) * price
        if avg_vol_usd < MIN_AVG_VOLUME_USD:
            return None

        prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
        price_change_pct = abs(price - prev_close) / prev_close if prev_close > 0 else 0
        if price_change_pct > 0.40:
            print(f"  {symbol}: rejected -- suspicious price move ({price_change_pct*100:.0f}%)")
            return None

        macd, signal, hist = get_macd(close)
        try:
            macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            signal_line = macd_line.ewm(span=9).mean()
            hist_series = macd_line - signal_line
            h_yesterday = float(hist_series.iloc[-2])
        except Exception:
            h_yesterday = hist
        macd_rising = bool(hist > h_yesterday)

        pct_b, bb_width = get_bollinger(close)
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        if price > ma20 > ma50: daily_trend = "uptrend"
        elif price < ma20 < ma50: daily_trend = "downtrend"
        else: daily_trend = "sideways"

        weekly_trend = get_weekly_trend(ticker)

        chg_1d  = round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2) if len(close) >= 2 else 0.0
        chg_7d  = round(float((close.iloc[-1] - close.iloc[-8]) / close.iloc[-8] * 100), 2) if len(close) >= 8 else 0.0
        chg_30d = round(float((close.iloc[-1] - close.iloc[-31]) / close.iloc[-31] * 100), 2) if len(close) >= 31 else 0.0
        chg_90d = round(float((close.iloc[-1] - close.iloc[-91]) / close.iloc[-91] * 100), 2) if len(close) >= 91 else 0.0

        # Distance from key levels
        dist_from_ma200 = round((price - ma200) / ma200 * 100, 1) if ma200 else None
        dist_from_ma50  = round((price - ma50) / ma50 * 100, 1)
        support_30d = round(float(close.iloc[-30:].min()), 8)
        resistance_30d = round(float(close.iloc[-30:].max()), 8)

        # Recent candle structure: last 3 closes
        last3 = [round(float(close.iloc[i]), 8) for i in [-3, -2, -1]]

        return {
            "symbol": symbol,
            "price": price,
            "chg_1d": chg_1d,
            "chg_7d": chg_7d,
            "chg_30d": chg_30d,
            "chg_90d": chg_90d,
            "rsi_14": get_rsi(close, 14),
            "rsi_7": get_rsi(close, 7),
            "macd_hist": hist,
            "macd_rising": macd_rising,
            "pct_b": pct_b,
            "bb_width": bb_width,
            "atr_pct": get_atr_pct(data),
            "vol_ratio": get_volume_ratio(data),
            "avg_vol_usd_m": round(avg_vol_usd / 1_000_000, 1),
            "weekly_trend": weekly_trend,
            "daily_trend": daily_trend,
            "ma20": round(ma20, 6),
            "ma50": round(ma50, 6),
            "ma200": round(ma200, 6) if ma200 else None,
            "dist_from_ma50_pct": dist_from_ma50,
            "dist_from_ma200_pct": dist_from_ma200,
            "above_ma200": bool(price > ma200) if ma200 else False,
            "support_30d": support_30d,
            "resistance_30d": resistance_30d,
            "last3_closes": last3,
        }
    except Exception as e:
        print(f"  {symbol}: skipped ({e})")
        return None

# -- PORTFOLIO HELPERS ------------------------------------------------------
def load_from_github(filename, default):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        r = req.get(url, headers=headers)
        if r.status_code == 200:
            return json.loads(base64.b64decode(r.json()["content"]).decode())
    except Exception:
        pass
    return default

def load_portfolio():
    p = load_from_github(PORTFOLIO_FILE, None)
    if not p or not isinstance(p, dict):
        p = {"cash": START_BALANCE, "positions": {}}
        save_portfolio(p)
    if not isinstance(p.get("positions"), dict):
        p["positions"] = {}
    return p

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

def load_trades():
    result = load_from_github(TRADES_FILE, [])
    if not isinstance(result, list):
        return []
    return result

def save_trades(t):
    with open(TRADES_FILE, "w") as f:
        json.dump(t, f, indent=2)

def get_total_balance(portfolio, mdmap):
    total = portfolio["cash"]
    for symbol, pos in portfolio["positions"].items():
        price = mdmap.get(symbol, {}).get("price", pos["entry_price"])
        if pos["type"] == "long":
            total += pos["amount"] * price
        else:
            total += pos["amount_usd"] + (pos["entry_price"] - price) * pos["amount"]
    return round(total, 2)

def hours_since_open(pos):
    try:
        return (datetime.now() - datetime.strptime(pos["opened_at"], "%Y-%m-%d %H:%M")).total_seconds() / 3600
    except Exception:
        return 9999

def recent_stop_loss(symbol, trades):
    """Returns True if symbol hit a stop-loss within STOP_LOSS_COOLDOWN_HOURS."""
    cutoff = datetime.now() - timedelta(hours=STOP_LOSS_COOLDOWN_HOURS)
    for t in reversed(trades):
        if t.get("symbol") == symbol and "Stop-loss" in t.get("reason", ""):
            try:
                if datetime.strptime(t["time"], "%Y-%m-%d %H:%M") > cutoff:
                    return True
            except Exception:
                pass
    return False

def recent_any_close(symbol, trades):
    """Returns True if symbol was closed (for any reason) within CLOSE_COOLDOWN_HOURS."""
    cutoff = datetime.now() - timedelta(hours=CLOSE_COOLDOWN_HOURS)
    for t in reversed(trades):
        if t.get("symbol") == symbol and t.get("action") == "close":
            try:
                if datetime.strptime(t["time"], "%Y-%m-%d %H:%M") > cutoff:
                    return True
            except Exception:
                pass
    return False

def recent_trade_history(trades, n=10):
    """Returns last n closed trades as summary text for context."""
    closed = [t for t in trades if t.get("pnl") is not None][-n:]
    if not closed:
        return "No closed trades yet."
    lines = []
    for t in closed:
        lines.append(
            f"  {t['time']} {t['symbol']:7s} {t['action']:10s} "
            f"PnL: {t.get('pnl', 0):+.2f} | {t.get('reason','')[:60]}"
        )
    return "\n".join(lines)

# -- HARD STOP-LOSS / TAKE-PROFIT -------------------------------------------
def check_hard_exits(portfolio, mdmap, trades):
    executed = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    to_close = []

    for symbol, pos in portfolio["positions"].items():
        price = mdmap.get(symbol, {}).get("price", pos["entry_price"])
        entry = pos["entry_price"]
        last_known = pos.get("last_known_price", entry)
        peak = pos.get("peak_price", entry)

        drop_from_last = (last_known - price) / last_known if last_known > 0 else 0
        if drop_from_last > MAX_SINGLE_RUN_DROP:
            print(f"  ! {symbol}: price ${price:.4f} is {drop_from_last*100:.0f}% below last known "
                  f"${last_known:.4f} -- likely bad data, skipping exit check")
            continue

        pos["last_known_price"] = price

        if pos["type"] == "long" and price > peak:
            pos["peak_price"] = price
            peak = price

        reason = None
        if pos["type"] == "long":
            pnl_pct = (price - entry) / entry
            trail = (peak - price) / peak if peak > entry else 0
            if pnl_pct <= -STOP_LOSS_PCT: reason = f"Stop-loss: {pnl_pct*100:.1f}%"
            elif pnl_pct >= TAKE_PROFIT_PCT: reason = f"Take-profit: +{pnl_pct*100:.1f}%"
            elif trail >= TRAILING_STOP_PCT and pnl_pct > 0.02: reason = f"Trailing stop: -{trail*100:.1f}% from peak"
        else:
            pnl_pct = (entry - price) / entry
            if pnl_pct <= -STOP_LOSS_PCT: reason = f"Stop-loss: {pnl_pct*100:.1f}%"
            elif pnl_pct >= TAKE_PROFIT_PCT: reason = f"Take-profit: +{pnl_pct*100:.1f}%"

        if reason:
            to_close.append((symbol, pos, price, reason))

    for symbol, pos, price, reason in to_close:
        if pos["type"] == "long":
            pnl = round(pos["amount"] * price - pos["amount_usd"], 2)
            portfolio["cash"] += pos["amount"] * price
        else:
            pnl = round((pos["entry_price"] - price) * pos["amount"], 2)
            portfolio["cash"] += pos["amount_usd"] + pnl
        del portfolio["positions"][symbol]
        ticker = UNIVERSE.get(symbol, symbol + "-USD")
        trades.append({"time": timestamp, "symbol": symbol, "ticker": ticker,
                        "action": "close", "price": price, "amount_usd": pos["amount_usd"],
                        "pnl": pnl, "confidence": 100, "reason": reason})
        executed.append(f"AUTO-CLOSE {symbol} PnL=${pnl:+.2f} | {reason}")
        print(f"  {symbol}: {reason} -> PnL ${pnl:+.2f}")

    return portfolio, executed

# -- AI ANALYSIS ------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert cryptocurrency swing trader managing a real portfolio.

Your job is to analyze the full market snapshot provided and make high-conviction trading decisions.
You have complete freedom to reason about what you see — trust your analysis over rigid rules.

HARD LIMITS (enforced by code after your response — you cannot override these):
- Max 3 open positions at once
- 25% of balance always kept as cash reserve
- Stop-loss: 7% | Take-profit: 15% | Trailing stop: 5% from peak
- Minimum hold: 24h before AI-initiated close (hard stops trigger anytime)
- 72h cooldown on re-entry after a stop-loss on the same symbol

YOUR ANALYTICAL PROCESS:
1. Read BTC and ETH first — they set the macro tone for all alts
2. Assess overall market regime (bull / bear / mixed / ranging)
3. Look for assets showing genuine strength or high-probability reversal setups
4. Consider: trend structure, momentum, volume confirmation, distance from key levels
5. Be selective — 0 trades is better than a forced trade
6. For open positions: assess whether the thesis still holds or if it's time to exit

WHAT MAKES A GOOD SETUP:
- Asset in weekly uptrend or convincing weekly base/reversal
- Pulling back to meaningful support (MA, BB lower band, prior structure)
- RSI cooling off but not in freefall
- Volume showing accumulation or at least not distribution
- BTC not in active breakdown

WHAT TO AVOID:
- Chasing momentum after a big move (high RSI + high BB%)
- Entering against the weekly trend
- Adding to a losing thesis
- Low-liquidity setups (low vol_ratio)

Respond ONLY in valid JSON. Be honest in your reasoning — if nothing looks good, return an empty actions array."""

def format_asset_block(m, is_open=False, pos=None, mdmap=None):
    tag = " [OPEN POSITION]" if is_open else ""
    pnl_line = ""
    if is_open and pos:
        price = m["price"]
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        held_h = hours_since_open(pos)
        pnl_line = f"\n    Position: entry ${pos['entry_price']} | now ${price} | PnL {pnl_pct:+.1f}% | held {held_h:.0f}h"

    return (
        f"\n{m['symbol']}{tag}"
        f"\n  Price: ${m['price']} | 1d: {m['chg_1d']:+.1f}% | 7d: {m['chg_7d']:+.1f}% | 30d: {m['chg_30d']:+.1f}% | 90d: {m['chg_90d']:+.1f}%"
        f"\n  Trend: weekly={m['weekly_trend']} daily={m['daily_trend']}"
        f"\n  RSI(14): {m['rsi_14']} | RSI(7): {m['rsi_7']} | MACD hist: {m['macd_hist']:+.6f} ({'rising' if m['macd_rising'] else 'falling'})"
        f"\n  BB%: {m['pct_b']:.2f} (width {m['bb_width']:.1f}%) | ATR%: {m['atr_pct']:.2f}%"
        f"\n  Vol ratio (5d/20d avg): {m['vol_ratio']:.2f} | Avg daily vol: ${m['avg_vol_usd_m']:.0f}M"
        f"\n  MA50 dist: {m['dist_from_ma50_pct']:+.1f}% | MA200 dist: {m['dist_from_ma200_pct']:+.1f}% ({'above' if m['above_ma200'] else 'below'} MA200)"
        f"\n  Support(30d): ${m['support_30d']} | Resistance(30d): ${m['resistance_30d']}"
        f"\n  Last 3 closes: {m['last3_closes']}"
        f"{pnl_line}"
    )

def analyze_swing(all_market_data, portfolio, total_balance, trades):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    open_syms = set(portfolio["positions"].keys())
    closeable = [s for s, p in portfolio["positions"].items() if hours_since_open(p) >= MIN_HOLD_HOURS]

    # BTC and ETH always first for macro context
    priority = ["BTC", "ETH"]
    ordered = [m for m in all_market_data if m["symbol"] in priority] + \
              [m for m in all_market_data if m["symbol"] not in priority]

    market_text = ""
    for m in ordered:
        is_open = m["symbol"] in open_syms
        pos = portfolio["positions"].get(m["symbol"])
        market_text += format_asset_block(m, is_open=is_open, pos=pos)

    user_prompt = f"""Current time: {now}

PORTFOLIO:
  Cash: ${portfolio['cash']:.2f} | Total balance: ${total_balance:.2f} | PnL vs start: ${total_balance - START_BALANCE:+.2f} ({(total_balance/START_BALANCE - 1)*100:+.1f}%)
  Open positions ({len(portfolio['positions'])}/{MAX_POSITIONS}): {list(open_syms) if open_syms else 'none'}
  Eligible for AI close (>={MIN_HOLD_HOURS}h held): {closeable if closeable else 'none'}

RECENT TRADE HISTORY (last 10 closed):
{recent_trade_history(trades)}

FULL MARKET SNAPSHOT ({len(all_market_data)} assets):
{market_text}

Based on the above, decide what actions to take (if any). Respond ONLY in this JSON format:
{{
  "market_regime": "bull | bear | mixed | ranging",
  "btc_assessment": "one sentence on BTC structure and what it means for alts",
  "reasoning": "2-3 sentences explaining your overall read and why you are or aren't trading",
  "actions": [
    {{
      "symbol": "ETH",
      "action": "open_long | open_short | close",
      "invest_pct": 30,
      "confidence": 75,
      "entry_reason": "specific technical and structural reasoning",
      "target_pct": 12,
      "invalidation": "what price action would prove this wrong"
    }}
  ],
  "top_watch": ["SYMBOL1", "SYMBOL2"],
  "avoid": ["SYMBOL3"]
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"  AI analysis failed: {e}")
        return None

# -- EXECUTE TRADES ---------------------------------------------------------
def execute_actions(portfolio, analysis, mdmap, trades):
    executed = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    regime = analysis.get("market_regime", "mixed")

    for action in analysis.get("actions", []):
        symbol = action["symbol"]
        act = action["action"]
        confidence = action.get("confidence", 0)
        md = mdmap.get(symbol)
        if not md:
            print(f"  No data for {symbol} -- skip")
            continue
        price = md["price"]
        ticker = UNIVERSE.get(symbol, symbol + "-USD")

        if act == "open_long":
            if len(portfolio["positions"]) >= MAX_POSITIONS:
                print(f"  Skip LONG {symbol} -- max positions"); continue
            if symbol in portfolio["positions"]:
                print(f"  Skip LONG {symbol} -- already open"); continue
            if confidence < MIN_CONFIDENCE:
                print(f"  Skip LONG {symbol} -- confidence {confidence}% < {MIN_CONFIDENCE}%"); continue
            if md["weekly_trend"] == "downtrend":
                print(f"  Skip LONG {symbol} -- weekly downtrend"); continue
            if recent_stop_loss(symbol, trades):
                print(f"  Skip LONG {symbol} -- stop-loss cooldown ({STOP_LOSS_COOLDOWN_HOURS}h)"); continue
            if recent_any_close(symbol, trades):
                print(f"  Skip LONG {symbol} -- re-entry cooldown ({CLOSE_COOLDOWN_HOURS}h after close)"); continue

            total_bal = get_total_balance(portfolio, mdmap)
            min_cash = total_bal * CASH_RESERVE_PCT
            usable_cash = portfolio["cash"] - min_cash
            if usable_cash < 20:
                print(f"  Skip LONG {symbol} -- cash reserve floor (keeping ${min_cash:.0f})"); continue

            invest_pct = min(action.get("invest_pct", 30), MAX_POSITION_PCT * 100)
            if regime == "bear":
                invest_pct = min(invest_pct, 20)
                print(f"  Bear regime: capping {symbol} position at 20%")
            amount_usd = round(min(portfolio["cash"] * (invest_pct / 100), usable_cash), 2)
            if amount_usd < 20:
                continue

            amount = amount_usd / price
            portfolio["cash"] -= amount_usd
            portfolio["positions"][symbol] = {
                "type": "long", "entry_price": price,
                "amount": amount, "amount_usd": amount_usd,
                "opened_at": timestamp, "peak_price": price,
                "target_pct": action.get("target_pct", TAKE_PROFIT_PCT * 100),
            }
            trades.append({"time": timestamp, "symbol": symbol, "ticker": ticker,
                            "action": "open_long", "price": price, "amount_usd": amount_usd,
                            "confidence": confidence, "reason": action.get("entry_reason", "")})
            executed.append(f"LONG {symbol} ${amount_usd:.0f} @ ${price}")
            print(f"  LONG {symbol} | ${amount_usd:.0f} @ ${price} | conf {confidence}%")

        elif act == "open_short":
            if len(portfolio["positions"]) >= MAX_POSITIONS: continue
            if symbol in portfolio["positions"]: continue
            if confidence < MIN_CONFIDENCE: continue
            if md["weekly_trend"] != "downtrend":
                print(f"  Skip SHORT {symbol} -- not weekly downtrend"); continue
            if recent_stop_loss(symbol, trades):
                print(f"  Skip SHORT {symbol} -- stop-loss cooldown"); continue
            if recent_any_close(symbol, trades):
                print(f"  Skip SHORT {symbol} -- re-entry cooldown ({CLOSE_COOLDOWN_HOURS}h after close)"); continue

            total_bal = get_total_balance(portfolio, mdmap)
            usable_cash = portfolio["cash"] - total_bal * CASH_RESERVE_PCT
            if usable_cash < 20: continue

            invest_pct = min(action.get("invest_pct", 25), MAX_POSITION_PCT * 100)
            amount_usd = round(min(portfolio["cash"] * (invest_pct / 100), usable_cash), 2)
            if amount_usd < 20: continue
            amount = amount_usd / price
            portfolio["cash"] -= amount_usd
            portfolio["positions"][symbol] = {
                "type": "short", "entry_price": price,
                "amount": amount, "amount_usd": amount_usd,
                "opened_at": timestamp,
            }
            trades.append({"time": timestamp, "symbol": symbol, "ticker": ticker,
                            "action": "open_short", "price": price, "amount_usd": amount_usd,
                            "confidence": confidence, "reason": action.get("entry_reason", "")})
            executed.append(f"SHORT {symbol} ${amount_usd:.0f} @ ${price}")
            print(f"  SHORT {symbol} | ${amount_usd:.0f} @ ${price} | conf {confidence}%")

        elif act == "close" and symbol in portfolio["positions"]:
            pos = portfolio["positions"][symbol]
            held_h = hours_since_open(pos)
            if held_h < MIN_HOLD_HOURS:
                print(f"  Skip CLOSE {symbol} -- only {held_h:.0f}h (min {MIN_HOLD_HOURS}h)"); continue

            if pos["type"] == "long":
                pnl = round(pos["amount"] * price - pos["amount_usd"], 2)
                portfolio["cash"] += pos["amount"] * price
            else:
                pnl = round((pos["entry_price"] - price) * pos["amount"], 2)
                portfolio["cash"] += pos["amount_usd"] + pnl
            del portfolio["positions"][symbol]
            trades.append({"time": timestamp, "symbol": symbol, "ticker": ticker,
                            "action": "close", "price": price, "amount_usd": pos["amount_usd"],
                            "pnl": pnl, "confidence": confidence,
                            "reason": action.get("entry_reason", action.get("reason", ""))})
            executed.append(f"CLOSE {symbol} PnL=${pnl:+.2f}")
            print(f"  CLOSE {symbol} | PnL ${pnl:+.2f} @ ${price}")

    return portfolio, executed

# -- MAIN -------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Swing Trader – Full market scan")
print(f"{'='*70}")

# 1. Build dynamic universe
print(f"\n  Fetching dynamic universe (top {UNIVERSE_MAX_ASSETS} by volume)…")
UNIVERSE = fetch_dynamic_universe(UNIVERSE_MAX_ASSETS)
if not UNIVERSE:
    print("  Warning: dynamic fetch failed, using fallback universe")
    UNIVERSE = {
        "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
        "BNB": "BNB-USD", "XRP": "XRP-USD", "DOGE": "DOGE-USD",
        "ADA": "ADA-USD", "AVAX": "AVAX-USD", "LINK": "LINK-USD",
        "DOT": "DOT-USD",
    }
print(f"  Universe: {len(UNIVERSE)} tokens")

# Scan universe
print(f"  Scanning market data…")
all_data = []
mdmap = {}
for symbol, ticker in UNIVERSE.items():
    m = get_market_data(symbol, ticker)
    if m:
        all_data.append(m)
        mdmap[symbol] = m

print(f"  Valid: {len(all_data)} assets")

# 2. Load portfolio and trades
portfolio = load_portfolio()
trades = load_trades()
total_balance = get_total_balance(portfolio, mdmap)

print(f"\n  Cash: ${portfolio['cash']:.2f} | Balance: ${total_balance:.2f}"
      f" | PnL: ${total_balance - START_BALANCE:+.2f}"
      f" | Positions: {len(portfolio['positions'])}")
if portfolio["positions"]:
    for sym, pos in portfolio["positions"].items():
        price = mdmap.get(sym, {}).get("price", pos["entry_price"])
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        held_h = hours_since_open(pos)
        print(f"    {sym}: {pos['type'].upper()} @ ${pos['entry_price']}"
              f" -> ${price} | {pnl_pct:+.1f}% | {held_h:.0f}h held")

# 3. Hard exits (SL / TP / trailing)
print("\n  Checking stop-loss / take-profit…")
portfolio, auto_closed = check_hard_exits(portfolio, mdmap, trades)
if not auto_closed:
    print("  None triggered.")

# 4. AI analysis — full market, no pre-filtering
print(f"\n  AI analyzing full {len(all_data)}-asset market…")
analysis = analyze_swing(all_data, portfolio, total_balance, trades)
if analysis is None:
    print("  Aborting run — portfolio unchanged.")
    print(f"{'='*70}\n")
    exit(1)

portfolio, executed = execute_actions(portfolio, analysis, mdmap, trades)
total_balance = get_total_balance(portfolio, mdmap)

# 5. Save & push
save_portfolio(portfolio)
save_trades(trades)
push_to_github(TRADES_FILE)
push_to_github(PORTFOLIO_FILE)

# 6. Summary
print(f"\n  Regime: {analysis.get('market_regime','?').upper()}")
print(f"  BTC:    {analysis.get('btc_assessment','')}")
print(f"  Read:   {analysis.get('reasoning','')}")
print(f"  Watch:  {analysis.get('top_watch', [])}")
print(f"  Avoid:  {analysis.get('avoid', [])}")
print(f"  Balance: ${total_balance:.2f} | Positions open: {len(portfolio['positions'])}")
if executed:
    for e in executed: print(f"  -> {e}")
else:
    print("  No trades – waiting for quality setups.")
print(f"{'='*70}\n")
