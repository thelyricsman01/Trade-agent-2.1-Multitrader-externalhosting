import anthropic

import json

import os

import yfinance as yf

import pandas as pd

import numpy as np

import base64

import requests as req

from datetime import datetime

api_key = os.environ.get("ANTHROPIC_API_KEY")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

GITHUB_REPO = "thelyricsman01/Trade-agent-2.1-Multitrader-externalhosting"

GITHUB_BRANCH = "main"

client = anthropic.Anthropic(api_key=api_key)

TRADES_FILE = "trades.json"

PORTFOLIO_FILE = "portfolio.json"

START_BALANCE = 2000.0

# -- SWING TRADING PARAMETERS -----------------------------------------------
STOP_LOSS_PCT = 0.07
TAKE_PROFIT_PCT = 0.15
TRAILING_STOP_PCT = 0.05
MIN_HOLD_HOURS = 24
MAX_POSITIONS = 3
MAX_POSITION_PCT = 0.40
MIN_ENTRY_SCORE = 60  # Relaxed from 65

# -- HARD ENTRY FILTERS (enforced in code, not just prompt) -----------------
MIN_VOL_RATIO = 0.8        # Relaxed from 1.0
MAX_ENTRY_BB_PCT = 0.55    # Relaxed from 0.45
MAX_ENTRY_RSI = 62         # Relaxed from 55
CASH_RESERVE_PCT = 0.25    # Always keep 25% cash

# -- ASSET UNIVERSE ---------------------------------------------------------
UNIVERSE = {
    # Large cap
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    # Layer 1s
    "SOL": "SOL-USD",
    "AVAX": "AVAX-USD",
    "ADA": "ADA-USD",
    "NEAR": "NEAR-USD",
    "DOT": "DOT-USD",
    "ATOM": "ATOM-USD",
    "APT": "APT-USD",
    "SUI": "SUI-USD",
    # Exchange tokens
    "BNB": "BNB-USD",
    # DeFi
    "LINK": "LINK-USD",
    "UNI": "UNI-USD",
    "AAVE": "AAVE-USD",
    "MKR": "MKR-USD",
    # Payments
    "XRP": "XRP-USD",
    "LTC": "LTC-USD",
    "XLM": "XLM-USD",
    # Meme / high beta
    "DOGE": "DOGE-USD",
    "SHIB": "SHIB-USD",
    "PEPE": "PEPE-USD",
    # Layer 2
    "ARB": "ARB-USD",
    "OP": "OP-USD",
    "MATIC": "MATIC-USD",
    # AI / infra
    "FET": "FET-USD",
    "RENDER": "RENDER-USD",
    # Momentum alts
    "INJ": "INJ-USD",
    "SEI": "SEI-USD",
    "WIF": "WIF-USD",
    "BONK": "BONK-USD",
    "JTO": "JTO-USD",
    "TIA": "TIA-USD",
}

MIN_AVG_VOLUME_USD = 5_000_000

# -- GITHUB HELPERS ---------------------------------------------------------
def push_to_github(filename):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        with open(filename, "r") as f:
            content = f.read()
        encoded = base64.b64encode(content.encode()).decode()
        response = req.get(url, headers=headers)
        sha = response.json().get("sha") if response.status_code == 200 else None
        data = {"message": f"Update {filename}", "content": encoded, "branch": GITHUB_BRANCH}
        if sha:
            data["sha"] = sha
        req.put(url, headers=headers, json=data)
        print(f"  Pushed {filename} to GitHub")
    except Exception as e:
        print(f"  Error pushing {filename}: {e}")

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
        w = yf.download(ticker, period="6mo", interval="1wk", progress=False, auto_adjust=True)
        if isinstance(w.columns, pd.MultiIndex):
            w.columns = w.columns.get_level_values(0)
        if len(w) < 8:
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

def score_asset(m):
    """Heuristic swing-entry score 0-100. Higher = better setup right now."""
    score = 0

    if m["weekly_trend"] == "uptrend": score += 30
    elif m["weekly_trend"] == "sideways": score += 10
    else: score -= 20

    rsi = m["rsi_14"]
    if 35 <= rsi <= 60: score += 20    # Relaxed upper bound from 55 to 60
    elif 30 <= rsi < 35: score += 12
    elif 60 < rsi <= 68: score += 8    # Slightly extended range
    elif rsi > 72: score -= 15         # Relaxed overbought threshold from 70 to 72
    elif rsi < 25: score -= 5

    if m["macd_rising"] and m["macd_hist"] > 0: score += 20
    elif m["macd_rising"]: score += 10
    elif m["macd_hist"] < 0 and not m["macd_rising"]: score -= 10

    pb = m["pct_b"]
    if pb < 0.2: score += 15
    elif pb < 0.4: score += 8
    elif pb < 0.55: score += 4         # New: small bonus for midrange (was 0 points before)
    elif pb > 0.8: score -= 10

    if m["vol_ratio"] >= 1.5: score += 10
    elif m["vol_ratio"] >= 1.2: score += 5
    elif m["vol_ratio"] >= 0.8: score += 2  # New: small credit for near-avg volume
    elif m["vol_ratio"] < 0.6: score -= 5   # Only penalise very low volume

    chg7 = m["chg_7d"]
    if -10 <= chg7 <= -3: score += 8
    elif -3 < chg7 <= 2: score += 5
    elif chg7 > 15: score -= 10

    if m.get("above_ma200"): score += 5
    return max(0, min(100, score))

def get_market_data(symbol, ticker):
    try:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) < 50:
            return None

        close = data["Close"]
        price = round(float(close.iloc[-1]), 8)
        avg_vol_usd = float(data["Volume"].rolling(20).mean().iloc[-1]) * price
        if avg_vol_usd < MIN_AVG_VOLUME_USD:
            return None

        # -- DATA SANITY CHECK -- reject bad yfinance data
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
        price_change_pct = abs(price - prev_close) / prev_close if prev_close > 0 else 0
        if price_change_pct > 0.40:
            print(f"  {symbol}: rejected -- suspicious price move ({price_change_pct*100:.0f}% vs prev close, likely bad data)")
            return None

        macd, signal, hist = get_macd(close)
        h_today = hist
        h_yesterday = float(
            (close.ewm(span=12).mean() - close.ewm(span=26).mean() -
             (close.ewm(span=12).mean() - close.ewm(span=26).mean()).ewm(span=9).mean()).iloc[-2]
        )
        macd_rising = bool(h_today > h_yesterday)

        pct_b, bb_width = get_bollinger(close)
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        if price > ma20 > ma50: daily_trend = "uptrend"
        elif price < ma20 < ma50: daily_trend = "downtrend"
        else: daily_trend = "sideways"

        weekly_trend = get_weekly_trend(ticker)

        chg_7d = round(float((close.iloc[-1] - close.iloc[-8]) / close.iloc[-8] * 100), 2) if len(close) >= 8 else 0.0
        chg_30d = round(float((close.iloc[-1] - close.iloc[-31]) / close.iloc[-31] * 100), 2) if len(close) >= 31 else 0.0

        m = {
            "symbol": symbol,
            "ticker": ticker,
            "price": price,
            "rsi_14": get_rsi(close, 14),
            "rsi_7": get_rsi(close, 7),
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
            "macd_rising": macd_rising,
            "pct_b": pct_b,
            "bb_width": bb_width,
            "atr_pct": get_atr_pct(data),
            "vol_ratio": get_volume_ratio(data),
            "weekly_trend": weekly_trend,
            "daily_trend": daily_trend,
            "ma20": round(ma20, 6),
            "ma50": round(ma50, 6),
            "ma200": round(ma200, 6) if ma200 else None,
            "above_ma200": bool(price > ma200) if ma200 else False,
            "support": round(float(close.iloc[-30:].min()), 8),
            "resistance": round(float(close.iloc[-30:].max()), 8),
            "chg_7d": chg_7d,
            "chg_30d": chg_30d,
            "avg_vol_usd_m": round(avg_vol_usd / 1_000_000, 1),
        }
        m["score"] = score_asset(m)
        return m
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
    if not p:
        p = {"cash": START_BALANCE, "positions": {}}
        save_portfolio(p)
    return p

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

def load_trades():
    return load_from_github(TRADES_FILE, [])

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

# -- HARD STOP-LOSS / TAKE-PROFIT -------------------------------------------
MAX_SINGLE_RUN_DROP = 0.20

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

# -- AI SWING ANALYSIS ------------------------------------------------------
def analyze_swing(top_candidates, portfolio, total_balance):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    positions_summary = ""
    for symbol, pos in portfolio["positions"].items():
        md = next((m for m in top_candidates if m["symbol"] == symbol), None)
        price = md["price"] if md else pos["entry_price"]
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        held_h = hours_since_open(pos)
        positions_summary += (
            f"\n  - {symbol}: {pos['type'].upper()} entry ${pos['entry_price']}"
            f" | now ${price} | PnL {pnl_pct:+.1f}% | held {held_h:.0f}h"
        )

    candidates_text = ""
    open_syms = set(portfolio["positions"].keys())
    for m in top_candidates:
        tag = " [OPEN]" if m["symbol"] in open_syms else ""
        candidates_text += (
            f"\n  {m['symbol']:7s} score:{m['score']:3d} | ${m['price']}"
            f" | W:{m['weekly_trend']:9s} D:{m['daily_trend']:9s}"
            f" | RSI:{m['rsi_14']:5.1f} | MACDhist:{m['macd_hist']:+.6f} rising:{m['macd_rising']}"
            f" | BB%:{m['pct_b']:.2f} | vol:{m['vol_ratio']:.2f}"
            f" | 7d:{m['chg_7d']:+.1f}% 30d:{m['chg_30d']:+.1f}%"
            f" | {'^MA200' if m['above_ma200'] else 'vMA200'}"
            f" | ${m['avg_vol_usd_m']:.0f}M/day{tag}"
        )

    closeable = [s for s, p in portfolio["positions"].items() if hours_since_open(p) >= MIN_HOLD_HOURS]

    prompt = f"""You are a disciplined swing trader. Current time: {now}

RULES:
1. Only open longs in weekly UPTREND or sideways with strong signals. Never in downtrends.
2. Entry requires ALL of: RSI <= {MAX_ENTRY_RSI} + MACD histogram rising + BB% <= {MAX_ENTRY_BB_PCT} (near/mid support) + vol_ratio >= {MIN_VOL_RATIO}
3. Never open if market_regime is bear or confidence < {MIN_ENTRY_SCORE}
4. Max {MAX_POSITIONS} open positions. Prefer 1-2 high-conviction trades over many mediocre ones.
5. Only suggest CLOSE for positions held >= {MIN_HOLD_HOURS}h: {closeable if closeable else 'none eligible yet'}
6. Do NOT suggest closing positions held < {MIN_HOLD_HOURS}h – hard stops handle emergencies.
7. Target +10-15% per trade. Do not chase small moves or open positions just because cash is available.
8. NEVER rationalize around entry rules. If vol_ratio < {MIN_VOL_RATIO}, BB% > {MAX_ENTRY_BB_PCT}, or RSI > {MAX_ENTRY_RSI}, skip the entry.
9. Pick the BEST setup from the scan – it is better to do nothing than to enter a weak setup.

PORTFOLIO:
- Cash: ${portfolio['cash']:.2f} | Balance: ${total_balance:.2f} | PnL: ${total_balance - START_BALANCE:+.2f}
- Open ({len(portfolio['positions'])}): {positions_summary if portfolio['positions'] else 'None'}

TOP CANDIDATES (scored from {len(top_candidates)}-asset scan, sorted by swing score):
{candidates_text}

Respond ONLY in valid JSON:
{{
  "actions": [
    {{
      "symbol": "ETH",
      "action": "open_long | open_short | close",
      "invest_pct": 35,
      "confidence": 78,
      "entry_reason": "specific technical confluence",
      "target_pct": 13,
      "invalidation": "what would prove this trade wrong"
    }}
  ],
  "market_regime": "bull | bear | mixed | ranging",
  "btc_weekly_bias": "bullish | bearish | neutral",
  "top_pick": "SYMBOL",
  "summary": "one sentence"
}}

Omit assets you want to skip – only include real action decisions.
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

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
            if regime == "bear":
                print(f"  Skip LONG {symbol} -- bear regime"); continue
            if len(portfolio["positions"]) >= MAX_POSITIONS:
                print(f"  Skip LONG {symbol} -- max positions"); continue
            if symbol in portfolio["positions"]:
                print(f"  Skip LONG {symbol} -- already open"); continue
            if confidence < MIN_ENTRY_SCORE:
                print(f"  Skip LONG {symbol} -- confidence {confidence}%"); continue
            if md["weekly_trend"] == "downtrend":
                print(f"  Skip LONG {symbol} -- weekly downtrend"); continue

            # -- HARD ENTRY FILTERS --
            if md["vol_ratio"] < MIN_VOL_RATIO:
                print(f"  Skip LONG {symbol} -- vol_ratio {md['vol_ratio']:.2f} < {MIN_VOL_RATIO} (hard gate)"); continue
            if md["pct_b"] > MAX_ENTRY_BB_PCT:
                print(f"  Skip LONG {symbol} -- BB% {md['pct_b']:.2f} > {MAX_ENTRY_BB_PCT} (not near support)"); continue
            if md["rsi_14"] > MAX_ENTRY_RSI:
                print(f"  Skip LONG {symbol} -- RSI {md['rsi_14']:.1f} > {MAX_ENTRY_RSI} (not a dip)"); continue

            # -- CASH RESERVE --
            total_bal = get_total_balance(portfolio, mdmap)
            min_cash = total_bal * CASH_RESERVE_PCT
            usable_cash = portfolio["cash"] - min_cash
            if usable_cash < 20:
                print(f"  Skip LONG {symbol} -- cash reserve floor (keeping ${min_cash:.0f})"); continue

            invest_pct = min(action.get("invest_pct", 30), MAX_POSITION_PCT * 100)
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
            if confidence < MIN_ENTRY_SCORE: continue
            if md["weekly_trend"] != "downtrend":
                print(f"  Skip SHORT {symbol} -- not weekly downtrend"); continue

            invest_pct = min(action.get("invest_pct", 25), MAX_POSITION_PCT * 100)
            amount_usd = round(portfolio["cash"] * (invest_pct / 100), 2)
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
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Swing Trader – Dynamic universe scan")
print(f"{'='*70}")

# 1. Scan the full universe
print(f"\n  Scanning {len(UNIVERSE)} assets…")
all_data = []
mdmap = {}
for symbol, ticker in UNIVERSE.items():
    m = get_market_data(symbol, ticker)
    if m:
        all_data.append(m)
        mdmap[symbol] = m

print(f"  Valid: {len(all_data)} assets")

# 2. Score and rank
all_data.sort(key=lambda x: x["score"], reverse=True)
TOP_N = 15

portfolio_peek = load_from_github(PORTFOLIO_FILE, {"cash": START_BALANCE, "positions": {}})
open_symbols = set(portfolio_peek.get("positions", {}).keys())
top_set = {m["symbol"] for m in all_data[:TOP_N]}
extra_open = [m for m in all_data if m["symbol"] in open_symbols and m["symbol"] not in top_set]
top_candidates = all_data[:TOP_N] + extra_open

print(f"\n  Top {TOP_N} by swing score:")
print(f"  {'Sym':<8} {'Sc':>3} {'Weekly':>9} {'RSI':>5} {'BB%':>5} {'Vol':>5} {'7d%':>7} {'MA200':>6}")
for m in all_data[:TOP_N]:
    flag = " ★" if m["symbol"] in open_symbols else ""
    print(f"  {m['symbol']:<8} {m['score']:>3} {m['weekly_trend']:>9}"
          f" {m['rsi_14']:>5.1f} {m['pct_b']:>5.2f} {m['vol_ratio']:>5.2f}"
          f" {m['chg_7d']:>+6.1f}% {'^' if m['above_ma200'] else 'v'}{flag}")

# 3. Load portfolio
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

# 4. Hard exits (SL / TP / trailing)
print("\n  Checking stop-loss / take-profit…")
portfolio, auto_closed = check_hard_exits(portfolio, mdmap, trades)
if not auto_closed:
    print("  None triggered.")

# 5. AI analysis
print(f"\n  AI analyzing top {len(top_candidates)} candidates…")
analysis = analyze_swing(top_candidates, portfolio, total_balance)
portfolio, executed = execute_actions(portfolio, analysis, mdmap, trades)
total_balance = get_total_balance(portfolio, mdmap)

# 6. Save & push
save_portfolio(portfolio)
save_trades(trades)
push_to_github(TRADES_FILE)
push_to_github(PORTFOLIO_FILE)

# 7. Summary
print(f"\n  Regime: {analysis.get('market_regime','?').upper()}"
      f" | BTC bias: {analysis.get('btc_weekly_bias','?')}"
      f" | Top pick: {analysis.get('top_pick','?')}")
print(f"  {analysis.get('summary','')}")
print(f"  Balance: ${total_balance:.2f} | Positions open: {len(portfolio['positions'])}")
if executed:
    for e in executed: print(f"  -> {e}")
else:
    print("  No trades – waiting for quality setups.")
print(f"{'='*70}\n")