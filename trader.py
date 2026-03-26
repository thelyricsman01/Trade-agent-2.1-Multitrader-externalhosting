from dotenv import load_dotenv
load_dotenv()

import anthropic
import json
import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

api_key = "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"
client  = anthropic.Anthropic(api_key=api_key)

INTERVAL     = 30 * 60  # 30 minutter
TRADES_FILE  = "trades.json"
PORTFOLIO_FILE = "portfolio.json"
START_BALANCE  = 2000.0

ASSETS = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "BNB":  "BNB-USD",
    "XRP":  "XRP-USD",
    "DOGE": "DOGE-USD",
    "ADA":  "ADA-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
}

# ── TEKNISKE INDIKATORER ─────────────────────────────────
def get_rsi(data, period=14):
    delta = data["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

def get_macd(data):
    exp12  = data["Close"].ewm(span=12).mean()
    exp26  = data["Close"].ewm(span=26).mean()
    macd   = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    return round(float(macd.iloc[-1]), 2), round(float(signal.iloc[-1]), 2)

def get_trend(data):
    ma20 = data["Close"].rolling(20).mean().iloc[-1]
    ma50 = data["Close"].rolling(50).mean().iloc[-1]
    cur  = data["Close"].iloc[-1]
    if cur > ma20 > ma50:   return "uptrend"
    elif cur < ma20 < ma50: return "downtrend"
    else:                   return "sideways"

def get_market_data(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if len(data) < 50:
            return None
        price        = round(float(data["Close"].iloc[-1]), 6)
        rsi          = get_rsi(data)
        trend        = get_trend(data)
        macd, signal = get_macd(data)
        change_24h   = round(float((data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100), 2)
        return {
            "ticker":    ticker,
            "price":     price,
            "rsi":       rsi,
            "trend":     trend,
            "macd":      macd,
            "signal":    signal,
            "change_24h": change_24h,
        }
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None

# ── PORTEFØLJE ───────────────────────────────────────────
def load_portfolio():
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        portfolio = {
            "cash":      START_BALANCE,
            "positions": {}
        }
        save_portfolio(portfolio)
        return portfolio

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def get_total_balance(portfolio, market_data):
    total = portfolio["cash"]
    for symbol, pos in portfolio["positions"].items():
        price = next((m["price"] for m in market_data if m and m["ticker"] == ASSETS[symbol]), pos["entry_price"])
        if pos["type"] == "long":
            total += pos["amount"] * price
        elif pos["type"] == "short":
            total += pos["amount_usd"] + (pos["entry_price"] - price) * pos["amount"]
    return round(total, 2)

def load_trades():
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)

# ── AI ANALYSE ───────────────────────────────────────────
def analyze_all(market_data, portfolio, total_balance):
    positions_summary = ""
    for symbol, pos in portfolio["positions"].items():
        positions_summary += f"\n  - {symbol}: {pos['type'].upper()} | Entry: ${pos['entry_price']} | Amount: {pos['amount']} | Invested: ${pos['amount_usd']}"

    assets_summary = ""
    for m in market_data:
        if m:
            assets_summary += f"\n  - {m['ticker']}: Price=${m['price']} | RSI={m['rsi']} | Trend={m['trend']} | MACD={m['macd']} | Signal={m['signal']} | 24h={m['change_24h']}%"

    prompt = f"""
    You are an aggressive but risk-aware crypto portfolio manager.
    You manage a ${START_BALANCE} portfolio and must maximize returns.

    CURRENT PORTFOLIO:
    - Cash available: ${portfolio['cash']}
    - Total balance: ${total_balance}
    - PnL vs start: ${round(total_balance - START_BALANCE, 2)}
    - Open positions: {len(portfolio['positions'])} {positions_summary if portfolio['positions'] else 'None'}

    MARKET DATA:
    {assets_summary}

    YOUR TASK:
    Analyze all assets and decide what to do. You can:
    - Open a LONG position (buy, expecting price to rise)
    - Open a SHORT position (expecting price to fall)
    - CLOSE an existing position (take profit or cut loss)
    - Do nothing on an asset (omit it from response)

    RULES:
    - Total portfolio cannot exceed ${START_BALANCE}
    - Max 40% of total cash per new position
    - Only open new positions if cash is available
    - Consider RSI (oversold <30 = buy signal, overbought >70 = sell/short signal)
    - Consider MACD crossover and trend direction
    - Close positions if trend reverses or RSI signals opposite direction
    - Be decisive - avoid holding too many losing positions

    Respond ONLY in JSON format:
    {{
      "actions": [
        {{
          "symbol": "BTC",
          "action": "open_long, open_short, close or skip",
          "invest_pct": 0-40,
          "confidence": 0-100,
          "reason": "brief explanation"
        }}
      ],
      "market_outlook": "bullish, bearish or mixed",
      "summary": "one sentence portfolio summary"
    }}
    Only include assets where action is NOT skip.
    """

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

# ── UTFØR HANDLER ────────────────────────────────────────
def execute_actions(portfolio, actions, market_data):
    trades   = load_trades()
    executed = []

    for action in actions["actions"]:
        symbol = action["symbol"]
        act    = action["action"]

        market = next((m for m in market_data if m and m["ticker"] == ASSETS.get(symbol)), None)
        if not market:
            continue

        price      = market["price"]
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M")

        if act == "open_long" and symbol not in portfolio["positions"]:
            invest_pct = min(action["invest_pct"], 40)
            amount_usd = round(portfolio["cash"] * (invest_pct / 100), 2)
            if amount_usd < 10:
                continue
            amount = amount_usd / price
            portfolio["cash"]                -= amount_usd
            portfolio["positions"][symbol]    = {
                "type":        "long",
                "entry_price": price,
                "amount":      amount,
                "amount_usd":  amount_usd,
            }
            trade = {
                "time": timestamp, "symbol": symbol, "ticker": ASSETS[symbol],
                "action": "open_long", "price": price, "amount_usd": amount_usd,
                "confidence": action["confidence"], "reason": action["reason"],
            }
            trades.append(trade)
            executed.append(f"LONG {symbol} ${amount_usd}")
            print(f"  ✅ LONG {symbol} | ${amount_usd} at ${price}")

        elif act == "open_short" and symbol not in portfolio["positions"]:
            invest_pct = min(action["invest_pct"], 40)
            amount_usd = round(portfolio["cash"] * (invest_pct / 100), 2)
            if amount_usd < 10:
                continue
            amount = amount_usd / price
            portfolio["cash"]                -= amount_usd
            portfolio["positions"][symbol]    = {
                "type":        "short",
                "entry_price": price,
                "amount":      amount,
                "amount_usd":  amount_usd,
            }
            trade = {
                "time": timestamp, "symbol": symbol, "ticker": ASSETS[symbol],
                "action": "open_short", "price": price, "amount_usd": amount_usd,
                "confidence": action["confidence"], "reason": action["reason"],
            }
            trades.append(trade)
            executed.append(f"SHORT {symbol} ${amount_usd}")
            print(f"  ✅ SHORT {symbol} | ${amount_usd} at ${price}")

        elif act == "close" and symbol in portfolio["positions"]:
            pos = portfolio["positions"][symbol]
            if pos["type"] == "long":
                proceeds = pos["amount"] * price
                pnl      = round(proceeds - pos["amount_usd"], 2)
                portfolio["cash"] += proceeds
            else:
                pnl      = round((pos["entry_price"] - price) * pos["amount"], 2)
                portfolio["cash"] += pos["amount_usd"] + pnl

            del portfolio["positions"][symbol]
            trade = {
                "time": timestamp, "symbol": symbol, "ticker": ASSETS[symbol],
                "action": "close", "price": price, "amount_usd": pos["amount_usd"],
                "pnl": pnl, "confidence": action["confidence"], "reason": action["reason"],
            }
            trades.append(trade)
            executed.append(f"CLOSE {symbol} PnL=${pnl}")
            print(f"  ✅ CLOSE {symbol} | PnL: ${pnl} at ${price}")

    save_trades(trades)
    return portfolio, executed

# ── HOVEDLOOP ────────────────────────────────────────────
print("🚀 Multi-Asset Crypto Trader started")
print(f"   Analyzing {len(ASSETS)} assets every 30 minutes")
print("   Press Ctrl+C to stop.\n")

while True:
    try:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Fetching market data...")

        market_data = []
        for symbol, ticker in ASSETS.items():
            data = get_market_data(ticker)
            if data:
                market_data.append(data)
                print(f"  {symbol}: ${data['price']} | RSI: {data['rsi']} | {data['trend']}")

        portfolio     = load_portfolio()
        total_balance = get_total_balance(portfolio, market_data)

        print(f"\n  Cash: ${portfolio['cash']} | Balance: ${total_balance} | Positions: {len(portfolio['positions'])}")
        print("\n  Analyzing with AI...")

        analysis              = analyze_all(market_data, portfolio, total_balance)
        portfolio, executed   = execute_actions(portfolio, analysis, market_data)
        total_balance         = get_total_balance(portfolio, market_data)

        save_portfolio(portfolio)

        print(f"\n  Outlook: {analysis['market_outlook'].upper()}")
        print(f"  Summary: {analysis['summary']}")
        print(f"  New balance: ${total_balance}")
        if executed:
            print(f"  Executed: {', '.join(executed)}")
        else:
            print("  No trades executed this round.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n  Next analysis in 30 minutes...")
    time.sleep(INTERVAL)