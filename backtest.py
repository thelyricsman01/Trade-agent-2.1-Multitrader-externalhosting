"""
Simple backtest for the swing trading score system.
Tests how well the score_asset() heuristic predicts 15-day forward returns.

Usage: python backtest.py
"""

import requests as req
import pandas as pd
import numpy as np
from datetime import datetime

BINANCE_BASE = "https://api.binance.com"

UNIVERSE = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "AVAX": "AVAX-USD",
    "ADA": "ADA-USD", "NEAR": "NEAR-USD", "DOT": "DOT-USD", "ATOM": "ATOM-USD",
    "BNB": "BNB-USD", "LINK": "LINK-USD", "AAVE": "AAVE-USD", "XRP": "XRP-USD",
    "LTC": "LTC-USD", "XLM": "XLM-USD", "DOGE": "DOGE-USD", "ARB": "ARB-USD",
    "OP": "OP-USD", "FET": "FET-USD", "RENDER": "RENDER-USD", "INJ": "INJ-USD",
    "SEI": "SEI-USD", "TIA": "TIA-USD",
}

MIN_ENTRY_SCORE = 55
STOP_LOSS_PCT = 0.07
TAKE_PROFIT_PCT = 0.15
HOLD_DAYS = 15

def to_binance_symbol(ticker):
    return ticker.replace("-USD", "USDT")

def binance_klines(binance_symbol, interval="1d", limit=500):
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
    except Exception as e:
        print(f"  Binance fetch error: {e}")
        return None

def get_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs)))

def get_macd_hist(series):
    macd = series.ewm(span=12).mean() - series.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    return macd - signal

def get_bb_pct(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower + 1e-12)

def score_row(close, vol, i):
    """Score at day i using the same logic as score_asset()."""
    if i < 50:
        return None
    window = close.iloc[:i+1]
    vol_window = vol.iloc[:i+1]

    rsi = float(get_rsi(window).iloc[-1])
    hist = float(get_macd_hist(window).iloc[-1])
    hist_prev = float(get_macd_hist(window).iloc[-2]) if len(window) > 1 else hist
    pct_b = float(get_bb_pct(window).iloc[-1])
    vol_ratio = float(vol_window.iloc[-5:].mean() / (vol_window.rolling(20).mean().iloc[-1] + 1e-12))
    ma20 = float(window.rolling(20).mean().iloc[-1])
    ma50 = float(window.rolling(50).mean().iloc[-1])
    ma200 = float(window.rolling(200).mean().iloc[-1]) if len(window) >= 200 else None
    price = float(window.iloc[-1])
    chg_7d = float((window.iloc[-1] - window.iloc[-8]) / window.iloc[-8] * 100) if len(window) >= 8 else 0

    if price > ma20 > ma50: weekly_trend = "uptrend"
    elif price < ma20 < ma50: weekly_trend = "downtrend"
    else: weekly_trend = "sideways"

    score = 0
    if weekly_trend == "uptrend": score += 30
    elif weekly_trend == "sideways": score += 10
    else: score -= 20

    if 35 <= rsi <= 60: score += 20
    elif 30 <= rsi < 35: score += 12
    elif 60 < rsi <= 68: score += 8
    elif rsi > 72: score -= 15
    elif rsi < 25: score -= 5

    macd_rising = hist > hist_prev
    if macd_rising and hist > 0: score += 20
    elif macd_rising: score += 10
    elif hist < 0 and not macd_rising: score -= 10

    if pct_b < 0.2: score += 15
    elif pct_b < 0.4: score += 8
    elif pct_b < 0.55: score += 4
    elif pct_b > 0.8: score -= 10

    if vol_ratio >= 1.5: score += 10
    elif vol_ratio >= 1.2: score += 5
    elif vol_ratio >= 0.8: score += 2
    elif vol_ratio < 0.6: score -= 5

    if -10 <= chg_7d <= -3: score += 8
    elif -3 < chg_7d <= 2: score += 5
    elif chg_7d > 15: score -= 10

    if ma200 and price > ma200: score += 5

    return max(0, min(100, score)), rsi, pct_b, vol_ratio

def simulate_trade(close, entry_i):
    """Simulate a trade from entry_i, returns pnl_pct and exit reason."""
    entry = float(close.iloc[entry_i])
    for j in range(1, HOLD_DAYS + 1):
        if entry_i + j >= len(close):
            break
        price = float(close.iloc[entry_i + j])
        pnl = (price - entry) / entry
        if pnl <= -STOP_LOSS_PCT:
            return pnl, "stop-loss"
        if pnl >= TAKE_PROFIT_PCT:
            return pnl, "take-profit"
    final = float(close.iloc[min(entry_i + HOLD_DAYS, len(close) - 1)])
    return (final - entry) / entry, "time-exit"

def backtest_symbol(symbol, ticker):
    bs = to_binance_symbol(ticker)
    data = binance_klines(bs, "1d", 500)
    if data is None or len(data) < 100:
        return []

    close = data["Close"]
    vol = data["Volume"]
    results = []

    step = 5  # check every 5 days to avoid overlapping trades
    i = 50
    while i < len(close) - HOLD_DAYS - 1:
        result = score_row(close, vol, i)
        if result is None:
            i += step
            continue
        score, rsi, pct_b, vol_ratio = result

        if score >= MIN_ENTRY_SCORE and rsi <= 55 and pct_b <= 0.55 and vol_ratio >= 1.0:
            pnl, reason = simulate_trade(close, i)
            results.append({
                "symbol": symbol,
                "date": str(close.index[i].date()),
                "score": score,
                "rsi": round(rsi, 1),
                "pct_b": round(pct_b, 2),
                "vol_ratio": round(vol_ratio, 2),
                "pnl_pct": round(pnl * 100, 2),
                "win": pnl > 0,
                "reason": reason,
            })
            i += HOLD_DAYS  # skip forward to avoid overlapping
        else:
            i += step

    return results

# -- RUN BACKTEST -----------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Backtest: score >= {MIN_ENTRY_SCORE}, SL {STOP_LOSS_PCT*100:.0f}%, TP {TAKE_PROFIT_PCT*100:.0f}%, hold {HOLD_DAYS}d max")
print(f"  Assets: {len(UNIVERSE)} | Data: last 500 days from Binance")
print(f"{'='*65}\n")

all_results = []
for symbol, ticker in UNIVERSE.items():
    print(f"  {symbol}...", end=" ", flush=True)
    results = backtest_symbol(symbol, ticker)
    all_results.extend(results)
    print(f"{len(results)} signals")

if not all_results:
    print("\n  No signals found. Try lowering MIN_ENTRY_SCORE.")
else:
    df = pd.DataFrame(all_results)
    wins = df["win"].sum()
    total = len(df)
    win_rate = wins / total * 100
    avg_pnl = df["pnl_pct"].mean()
    avg_win = df[df["win"]]["pnl_pct"].mean() if wins > 0 else 0
    avg_loss = df[~df["win"]]["pnl_pct"].mean() if (total - wins) > 0 else 0

    print(f"\n{'='*65}")
    print(f"  RESULTS ({total} simulated trades across {df['symbol'].nunique()} assets)")
    print(f"{'='*65}")
    print(f"  Win rate:    {win_rate:.1f}%  ({wins}/{total})")
    print(f"  Avg PnL:     {avg_pnl:+.2f}%")
    print(f"  Avg win:     {avg_win:+.2f}%")
    print(f"  Avg loss:    {avg_loss:+.2f}%")
    print(f"  Reward/risk: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")

    print(f"\n  Exit reasons:")
    for reason, count in df["reason"].value_counts().items():
        print(f"    {reason:<15} {count:>4} ({count/total*100:.1f}%)")

    print(f"\n  Top 5 symbols by avg PnL:")
    sym_stats = df.groupby("symbol").agg(
        trades=("pnl_pct","count"),
        win_rate=("win","mean"),
        avg_pnl=("pnl_pct","mean")
    ).sort_values("avg_pnl", ascending=False)
    for sym, row in sym_stats.head(5).iterrows():
        print(f"    {sym:<8} {row['trades']:>3} trades | WR {row['win_rate']*100:.0f}% | avg {row['avg_pnl']:+.2f}%")

    print(f"\n  Bottom 5 symbols by avg PnL:")
    for sym, row in sym_stats.tail(5).iterrows():
        print(f"    {sym:<8} {row['trades']:>3} trades | WR {row['win_rate']*100:.0f}% | avg {row['avg_pnl']:+.2f}%")

    print(f"\n  Full results saved to backtest_results.csv")
    df.to_csv("backtest_results.csv", index=False)

print(f"\n{'='*65}\n")
