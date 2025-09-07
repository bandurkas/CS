#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH/USDT Scalper Notifier (Bybit, 15m & 1h)
- No TradingView alerts needed.
- Fetches OHLCV from Bybit via CCXT (public API).
- Re-creates main parts of the Pine logic and sends Telegram messages on:
  * Signal LONG/SHORT (on bar close)
  * Retest limit "filled" (touch)
  * Retest "expired" (not touched within dynamic lifetime)

Requirements:
    pip install ccxt pandas numpy requests

Config:
    Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID below (from your Telegram bot).
    Optionally tweak parameters in "USER CONFIG".
"""
import time
import json
import math
import os
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd
import ccxt

# -------------------- USER CONFIG --------------------
TELEGRAM_TOKEN   = os.getenv("TG_TOKEN",   "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

SYMBOL           = "ETH/USDT:USDT"  # Bybit USDT Perp in CCXT
TIMEFRAMES       = ["15m", "1h"]    # Which timeframes to monitor
LOOP_SLEEP_SEC   = 30               # main loop sleep

# Indicator params (mirror of Pine where possible)
emaLenFast  = 20
emaLenSlow  = 50
devLen      = 100
sigmaK      = 1.0
volMult     = 1.5
sweepLen    = 20
minDevPctVW = 0.15

atrLen      = 14
atrMult15   = 1.0
atrMult60   = 1.5
padSwingATR = 0.2

closeOnly   = True                  # use bar close only
entryMode   = "On retest only"      # ["On retest only", "At bar close"]

# Adaptive Cancel
adaptiveCancel = True
nTRLen        = 14
cancelScale   = 1.5
cancelMin     = 2
cancelMax     = 8

# -----------------------------------------------------

STATE_FILE = "eth_notifier_state.json"  # local persistence


def send_telegram(msg: str):
    if "PUT_YOUR_TELEGRAM" in TELEGRAM_TOKEN or "PUT_YOUR_CHAT_ID" in TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured. Message:", msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print("[ERR] Telegram send failed:", r.text)
    except Exception as e:
        print("[ERR] Telegram send exception:", e)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    trv = tr(high, low, close)
    # Pine ta.atr uses RMA; approximate with EMA of TR
    return trv.ewm(alpha=1/length, adjust=False).mean()


def rolling_vwap(tp: pd.Series, vol: pd.Series, window: int) -> pd.Series:
    # Rolling VWAP over fixed window (not day-anchored). Good enough for notifier.
    pv = tp * vol
    c_pv = pv.cumsum()
    c_v  = vol.cumsum()
    pv_w = c_pv - c_pv.shift(window, fill_value=0)
    v_w  = c_v  - c_v.shift(window,  fill_value=0)
    vwap = pv_w / (v_w.replace(0, np.nan))
    return vwap


def fetch_ohlcv_bybit(exchange, symbol: str, timeframe: str, limit: int = 500):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params={"category": "linear"})
    except Exception as e:
        # Fallback attempt without category (older ccxt)
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def timeframe_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 1


def compute_signals(df: pd.DataFrame, tf: str):
    """
    Returns dict with signals on the last CLOSED bar (index -2):
      {
        'sigLong': bool,
        'sigShort': bool,
        'retestLong': float or None,
        'retestShort': float or None,
        'close': float,
        'bar_time': int (ms),
        'expiryBarsL': int or None,
        'expiryBarsS': int or None,
      }
    """
    i = len(df) - 2  # last closed bar index
    if i < max(emaLenSlow, devLen, sweepLen, atrLen, 20) + 5:
        return None

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    hlc3  = (high + low + close) / 3

    emaFast = ema(close, emaLenFast)
    emaSlow = ema(close, emaLenSlow)
    trendLong  = emaFast > emaSlow
    trendShort = emaFast < emaSlow

    vwap = rolling_vwap(hlc3, vol, devLen)
    dev  = close - vwap
    stdevDev = dev.rolling(devLen).std()
    bandUp   = vwap + sigmaK * stdevDev
    bandDn   = vwap - sigmaK * stdevDev

    atrv = atr(high, low, close, atrLen)
    atrMult = atrMult15 if tf == "15m" else (atrMult60 if tf == "1h" else atrMult15)

    # prevHigh/prevLow over sweepLen EXCLUDING current bar
    prevHigh = high.shift(1).rolling(sweepLen).max()
    prevLow  = low.shift(1).rolling(sweepLen).min()
    volOK    = vol > vol.rolling(20).mean() * volMult

    # A) Sweep & Reversal
    longSweep  = (low < prevLow) & (close > prevLow) & volOK
    shortSweep = (high > prevHigh) & (close < prevHigh) & volOK

    # B) VWAP Mean Revert
    devPct = (close - vwap).abs() / vwap.abs() * 100.0
    longVMR  = (close.shift(1) < bandDn.shift(1)) & (close > bandDn) & (devPct >= minDevPctVW)
    shortVMR = (close.shift(1) > bandUp.shift(1)) & (close < bandUp) & (devPct >= minDevPctVW)

    # C) Momentum Pullback
    rsiv = rsi(close, 14)
    crossUp   = (close.shift(1) <= emaFast.shift(1)) & (close > emaFast)
    crossDown = (close.shift(1) >= emaFast.shift(1)) & (close < emaFast)
    longMom  = trendLong & crossUp  & (low <= emaFast) & (rsiv > 50)
    shortMom = trendShort & crossDown & (high >= emaFast) & (rsiv < 50)

    # Select per Pine order (All)
    longSel  = longSweep | longVMR | longMom
    shortSel = shortSweep | shortVMR | shortMom

    sigLong  = bool(longSel.iat[i])
    sigShort = bool(shortSel.iat[i])

    retestLong = None
    retestShort = None
    if sigLong:
        # Prioritize which setup fired
        if bool(longSweep.iat[i]):      retestLong = float(prevLow.iat[i])
        elif bool(longVMR.iat[i]):      retestLong = float(bandDn.iat[i])
        elif bool(longMom.iat[i]):      retestLong = float(emaFast.iat[i])
    if sigShort:
        if bool(shortSweep.iat[i]):     retestShort = float(prevHigh.iat[i])
        elif bool(shortVMR.iat[i]):     retestShort = float(bandUp.iat[i])
        elif bool(shortMom.iat[i]):     retestShort = float(emaFast.iat[i])

    # Dynamic cancel bars
    expiryBarsL = None
    expiryBarsS = None
    if adaptiveCancel:
        trv = tr(high, low, close)
        avgTR = trv.rolling(nTRLen).mean()
        if sigLong and retestLong is not None:
            dist = abs(float(close.iat[i]) - retestLong)
            e = 0 if avgTR.iat[i] in (0, None, np.nan) else math.ceil(dist / float(avgTR.iat[i]))
            cb = int(max(cancelMin, min(cancelMax, round(cancelScale * e))))
            expiryBarsL = cb
        if sigShort and retestShort is not None:
            dist = abs(float(close.iat[i]) - retestShort)
            e = 0 if avgTR.iat[i] in (0, None, np.nan) else math.ceil(dist / float(avgTR.iat[i]))
            cb = int(max(cancelMin, min(cancelMax, round(cancelScale * e))))
            expiryBarsS = cb

    return {
        "sigLong": sigLong,
        "sigShort": sigShort,
        "retestLong": retestLong,
        "retestShort": retestShort,
        "close": float(close.iat[i]),
        "bar_time": int(df["timestamp"].iat[i]),
        "expiryBarsL": expiryBarsL,
        "expiryBarsS": expiryBarsS,
    }


def ts_ms_to_str(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def main():
    state = load_state()
    exchange = ccxt.bybit({"enableRateLimit": True})
    print("[INFO] Started ETH notifier. Press Ctrl+C to stop.")

    while True:
        try:
            for tf in TIMEFRAMES:
                # init bucket
                if tf not in state:
                    state[tf] = {
                        "last_signal_ts": None,
                        "pending": None  # {'side','retest','expiry_ts','signal_ts'}
                    }

                # fetch
                raw = fetch_ohlcv_bybit(exchange, SYMBOL, tf, limit=600)
                if not raw or len(raw) < 100:
                    print(f"[WARN] No OHLCV for {tf}")
                    continue

                df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
                sig = compute_signals(df, tf)
                if sig is None:
                    continue

                last_signal_ts = state[tf]["last_signal_ts"]
                # check signal on last closed bar
                if closeOnly:
                    sig_bar_ts = sig["bar_time"]
                    if sig_bar_ts != last_signal_ts:
                        # new bar computed
                        if sig["sigLong"]:
                            msg = f"🟢 <b>LONG SIGNAL</b> ({tf}) @ {ts_ms_to_str(sig_bar_ts)}\n" \
                                  f"Close: {sig['close']:.2f}\n" \
                                  f"Retest: {sig['retestLong']:.2f if sig['retestLong'] is not None else float('nan')}\n"
                            send_telegram(msg)
                            # create pending if retest mode
                            if entryMode == "On retest only" and sig["retestLong"] is not None:
                                m = timeframe_minutes(tf)
                                bars = sig["expiryBarsL"] if sig["expiryBarsL"] is not None else cancelMin
                                expiry_ts = sig_bar_ts + bars * m * 60 * 1000
                                state[tf]["pending"] = {"side":"L","retest":sig["retestLong"],"expiry_ts":expiry_ts,"signal_ts":sig_bar_ts}
                                send_telegram(f"⏳ ({tf}) Pending RETEST LONG @ {sig['retestLong']:.2f} | expires in ~{bars} bars")
                            elif entryMode == "At bar close":
                                send_telegram(f"✅ ({tf}) Filled LONG at close ~ {sig['close']:.2f}")
                        if sig["sigShort"]:
                            msg = f"🔻 <b>SHORT SIGNAL</b> ({tf}) @ {ts_ms_to_str(sig_bar_ts)}\n" \
                                  f"Close: {sig['close']:.2f}\n" \
                                  f"Retest: {sig['retestShort']:.2f if sig['retestShort'] is not None else float('nan')}\n"
                            send_telegram(msg)
                            if entryMode == "On retest only" and sig["retestShort"] is not None:
                                m = timeframe_minutes(tf)
                                bars = sig["expiryBarsS"] if sig["expiryBarsS"] is not None else cancelMin
                                expiry_ts = sig_bar_ts + bars * m * 60 * 1000
                                state[tf]["pending"] = {"side":"S","retest":sig["retestShort"],"expiry_ts":expiry_ts,"signal_ts":sig_bar_ts}
                                send_telegram(f"⏳ ({tf}) Pending RETEST SHORT @ {sig['retestShort']:.2f} | expires in ~{bars} bars")
                            elif entryMode == "At bar close":
                                send_telegram(f"✅ ({tf}) Filled SHORT at close ~ {sig['close']:.2f}")

                        state[tf]["last_signal_ts"] = sig_bar_ts

                # monitor pending for retest fill or expiry
                pend = state[tf]["pending"]
                if pend is not None:
                    # Current bar (latest, still forming) touches?
                    last = df.iloc[-1]
                    now_ts = int(last["timestamp"])
                    if pend["side"] == "L":
                        if last["low"] <= pend["retest"]:
                            send_telegram(f"✅ ({tf}) RETEST LONG filled @ ~{pend['retest']:.2f} (touch)")
                            state[tf]["pending"] = None
                        elif now_ts >= pend["expiry_ts"]:
                            send_telegram(f"⌛️ ({tf}) RETEST LONG expired (no touch)")
                            state[tf]["pending"] = None
                    else:
                        if last["high"] >= pend["retest"]:
                            send_telegram(f"✅ ({tf}) RETEST SHORT filled @ ~{pend['retest']:.2f} (touch)")
                            state[tf]["pending"] = None
                        elif now_ts >= pend["expiry_ts"]:
                            send_telegram(f"⌛️ ({tf}) RETEST SHORT expired (no touch)")
                            state[tf]["pending"] = None

                save_state(state)
                time.sleep(1)  # slight pacing per timeframe

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
            break
        except Exception as e:
            print("[ERR] Loop exception:", e)
            time.sleep(5)

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()
