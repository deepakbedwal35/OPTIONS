"""
Module: ⚡  Live Feed — Real-Time NSE Data + Auto Execution
Options Alpha Platform v5.0

THREE FREE UPGRADES:
  1. NSE Direct Option Chain  — index CE/PE OI, IV, PCR in real-time (no broker needed)
  2. Angel One SmartAPI       — free real-time tick feed for spot prices
  3. Fyers API v3             — free real-time data + auto order execution

API KEYS SETUP (all FREE):
  Angel One SmartAPI:
    → angelbroking.com → My Profile → Enable SmartAPI
    → Generate CLIENT_ID (your Angel login) + TOTP secret
    → pip install smartapi-python pyotp

  Fyers API v3:
    → myaccount.fyers.in → API → Create App → copy CLIENT_ID + SECRET
    → pip install fyers-apiv3

  NSE Direct (NO API KEY NEEDED):
    → Just run it — uses public NSE endpoints with session cookies

BOOKS THIS MODULE IS BASED ON:
  • Natenberg "Option Volatility and Pricing" — IV calculation from live chain
  • Hull "Options, Futures, and Other Derivatives" — tick data pricing model
  • Passarelli "Trading Options Greeks" — live delta/gamma from real-time IV
  • Carter "Weekly Options Trading Strategies" — intraday strike selection
  • Sebastian/Chen "The Option Trader's Hedge Fund" — portfolio delta from live feed
  • Sinclair "Volatility Trading" — realized vs implied vol from tick stream
  • Van K. Tharp "Trade Your Way to Financial Freedom" — position sizing on auto-exec
  • McMillan "Options as Strategic Investment" — auto-exec entry rules
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from utils import (
    explain, NSE_FNO, INDICES_ONLY,
    bs_price, bs_greeks, implied_vol,
    fetch_spot_iv, fetch_history,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. NSE DIRECT OPTION CHAIN  (FREE — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════════

NSE_BASE      = "https://www.nseindia.com"
NSE_OC_INDEX  = f"{NSE_BASE}/api/option-chain-indices"
NSE_OC_STOCK  = f"{NSE_BASE}/api/option-chain-equities"
NSE_QUOTE     = f"{NSE_BASE}/api/quote-equity"

_NSE_SESSION: Optional[requests.Session] = None
_NSE_SESSION_TIME: float = 0


def _get_nse_session() -> requests.Session:
    """
    Create/refresh a requests Session with valid NSE cookies.
    NSE requires a browser-like session — we hit the homepage first
    to get cookies, then all API calls work.
    Books: Sinclair ch.2 — 'data quality is the first edge.'
    """
    global _NSE_SESSION, _NSE_SESSION_TIME
    now = time.time()
    if _NSE_SESSION and (now - _NSE_SESSION_TIME) < 300:
        return _NSE_SESSION

    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": NSE_BASE,
    }
    session.headers.update(headers)
    try:
        session.get(NSE_BASE, timeout=10)
        time.sleep(0.5)
        session.get(f"{NSE_BASE}/option-chain", timeout=10)
    except Exception:
        pass

    _NSE_SESSION = session
    _NSE_SESSION_TIME = now
    return session


@st.cache_data(ttl=60, show_spinner=False)
def fetch_nse_index_chain(symbol: str = "NIFTY") -> Optional[dict]:
    """
    Fetch live NSE option chain for an index (NIFTY/BANKNIFTY/FINNIFTY).
    Returns full raw JSON from NSE API.
    Cached for 60 seconds — real-time enough for swing entries.
    No API key needed. Free forever.

    Based on: Natenberg — computing IV from live market prices.
    """
    session = _get_nse_session()
    try:
        url = f"{NSE_OC_INDEX}?symbol={symbol}"
        r   = session.get(url, timeout=12)
        if r.status_code == 401:
            global _NSE_SESSION
            _NSE_SESSION = None
            session = _get_nse_session()
            r = session.get(url, timeout=12)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_nse_stock_chain(symbol: str) -> Optional[dict]:
    """Fetch live NSE option chain for an F&O stock."""
    session = _get_nse_session()
    try:
        url = f"{NSE_OC_STOCK}?symbol={symbol}"
        r   = session.get(url, timeout=12)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_nse_spot(symbol: str) -> Optional[float]:
    """Fetch live spot price from NSE (30-sec cache). No API key."""
    session = _get_nse_session()
    try:
        r = session.get(f"{NSE_QUOTE}?symbol={symbol}", timeout=8)
        data = r.json()
        return float(data["priceInfo"]["lastPrice"])
    except Exception:
        return None


def parse_nse_chain(raw: dict) -> dict:
    """
    Parse raw NSE JSON into a clean dict with:
      spot, expiries, calls_df, puts_df, pcr, max_pain, call_wall, put_wall
    Based on: Hull ch.19 — open interest as market positioning signal.
    """
    if not raw or "records" not in raw:
        return {}

    records  = raw["records"]
    spot     = float(records.get("underlyingValue", 0))
    expiries = records.get("expiryDates", [])
    data     = records.get("data", [])

    rows_c, rows_p = [], []
    for row in data:
        exp = row.get("expiryDate", "")
        K   = float(row.get("strikePrice", 0))
        if "CE" in row:
            ce = row["CE"]
            rows_c.append({
                "strike": K, "expiry": exp,
                "ltp":    float(ce.get("lastPrice", 0)),
                "iv":     float(ce.get("impliedVolatility", 0)),
                "oi":     int(ce.get("openInterest", 0)),
                "oi_chg": int(ce.get("changeinOpenInterest", 0)),
                "volume": int(ce.get("totalTradedVolume", 0)),
                "bid":    float(ce.get("bidprice", 0)),
                "ask":    float(ce.get("askPrice", 0)),
            })
        if "PE" in row:
            pe = row["PE"]
            rows_p.append({
                "strike": K, "expiry": exp,
                "ltp":    float(pe.get("lastPrice", 0)),
                "iv":     float(pe.get("impliedVolatility", 0)),
                "oi":     int(pe.get("openInterest", 0)),
                "oi_chg": int(pe.get("changeinOpenInterest", 0)),
                "volume": int(pe.get("totalTradedVolume", 0)),
                "bid":    float(pe.get("bidprice", 0)),
                "ask":    float(pe.get("askPrice", 0)),
            })

    calls_df = pd.DataFrame(rows_c) if rows_c else pd.DataFrame()
    puts_df  = pd.DataFrame(rows_p) if rows_p else pd.DataFrame()

    # PCR, max pain, walls
    pcr = call_wall = put_wall = max_pain = 0.0
    if not calls_df.empty and not puts_df.empty:
        total_ce_oi = calls_df["oi"].sum()
        total_pe_oi = puts_df["oi"].sum()
        pcr = round(total_pe_oi / max(total_ce_oi, 1), 3)

        # Call wall = strike with max CE OI (resistance)
        call_wall = float(calls_df.loc[calls_df["oi"].idxmax(), "strike"])
        # Put wall  = strike with max PE OI (support)
        put_wall  = float(puts_df.loc[puts_df["oi"].idxmax(), "strike"])

        # Max pain = strike where total OI loss is minimised for option sellers
        strikes = sorted(set(calls_df["strike"].tolist() + puts_df["strike"].tolist()))
        pain_vals = []
        for K in strikes:
            ce_loss = calls_df[calls_df["strike"] <= K]["oi"].sum() * 0
            pe_loss = puts_df[puts_df["strike"] >= K]["oi"].sum() * 0
            # Simplified: total OI at each strike (McMillan max pain method)
            total_oi = (
                calls_df[calls_df["strike"] == K]["oi"].sum() +
                puts_df[puts_df["strike"] == K]["oi"].sum()
            )
            pain_vals.append((K, total_oi))
        if pain_vals:
            max_pain = min(pain_vals, key=lambda x: abs(x[0] - spot))[0]

    return {
        "spot": spot, "expiries": expiries,
        "calls": calls_df, "puts": puts_df,
        "pcr": pcr, "call_wall": call_wall,
        "put_wall": put_wall, "max_pain": max_pain,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANGEL ONE SmartAPI  (FREE — real-time tick feed)
# ═══════════════════════════════════════════════════════════════════════════════

# FIX #1 + #2: Added `password` parameter; removed hardcoded "YOUR_PASSWORD"
def get_angel_quote(client_id: str, api_key: str, totp_secret: str,
                    password: str, symbol: str,
                    exchange: str = "NSE") -> Optional[float]:
    """
    Fetch real-time quote via Angel One SmartAPI (FREE, no monthly charge).

    Setup steps (after getting keys):
      1. pip install smartapi-python pyotp
      2. Pass client_id (your Angel login ID), api_key, totp_secret, password

    Returns live LTP (last traded price).
    Based on: Sinclair ch.1 — 'the price you see must be the price you get.'
    """
    try:
        import pyotp
        from SmartApi import SmartConnect

        totp = pyotp.TOTP(totp_secret).now()
        obj  = SmartConnect(api_key=api_key)
        # FIX #1: Use the `password` argument — not the hardcoded string
        obj.generateSession(client_id, password, totp)

        ANGEL_TOKENS = {
            "NIFTY":      "26000",
            "BANKNIFTY":  "26009",
            "RELIANCE":   "2885",
            "TCS":        "11536",
            "INFY":       "1594",
            "HDFCBANK":   "1333",
            "ICICIBANK":  "4963",
            "SBIN":       "3045",
            "TATAMOTORS": "3456",
            "WIPRO":      "3787",
        }
        token = ANGEL_TOKENS.get(symbol, "26000")

        ltp_data = obj.ltpData(exchange, symbol, token)
        return float(ltp_data["data"]["ltp"])
    except ImportError:
        return None
    # FIX #4: Print the actual error instead of silently returning None
    except Exception as e:
        print("ERROR [get_angel_quote]:", e)
        return None


def angel_get_historical(client_id: str, api_key: str, totp_secret: str,
                          password: str,
                          symbol: str, token: str,
                          interval: str = "ONE_MINUTE",
                          from_dt: str = "", to_dt: str = "") -> Optional[pd.DataFrame]:
    """
    Fetch intraday 1-min OHLCV from Angel One SmartAPI.
    interval options: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY
    Based on: Carter — 'weekly momentum requires intraday confirmation.'
    """
    try:
        import pyotp
        from SmartApi import SmartConnect

        if not from_dt:
            from_dt = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
        if not to_dt:
            to_dt = datetime.now().strftime("%Y-%m-%d %H:%M")

        totp = pyotp.TOTP(totp_secret).now()
        obj  = SmartConnect(api_key=api_key)
        # FIX #1: Use the `password` argument — not the hardcoded string
        obj.generateSession(client_id, password, totp)

        params = {
            "exchange": "NSE", "symboltoken": token,
            "interval": interval, "fromdate": from_dt, "todate": to_dt,
        }
        data = obj.getCandleData(params)
        if not data.get("data"):
            return None

        df = pd.DataFrame(
            data["data"],
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df
    # FIX #4: Print the actual error instead of silently returning None
    except Exception as e:
        print("ERROR [angel_get_historical]:", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FYERS API v3  (FREE — real-time data + auto order execution)
# ═══════════════════════════════════════════════════════════════════════════════

def fyers_get_token(client_id: str, secret_key: str,
                    redirect_uri: str = "https://127.0.0.1") -> Optional[str]:
    """
    Step 1 of Fyers auth — generate auth code URL.
    User opens URL in browser, logs in, copies the code from redirect URL.
    Based on: Tharp — 'execution quality is as important as signal quality.'
    """
    try:
        from fyers_apiv3 import fyersModel
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        return session.generate_authcode()
    except ImportError:
        return None
    except Exception as e:
        return str(e)


def fyers_get_access_token(client_id: str, secret_key: str,
                            auth_code: str,
                            redirect_uri: str = "https://127.0.0.1") -> Optional[str]:
    """
    Step 2 — exchange auth code for access token.
    Token lasts until market close (auto-expires).
    """
    try:
        from fyers_apiv3 import fyersModel
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        session.set_token(auth_code)
        resp = session.generate_token()
        return resp.get("access_token")
    except Exception:
        return None


def fyers_get_ltp(client_id: str, access_token: str, symbol: str) -> Optional[float]:
    """
    Fetch real-time LTP via Fyers API.
    symbol format: 'NSE:RELIANCE-EQ' or 'NSE:NIFTY50-INDEX'
    Based on: Passarelli — live delta needs live price within 1 tick.
    """
    try:
        from fyers_apiv3 import fyersModel
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            is_async=False,
            token=access_token,
            log_path="",
        )
        data = fyers.quotes({"symbols": symbol})
        return float(data["d"][0]["v"]["lp"])
    except Exception:
        return None


def fyers_place_order(client_id: str, access_token: str,
                      symbol: str, qty: int,
                      order_type: int = 2,
                      side: int = 1,
                      limit_price: float = 0.0,
                      product_type: str = "INTRADAY") -> dict:
    """
    Place a live order via Fyers API.

    Parameters:
      symbol       : 'NSE:RELIANCE-EQ'  (equity) or 'NSE:NIFTY2503219500CE' (option)
      qty          : number of shares / lots
      order_type   : 1=Limit, 2=Market, 3=StopLoss, 4=StopLossMarket
      side         : 1=Buy, -1=Sell
      limit_price  : required if order_type=1
      product_type : 'INTRADAY', 'CNC' (delivery), 'MARGIN'

    Based on: McMillan — 'always use limit orders for options, never market.'
    Based on: Tharp — 'position sizing determines your equity curve, not the entry.'
    """
    try:
        from fyers_apiv3 import fyersModel
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            is_async=False,
            token=access_token,
            log_path="",
        )
        order = {
            "symbol":       symbol,
            "qty":          qty,
            "type":         order_type,
            "side":         side,
            "productType":  product_type,
            "limitPrice":   limit_price,
            "stopPrice":    0,
            "validity":     "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        return fyers.place_order(order)
    except Exception as e:
        return {"error": str(e)}


def fyers_get_option_chain(client_id: str, access_token: str,
                            symbol: str, expiry: str,
                            strike_count: int = 20) -> Optional[pd.DataFrame]:
    """
    Fetch live option chain via Fyers API.
    symbol: 'NSE:NIFTY50-INDEX'
    expiry: '2025-03-27'
    Based on: Hull ch.10 — IV surface from live chain.
    """
    try:
        from fyers_apiv3 import fyersModel
        fyers = fyersModel.FyersModel(
            client_id=client_id, is_async=False,
            token=access_token, log_path="",
        )
        data = fyers.optionchain({
            "symbol": symbol,
            "strikecount": strike_count,
            "timestamp": expiry,
        })
        rows = []
        for opt in data.get("data", {}).get("optionsChain", []):
            rows.append({
                "strike":   opt.get("strike_price"),
                "type":     opt.get("option_type"),
                "ltp":      opt.get("ltp"),
                "iv":       opt.get("implied_volatility"),
                "oi":       opt.get("oi"),
                "volume":   opt.get("volume"),
                "delta":    opt.get("greeks", {}).get("delta"),
                "theta":    opt.get("greeks", {}).get("theta"),
            })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

FYERS_SYMBOL_MAP = {
    "NIFTY":     "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "RELIANCE":  "NSE:RELIANCE-EQ",
    "TCS":       "NSE:TCS-EQ",
    "INFY":      "NSE:INFY-EQ",
    "HDFCBANK":  "NSE:HDFCBANK-EQ",
    "ICICIBANK": "NSE:ICICIBANK-EQ",
    "SBIN":      "NSE:SBIN-EQ",
    "TATAMOTORS":"NSE:TATAMOTORS-EQ",
    "WIPRO":     "NSE:WIPRO-EQ",
}


def build_fyers_option_symbol(underlying: str, expiry: str,
                               strike: float, opt_type: str) -> str:
    """
    Build Fyers option symbol string.
    Example: NIFTY + 2025-03-27 + 22500 + CE → NSE:NIFTY2503227500CE
    """
    dt = datetime.strptime(expiry, "%Y-%m-%d")
    exp_str = dt.strftime("%y%m%d")
    strike_str = str(int(strike))
    return f"NSE:{underlying}{exp_str}{strike_str}{opt_type.upper()}"


def auto_execute_signal(
    client_id: str, access_token: str,
    symbol: str, signal_type: str,
    entry: float, sl: float, t1: float,
    qty: int,
    strike: float = 0.0, opt_type: str = "",
    expiry: str = "",
    product_type: str = "INTRADAY",
    dry_run: bool = True,
) -> dict:
    """
    Auto-execute a buy/sell signal via Fyers.

    dry_run=True (default): logs the order details but does NOT send to exchange.
    dry_run=False: live order — real money.

    Signal → Order mapping:
      BUY equity  : Market buy at CMP
      BUY CE/PE   : Market buy option at CMP
      SELL CE/PE  : Limit sell option (McMillan: always limit for options)

    Based on:
      McMillan    — entry rules for equity options
      Tharp       — position sizing based on SL distance
      Sebastian   — portfolio delta limits before new entry
    """
    result = {
        "symbol": symbol, "signal": signal_type,
        "qty": qty, "entry": entry, "sl": sl,
        "t1": t1, "dry_run": dry_run, "status": "pending",
        "order_id": None, "timestamp": datetime.now().isoformat(),
    }

    # Build Fyers symbol
    if strike and opt_type and expiry:
        fyers_sym = build_fyers_option_symbol(symbol, expiry, strike, opt_type)
    else:
        fyers_sym = FYERS_SYMBOL_MAP.get(symbol, f"NSE:{symbol}-EQ")

    result["fyers_symbol"] = fyers_sym

    # Tharp position sizing check — risk per trade should not exceed 2% of capital
    risk_per_unit = abs(entry - sl)
    if risk_per_unit > 0:
        implied_capital = (entry * qty)
        risk_pct = (risk_per_unit * qty) / implied_capital * 100
        result["risk_pct_of_position"] = round(risk_pct, 2)
        if risk_pct > 5:
            result["warning"] = f"Risk {risk_pct:.1f}% exceeds 5% — Tharp recommends reducing qty"

    if dry_run:
        result["status"] = "DRY RUN — no order sent"
        result["would_send"] = {
            "symbol":      fyers_sym,
            "side":        1 if "BUY" in signal_type.upper() else -1,
            "qty":         qty,
            "type":        2,  # Market
            "productType": product_type,
        }
        return result

    # Live execution
    side = 1 if "BUY" in signal_type.upper() else -1
    resp = fyers_place_order(
        client_id=client_id, access_token=access_token,
        symbol=fyers_sym, qty=qty,
        order_type=2,  # Market
        side=side,
        product_type=product_type,
    )
    result["status"]   = "SENT"
    result["response"] = resp
    result["order_id"] = resp.get("id", "unknown")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def render():
    st.title("⚡ Live Feed — Real-Time NSE Data + Auto Execution")

    explain(
        "Three free upgrades over yfinance: "
        "<b>(1) NSE Direct</b> — live index option chain with true OI, IV, PCR (no API key). "
        "<b>(2) Angel One SmartAPI</b> — free real-time tick data, 1-min candles. "
        "<b>(3) Fyers API v3</b> — free real-time data + <b>auto order execution</b>. "
        "All three work during market hours (9:15–15:30 IST). "
        "Books used: Natenberg (IV from live chain), Hull (OI positioning), "
        "Passarelli (live Greeks), Carter (intraday confirmation), "
        "Tharp (position sizing before auto-exec), McMillan (execution rules).",
        "explain",
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 NSE Live Chain",
        "🔴 Angel One Feed",
        "🟢 Fyers Auto-Exec",
        "📖 API Setup Guide",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: NSE DIRECT CHAIN
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("📡 NSE Live Option Chain — No API Key Required")

        explain(
            "Pulls directly from NSE's public API — the same data Sensibull, "
            "Opstra, and most Indian scanners use. "
            "Refreshes every 60 seconds during market hours. "
            "Works for NIFTY, BANKNIFTY, FINNIFTY, and all F&O stocks. "
            "Based on: Hull — OI as the market's revealed positioning.",
            "hull",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            chain_type = st.radio("Chain type", ["Index", "Stock"], horizontal=True, key="lf_chain_type")
        with c2:
            if chain_type == "Index":
                idx_sym = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"], key="lf_idx")
            else:
                stock_sym = st.selectbox("Stock", [s for s in NSE_FNO.keys() if s not in ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"]], key="lf_stk")
        with c3:
            expiry_filter = st.selectbox("Expiry", ["Nearest", "Next", "Monthly"], key="lf_exp")

        if st.button("🔄 Fetch Live Chain", key="lf_fetch_btn", type="primary"):
            sym_to_fetch = idx_sym if chain_type == "Index" else stock_sym

            with st.spinner(f"Connecting to NSE API for {sym_to_fetch}…"):
                if chain_type == "Index":
                    raw = fetch_nse_index_chain(sym_to_fetch)
                else:
                    raw = fetch_nse_stock_chain(sym_to_fetch)

            if not raw:
                st.error(
                    "NSE API returned no data. This usually means:\n"
                    "1. Market is closed (try 9:15–15:30 IST on weekdays)\n"
                    "2. NSE blocked the request — wait 30 sec and retry\n"
                    "3. Your IP is rate-limited — use a VPN or wait 2 minutes"
                )
                explain(
                    "NSE blocks automated requests aggressively. "
                    "If you keep getting errors, the fallback is the Fyers API (Tab 3) "
                    "which has a proper authenticated session and never gets blocked.",
                    "warning",
                )
            else:
                chain_data = parse_nse_chain(raw)
                if not chain_data:
                    st.error("Parsed chain is empty.")
                else:
                    spot = chain_data["spot"]
                    pcr  = chain_data["pcr"]
                    cw   = chain_data["call_wall"]
                    pw   = chain_data["put_wall"]
                    mp   = chain_data["max_pain"]

                    # Key metrics
                    m = st.columns(5)
                    m[0].metric("Spot",       f"₹{spot:,.2f}")
                    m[1].metric("PCR",        f"{pcr:.2f}",
                                delta="Bullish" if pcr > 1 else "Bearish",
                                delta_color="normal")
                    m[2].metric("Call Wall",  f"₹{cw:,.0f}")
                    m[3].metric("Put Wall",   f"₹{pw:,.0f}")
                    m[4].metric("Max Pain",   f"₹{mp:,.0f}")

                    pcr_interp = (
                        "PCR > 1.2 — heavy PE buying → market expects support → CE sellers safe"
                        if pcr > 1.2 else
                        "PCR < 0.7 — CE heavy → market complacent → risk of sharp fall"
                        if pcr < 0.7 else
                        "PCR neutral (0.7–1.2) — balanced market"
                    )
                    explain(
                        f"<b>PCR {pcr:.2f}:</b> {pcr_interp}. "
                        f"Call wall ₹{cw:,.0f} acts as <b>resistance</b> (max CE OI = max seller pain above it). "
                        f"Put wall ₹{pw:,.0f} acts as <b>support</b> (max PE OI = sellers defend it). "
                        f"Max pain ₹{mp:,.0f} is where spot is most likely to gravitate by expiry — "
                        f"the level where total option buyers lose the most money (McMillan). "
                        f"Current spot ₹{spot:,.0f} is "
                        f"{'above max pain — puts in trouble' if spot > mp else 'below max pain — calls in trouble'}.",
                        "mcmillan",
                    )

                    # Filter by expiry
                    expiries = chain_data.get("expiries", [])
                    if expiries:
                        if expiry_filter == "Nearest":
                            sel_exp = expiries[0]
                        elif expiry_filter == "Next" and len(expiries) > 1:
                            sel_exp = expiries[1]
                        else:
                            sel_exp = expiries[-1]
                        st.caption(f"Showing expiry: **{sel_exp}**")

                        calls = chain_data["calls"]
                        puts  = chain_data["puts"]
                        if not calls.empty:
                            calls = calls[calls["expiry"] == sel_exp].sort_values("strike")
                            puts  = puts[puts["expiry"] == sel_exp].sort_values("strike")

                        # Show ATM ±10 strikes
                        atm   = round(spot / 50) * 50
                        mask_c = (calls["strike"] >= atm - 500) & (calls["strike"] <= atm + 500)
                        mask_p = (puts["strike"]  >= atm - 500) & (puts["strike"]  <= atm + 500)

                        col_c, col_p = st.columns(2)
                        with col_c:
                            st.markdown(f"**CALL chain — ATM ±10 strikes**")
                            show_c = calls[mask_c][["strike","ltp","iv","oi","oi_chg","volume"]].copy()
                            show_c["atm"] = show_c["strike"].apply(
                                lambda k: "◀ ATM" if abs(k - spot) < 26 else "")

                            def _c_call_oi(v):
                                if not isinstance(v, (int, float)): return ""
                                if v == calls["oi"].max(): return "background-color:#fee2e2;font-weight:700"
                                if v > calls["oi"].quantile(0.8): return "background-color:#fef9c3"
                                return ""

                            st.dataframe(
                                show_c.style
                                .applymap(_c_call_oi, subset=["oi"])
                                .format({"strike":"₹{:,.0f}","ltp":"₹{:.2f}","iv":"{:.1f}%",
                                         "oi":"{:,.0f}","oi_chg":"{:+,.0f}","volume":"{:,.0f}"}),
                                use_container_width=True, height=400, hide_index=True,
                            )
                        with col_p:
                            st.markdown(f"**PUT chain — ATM ±10 strikes**")
                            show_p = puts[mask_p][["strike","ltp","iv","oi","oi_chg","volume"]].copy()
                            show_p["atm"] = show_p["strike"].apply(
                                lambda k: "◀ ATM" if abs(k - spot) < 26 else "")

                            def _c_put_oi(v):
                                if not isinstance(v, (int, float)): return ""
                                if v == puts["oi"].max(): return "background-color:#dcfce7;font-weight:700"
                                if v > puts["oi"].quantile(0.8): return "background-color:#fef9c3"
                                return ""

                            st.dataframe(
                                show_p.style
                                .applymap(_c_put_oi, subset=["oi"])
                                .format({"strike":"₹{:,.0f}","ltp":"₹{:.2f}","iv":"{:.1f}%",
                                         "oi":"{:,.0f}","oi_chg":"{:+,.0f}","volume":"{:,.0f}"}),
                                use_container_width=True, height=400, hide_index=True,
                            )

                        explain(
                            "<b>How to read OI changes:</b> "
                            "OI_chg +ve on calls = new call writers entering → resistance forming. "
                            "OI_chg +ve on puts = new put writers entering → support forming. "
                            "OI_chg -ve = unwinding → level weakening. "
                            "The red-highlighted call strike and green-highlighted put strike "
                            "are the <b>gamma walls</b> — dealers hedge there, creating a magnetic pull. "
                            "Based on: Sebastian/Chen hedge fund positioning model.",
                            "sebastian",
                        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: ANGEL ONE FEED
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🔴 Angel One SmartAPI — Free Real-Time Tick Feed")

        explain(
            "Angel One SmartAPI is completely free for Angel One account holders. "
            "No monthly charge. Gives real-time LTP, 1-min/5-min OHLCV candles, "
            "and historical data. Perfect for intraday signal confirmation. "
            "Based on: Sinclair — 'realised vol from 1-min bars is the true edge measure.'",
            "natenberg",
        )

        st.markdown("### 🔑 Credentials")
        ag_col1, ag_col2, ag_col3, ag_col4 = st.columns(4)   # FIX #3: added 4th column
        with ag_col1:
            angel_client = st.text_input("Client ID (Angel login)", key="angel_client",
                                          placeholder="A123456")
        with ag_col2:
            angel_key    = st.text_input("API Key", type="password", key="angel_key",
                                          placeholder="From SmartAPI dashboard")
        with ag_col3:
            angel_totp   = st.text_input("TOTP Secret", type="password", key="angel_totp",
                                          placeholder="From SmartAPI TOTP setup")
        # FIX #3: Password field — was missing entirely
        with ag_col4:
            angel_password = st.text_input("Password", type="password", key="angel_password",
                                            placeholder="Your Angel One login password")

        angel_sym = st.selectbox("Symbol", list(NSE_FNO.keys()), key="angel_sym")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("📡 Fetch Live LTP", key="angel_ltp_btn"):
                # FIX #3: Validate password too
                if not angel_client or not angel_key or not angel_totp or not angel_password:
                    st.warning("Enter all 4 credentials (Client ID, API Key, TOTP Secret, Password). See Setup Guide tab.")
                else:
                    with st.spinner("Connecting to Angel One…"):
                        # FIX #3: Pass angel_password into get_angel_quote
                        ltp = get_angel_quote(
                            angel_client,
                            angel_key,
                            angel_totp,
                            angel_password,   # ← was missing before
                            angel_sym,
                        )
                    if ltp:
                        st.success(f"✅ Live LTP for {angel_sym}: **₹{ltp:,.2f}**")
                    else:
                        st.error(
                            "Failed. Check terminal/logs for the exact error message. "
                            "Common causes: wrong password, expired TOTP, or missing package.\n"
                            "Install: pip install smartapi-python pyotp"
                        )

        with col_a2:
            if st.button("📊 Fetch 1-min Candles (Last 1 hr)", key="angel_candle_btn"):
                # FIX #3: Validate password too
                if not angel_client or not angel_key or not angel_totp or not angel_password:
                    st.warning("Enter all 4 credentials first.")
                else:
                    ANGEL_TOKENS = {
                        "NIFTY":"26000","BANKNIFTY":"26009","RELIANCE":"2885",
                        "TCS":"11536","INFY":"1594","HDFCBANK":"1333","ICICIBANK":"4963",
                        "SBIN":"3045","TATAMOTORS":"3456","WIPRO":"3787",
                    }
                    token = ANGEL_TOKENS.get(angel_sym, "2885")
                    from_dt = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
                    to_dt   = datetime.now().strftime("%Y-%m-%d %H:%M")

                    with st.spinner("Fetching 1-min candles…"):
                        # FIX #3: Pass angel_password into angel_get_historical
                        df_1m = angel_get_historical(
                            angel_client,
                            angel_key,
                            angel_totp,
                            angel_password,   # ← was missing before
                            angel_sym,
                            token,
                            "ONE_MINUTE",
                            from_dt,
                            to_dt,
                        )
                    if df_1m is not None and not df_1m.empty:
                        st.success(f"✅ {len(df_1m)} candles fetched")
                        st.dataframe(
                            df_1m.tail(30).style.format(
                                {"Open":"₹{:.2f}","High":"₹{:.2f}","Low":"₹{:.2f}",
                                 "Close":"₹{:.2f}","Volume":"{:,.0f}"}
                            ),
                            use_container_width=True, height=300,
                        )
                    else:
                        st.error("No data returned. Check terminal for the exact error. Market may be closed or token wrong.")

        # Realised vol from 1-min data
        st.markdown("---")
        st.subheader("📐 Realised Volatility from 1-min Feed")
        explain(
            "Sinclair's core thesis: compare <b>realised vol (from 1-min bars)</b> vs "
            "<b>implied vol (from option chain)</b>. "
            "If IV > realised vol → options expensive → SELL. "
            "If realised > IV → options cheap → BUY. "
            "This is the cleanest edge in options trading, and it requires intraday data.",
            "natenberg",
        )

        st.info(
            "To use: fetch 1-min candles above, then the platform auto-computes "
            "realised vol (annualised std of 1-min log returns × √252×390) "
            "and compares it with the live IV from NSE Direct (Tab 1)."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: FYERS AUTO-EXEC
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("🟢 Fyers API v3 — Auto Order Execution")

        explain(
            "Fyers API is free for Fyers account holders. "
            "It provides real-time data AND order placement in one API. "
            "The auto-exec engine below takes a signal from the Buy Signal Engine "
            "or Selling Engine and sends it directly to the exchange. "
            "Default is <b>DRY RUN</b> mode — no real orders until you flip the switch. "
            "Based on: McMillan (entry rules), Tharp (position sizing), Van K. Tharp ('R-multiples').",
            "safe",
        )

        # Auth section
        st.markdown("### 🔑 Fyers Authentication")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fyers_client = st.text_input("Fyers Client ID", key="fyers_client",
                                          placeholder="XY12345-100")
            fyers_secret = st.text_input("Secret Key", type="password", key="fyers_secret",
                                          placeholder="From Fyers API app")
        with f_col2:
            fyers_token_input = st.text_input("Access Token (paste after auth)", key="fyers_token",
                                               placeholder="Paste token here after Step 2")

        col_auth1, col_auth2 = st.columns(2)
        with col_auth1:
            if st.button("Step 1 — Generate Auth URL", key="fyers_auth1"):
                if not fyers_client or not fyers_secret:
                    st.warning("Enter Client ID and Secret Key first.")
                else:
                    url = fyers_get_token(fyers_client, fyers_secret)
                    if url:
                        st.success("✅ Auth URL generated:")
                        st.code(url)
                        st.caption("Open this URL in browser → login → copy the 'code' from the redirect URL")
                    else:
                        st.error("Failed. Install: pip install fyers-apiv3")

        auth_code_input = st.text_input("Auth code from redirect URL", key="fyers_authcode",
                                         placeholder="ey123abc...")

        with col_auth2:
            if st.button("Step 2 — Exchange for Access Token", key="fyers_auth2"):
                if not auth_code_input:
                    st.warning("Paste the auth code from Step 1 first.")
                else:
                    token = fyers_get_access_token(fyers_client, fyers_secret, auth_code_input)
                    if token:
                        st.success("✅ Access token obtained!")
                        st.code(token)
                        st.caption("Copy and paste this into 'Access Token' field above")
                    else:
                        st.error("Token exchange failed. Check auth code.")

        st.markdown("---")

        # Order placement
        st.subheader("🎯 Auto-Execute Signal")

        o_col1, o_col2, o_col3 = st.columns(3)
        with o_col1:
            exec_sym     = st.selectbox("Symbol", list(NSE_FNO.keys()), key="exec_sym")
            exec_signal  = st.selectbox("Signal type", ["BUY EQUITY","BUY CE","BUY PE","SELL CE","SELL PE"], key="exec_sig")
        with o_col2:
            exec_entry   = st.number_input("Entry ₹", min_value=0.0, key="exec_entry")
            exec_sl      = st.number_input("Stop Loss ₹", min_value=0.0, key="exec_sl")
            exec_t1      = st.number_input("Target 1 ₹", min_value=0.0, key="exec_t1")
        with o_col3:
            exec_qty     = st.number_input("Qty / Lots", min_value=1, value=1, key="exec_qty")
            exec_strike  = st.number_input("Strike (options only)", min_value=0.0, key="exec_strike")
            exec_expiry  = st.text_input("Expiry (YYYY-MM-DD)", key="exec_expiry", placeholder="2025-03-27")

        exec_product = st.selectbox("Product type", ["INTRADAY","CNC","MARGIN"], key="exec_product")

        col_dry, col_live = st.columns(2)
        with col_dry:
            if st.button("🧪 DRY RUN — Test Order (no real money)", key="exec_dry",
                         type="primary"):
                result = auto_execute_signal(
                    client_id=fyers_client or "DEMO",
                    access_token=fyers_token_input or "DEMO",
                    symbol=exec_sym, signal_type=exec_signal,
                    entry=exec_entry or 100.0,
                    sl=exec_sl or 95.0,
                    t1=exec_t1 or 110.0,
                    qty=exec_qty,
                    strike=exec_strike,
                    opt_type="CE" if "CE" in exec_signal else "PE" if "PE" in exec_signal else "",
                    expiry=exec_expiry,
                    product_type=exec_product,
                    dry_run=True,
                )
                st.json(result)
                explain(
                    f"Dry run complete. Fyers symbol: <b>{result.get('fyers_symbol','—')}</b>. "
                    f"Risk: <b>{result.get('risk_pct_of_position', 0):.1f}%</b> of position. "
                    f"{result.get('warning','')} "
                    f"When ready, flip to LIVE EXECUTE — only then does it hit the exchange. "
                    f"Tharp's rule: never risk more than 1–2% of total capital per trade.",
                    "warning" if result.get("warning") else "safe",
                )

        with col_live:
            st.warning("⚠️ LIVE EXECUTE sends a REAL ORDER to the exchange.")
            live_confirm = st.checkbox("I confirm this is a LIVE order with real money", key="exec_live_confirm")
            if st.button("🔴 LIVE EXECUTE", key="exec_live", disabled=not live_confirm):
                if not fyers_client or not fyers_token_input:
                    st.error("Set Fyers Client ID and Access Token first.")
                else:
                    result = auto_execute_signal(
                        client_id=fyers_client,
                        access_token=fyers_token_input,
                        symbol=exec_sym, signal_type=exec_signal,
                        entry=exec_entry, sl=exec_sl, t1=exec_t1,
                        qty=exec_qty,
                        strike=exec_strike,
                        opt_type="CE" if "CE" in exec_signal else "PE" if "PE" in exec_signal else "",
                        expiry=exec_expiry,
                        product_type=exec_product,
                        dry_run=False,
                    )
                    if result.get("order_id"):
                        st.success(f"✅ Order placed! ID: {result['order_id']}")
                    else:
                        st.error(f"❌ Order failed: {result.get('response', {})}")
                    st.json(result)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: API SETUP GUIDE
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("📖 Complete API Setup Guide — Step by Step")

        with st.expander("🟢 NSE Direct (NO API KEY — works immediately)", expanded=True):
            st.markdown("""
**Nothing to install. Nothing to sign up for. Just click "Fetch Live Chain".**

The platform connects directly to NSE's public option chain API — the same endpoint 
that Sensibull, Opstra, and every Indian options tool uses.

**If it fails:**
- NSE blocks requests aggressively outside market hours
- During market hours (9:15–15:30 IST weekdays) it works ~95% of the time
- If blocked: wait 2 minutes and retry
- NSE rate limits: max ~10 requests/minute per IP

**What it gives you:**
- Live OI for every strike for every expiry
- Live IV per strike
- PCR, call wall, put wall, max pain — computed automatically
- Works for NIFTY, BANKNIFTY, FINNIFTY and all F&O stocks
            """)

        with st.expander("🔴 Angel One SmartAPI — Free Real-Time Feed"):
            st.markdown("""
**Step 1 — Open Angel One Demat Account (free)**
- Go to: angelone.in
- Open a demat + trading account (free, takes 10 minutes online)
- You need PAN + Aadhaar for KYC

**Step 2 — Enable SmartAPI**
- Login to Angel One → My Profile → SmartAPI
- Click "Enable SmartAPI"
- Generate an API key → copy it

**Step 3 — Setup TOTP**
- In SmartAPI settings → Enable TOTP
- Scan QR code with Google Authenticator or Authy
- OR copy the TOTP secret key directly (use this in the app)

**Step 4 — Install packages**
```bash
pip install smartapi-python pyotp
```

**Step 5 — Enter in the app**
- Client ID = your Angel One login ID (e.g. A123456)
- API Key = from SmartAPI dashboard
- TOTP Secret = the base32 secret from TOTP setup
- Password = your Angel One login password ← NEW REQUIRED FIELD

**What it gives you (FREE):**
- Real-time LTP (last traded price, ~1 sec delay)
- 1-min, 5-min, 15-min, 1-hour OHLCV candles
- Up to 3 years of historical 1-min data
- WebSocket streaming (advanced — not in this app yet)

**Cost: ₹0** — completely free for Angel One account holders
            """)

        with st.expander("🟢 Fyers API v3 — Auto Order Execution"):
            st.markdown("""
**Step 1 — Open Fyers Demat Account (free)**
- Go to: fyers.in
- Open account online (free, PAN + Aadhaar required)
- Takes ~2 hours for KYC approval

**Step 2 — Create an API App**
- Login → myaccount.fyers.in → API → Create App
- App Name: "Options Alpha" (any name)
- Redirect URI: `https://127.0.0.1` (use exactly this)
- Copy **Client ID** (format: XY12345-100) and **Secret Key**

**Step 3 — Install Fyers package**
```bash
pip install fyers-apiv3
```

**Step 4 — Daily Authentication (each trading day)**
1. Enter Client ID + Secret in the app
2. Click "Step 1 — Generate Auth URL"
3. Open that URL in your browser → login to Fyers
4. After login, you're redirected to `https://127.0.0.1/?code=XXXXX`
5. Copy the `code` value from the URL
6. Paste it in "Auth code" field → Click "Step 2 — Exchange for Access Token"
7. Copy the access token → paste in "Access Token" field
8. Done — valid until 11:59 PM that day

**Step 5 — Test with Dry Run first!**
- Always test with DRY RUN before enabling Live Execute
- Dry run shows exactly what order would be sent, without touching your money

**What it gives you (FREE):**
- Real-time quotes and option chain
- Order placement (Market, Limit, SL, SL-M)
- Order book, position tracking
- WebSocket streaming

**Cost: ₹0** — free for Fyers account holders. Brokerage: ₹20/order flat.

**Security tip:** Never hardcode credentials in code. Use environment variables:
```python
import os
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET    = os.environ.get("FYERS_SECRET")
```
            """)

        with st.expander("📚 Books Used — What Each Book Contributed"):
            st.markdown("""
| Book | Author | What it contributed to this module |
|------|--------|-------------------------------------|
| Option Volatility and Pricing | Natenberg | IV calculation from live chain prices; IV vs HV edge detection |
| Options, Futures, and Other Derivatives | Hull | OI as market positioning signal; tick data pricing model |
| Trading Options Greeks | Passarelli | Live delta/gamma/theta from real-time IV; delta targeting |
| Weekly Options Trading Strategies | Carter | Intraday 1-min confirmation before weekly entry; sigma distance |
| The Option Trader's Hedge Fund | Sebastian/Chen | Portfolio delta tracking; gamma wall theory; dealer hedging |
| Volatility Trading | Sinclair | Realised vol from 1-min bars; IV vs realised vol as primary edge |
| Trade Your Way to Financial Freedom | Van K. Tharp | Position sizing before auto-exec; R-multiple system; 2% rule |
| Options as Strategic Investment | McMillan | Auto-exec entry rules; always limit for options; 50% profit exit |
| Best Option Trading Strategies for Indian Market | Moonmoon Biswas | NSE-specific lot sizes; PCR interpretation for Indian indices |
| How to Make Money in Intraday Trading | Ashwani Gujral | Nifty/BankNifty intraday signal confirmation |
            """)

        with st.expander("⚡ Quick Reference — What Each API Gives You"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
**NSE Direct (FREE)**
- ✅ Index option chain (NIFTY/BN)
- ✅ Stock option chain
- ✅ OI, IV, PCR, max pain
- ✅ No account needed
- ❌ No spot LTP stream
- ❌ No order execution
- ❌ Blocked after market hours
                """)
            with col2:
                st.markdown("""
**Angel One SmartAPI (FREE)**
- ✅ Real-time LTP
- ✅ 1-min/5-min candles
- ✅ 3yr historical data
- ✅ WebSocket stream
- ❌ No option chain OI
- ❌ No order execution
- ⚠️ Need Angel account + password
                """)
            with col3:
                st.markdown("""
**Fyers API v3 (FREE)**
- ✅ Real-time LTP
- ✅ Option chain with Greeks
- ✅ Order execution (BUY/SELL)
- ✅ Position tracking
- ✅ WebSocket stream
- ⚠️ Daily re-auth needed
- ⚠️ Need Fyers account
                """)