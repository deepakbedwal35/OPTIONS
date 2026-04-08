"""
Module: 🟢  Buy Signal Engine  v2
Options Alpha Platform v6.0

Fixes vs v1:
  1. NO PAGE REFRESH on Telegram send — results cached in st.session_state.
     Clicking Send never re-runs analysis or wipes the output.
  2. DTE INPUT REMOVED — expiry and premium come from the live option chain.
     DTE is only a BS fallback when chain is unavailable.
  3. SECTOR SCAN shows Strike, Live Premium, Δ, Θ per stock from real chain.
  4. Telegram reads credentials from alerts_journal.py global config.
     No need to type token/chat in the UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, time
from datetime import datetime

warnings.filterwarnings("ignore")

from utils import (
    explain, NSE_FNO, SECTORS,
    bs_price, bs_greeks, implied_vol,
    fetch_history, fetch_spot_iv, fetch_option_chain,
    compute_monthly_winrate, compute_buy_score,
)
from alerts_journal import (
    send_dual_alert, add_paper_trade, render_journal_tab,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED,
)

_SS = "buy_engine_result"   # session_state key for cached analysis


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _strike_step(spot):
    if spot > 40000: return 100
    if spot > 5000:  return 50
    if spot > 1000:  return 10
    return 5


def _best_strike_from_chain(chain, spot, opt_type, target_delta=0.45):
    """
    Find best strike from live yfinance chain.
    Returns dict: strike, premium, delta, theta, vega, iv, oi
    Returns {} on failure.
    """
    try:
        df = chain.calls if opt_type == "call" else chain.puts
        if df is None or df.empty:
            return {}
        r = 0.065
        T = max(30 / 365, 0.001)
        results = []
        for _, row in df.iterrows():
            K    = float(row["strike"])
            bid  = float(row.get("bid", 0) or 0)
            ask  = float(row.get("ask", 0) or 0)
            last = float(row.get("lastPrice", 0) or 0)
            mid  = (bid + ask) / 2 if (bid + ask) > 0 else last
            oi   = int(row.get("openInterest", 0) or 0)
            miv  = float(row.get("impliedVolatility", 0) or 0)
            if mid < 0.5 or miv <= 0:
                continue
            iv_use = max(miv, 0.05)
            g = bs_greeks(spot, K, T, r, iv_use, opt_type)
            results.append({
                "strike":  K,
                "premium": round(mid, 2),
                "delta":   round(g["delta"], 3),
                "theta":   round(g["theta"], 3),
                "vega":    round(g["vega"], 3),
                "iv":      round(iv_use * 100, 1),
                "oi":      oi,
                "_dd":     abs(abs(g["delta"]) - target_delta),
            })
        if not results:
            return {}
        results.sort(key=lambda x: x["_dd"])
        best = results[0]
        best.pop("_dd")
        return best
    except Exception:
        return {}


def _bs_fallback(spot, iv, opt_type, target_delta=0.45, dte=30):
    """BS-model strike when no chain data."""
    T    = max(dte / 365, 0.001)
    r    = 0.065
    step = _strike_step(spot)
    lo   = spot * (0.90 if opt_type == "call" else 0.80)
    hi   = spot * (1.12 if opt_type == "call" else 1.01)
    best_k, best_diff = spot, 999
    for k in np.arange(lo, hi, step):
        k = round(k / step) * step
        g = bs_greeks(spot, k, T, r, iv / 100, opt_type)
        diff = abs(abs(g["delta"]) - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    g    = bs_greeks(spot, best_k, T, r, iv / 100, opt_type)
    prem = bs_price(spot, best_k, T, r, iv / 100, opt_type)
    return {
        "strike":  best_k,
        "premium": round(prem, 2),
        "delta":   round(g["delta"], 3),
        "theta":   round(g["theta"], 3),
        "vega":    round(g["vega"], 3),
        "iv":      round(iv, 1),
        "oi":      0,
    }


def _opt_levels(premium):
    return {
        "opt_sl": round(premium * 0.60, 2),   # −40%
        "opt_t1": round(premium * 1.60, 2),   # +60%
        "opt_t2": round(premium * 2.20, 2),   # +120%
    }


def _indicators(hist):
    close = hist["Close"]; vol = hist["Volume"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
    ema12 = close.ewm(span=12).mean(); ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26; sig = macd.ewm(span=9).mean()
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    spot  = float(close.iloc[-1])
    high  = hist["High"]; low = hist["Low"]
    tr    = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr   = float(tr.rolling(14).mean().iloc[-1])
    avg_v = float(vol.tail(20).mean())
    cur_v = float(vol.iloc[-1])
    highs = hist["High"].tail(20).values; lows = hist["Low"].tail(20).values
    hh = highs[-1] > highs[:10].max(); hl = lows[-1] > lows[:10].min()
    dow = "UPTREND" if (hh and hl) else "DOWNTREND" if (not hh and not hl) else "SIDEWAYS"
    return {
        "spot": spot, "rsi": rsi,
        "macd": float(macd.iloc[-1]), "signal": float(sig.iloc[-1]),
        "macd_cross": float(macd.iloc[-1]) > float(sig.iloc[-1]),
        "ema20": ema20, "ema50": ema50, "atr": atr,
        "vol_surge": cur_v / max(avg_v, 1),
        "dow": dow,
        "support":    float(hist["Low"].tail(10).min()),
        "resistance": float(hist["High"].tail(20).max()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_buy_signal(ind, stats, wr=None, chain=None, expiry=""):
    spot = ind["spot"]; iv = stats["iv"]/100; hv = stats["hv20"]/100
    rsi  = ind["rsi"]; vs = ind["vol_surge"]
    rsi_ok = 40 <= rsi <= 65
    ema_ok = ind["ema20"] > ind["ema50"] and spot > ind["ema20"]
    dist_r = (ind["resistance"] - spot) / spot * 100 if ind["resistance"] > 0 else 99

    dow_score    = {"UPTREND":6,"SIDEWAYS":2,"DOWNTREND":0}.get(ind["dow"],0)
    mom_score    = (2 if rsi_ok else 0) + (2 if ind["macd_cross"] else 0) + (2 if ema_ok else 0)
    vol_score    = min(6,(3 if vs>=1.5 else 2 if vs>=1.2 else 1)+
                       (3 if vs>=1.2 and rsi>50 else 1 if vs>=1.0 else 0))
    struct_score = min(6,(2 if spot>ind["ema20"] else 0)+
                       (2 if spot>ind["ema50"] else 0)+
                       (2 if dist_r>3 else 1 if dist_r>1 else 0))
    opt_score    = 0
    opt_score   += (2 if iv < hv else 1 if stats["ivr"] < 40 else 0)
    if wr:
        up = wr["summary"].get("up_rate",50)
        opt_score += (2 if up>=55 else 1 if up>=50 else 0)
    if chain is not None:
        try:
            pcr = chain.puts["openInterest"].sum()/max(chain.calls["openInterest"].sum(),1)
            opt_score += (2 if pcr<0.7 else 1 if pcr<0.9 else 0)
        except Exception: pass
    opt_score = min(opt_score, 6)

    total = dow_score+mom_score+vol_score+struct_score+opt_score
    grade = "A+" if total>=26 else "A" if total>=22 else "B" if total>=18 else "C" if total>=14 else "D"

    return {
        "total":total,"grade":grade,
        "dow_score":dow_score,"mom_score":mom_score,
        "vol_score":vol_score,"struct_score":struct_score,"opt_score":opt_score,
        "dow_text": {"UPTREND":"Confirmed uptrend (HH+HL) ✅",
                     "DOWNTREND":"Downtrend — high risk counter-trend",
                     "SIDEWAYS":"Sideways — breakout watch"}.get(ind["dow"],""),
        "rsi_text": f"RSI {rsi:.0f} ({'healthy' if rsi_ok else 'overbought' if rsi>70 else 'oversold'})",
        "macd_text": "MACD bullish crossover ✅" if ind["macd_cross"] else "MACD below signal",
        "vol_text": f"Volume {vs:.1f}× avg ({'accumulation ✅' if vs>=1.5 else 'moderate'})",
        "struct_text": f"₹{spot:.2f} {'above' if spot>ind['ema20'] else 'below'} EMA20 ₹{ind['ema20']:.2f}",
    }


def _buy_levels(spot, atr, grade):
    sl_m  = {"A+":1.5,"A":1.5,"B":1.2,"C":1.0,"D":0.8}.get(grade,1.5)
    t1_r  = {"A+":2.0,"A":2.0,"B":1.8,"C":1.5,"D":1.2}.get(grade,2.0)
    t2_r  = {"A+":3.5,"A":3.0,"B":2.5,"C":2.0,"D":1.5}.get(grade,3.0)
    sl    = round(spot - atr*sl_m, 2)
    risk  = spot - sl
    return {"entry":spot,"sl":sl,"t1":round(spot+risk*t1_r,2),"t2":round(spot+risk*t2_r,2),
            "risk":risk,"rr_t1":t1_r,"rr_t2":t2_r}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render():
    st.title("🟢 Buy Signal Engine")
    explain(
        "30-point scoring engine. Strike and premium are pulled from the "
        "<b>live option chain</b> — no DTE input needed. "
        "Telegram uses credentials from <b>alerts_journal.py</b>. "
        "Clicking Send <b>never refreshes</b> the analysis — results are cached.",
        "safe",
    )

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### ⚙️ Parameters")
        sym        = st.selectbox("Stock / Index", list(NSE_FNO.keys()), key="buy_sym")
        trade_type = st.radio("Trade type",
                              ["Equity Buy","CE Buy (Call)","PE Buy (Put)"],
                              key="buy_type")
        target_delta = st.slider("Target Δ", 0.30, 0.70, 0.45, 0.05,
                                  key="buy_delta",
                                  help="0.45 = near-ATM. Lower = cheaper OTM, more leverage")
        st.markdown("---")

        tg_ok = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and TELEGRAM_ENABLED)
        st.markdown(
            f'<div style="padding:8px 12px;border-radius:8px;font-size:.82rem;'
            f'background:{"#dcfce7" if tg_ok else "#fee2e2"};'
            f'color:{"#166534" if tg_ok else "#991b1b"};font-weight:600;">'
            f'📱 Telegram {"Ready ✅" if tg_ok else "Not configured ❌"}</div>',
            unsafe_allow_html=True,
        )
        if not tg_ok:
            st.caption("Set TELEGRAM_BOT_TOKEN in alerts_journal.py")

        send_chart   = st.checkbox("Include chart image", value=True, key="buy_chart")
        auto_journal = st.checkbox("Auto-log to journal", value=True, key="buy_journal")
        st.markdown("---")

        if st.button("🔍 Generate Signal", type="primary", key="buy_analyse"):
            _do_analyse(sym, trade_type, target_delta, auto_journal)

    # Always render cached result — safe on every Streamlit re-run
    if _SS in st.session_state:
        with col_main:
            _render_result(send_chart, tg_ok)

    # ── Sector scanner ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔭 Sector Buy Signal Scanner")
    explain(
        "Every stock scored and the best strike picked from the live chain. "
        "Premium, Δ and Θ columns are real market values — not theoretical.",
        "explain",
    )
    sc1, sc2, sc3 = st.columns(3)
    with sc1: sector_sel  = st.selectbox("Sector", list(SECTORS.keys()), key="buy_sector")
    with sc2: min_score_f = st.slider("Min score", 0, 30, 18, key="buy_minscore")
    with sc3: grade_f     = st.multiselect("Grades", ["A+","A","B","C","D"],
                                            default=["A+","A"], key="buy_gradef")

    scan_type = st.radio("Scan for", ["CE (bullish)","PE (bearish)","Equity only"],
                          horizontal=True, key="buy_scantype")

    if st.button("🚀 Scan Sector", key="buy_scan_btn"):
        _run_sector_scan(sector_sel, min_score_f, grade_f, scan_type, target_delta)

    st.markdown("---")
    render_journal_tab()


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE — runs once, stores in session_state
# ═══════════════════════════════════════════════════════════════════════════════

def _do_analyse(sym, trade_type, target_delta, auto_journal):
    ticker = NSE_FNO[sym]
    with st.spinner(f"Fetching live data for {sym}…"):
        stats         = fetch_spot_iv(ticker)
        hist          = fetch_history(ticker, period="1y")
        wr            = compute_monthly_winrate(ticker)
        chain, expiry = fetch_option_chain(ticker)

    if not stats or hist is None:
        st.error("Could not fetch data.")
        return

    ind  = _indicators(hist)
    sc   = score_buy_signal(ind, stats, wr, chain, expiry or "")
    lvl  = _buy_levels(ind["spot"], ind["atr"], sc["grade"])

    # Option strike — real chain first, BS fallback
    opt_data = {}
    if "CE" in trade_type or "PE" in trade_type:
        ots = "call" if "CE" in trade_type else "put"
        otl = "CE"   if "CE" in trade_type else "PE"
        if chain is not None:
            opt_data = _best_strike_from_chain(chain, ind["spot"], ots, target_delta)
        if not opt_data:
            opt_data = _bs_fallback(ind["spot"], stats["iv"], ots, target_delta)
        if opt_data:
            opt_data["opt_type"] = otl
            opt_data.update(_opt_levels(opt_data["premium"]))
            opt_data["expiry"] = expiry or ""

    pcr_val = 0.0
    if chain is not None:
        try:
            pcr_val = chain.puts["openInterest"].sum() / max(chain.calls["openInterest"].sum(), 1)
        except Exception: pass

    if auto_journal:
        add_paper_trade(
            symbol=sym, entry=lvl["entry"], sl=lvl["sl"],
            t1=lvl["t1"], t2=lvl["t2"],
            score=sc["total"], grade=sc["grade"],
            setup_type=f"{trade_type} | {ind['dow']} | RSI {ind['rsi']:.0f}",
            strike_price=opt_data.get("strike",0),
            option_type=opt_data.get("opt_type",""),
        )

    st.session_state[_SS] = {
        "sym":sym,"trade_type":trade_type,
        "spot":ind["spot"],"iv":stats["iv"],"atr":ind["atr"],
        "sc":sc,"lvl":lvl,"ind":ind,"stats":stats,
        "hist":hist,"opt_data":opt_data,"pcr_val":pcr_val,
        "expiry":expiry or "",
    }
    st.success(f"✅ {sym} — {sc['grade']} signal ({sc['total']}/30) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER RESULT — reads session_state, safe on every re-run
# ═══════════════════════════════════════════════════════════════════════════════

def _render_result(send_chart, tg_ok):
    r        = st.session_state[_SS]
    sym      = r["sym"]; sc = r["sc"]; lvl = r["lvl"]
    ind      = r["ind"]; stats = r["stats"]; hist = r["hist"]
    opt_data = r["opt_data"]; spot = r["spot"]
    iv       = r["iv"]; pcr_val = r["pcr_val"]; expiry = r["expiry"]
    grade    = sc["grade"]; total = sc["total"]
    gc = {"A+":"#15803d","A":"#16a34a","B":"#d97706","C":"#ea580c","D":"#dc2626"}.get(grade,"#64748b")

    m = st.columns(6)
    m[0].metric("Symbol",   sym)
    m[1].metric("CMP",      f"₹{spot:,.2f}")
    m[2].metric("Score",    f"{total}/30")
    m[3].metric("Grade",    grade)
    m[4].metric("RSI",      f"{ind['rsi']:.1f}")
    m[5].metric("Vol ×",    f"{ind['vol_surge']:.1f}")

    emoji = "🚀" if grade=="A+" else "✅" if grade=="A" else "🟡"
    st.markdown(f"""
    <div class="score-card" style="border-top:4px solid {gc};">
      <div class="score-title">{emoji} {sym} — {grade} Buy Signal ({total}/30)</div>
      <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap">
        <div><div style="font-size:.73rem;color:#64748b">Entry</div>
             <div class="score-val">₹{lvl['entry']:,.2f}</div></div>
        <div><div style="font-size:.73rem;color:#64748b">Stop Loss</div>
             <div class="score-val" style="color:#dc2626">₹{lvl['sl']:,.2f}</div></div>
        <div><div style="font-size:.73rem;color:#64748b">T1</div>
             <div class="score-val" style="color:#16a34a">₹{lvl['t1']:,.2f}</div></div>
        <div><div style="font-size:.73rem;color:#64748b">T2</div>
             <div class="score-val" style="color:#15803d">₹{lvl['t2']:,.2f}</div></div>
        <div><div style="font-size:.73rem;color:#64748b">R:R T1 / T2</div>
             <div class="score-val">1:{lvl['rr_t1']} / 1:{lvl['rr_t2']}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.subheader("📊 Scoring Breakdown")
    for name, pts, mx, desc in [
        ("Dow Theory",      sc["dow_score"],    6, sc["dow_text"]),
        ("Momentum",        sc["mom_score"],    6, f"{sc['rsi_text']} | {sc['macd_text']}"),
        ("Volume",          sc["vol_score"],    6, sc["vol_text"]),
        ("Price Structure", sc["struct_score"], 6, sc["struct_text"]),
        ("Options Context", sc["opt_score"],    6,
         f"IV {iv:.1f}% vs HV {stats['hv20']:.1f}% | IVR {stats['ivr']:.0f}% | PCR {pcr_val:.2f}"),
    ]:
        pct = pts/mx
        bc  = "#16a34a" if pct>=0.8 else "#d97706" if pct>=0.5 else "#dc2626"
        st.markdown(f"""
        <div style="margin:5px 0">
          <div style="display:flex;justify-content:space-between;font-size:.84rem">
            <span style="font-weight:600">{name}</span>
            <span style="color:{bc};font-weight:700">{pts}/{mx}</span>
          </div>
          <div style="background:#e2e8f0;border-radius:6px;height:7px;margin:3px 0">
            <div style="background:{bc};width:{int(pct*100)}%;height:7px;border-radius:6px"></div>
          </div>
          <div style="font-size:.75rem;color:#64748b">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Option card
    if opt_data:
        st.markdown("---")
        otl     = opt_data.get("opt_type","CE")
        strike  = opt_data.get("strike",0)
        premium = opt_data.get("premium",0)
        delta   = opt_data.get("delta",0)
        theta   = opt_data.get("theta",0)
        vega    = opt_data.get("vega",0)
        opt_iv  = opt_data.get("iv",iv)
        oi      = opt_data.get("oi",0)
        opt_sl  = opt_data.get("opt_sl",0)
        opt_t1  = opt_data.get("opt_t1",0)
        opt_t2  = opt_data.get("opt_t2",0)
        src     = "Live chain ✅" if oi>0 else "BS model estimate"

        st.subheader(f"🎯 Option Trade — {sym} {int(strike)} {otl}")
        o = st.columns(7)
        o[0].metric("Strike",       f"₹{int(strike):,}")
        o[1].metric("Type",         otl)
        o[2].metric("Live Premium", f"₹{premium:.2f}")
        o[3].metric("Delta",        f"{delta:.3f}")
        o[4].metric("Theta/day",    f"₹{theta:.3f}")
        o[5].metric("Vega",         f"₹{vega:.3f}")
        o[6].metric("OI / Source",  f"{oi:,}" if oi else src)

        exp_html = f'<div><div style="font-size:.73rem;color:#64748b">Expiry</div><div style="font-size:1rem;font-weight:700">{expiry}</div></div>' if expiry else ""
        st.markdown(f"""
        <div class="score-card" style="border-top:4px solid #4f46e5;">
          <div class="score-title">📌 Option Levels — {src}</div>
          <div style="display:flex;gap:24px;margin-top:8px;flex-wrap:wrap">
            <div><div style="font-size:.73rem;color:#64748b">Buy @ entry</div>
                 <div class="score-val">₹{premium:.2f}</div></div>
            <div><div style="font-size:.73rem;color:#64748b">SL −40%</div>
                 <div class="score-val" style="color:#dc2626">₹{opt_sl:.2f}</div></div>
            <div><div style="font-size:.73rem;color:#64748b">T1 +60%</div>
                 <div class="score-val" style="color:#16a34a">₹{opt_t1:.2f}</div></div>
            <div><div style="font-size:.73rem;color:#64748b">T2 +120%</div>
                 <div class="score-val" style="color:#15803d">₹{opt_t2:.2f}</div></div>
            {exp_html}
          </div>
        </div>""", unsafe_allow_html=True)

        explain(
            f"Strike {int(strike)} {otl} — Δ={delta:.3f} (near-ATM). "
            f"IV {opt_iv:.1f}% vs HV {stats['hv20']:.1f}% — "
            f"{'options CHEAP ✅ (Natenberg buy edge)' if opt_iv < stats['hv20'] else 'options rich, strong directional signal needed'}. "
            f"Theta cost ₹{abs(theta):.3f}/day. "
            f"<b>Exit if premium hits ₹{opt_sl:.2f} (−40%). No exceptions.</b>",
            "hull",
        )

        # P&L chart
        T_p  = max(30/365, 0.001); r_p = 0.065
        iv_p = max(opt_iv/100, 0.05); ot_s = "call" if otl=="CE" else "put"
        xs   = np.linspace(spot*0.85, spot*1.15, 80)
        pnl  = [bs_price(s,strike,T_p,r_p,iv_p,ot_s)-premium for s in xs]
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=list(xs),y=pnl,mode="lines",
                                  line=dict(color="#4f46e5",width=2.5),
                                  fill="tozeroy",fillcolor="rgba(79,70,229,0.07)"))
        for xv,col,lbl in [(spot,"#0f172a","CMP"),(lvl["t1"],"#16a34a","T1"),(lvl["t2"],"#15803d","T2")]:
            fig.add_vline(x=xv,line=dict(color=col,dash="dash"),annotation_text=lbl)
        fig.add_hline(y=0,line=dict(color="#94a3b8",width=0.8))
        fig.update_layout(title=f"P&L — {sym} {int(strike)} {otl}",
                          xaxis_title="Spot (₹)",yaxis_title="P&L (₹)",
                          template="plotly_white",height=260)
        st.plotly_chart(fig, use_container_width=True)

    # Chart
    st.subheader("📈 Price Chart")
    _chart(hist, sym, lvl, ind)

    # ── Telegram — this button does NOT cause page refresh ────────────────────
    st.markdown("---")
    st.subheader("📱 Send Telegram Alert")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🚀 Send Alert (Signal + Analysis + Chart)",
                     key="buy_send_tg", type="primary", disabled=not tg_ok):
            with st.spinner("Sending Telegram alerts…"):
                res = send_dual_alert(
                    symbol=sym, price=spot, score=total, grade=grade,
                    entry=lvl["entry"], sl=lvl["sl"],
                    t1=lvl["t1"], t2=lvl["t2"],
                    dow_signal=ind["dow"], rsi=ind["rsi"],
                    strike_price=opt_data.get("strike",0),
                    option_type=opt_data.get("opt_type",""),
                    expiry=expiry, pcr=pcr_val, atm_iv=iv,
                    macd_signal=sc["macd_text"],
                    volume_signal=sc["vol_text"],
                    pattern=sc["struct_text"],
                    support_level=ind["support"],
                    resistance_level=ind["resistance"],
                    atr=ind["atr"], risk_reward_t1=lvl["rr_t1"],
                    sector_rank=stats.get("sector",""),
                    send_chart=send_chart,
                )
            if res.get("skipped"):
                st.info(f"⏭ {res['skipped']}")
            else:
                if res.get("alert1"): st.success("✅ Alert 1 sent")
                if res.get("alert2"): st.success("✅ Alert 2 sent")
                if res.get("chart"):  st.success("✅ Chart sent")
                if not res.get("alert1"):
                    st.error("❌ Failed — check TELEGRAM_BOT_TOKEN in alerts_journal.py")
        if not tg_ok:
            st.caption("Configure TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in alerts_journal.py")

    with col_b:
        if st.button("🗑️ Clear / New Signal", key="buy_clear"):
            if _SS in st.session_state:
                del st.session_state[_SS]
            st.rerun()

    # Book explanations
    with st.expander("📚 Why each dimension matters"):
        explain(f"<b>Dow Theory {sc['dow_score']}/6:</b> {sc['dow_text']}. Fighting the primary trend is the #1 reason retail traders lose on options.", "explain")
        explain(f"<b>Momentum {sc['mom_score']}/6:</b> {sc['rsi_text']}. {sc['macd_text']}. Passarelli: delta moves with momentum, not just price.", "explain")
        explain(f"<b>Volume {sc['vol_score']}/6:</b> {sc['vol_text']}. McMillan: volume is the only confirmation that price is real.", "mcmillan")
        explain(f"<b>Structure {sc['struct_score']}/6:</b> {sc['struct_text']}. Support ₹{ind['support']:.2f} | Resistance ₹{ind['resistance']:.2f}.", "hull")
        explain(f"<b>Options {sc['opt_score']}/6:</b> IV {iv:.1f}% vs HV {stats['hv20']:.1f}%. {'IV < HV → buy edge ✅ (Natenberg)' if iv < stats['hv20'] else 'IV > HV → options costly'}.", "natenberg")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════════════════════════

def _chart(hist, sym, lvl, ind):
    df    = hist.tail(60).copy()
    df.index = pd.to_datetime(df.index)
    close = df["Close"]
    fig   = make_subplots(rows=2,cols=1,shared_xaxes=True,
                          vertical_spacing=0.04,row_heights=[0.72,0.28])
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=close,name=sym,
        increasing_line_color="#22c55e",decreasing_line_color="#ef4444",
    ),row=1,col=1)
    for span,col,nm in [(20,"#3b82f6","EMA20"),(50,"#f97316","EMA50")]:
        fig.add_trace(go.Scatter(x=df.index,y=close.ewm(span=span).mean(),
                                  name=nm,line=dict(color=col,width=1.5)),row=1,col=1)
    for lvl_v,col,lbl in [(lvl["entry"],"#FFD700",f"Entry ₹{lvl['entry']:.0f}"),
                           (lvl["sl"],"#ef4444",f"SL ₹{lvl['sl']:.0f}"),
                           (lvl["t1"],"#22c55e",f"T1 ₹{lvl['t1']:.0f}"),
                           (lvl["t2"],"#15803d",f"T2 ₹{lvl['t2']:.0f}")]:
        fig.add_hline(y=lvl_v,line=dict(color=col,dash="dash",width=1.5),
                      annotation_text=lbl,annotation_position="right",row=1,col=1)
    vc = ["#22c55e" if c>=o else "#ef4444" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Vol",
                          marker_color=vc,opacity=0.6),row=2,col=1)
    fig.update_layout(title=f"{sym} — 60-day chart",xaxis_rangeslider_visible=False,
                      template="plotly_white",height=500,
                      legend=dict(x=0.01,y=0.99),yaxis2_title="Volume")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sector_scan(sector, min_score, grade_filter, scan_type, target_delta):
    syms     = SECTORS[sector]
    ots      = "call" if "CE" in scan_type else "put" if "PE" in scan_type else None
    otl      = "CE" if ots=="call" else "PE" if ots=="put" else "—"
    results  = []
    pb       = st.progress(0, text="Scanning…")

    for i, sym in enumerate(syms):
        ticker = NSE_FNO[sym]
        pb.progress((i+1)/len(syms), text=f"Scanning {sym}…")
        try:
            stats         = fetch_spot_iv(ticker)
            hist          = fetch_history(ticker, period="1y")
            wr            = compute_monthly_winrate(ticker)
            chain, expiry = fetch_option_chain(ticker)
            if not stats or hist is None: continue
            ind = _indicators(hist)
            sc  = score_buy_signal(ind, stats, wr, chain)
            lvl = _buy_levels(ind["spot"], ind["atr"], sc["grade"])

            strike = premium = delta = theta = oi_v = 0.0
            if ots:
                od = {}
                if chain is not None:
                    od = _best_strike_from_chain(chain, ind["spot"], ots, target_delta)
                if not od:
                    od = _bs_fallback(ind["spot"], stats["iv"], ots, target_delta)
                strike  = od.get("strike", 0)
                premium = od.get("premium", 0)
                delta   = od.get("delta", 0)
                theta   = od.get("theta", 0)
                oi_v    = od.get("oi", 0)

            results.append({
                "Symbol":  sym, "Score": sc["total"], "Grade": sc["grade"],
                "CMP":     round(ind["spot"],2), "RSI": round(ind["rsi"],1),
                "Vol ×":   round(ind["vol_surge"],2), "Dow": ind["dow"],
                "EMA✓":    "✅" if ind["ema20"]>ind["ema50"] and ind["spot"]>ind["ema20"] else "❌",
                "Entry":   lvl["entry"], "SL": lvl["sl"],
                "T1":      lvl["t1"], "T2": lvl["t2"],
                "Opt":     otl, "Strike": int(strike) if strike else 0,
                "Premium": premium, "Δ": delta, "Θ/day": theta, "OI": int(oi_v),
            })
        except Exception: continue
        time.sleep(0.04)

    pb.empty()
    if not results:
        st.warning("No results — lower min score or change sector.")
        return

    df = pd.DataFrame(results)
    df = df[df["Score"] >= min_score]
    if grade_filter: df = df[df["Grade"].isin(grade_filter)]
    df = df.sort_values("Score", ascending=False)
    if df.empty:
        st.warning("No stocks passed the filter.")
        return

    eq_c  = ["Symbol","Score","Grade","CMP","RSI","Vol ×","Dow","EMA✓","Entry","SL","T1","T2"]
    oc    = ["Opt","Strike","Premium","Δ","Θ/day","OI"] if ots else []
    cols  = eq_c + oc

    def _cs(v):
        if not isinstance(v,(int,float)): return ""
        if v>=26: return "background-color:#dcfce7;color:#166534;font-weight:700"
        if v>=22: return "background-color:#f0fdf4;color:#166534"
        if v>=18: return "background-color:#fef9c3;color:#854d0e"
        return "background-color:#fee2e2;color:#991b1b"
    def _cg(v):
        return {"A+":"background-color:#dcfce7;color:#166534;font-weight:700",
                "A":"background-color:#f0fdf4;color:#166534",
                "B":"background-color:#fef9c3;color:#854d0e",
                "C":"background-color:#fee2e2;color:#991b1b"}.get(v,"")

    fmt = {"CMP":"₹{:,.2f}","Entry":"₹{:,.2f}","SL":"₹{:,.2f}",
           "T1":"₹{:,.2f}","T2":"₹{:,.2f}","Vol ×":"{:.2f}×",
           "Premium":"₹{:.2f}","Δ":"{:.3f}","Θ/day":"₹{:.3f}","OI":"{:,.0f}"}

    st.dataframe(
        df[cols].style
        .applymap(_cs, subset=["Score"])
        .applymap(_cg, subset=["Grade"])
        .format({k:v for k,v in fmt.items() if k in cols}),
        use_container_width=True, height=500, hide_index=True,
    )
    st.success(f"✅ {len(df)} signals in {sector} — {otl} strikes from live chain")

    explain(
        "Premium = real market mid-price (bid+ask)÷2, OI=0 means BS estimate. "
        "Δ = option moves this much per ₹1 in the stock. "
        "Θ = daily decay cost. Exit any option if premium falls 40% from your entry.",
        "mcmillan",
    )

    tg_ok = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and TELEGRAM_ENABLED)
    if tg_ok and len(df):
        if st.button("📱 Send Scan Summary", key="buy_scan_tg"):
            from alerts_journal import send_telegram, format_scan_summary
            top = [{"symbol":r["Symbol"],"price":r["CMP"],"score":r["Score"],
                    "grade":r["Grade"],"change":0.0,"dow":{"signal":r["Dow"]}}
                   for _,r in df.head(10).iterrows()]
            ok = send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, format_scan_summary(top))
            st.success("✅ Sent!" if ok else "❌ Failed")
