"""
Module: 📈  Payoff Builder
Options Alpha Platform v5.0 — Individual Module
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, time
from datetime import datetime
warnings.filterwarnings("ignore")

from utils import (
    explain, NSE_FNO, SECTORS, INDICES_ONLY, ALL_STOCKS_ONLY,
    bs_price, bs_greeks, bs_d1d2, implied_vol,
    fetch_history, fetch_spot_iv, fetch_option_chain,
    compute_monthly_winrate, build_scanner_row, get_yf
)


def render():
    st.title("📈 Multi-Leg Payoff Builder")

    explain(
        "Build any strategy by stacking legs. The diagram shows P&L at expiry (solid) and "
        "today (dashed via Black-Scholes). Gap between the two curves = remaining time value. "
        "McMillan: always draw your payoff diagram BEFORE entering a trade. "
        "Know your max loss, max profit, and both breakevens before pressing the button.",
        "mcmillan",
    )

    if "pb_legs" not in st.session_state:
        st.session_state.pb_legs = []

    # Quick load real data
    sym_pb = st.selectbox("Pre-fill spot from live data", ["Manual"] + list(NSE_FNO.keys()), key="pb_sym")
    if sym_pb != "Manual":
        with st.spinner("Loading..."):
            stats_pb = fetch_spot_iv(NSE_FNO[sym_pb])
        default_spot = int(stats_pb["spot"]) if stats_pb else 18000
        default_iv   = round(stats_pb["iv"],1) if stats_pb else 18.0
    else:
        default_spot, default_iv = 18000, 18.0

    col_add, col_chart = st.columns([1, 2])

    with col_add:
        st.subheader("Add Leg")
        l_type   = st.selectbox("Type", ["Long Call","Short Call","Long Put","Short Put"])
        l_strike = st.number_input("Strike (₹)", value=default_spot, step=50)
        l_prem   = st.number_input("Premium (₹)", value=200, step=5, min_value=1)
        l_qty    = st.number_input("Lots", value=1, min_value=1, max_value=20)
        l_iv     = st.number_input("IV (%)", value=default_iv, step=0.5) / 100
        l_dte    = st.number_input("DTE (days)", value=30, min_value=1, max_value=365)

        c1b, c2b = st.columns(2)
        if c1b.button("➕ Add", type="primary"):
            st.session_state.pb_legs.append(dict(
                type=l_type, strike=l_strike, prem=l_prem,
                qty=l_qty, iv=l_iv, dte=l_dte
            ))
        if c2b.button("🗑️ Clear"):
            st.session_state.pb_legs = []

        # Templates
        st.markdown("**Templates (NIFTY-style):**")
        tpls = {
            "Iron Condor": [
                {"type":"Short Put",  "strike":default_spot-1500, "prem":140, "qty":1, "iv":0.19,"dte":30},
                {"type":"Long Put",   "strike":default_spot-2000, "prem": 60, "qty":1, "iv":0.21,"dte":30},
                {"type":"Short Call", "strike":default_spot+1500, "prem":130, "qty":1, "iv":0.17,"dte":30},
                {"type":"Long Call",  "strike":default_spot+2000, "prem": 55, "qty":1, "iv":0.17,"dte":30},
            ],
            "Short Strangle": [
                {"type":"Short Put",  "strike":default_spot-1500, "prem":160, "qty":1, "iv":0.19,"dte":30},
                {"type":"Short Call", "strike":default_spot+1500, "prem":150, "qty":1, "iv":0.17,"dte":30},
            ],
            "Bull Call Spread": [
                {"type":"Long Call",  "strike":default_spot,      "prem":320, "qty":1, "iv":0.18,"dte":30},
                {"type":"Short Call", "strike":default_spot+1000, "prem":160, "qty":1, "iv":0.17,"dte":30},
            ],
            "Long Straddle": [
                {"type":"Long Call", "strike":default_spot, "prem":300, "qty":1, "iv":0.18,"dte":30},
                {"type":"Long Put",  "strike":default_spot, "prem":280, "qty":1, "iv":0.18,"dte":30},
            ],
        }
        for tname, tlegs in tpls.items():
            if st.button(tname, key=f"tpl_{tname}"):
                st.session_state.pb_legs = tlegs

        if st.session_state.pb_legs:
            st.markdown("**Legs:**")
            for i, leg in enumerate(st.session_state.pb_legs):
                c_l, c_x = st.columns([5, 1])
                c_l.caption(f"{leg['qty']}× {leg['type']} K={leg['strike']} P={leg['prem']}")
                if c_x.button("✕", key=f"rm_{i}"):
                    st.session_state.pb_legs.pop(i)
                    st.rerun()

    with col_chart:
        if not st.session_state.pb_legs:
            st.info("Add legs or pick a template on the left.")
        else:
            center = np.mean([l["strike"] for l in st.session_state.pb_legs])
            xs = np.linspace(center * 0.78, center * 1.22, 400)
            r_pb = 0.065

            def expiry_pnl(leg, S):
                K, p, q = leg["strike"], leg["prem"], leg["qty"]
                if leg["type"] == "Long Call":  return (max(S-K,0)-p)*q
                if leg["type"] == "Short Call": return (p-max(S-K,0))*q
                if leg["type"] == "Long Put":   return (max(K-S,0)-p)*q
                if leg["type"] == "Short Put":  return (p-max(K-S,0))*q
                return 0

            def now_pnl(leg, S):
                K, q = leg["strike"], leg["qty"]
                T = max(leg["dte"]/365, 0.001)
                ot = "call" if "Call" in leg["type"] else "put"
                val_now = bs_price(S, K, T, r_pb, leg["iv"], ot)
                sign = 1 if leg["type"].startswith("Long") else -1
                return sign * (val_now - leg["prem"]) * q

            pnl_exp = np.array([sum(expiry_pnl(l, s) for l in st.session_state.pb_legs) for s in xs])
            pnl_now = np.array([sum(now_pnl(l, s) for l in st.session_state.pb_legs) for s in xs])

            # Stats
            max_p = np.max(pnl_exp)
            min_p = np.min(pnl_exp)
            net_p = sum((-1 if l["type"].startswith("Long") else 1)*l["prem"]*l["qty"] for l in st.session_state.pb_legs)
            bes = []
            for i in range(1, len(pnl_exp)):
                if pnl_exp[i-1]*pnl_exp[i] < 0:
                    be = xs[i-1] - pnl_exp[i-1]*(xs[i]-xs[i-1])/(pnl_exp[i]-pnl_exp[i-1])
                    bes.append(be)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Max Profit",  "Unlimited" if max_p>1e5 else f"₹{max_p:,.0f}")
            m2.metric("Max Loss",    "Unlimited" if min_p<-1e5 else f"₹{min_p:,.0f}")
            m3.metric("Net Premium", f"{'+'if net_p>=0 else ''}₹{net_p:,.0f}")
            m4.metric("Breakeven(s)", " / ".join([f"₹{b:,.0f}" for b in bes]) if bes else "—")

            fig_pb = go.Figure()
            fig_pb.add_trace(go.Scatter(x=xs, y=pnl_exp, name="At Expiry",
                                         line=dict(color="#4f46e5", width=2.5)))
            fig_pb.add_trace(go.Scatter(x=xs, y=pnl_now, name="Today (BS)",
                                         line=dict(color="#f59e0b", width=2, dash="dash")))
            fig_pb.add_hline(y=0, line=dict(color="#94a3b8", dash="dot", width=1))
            for be in bes:
                fig_pb.add_vline(x=be, line=dict(color="#f97316", dash="dash", width=1))
            # profit zone shading
            fig_pb.add_traces([
                go.Scatter(x=xs, y=np.where(pnl_exp >= 0, pnl_exp, 0),
                           fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
                           line=dict(width=0), showlegend=False),
                go.Scatter(x=xs, y=np.where(pnl_exp < 0, pnl_exp, 0),
                           fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                           line=dict(width=0), showlegend=False),
            ])
            fig_pb.update_layout(xaxis_title="Spot (₹)", yaxis_title="P&L (₹)",
                                  template="plotly_white", height=400,
                                  legend=dict(x=0.01, y=0.99), hovermode="x unified")
            st.plotly_chart(fig_pb, use_container_width=True)

            explain(
                "Gap between solid (expiry) and dashed (today) lines = time value remaining. "
                "As days pass, the dashed line moves toward the solid line — this is theta decay. "
                "For premium sellers: you want the dashed line to start above zero and the solid line to end above zero. "
                "For buyers: you need the stock to move far enough that the solid line ends above zero.",
                "mcmillan",
            )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 4 — GREEKS DASHBOARD  ███
    # ═══════════════════════════════════════════════════════════════
