"""
Module: ⚙️  BS Pricer + Edge
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
    st.title("⚙️ Black-Scholes Pricer + Theoretical Edge")

    explain(
        "Natenberg's core concept: every option has a <b>theoretical value</b> (BS price at your IV estimate) "
        "and a <b>market price</b>. The difference is <b>theoretical edge</b>. "
        "Positive edge = market overpaying → SELL. Negative edge = market underpaying → BUY. "
        "Professional traders only trade when they have measurable edge.",
        "natenberg",
    )

    sym_bs = st.selectbox("Pre-fill from live data", ["Manual"] + list(NSE_FNO.keys()), key="bs_sym")
    if sym_bs != "Manual":
        with st.spinner("Loading..."):
            stats_bs = fetch_spot_iv(NSE_FNO[sym_bs])
        dS   = stats_bs["spot"]   if stats_bs else 18000.0
        dIV  = stats_bs["iv"]     if stats_bs else 18.0
    else:
        dS, dIV = 18000.0, 18.0

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Inputs")
        bsS  = st.number_input("Spot S (₹)",     value=float(dS),  step=10.0)
        bsK  = st.number_input("Strike K (₹)",   value=float(dS),  step=50.0)
        bsIV = st.number_input("Your IV σ (%)",  value=float(dIV), step=0.5)
        bsDTE= st.number_input("DTE (days)",      value=30,         min_value=1)
        bsR  = st.number_input("Risk-free r (%)", value=6.5,        step=0.1)
        mktC = st.number_input("Market Call price (₹) for edge calc", value=0.0, step=1.0)
        mktP = st.number_input("Market Put price (₹) for edge calc",  value=0.0, step=1.0)

    T_b = max(bsDTE/365, 0.001)
    r_b = bsR / 100
    iv_b = bsIV / 100
    d1v, d2v = bs_d1d2(bsS, bsK, T_b, r_b, iv_b)
    cP = bs_price(bsS, bsK, T_b, r_b, iv_b, "call")
    pP = bs_price(bsS, bsK, T_b, r_b, iv_b, "put")

    with c2:
        st.subheader("Results")
        r1, r2 = st.columns(2)
        r1.metric("Call (BS)",  f"₹{cP:.2f}")
        r2.metric("Put (BS)",   f"₹{pP:.2f}")
        r1.metric("d₁",         f"{d1v:.4f}")
        r2.metric("d₂",         f"{d2v:.4f}")
        r1.metric("N(d₁)",      f"{norm.cdf(d1v):.4f}")
        r2.metric("N(d₂)",      f"{norm.cdf(d2v):.4f}")

        explain(
            f"<b>Step-by-step derivation:</b><br>"
            f"d₁ = [ln({bsS:.0f}/{bsK:.0f}) + ({bsR:.1f}% + σ²/2)×{bsDTE}d/365] / (σ√T) = <b>{d1v:.4f}</b><br>"
            f"d₂ = d₁ − σ√T = {d1v:.4f} − {iv_b:.4f}×{T_b**0.5:.4f} = <b>{d2v:.4f}</b><br>"
            f"Call = {bsS:.0f}×N({d1v:.4f}) − {bsK:.0f}×e^(−{r_b:.4f}×{T_b:.4f})×N({d2v:.4f}) = <b>₹{cP:.2f}</b>",
            "natenberg",
        )

        if mktC > 0:
            iv_mkt_c = implied_vol(bsS, bsK, T_b, r_b, mktC, "call") * 100
            edge_c = mktC - cP
            style  = "safe" if edge_c > 0 else "warning"
            explain(
                f"Market call price ₹{mktC:.2f} → Market IV = <b>{iv_mkt_c:.2f}%</b> "
                f"vs your model IV {bsIV:.1f}%.<br>"
                f"<b>Theoretical Edge = {'+'if edge_c>=0 else ''}₹{edge_c:.2f}</b> — "
                f"{'Market OVERPAYING → SELL the call' if edge_c>0 else 'Market UNDERPAYING → BUY the call'}",
                style,
            )
        if mktP > 0:
            iv_mkt_p = implied_vol(bsS, bsK, T_b, r_b, mktP, "put") * 100
            edge_p = mktP - pP
            style  = "safe" if edge_p > 0 else "warning"
            explain(
                f"Market put price ₹{mktP:.2f} → Market IV = <b>{iv_mkt_p:.2f}%</b>.<br>"
                f"<b>Theoretical Edge = {'+'if edge_p>=0 else ''}₹{edge_p:.2f}</b> — "
                f"{'Market OVERPAYING → SELL the put' if edge_p>0 else 'Market UNDERPAYING → BUY the put'}",
                style,
            )

    st.markdown("---")
    st.subheader("Sensitivity Grid — Call Price (IV% × DTE)")
    ivs_g  = [8,12,15,18,20,25,30,40,50]
    dtes_g = [5,10,15,21,30,45,60,90]
    grid   = pd.DataFrame(
        {f"{d}d": [f"₹{bs_price(bsS,bsK,d/365,r_b,i/100,'call'):.1f}" for i in ivs_g]
         for d in dtes_g},
        index=[f"σ={i}%" for i in ivs_g],
    )
    st.dataframe(grid, use_container_width=True)
    explain(
        "Each cell = theoretical call price at that IV and DTE. "
        "The bottom row (high IV) vs top row (low IV) shows how much more you receive/pay for vol. "
        "Natenberg: when selling, pick a row near current IV and verify market price > that cell → edge confirmed.",
        "natenberg",
    )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 6 — VOLATILITY LAB  ███
    # ═══════════════════════════════════════════════════════════════
