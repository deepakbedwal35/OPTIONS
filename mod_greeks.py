"""
Module: 🔢  Greeks Dashboard
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
    st.title("🔢 Greeks Dashboard — Passarelli Framework")

    explain(
        "Passarelli: 'Greeks are the language of options. If you don't speak Greeks, "
        "you're trading blind.' This dashboard shows all first and higher-order Greeks "
        "calculated from real market data. Every number is explained in plain English.",
        "explain",
    )

    sym_g = st.selectbox("Stock (live data)", list(NSE_FNO.keys()))
    with st.spinner("Fetching..."):
        stats_g = fetch_spot_iv(NSE_FNO[sym_g])

    if not stats_g:
        st.error("Could not fetch data.")
        return

    S_g = stats_g["spot"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Live Spot",  f"₹{S_g:,.2f}")
    c2.metric("HV20",       f"{stats_g['hv20']:.1f}%")
    c3.metric("IV (approx)",f"{stats_g['iv']:.1f}%")
    c4.metric("IVR",        f"{stats_g['ivr']:.0f}%")

    col_sl, col_out = st.columns([1, 2])
    with col_sl:
        S   = st.slider("Spot", int(S_g*0.80), int(S_g*1.20), int(S_g), step=max(1,int(S_g*0.002)))
        K   = st.slider("Strike", int(S_g*0.80), int(S_g*1.20), int(S_g), step=max(1,int(S_g*0.002)))
        iv  = st.slider("IV (%)", 5.0, 80.0, stats_g["iv"], step=0.5) / 100
        dte = st.slider("DTE (days)", 1, 90, 30)
        opt_type = st.radio("Option type", ["call","put"], horizontal=True)

    T_g = max(dte/365, 0.001)
    r_g = 0.065
    g   = bs_greeks(S, K, T_g, r_g, iv, opt_type)
    price = bs_price(S, K, T_g, r_g, iv, opt_type)

    with col_out:
        m = st.columns(3)
        m[0].metric("Price",     f"₹{price:.2f}")
        m[1].metric("Δ Delta",   f"{g['delta']:.4f}")
        m[2].metric("Γ Gamma",   f"{g['gamma']:.6f}")
        m2 = st.columns(3)
        m2[0].metric("Θ Theta/d", f"₹{g['theta']:.4f}")
        m2[1].metric("V Vega/%",  f"₹{g['vega']:.4f}")
        m2[2].metric("Prob ITM",  f"{g['prob_itm']*100:.1f}%")

        explain(
            f"<b>Delta {g['delta']:.3f}</b>: every ₹1 move in {sym_g} changes this option's price by ₹{g['delta']:.3f}. "
            f"Also = {g['prob_itm']*100:.1f}% chance of expiring ITM. "
            f"<b>Cordier's sell-zone:</b> delta 0.10–0.20 (80–90% prob of expiring worthless). "
            f"Current delta is {'✅ in sell-zone' if 0.08 < abs(g['delta']) < 0.22 else '⚠️ outside sell-zone — check strike'}.",
            "cordier",
        )

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("First-Order Greeks")
        explain(
            "**Delta** = directional exposure. **Gamma** = how fast delta changes (risk near expiry). "
            "**Theta** = daily time decay ₹ (negative = you pay it daily if long). "
            "**Vega** = sensitivity to 1% change in IV. **Rho** = interest rate sensitivity (minor for equities).",
            "explain",
        )
        spot_range = np.linspace(S_g * 0.80, S_g * 1.20, 80)
        greek_sel = st.selectbox("Plot Greek vs Spot", ["delta","gamma","theta","vega"])
        yvals = [bs_greeks(s, K, T_g, r_g, iv, opt_type)[greek_sel] for s in spot_range]
        fig_gc = go.Figure()
        fig_gc.add_trace(go.Scatter(x=spot_range, y=yvals, line=dict(color="#4f46e5", width=2)))
        fig_gc.add_vline(x=S, line=dict(color="#ef4444", dash="dash"), annotation_text="Spot")
        fig_gc.add_vline(x=K, line=dict(color="#22c55e", dash="dash"), annotation_text="Strike")
        fig_gc.update_layout(xaxis_title="Spot (₹)", yaxis_title=greek_sel.capitalize(),
                              template="plotly_white", height=280)
        st.plotly_chart(fig_gc, use_container_width=True)

    with col_g2:
        st.subheader("Higher-Order Greeks (Passarelli Ch.8–10)")
        explain(
            "**Vanna** (dΔ/dσ): if IV spikes, your delta changes — your hedge becomes wrong. "
            "Critical for vol traders. "
            "**Charm** (dΔ/dt): delta bleeds daily. A 0.20 delta call today may be 0.15 delta tomorrow "
            "even if spot doesn't move — this is charm. "
            "**Volga** (dVega/dσ): convexity of vega — you profit from large IV moves when long volga.",
            "explain",
        )
        ho_data = {
            "Greek":       ["Vanna",         "Charm/day",      "Volga"],
            "Formula":     ["dΔ/dσ",          "dΔ/dt",          "dVega/dσ"],
            "Value":       [f"{g['vanna']:.4f}", f"{g['charm']:.6f}", f"{g['volga']:.4f}"],
            "Sign means":  [
                ("+ = delta rises with IV" if g['vanna']>0 else "- = delta falls with IV"),
                ("+ = delta grows over time" if g['charm']>0 else "- = delta shrinks over time"),
                ("+ = long vol-of-vol" if g['volga']>0 else "- = short vol-of-vol"),
            ],
        }
        st.dataframe(pd.DataFrame(ho_data), use_container_width=True, hide_index=True)

        # P&L attribution
        st.subheader("P&L Attribution (1-lot, 1-day, 1% moves)")
        lot = 50
        d_delta  = g["delta"] * S * 0.01 * lot
        d_gamma  = 0.5 * g["gamma"] * (S*0.01)**2 * lot
        d_theta  = g["theta"] * lot
        d_vega   = g["vega"]  * 1 * lot   # 1% IV move
        attr_df  = pd.DataFrame({
            "Source":  ["Delta (1% spot)", "Gamma (convexity)", "Theta (1 day)", "Vega (1% IV)"],
            "P&L (₹)": [d_delta, d_gamma, d_theta, d_vega],
        })
        fig_attr = px.bar(attr_df, x="Source", y="P&L (₹)",
                          color="P&L (₹)",
                          color_continuous_scale=["#ef4444","#94a3b8","#22c55e"],
                          title="Daily P&L Decomposition (₹ per lot)")
        fig_attr.update_layout(template="plotly_white", height=250, showlegend=False)
        st.plotly_chart(fig_attr, use_container_width=True)


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 5 — BS PRICER + EDGE  ███
    # ═══════════════════════════════════════════════════════════════
