"""
Module: 🎲  Probability Calculator
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
    st.title("🎲 Probability Calculator — Natenberg + Cordier")

    explain(
        "Natenberg: options ARE probability. Every option price embeds the market's probability estimate. "
        "This calculator combines the BS log-normal model with your actual historical data "
        "to give you the most accurate probability estimate possible.",
        "natenberg",
    )

    sym_pc = st.selectbox("Stock", list(NSE_FNO.keys()), key="pc_sym")
    with st.spinner("Loading..."):
        stats_pc = fetch_spot_iv(NSE_FNO[sym_pc])
        wr_pc    = compute_monthly_winrate(NSE_FNO[sym_pc])

    if not stats_pc:
        st.error("No data.")
        return

    S_pc = stats_pc["spot"]
    col_i, col_o = st.columns(2)
    with col_i:
        target_up = st.number_input("Upper target price (₹)", value=round(S_pc*1.05,0), step=50.0)
        target_dn = st.number_input("Lower target price (₹)", value=round(S_pc*0.95,0), step=50.0)
        iv_pc     = st.number_input("IV (%)", value=stats_pc["iv"], step=0.5)
        dte_pc    = st.number_input("DTE (days)", value=30, min_value=1)
        r_pc      = 0.065

    T_pc = max(dte_pc/365, 0.001)
    iv_v = iv_pc / 100
    d1_up, d2_up = bs_d1d2(S_pc, target_up, T_pc, r_pc, iv_v)
    d1_dn, d2_dn = bs_d1d2(S_pc, target_dn, T_pc, r_pc, iv_v)
    prob_above = (1 - norm.cdf(d2_up)) * 100
    prob_below = norm.cdf(d2_dn) * 100
    prob_between = 100 - prob_above - prob_below
    exp_price = S_pc * np.exp((r_pc - 0.5*iv_v**2)*T_pc)

    with col_o:
        st.metric("P(above upper)", f"{prob_above:.1f}%")
        st.metric("P(between)",     f"{prob_between:.1f}%",
                  delta="Iron condor profit zone" if prob_between > 60 else "Tight zone")
        st.metric("P(below lower)", f"{prob_below:.1f}%")
        st.metric("Expected price", f"₹{exp_price:,.0f}")

        explain(
            f"P(between) = {prob_between:.1f}% is the <b>theoretical probability of an iron condor profiting at expiry</b>. "
            f"For a short strangle: it equals the probability of full premium capture. "
            f"Cordier's minimum: P(between) ≥ 70% before selling a strangle.",
            "cordier",
        )

    # Distribution chart
    xs_pc = np.linspace(S_pc*0.70, S_pc*1.30, 200)
    pdf_vals = []
    for x in xs_pc:
        z = (np.log(x/S_pc) - (r_pc - 0.5*iv_v**2)*T_pc) / (iv_v*np.sqrt(T_pc))
        pdf_vals.append(norm.pdf(z) / (x * iv_v * np.sqrt(T_pc)))

    fig_prob = go.Figure()
    # Shade zones
    mask_below  = xs_pc <= target_dn
    mask_between= (xs_pc >= target_dn) & (xs_pc <= target_up)
    mask_above  = xs_pc >= target_up
    for mask, colour, label in [
        (mask_below,   "rgba(239,68,68,0.25)",   "Below — short put risk"),
        (mask_between, "rgba(34,197,94,0.20)",   "Flat zone — profit"),
        (mask_above,   "rgba(239,68,68,0.25)",   "Above — short call risk"),
    ]:
        if mask.any():
            fig_prob.add_trace(go.Scatter(
                x=xs_pc[mask], y=np.array(pdf_vals)[mask],
                fill="tozeroy", fillcolor=colour, line=dict(width=0), name=label,
            ))
    fig_prob.add_trace(go.Scatter(x=xs_pc, y=pdf_vals,
                                   line=dict(color="#1e293b", width=2), name="PDF"))
    fig_prob.add_vline(x=S_pc,       line=dict(color="black", dash="dot"),  annotation_text="Spot")
    fig_prob.add_vline(x=target_up,  line=dict(color="#ef4444", dash="dash"), annotation_text="Upper")
    fig_prob.add_vline(x=target_dn,  line=dict(color="#ef4444", dash="dash"), annotation_text="Lower")
    fig_prob.update_layout(title=f"Log-Normal Distribution — {sym_pc} ({dte_pc}-day horizon)",
                           xaxis_title="Price (₹)", yaxis_title="Probability Density",
                           template="plotly_white", height=340)
    st.plotly_chart(fig_prob, use_container_width=True)

    # Historical overlay
    if wr_pc:
        flat_hist = wr_pc["summary"]["flat_5pct"]
        up5_hist  = wr_pc["summary"]["up_5pct"]
        dn5_hist  = wr_pc["summary"]["dn_5pct"]
        st.subheader("Model vs Historical Comparison")
        comp_df = pd.DataFrame({
            "Zone":       ["Below lower (−5%)", "Flat (±5%)", "Above upper (+5%)"],
            "BS Model (%)":   [round(prob_below,1), round(prob_between,1), round(prob_above,1)],
            "Historical (%)": [round(dn5_hist,1),   round(flat_hist,1),   round(up5_hist,1)],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        explain(
            f"Model says flat = {prob_between:.1f}%, history says flat = {flat_hist:.1f}%. "
            f"{'They agree — strong confidence.' if abs(prob_between-flat_hist)<10 else 'They diverge — use the more conservative (lower) number when sizing your trade.'} "
            f"Natenberg: the BS model assumes log-normal returns; real stocks have fat tails, so historical data often shows more extreme moves than the model predicts.",
            "natenberg",
        )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 10 — ROLLING ENGINE  ███
    # ═══════════════════════════════════════════════════════════════
