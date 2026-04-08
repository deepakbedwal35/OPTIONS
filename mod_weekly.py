"""
Module: ⚡  Weekly Options Lab
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
    st.title("⚡ Weekly Options Lab — Jack Carter Framework")

    explain(
        "Jack Carter's <b>Weekly Options Trading Strategies</b>: weekly options decay 5× faster "
        "because theta is not linear — it accelerates near expiry. "
        "Carter's strategies: weekly iron condors, theta sprint strangles, momentum plays. "
        "<b>Passarelli warning:</b> gamma near expiry can wipe out a week's premium in one hour.",
        "explain",
    )

    sym_wk = st.selectbox("Stock / Index", list(NSE_FNO.keys()), key="wk_sym")
    with st.spinner("Loading..."):
        stats_wk = fetch_spot_iv(NSE_FNO[sym_wk])

    if not stats_wk:
        st.error("No data.")
        return

    S_wk  = stats_wk["spot"]
    iv_wk = stats_wk["iv"] / 100
    r_wk  = 0.065

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Spot",           f"₹{S_wk:,.2f}")
    c2.metric("IV",             f"{stats_wk['iv']:.1f}%")
    c3.metric("IVR",            f"{stats_wk['ivr']:.0f}%")
    c4.metric("Daily 1σ",       f"₹{S_wk*iv_wk/np.sqrt(252):,.0f}")
    c5.metric("Weekly 1σ",      f"₹{S_wk*iv_wk/np.sqrt(52):,.0f}")

    w1s = S_wk * iv_wk / np.sqrt(52)
    w2s = w1s * 2

    explain(
        f"Expected weekly move: <b>±₹{w1s:,.0f} ({w1s/S_wk*100:.1f}%)</b> at 1σ, "
        f"±₹{w2s:,.0f} ({w2s/S_wk*100:.1f}%) at 2σ. "
        f"Carter: sell weekly strangles outside 2σ — call above ₹{S_wk+w2s:,.0f}, put below ₹{S_wk-w2s:,.0f}.",
        "explain",
    )

    # Theta Sprint Calculator
    st.subheader("⏱️ Theta Sprint — Carter's Primary Weekly Strategy")
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        sprint_dte   = st.selectbox("Sell on (DTE)", [7,5,4], index=0)
        sprint_delta = st.slider("Target delta", 0.05, 0.25, 0.15)
    with col_ts2:
        T_sp = max(sprint_dte/365, 0.001)
        # Find call strike at delta
        lo_k, hi_k = S_wk*0.80, S_wk*1.30
        for _ in range(80):
            mid_k = (lo_k+hi_k)/2
            d = bs_greeks(S_wk,mid_k,T_sp,r_wk,iv_wk,"call")["delta"]
            if d > sprint_delta: lo_k=mid_k
            else: hi_k=mid_k
        ck = round(mid_k/max(50,int(S_wk*0.01)))*max(50,int(S_wk*0.01))
        # Find put strike at delta
        lo_k, hi_k = S_wk*0.70, S_wk*1.20
        for _ in range(80):
            mid_k = (lo_k+hi_k)/2
            d = abs(bs_greeks(S_wk,mid_k,T_sp,r_wk,iv_wk,"put")["delta"])
            if d < sprint_delta: hi_k=mid_k
            else: lo_k=mid_k
        pk = round(mid_k/max(50,int(S_wk*0.01)))*max(50,int(S_wk*0.01))
        cp_wk = bs_price(S_wk,ck,T_sp,r_wk,iv_wk,"call")
        pp_wk = bs_price(S_wk,pk,T_sp,r_wk,iv_wk,"put")
        tot_wk = cp_wk + pp_wk
        gc = bs_greeks(S_wk,ck,T_sp,r_wk,iv_wk,"call")
        gp = bs_greeks(S_wk,pk,T_sp,r_wk,iv_wk,"put")
        dt = abs(gc["theta"]) + abs(gp["theta"])
        st.markdown(f"""
        <div class="score-card">
          <div class="score-title">Carter Theta Sprint — {sprint_dte}DTE Strangle</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:8px">
            <div><div style="font-size:.72rem;color:#64748b">Short Call</div>
                 <div style="font-weight:700">₹{ck:,}</div>
                 <div style="font-size:.78rem">₹{cp_wk:.2f} | Δ={gc['delta']:.3f}</div></div>
            <div><div style="font-size:.72rem;color:#64748b">Short Put</div>
                 <div style="font-weight:700">₹{pk:,}</div>
                 <div style="font-size:.78rem">₹{pp_wk:.2f} | Δ={gp['delta']:.3f}</div></div>
            <div><div style="font-size:.72rem;color:#64748b">Total / Daily θ</div>
                 <div style="font-weight:700">₹{tot_wk:.2f}</div>
                 <div style="font-size:.78rem">₹{dt:.2f}/day</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Theta decay comparison
    st.subheader("Theta Decay: Weekly vs Monthly")
    dte_wk_r  = np.linspace(7,0,50)
    dte_mo_r  = np.linspace(30,0,50)
    tv_wk = [bs_price(S_wk,S_wk,max(d/365,0.001),r_wk,iv_wk,"call") for d in dte_wk_r]
    tv_mo = [bs_price(S_wk,S_wk,max(d/365,0.001),r_wk,iv_wk,"call") for d in dte_mo_r]
    fig_td = make_subplots(rows=1,cols=2,subplot_titles=["Weekly (7 DTE)","Monthly (30 DTE)"])
    fig_td.add_trace(go.Scatter(x=dte_wk_r,y=tv_wk,line=dict(color="#10b981",width=2),name="Weekly"),row=1,col=1)
    fig_td.add_trace(go.Scatter(x=dte_mo_r,y=tv_mo,line=dict(color="#4f46e5",width=2),name="Monthly"),row=1,col=2)
    for r,c in [(1,1),(1,2)]: fig_td.update_xaxes(autorange="reversed",row=r,col=c)
    fig_td.update_layout(template="plotly_white",height=260,showlegend=False)
    st.plotly_chart(fig_td, use_container_width=True)
    explain("Carter: weekly curve is steeper near expiry — most theta earned in last 2 days. "
            "Sell Monday, close Thursday. Passarelli: same gamma that accelerates theta accelerates losses.", "explain")

    # Gamma explosion chart
    st.subheader("⚠️ Gamma Explosion Near Expiry (Passarelli)")
    g_dtes = [30,21,14,7,5,3,2,1]
    g_vals = [bs_greeks(S_wk,S_wk,max(d/365,0.001),r_wk,iv_wk,"call")["gamma"] for d in g_dtes]
    fig_gx = go.Figure(go.Bar(x=[f"{d}d" for d in g_dtes],y=g_vals,
                               marker_color=["#22c55e" if d>=14 else "#f97316" if d>=5 else "#ef4444" for d in g_dtes]))
    fig_gx.update_layout(title="ATM Gamma vs DTE",yaxis_title="Gamma",template="plotly_white",height=240)
    st.plotly_chart(fig_gx, use_container_width=True)
    explain("Red bars (<5 DTE) = gamma danger zone. Carter rule: close weeklies by Thursday regardless of P&L. "
            "Never hold a short ATM option to expiry.", "danger")

    st.subheader("Carter's Weekly Playbook")
    for name, desc, style in [
        ("Theta Sprint Strangle", f"Sell {ck} call / {pk} put for ₹{tot_wk:.2f}. Close Thursday at 50%.", "safe"),
        ("Weekly Iron Condor", "Add wings for defined risk. Carter: use IC for NIFTY weeklies.", "explain"),
        ("Rolling Friday", "If Thursday position profitable >40%, close and re-sell next week's strangle.", "explain"),
        ("Event avoidance", "NEVER sell if earnings/RBI/FOMC falls in expiry week. Skip that week entirely.", "danger"),
    ]:
        explain(f"<b>{name}:</b> {desc}", style)


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE E — HEDGE FUND DASHBOARD  ███
    # ═══════════════════════════════════════════════════════════════
