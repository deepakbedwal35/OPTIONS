"""
Module: 📋  Covered Call Optimiser
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
    st.title("📋 Covered Call Optimiser — Ellman System")

    explain(
        "<b>Alan Ellman's system</b> classifies every covered call strike into three zones: "
        "<b>OTM</b> (growth + income), <b>ATM</b> (maximum income), <b>ITM</b> (maximum downside protection). "
        "This module runs Ellman's full selection process and calculates "
        "Static Return, If-Called Return, Downside Protection, and Breakeven — "
        "the four numbers Ellman says you must know before entering any covered call.",
        "explain",
    )

    sym_cc = st.selectbox("Stock", list(NSE_FNO.keys()), key="cc_sym")
    with st.spinner("Loading..."):
        stats_cc = fetch_spot_iv(NSE_FNO[sym_cc])

    if not stats_cc:
        st.error("No data.")
        return

    S_cc   = stats_cc["spot"]
    iv_cc  = stats_cc["iv"] / 100
    T_cc   = 30 / 365
    r_cc   = 0.065

    col_cc_i, col_cc_k = st.columns([1,2])
    with col_cc_i:
        cost_basis  = st.number_input("Cost basis / current price (₹)", value=float(S_cc), step=10.0)
        min_ret     = st.slider("Min monthly return target (%)", 1.0, 5.0, 2.0, step=0.5)
        protection  = st.slider("Priority: Income←→Protection", 0, 10, 5)

    with col_cc_k:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Spot",       f"₹{S_cc:,.2f}")
        c2.metric("Cost basis", f"₹{cost_basis:,.2f}")
        c3.metric("Unrealised", f"{(S_cc/cost_basis-1)*100:+.1f}%", delta_color="normal")
        c4.metric("IVR",        f"{stats_cc['ivr']:.0f}%")

    step_cc = max(50, int(S_cc*0.01))
    cc_rows = []
    for K in np.arange(S_cc*0.90, S_cc*1.15, step_cc):
        K = round(K/step_cc)*step_cc
        cp = bs_price(S_cc, K, T_cc, r_cc, iv_cc, "call")
        if cp < 1: continue
        g  = bs_greeks(S_cc, K, T_cc, r_cc, iv_cc, "call")
        mono = (K/S_cc-1)*100
        cat  = "OTM" if mono>1.5 else "ITM" if mono<-1.5 else "ATM"
        cc_rows.append({
            "Strike":            int(K),
            "Type":              cat,
            "Call Premium":      round(cp,2),
            "Static Ret (%)":    round(cp/cost_basis*100, 2),
            "If-Called Ret (%)": round((K-cost_basis+cp)/cost_basis*100, 2),
            "Annualised (%)":    round(cp/cost_basis*100*12, 1),
            "Downside Prot (%)": round(cp/S_cc*100, 2),
            "Breakeven (₹)":     round(S_cc-cp, 2),
            "Delta":             round(g["delta"],3),
            "Meets Target":      (cp/cost_basis*100) >= min_ret,
        })

    cc_df = pd.DataFrame(cc_rows)
    if cc_df.empty:
        st.warning("No strikes found.")
        return

    if protection >= 7:
        rec_df, rec_type = cc_df[cc_df["Type"]=="ITM"].sort_values("Downside Prot (%)",ascending=False), "ITM (max protection)"
    elif protection <= 3:
        rec_df, rec_type = cc_df[cc_df["Type"]=="OTM"].sort_values("If-Called Ret (%)",ascending=False), "OTM (growth+income)"
    else:
        rec_df, rec_type = cc_df[cc_df["Type"]=="ATM"].sort_values("Static Ret (%)",ascending=False), "ATM (max income)"
    if rec_df.empty:
        rec_df = cc_df.sort_values("Static Ret (%)",ascending=False)

    top = rec_df.iloc[0]
    explain(
        f"<b>Ellman Recommendation ({rec_type}):</b> "
        f"Sell the <b>₹{int(top['Strike']):,} {top['Type']} call</b> for ₹{top['Call Premium']:.2f}. "
        f"Static return: <b>{top['Static Ret (%)']:.2f}%/month ({top['Annualised (%)']:.1f}% pa)</b>. "
        f"Downside protection: {top['Downside Prot (%)']:.2f}% (breakeven ₹{top['Breakeven (₹)']:,.0f}). "
        f"If called away: {top['If-Called Ret (%)']:.2f}%.",
        "safe" if top["Meets Target"] else "warning",
    )

    # Three zones side by side
    st.subheader("Three-Zone Comparison (Ellman)")
    col_z1, col_z2, col_z3 = st.columns(3)
    for zone, col, bg in [("OTM",col_z1,"#f0fdf4"),("ATM",col_z2,"#dbeafe"),("ITM",col_z3,"#fef9c3")]:
        zr = cc_df[cc_df["Type"]==zone]
        if not zr.empty:
            z = zr.sort_values("Static Ret (%)",ascending=False).iloc[0]
            col.markdown(f"""
            <div style="background:{bg};border-radius:8px;padding:12px;text-align:center">
              <div style="font-weight:700">{zone}</div>
              <div style="font-size:1.3rem;font-weight:700;margin:4px 0">₹{int(z['Strike']):,}</div>
              <div style="font-size:.8rem">Premium: ₹{z['Call Premium']:.2f}</div>
              <div style="font-size:.8rem">Return: {z['Static Ret (%)']:.2f}%/mo</div>
              <div style="font-size:.8rem">Protection: {z['Downside Prot (%)']:.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.subheader("Full Strike Table — Ellman System")
    def colour_cc_row(row):
        bg = {"ATM":"#dbeafe","OTM":"#f0fdf4","ITM":"#fef9c3"}.get(row["Type"],"")
        return [f"background-color:{bg}"]*len(row) if bg else [""]*len(row)
    st.dataframe(
        cc_df.style.apply(colour_cc_row,axis=1)
        .applymap(lambda v: "color:#16a34a;font-weight:700" if v is True else "color:#dc2626" if v is False else "",subset=["Meets Target"])
        .format({"Strike":"₹{:,.0f}","Call Premium":"₹{:.2f}","Static Ret (%)":"{:.2f}%",
                 "If-Called Ret (%)":"{:.2f}%","Annualised (%)":"{:.1f}%",
                 "Downside Prot (%)":"{:.2f}%","Breakeven (₹)":"₹{:,.0f}","Delta":"{:.3f}"}),
        use_container_width=True, height=380, hide_index=True,
    )

    st.markdown("---")
    st.subheader("Ellman Exit Rules")
    for rule, desc, style in [
        ("20/10 Rule", "If call value drops to 20% of premium with ≥2 weeks left, buy back and re-sell. At 10% with <2 weeks, let expire.", "safe"),
        ("Mid-contract unwind", "If stock drops >8% in one week, buy back call and re-evaluate stock position.", "danger"),
        ("Roll up in bull market", "If stock rallies past strike, roll up to higher strike same expiry for small debit.", "warning"),
        ("Exit before earnings", "ALWAYS close covered calls before earnings if you want to hold the stock.", "danger"),
        ("Monthly cycle reset", "Reset every 4 weeks. Ellman's core discipline — never let positions drift.", "explain"),
    ]:
        explain(f"<b>{rule}:</b> {desc}", style)


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE D — WEEKLY OPTIONS LAB  ███
    # ═══════════════════════════════════════════════════════════════
