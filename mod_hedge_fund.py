"""
Module: 🛡️  Hedge Fund Dashboard
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
    st.title("🛡️ Portfolio Risk Dashboard — Sebastian/Chen + Vine")

    explain(
        "Sebastian & Chen's <b>Option Trader's Hedge Fund</b>: run your option book like a fund. "
        "Rules: (1) portfolio delta near zero, (2) net theta always positive, "
        "(3) net vega negative or flat, (4) gamma monitored daily. "
        "Vine adds: maximum per-position loss = 2% of portfolio.",
        "explain",
    )

    if "hf_positions" not in st.session_state:
        st.session_state.hf_positions = []

    col_hf_add, col_hf_port = st.columns([1,2])
    with col_hf_add:
        st.subheader("Add Position")
        hf_sym   = st.selectbox("Stock", list(NSE_FNO.keys()), key="hf_sym_add")
        hf_type  = st.selectbox("Option", ["Short Call","Short Put","Long Call","Long Put"], key="hf_type")
        hf_K     = st.number_input("Strike (₹)", value=18000, step=50, key="hf_k")
        hf_iv_p  = st.number_input("IV (%)", value=18.0, step=0.5, key="hf_iv") / 100
        hf_dte   = st.number_input("DTE", value=30, min_value=1, key="hf_dte")
        hf_lots  = st.number_input("Lots", value=1, min_value=1, key="hf_lots")
        hf_ls    = st.number_input("Lot size", value=50, min_value=1, key="hf_ls")

        if st.button("➕ Add to Portfolio", type="primary"):
            stats_hfa = fetch_spot_iv(NSE_FNO[hf_sym])
            if stats_hfa:
                st.session_state.hf_positions.append({
                    "sym":hf_sym,"type":hf_type,"K":hf_K,"iv":hf_iv_p,
                    "dte":hf_dte,"lots":hf_lots,"lot_size":hf_ls,"S":stats_hfa["spot"],
                })

        if st.button("NIFTY Iron Condor template"):
            st.session_state.hf_positions = [
                {"sym":"NIFTY","type":"Short Put","K":22000,"iv":0.19,"dte":30,"lots":2,"lot_size":50,"S":23000},
                {"sym":"NIFTY","type":"Long Put","K":21500,"iv":0.21,"dte":30,"lots":2,"lot_size":50,"S":23000},
                {"sym":"NIFTY","type":"Short Call","K":24000,"iv":0.17,"dte":30,"lots":2,"lot_size":50,"S":23000},
                {"sym":"NIFTY","type":"Long Call","K":24500,"iv":0.17,"dte":30,"lots":2,"lot_size":50,"S":23000},
            ]
        if st.button("🗑️ Clear"):
            st.session_state.hf_positions = []

        for i, pos in enumerate(st.session_state.hf_positions):
            cl, cx = st.columns([5,1])
            cl.caption(f"{pos['lots']}× {pos['type']} {pos['sym']} K={pos['K']}")
            if cx.button("✕", key=f"hf_rm_{i}"):
                st.session_state.hf_positions.pop(i)
                st.rerun()

    with col_hf_port:
        if not st.session_state.hf_positions:
            st.info("Add positions on the left or load a template.")
        else:
            r_hf = 0.065
            port_delta=port_gamma=port_theta=port_vega=0.0
            port_rows = []
            for pos in st.session_state.hf_positions:
                T_hf  = max(pos["dte"]/365,0.001)
                ot    = "call" if "Call" in pos["type"] else "put"
                sign  = -1 if "Short" in pos["type"] else 1
                mult  = pos["lots"]*pos["lot_size"]
                g     = bs_greeks(pos["S"],pos["K"],T_hf,r_hf,pos["iv"],ot)
                p     = bs_price(pos["S"],pos["K"],T_hf,r_hf,pos["iv"],ot)
                port_delta += sign*g["delta"]*mult
                port_gamma += sign*g["gamma"]*mult
                port_theta += sign*g["theta"]*mult
                port_vega  += sign*g["vega"]*mult
                port_rows.append({"Symbol":pos["sym"],"Position":pos["type"],"Strike":pos["K"],"Lots":pos["lots"],
                                   "Delta":round(sign*g["delta"]*mult,2),"Gamma":round(sign*g["gamma"]*mult,4),
                                   "Theta/d":round(sign*g["theta"]*mult,2),"Vega/%":round(sign*g["vega"]*mult,2),"Price":round(p,2)})

            dk = abs(port_delta)<50
            tk = port_theta>0
            vk = port_vega<=0
            gk = port_gamma>-0.01
            health = sum([dk,tk,vk,gk])*25
            hc = "#16a34a" if health>=75 else "#d97706" if health>=50 else "#dc2626"

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Net Delta",   f"{port_delta:+.2f}",   delta="✅ Neutral" if dk else "⚠️ Skewed",  delta_color="normal")
            k2.metric("Net Theta/d", f"₹{port_theta:+.2f}", delta="✅ Positive" if tk else "❌ Negative",delta_color="normal")
            k3.metric("Net Vega",    f"{port_vega:+.2f}",   delta="✅ Short" if vk else "⚠️ Long",      delta_color="normal")
            k4.metric("Net Gamma",   f"{port_gamma:+.4f}",  delta="✅ OK" if gk else "⚠️ High",         delta_color="normal")

            st.markdown(
                f'<div style="background:{hc}15;border:2px solid {hc};border-radius:8px;'
                f'padding:10px;margin:8px 0;text-align:center">'
                f'<b style="color:{hc}">Sebastian/Chen Health Score: {health}/100</b></div>',
                unsafe_allow_html=True,
            )

            port_df = pd.DataFrame(port_rows)
            st.dataframe(port_df.style.format({"Strike":"₹{:,.0f}","Delta":"{:+.2f}","Gamma":"{:+.4f}",
                                                "Theta/d":"₹{:+.2f}","Vega/%":"{:+.2f}","Price":"₹{:.2f}"}),
                         use_container_width=True, hide_index=True)

            if not dk:
                explain(f"Delta imbalance {port_delta:+.2f}. Sebastian: hedge when |delta| > 50. "
                        f"{'Buy a call' if port_delta<-50 else 'Buy a put or short call'} to rebalance.", "warning")
            if not tk:
                explain("Negative theta — you're paying decay. Add short premium (sell strangles/credit spreads).", "danger")
            if not vk:
                explain("Positive vega — hurt by IV falling. Sell more premium to go vega-neutral.", "warning")
            if health == 100:
                explain("✅ All Sebastian/Chen criteria met: delta neutral, positive theta, short vega, gamma manageable.", "safe")

            fig_hf = go.Figure(go.Bar(
                x=[r["Symbol"]+" "+r["Position"][:5] for r in port_rows],
                y=[r["Theta/d"] for r in port_rows],
                marker_color=["#22c55e" if v>0 else "#ef4444" for v in [r["Theta/d"] for r in port_rows]],
            ))
            fig_hf.update_layout(title="Theta by Position",yaxis_title="Daily θ (₹)",
                                  template="plotly_white",height=250,xaxis_tickangle=-30)
            st.plotly_chart(fig_hf, use_container_width=True)


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE F — BOOK STRATEGY MATRIX  ███
    # ═══════════════════════════════════════════════════════════════
