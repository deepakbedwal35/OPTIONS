"""
Module: 🚪  Exit Strategy Engine
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
    st.title("🚪 Exit Strategy Engine — 7-Book Framework")

    explain(
        "Most traders know how to enter. Almost none have a written exit plan. "
        "This module synthesises exit rules from 7 books: "
        "<b>Ellman</b> (covered call exits), <b>McMillan</b> (rolling rules), "
        "<b>Cordier</b> (50% rule), <b>Pezim</b> (stop-loss framework), "
        "<b>Sincere</b> (when to close early), <b>Vine</b> (position unwinding), "
        "<b>Duarte</b> (simple exit checklist). Enter your current position details below.",
        "explain",
    )

    col_ei, col_eo = st.columns([1,1])
    with col_ei:
        st.subheader("Your Current Position")
        exit_type   = st.selectbox("Position type",[
            "Short Call (naked/covered)","Short Put (naked/cash-secured)",
            "Short Strangle","Iron Condor","Long Call/Put (debit)","Calendar Spread",
        ], key="exit_type")
        entry_prem  = st.number_input("Premium received/paid (₹)", value=200.0, step=10.0)
        current_val = st.number_input("Current option value (₹)",  value=120.0, step=5.0)
        dte_entry   = st.number_input("DTE at entry",   value=30, min_value=1)
        dte_now     = st.number_input("DTE remaining",  value=18, min_value=0)
        delta_now_e = st.slider("Current delta (abs)", 0.01, 0.99, 0.25)
        iv_change   = st.slider("IV change since entry (%)", -50, 100, 0)
        has_event   = st.checkbox("Earnings/major event within DTE?", value=False)

    is_seller   = "Short" in exit_type or "Iron" in exit_type
    profit_pct  = ((entry_prem - current_val)/entry_prem*100) if is_seller else ((current_val - entry_prem)/entry_prem*100)
    time_elapsed= (1 - dte_now/max(dte_entry,1)) * 100

    with col_eo:
        st.subheader("P&L Status")
        pnl_color = "#16a34a" if profit_pct >= 0 else "#dc2626"
        st.markdown(f"""
        <div class="score-card">
          <div style="display:flex;gap:24px;flex-wrap:wrap">
            <div><div style="font-size:.75rem;color:#64748b">P&L</div>
                 <div class="score-val" style="color:{pnl_color}">{profit_pct:+.1f}%</div></div>
            <div><div style="font-size:.75rem;color:#64748b">Time elapsed</div>
                 <div class="score-val">{time_elapsed:.0f}%</div></div>
            <div><div style="font-size:.75rem;color:#64748b">DTE left</div>
                 <div class="score-val">{dte_now}d</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Theta decay curve
        dte_r = np.linspace(dte_entry, 0, 60)
        rem   = entry_prem * np.sqrt(np.maximum(dte_r / max(dte_entry,1), 0))
        fig_te = go.Figure()
        fig_te.add_trace(go.Scatter(x=dte_r, y=rem, line=dict(color="#4f46e5",width=2), name="Value"))
        fig_te.add_hline(y=entry_prem*0.5, line=dict(color="#22c55e",dash="dash"), annotation_text="50% target")
        fig_te.add_hline(y=entry_prem*0.2, line=dict(color="#f97316",dash="dash"), annotation_text="Ellman 80%")
        fig_te.add_vline(x=dte_now, line=dict(color="#ef4444",dash="dot"), annotation_text="Now")
        fig_te.update_layout(xaxis=dict(autorange="reversed"),xaxis_title="DTE",
                             yaxis_title="Value (₹)",template="plotly_white",height=240)
        st.plotly_chart(fig_te, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Exit Recommendations by Book")

    # Ellman
    if profit_pct >= 80 and is_seller:
        e_a,e_s = "CLOSE — 80%+ profit. Ellman: redeploy capital now.","safe"
    elif profit_pct >= 50 and is_seller:
        e_a,e_s = "CLOSE — 50% profit hit. Ellman: acceptable exit here.","safe"
    elif profit_pct <= -100 and is_seller:
        e_a,e_s = "CLOSE IMMEDIATELY — loss = 100% of premium. Ellman hard stop.","danger"
    elif delta_now_e > 0.30 and is_seller:
        e_a,e_s = f"ROLL OR CLOSE — Delta expanded to {delta_now_e:.2f} (>0.30). Ellman danger zone.","warning"
    else:
        e_a,e_s = f"HOLD — {profit_pct:.1f}% profit, delta {delta_now_e:.2f}. Healthy.","explain"
    explain(f"<b>Ellman (Exit Strategies for Covered Calls):</b> {e_a}", e_s)

    # McMillan
    if profit_pct >= 50 and is_seller:
        m_a,m_s = "CLOSE — 50% profit. McMillan proved this beats holding to expiry.","safe"
    elif dte_now <= 7 and is_seller:
        m_a,m_s = f"CLOSE — {dte_now} DTE left. Gamma risk too high. Premium not worth pin risk.","warning"
    elif has_event:
        m_a,m_s = "CLOSE BEFORE EVENT — McMillan: never hold short options through earnings.","danger"
    elif profit_pct <= -150:
        m_a,m_s = "CLOSE — McMillan 2× stop-loss hit.","danger"
    else:
        m_a,m_s = f"HOLD — {profit_pct:.1f}% profit. Within McMillan parameters.","explain"
    explain(f"<b>McMillan (Options as Strategic Investment):</b> {m_a}", "mcmillan")

    # Pezim
    if profit_pct <= -50 and is_seller:
        p_a,p_s = f"STOP LOSS — Pezim pre-set stop at 50% of premium. Close now.","danger"
    elif profit_pct >= 50 and is_seller:
        p_a,p_s = "TAKE PROFIT — Pezim: lock in 50% gain. Discipline over greed.","safe"
    elif iv_change > 20 and is_seller:
        p_a,p_s = f"CLOSE — IV rose {iv_change}%. Pezim: IV expansion >20% = defensive action.","danger"
    else:
        p_a,p_s = "HOLD — Within Pezim parameters.","explain"
    explain(f"<b>Pezim (How to Trade Options):</b> {p_a}", p_s)

    # Sincere
    if profit_pct >= 70 and is_seller:
        si_a,si_s = "CLOSE — Sincere: take 70% of max profit and walk away.","safe"
    elif time_elapsed >= 75 and profit_pct < 30 and is_seller:
        si_a,si_s = "REASSESS — 75% time elapsed, <30% profit. Underperforming.","warning"
    else:
        si_a,si_s = f"HOLD — {time_elapsed:.0f}% elapsed, {profit_pct:.1f}% captured.","explain"
    explain(f"<b>Sincere (Understanding Options):</b> {si_a}", si_s)

    # Vine
    if delta_now_e > 0.35 and is_seller:
        vi_a,vi_s = f"CLOSE — Delta {delta_now_e:.2f} breached Vine threshold (0.35).","danger"
    elif iv_change > 25 and is_seller:
        vi_a,vi_s = f"CLOSE — IV spike {iv_change}% triggers Vine vega blowup warning.","danger"
    else:
        vi_a,vi_s = "HOLD — All Vine risk metrics within limits.","explain"
    explain(f"<b>Vine (Options: Trading Strategy & Risk Management):</b> {vi_a}", vi_s)

    # Duarte checklist
    duarte_checks = [
        ("Profit > 50%",       profit_pct >= 50),
        ("Delta < 0.30",       delta_now_e < 0.30 if is_seller else True),
        ("DTE > 7",            dte_now > 7),
        ("No event in DTE",    not has_event),
        ("IV stable (<+20%)",  iv_change <= 20),
    ]
    passes = sum(1 for _,ok in duarte_checks if ok)
    explain(
        f"<b>Duarte (Trading Options for Dummies):</b> {passes}/5 checks pass. "
        f"{'✅ All clear — hold.' if passes==5 else '⚠️ '+str(5-passes)+' flags — review.' if passes>=3 else '🚨 '+str(5-passes)+' flags — close or defend.'}",
        "safe" if passes==5 else "warning" if passes>=3 else "danger",
    )
    for label, ok in duarte_checks:
        st.markdown(f'<div style="padding:3px 10px;font-size:.82rem">{"✅" if ok else "❌"} {label}</div>', unsafe_allow_html=True)

    # Consensus verdict
    styles = [e_s, m_s, p_s, si_s, vi_s]
    close_v = sum(1 for s in styles if s in ("danger","warning"))
    hold_v  = sum(1 for s in styles if s == "explain") + (1 if passes>=4 else 0)
    verdict = "CLOSE POSITION" if close_v>=3 else "DEFEND / ROLL" if close_v>=2 else "HOLD"
    vc = "#dc2626" if close_v>=3 else "#d97706" if close_v>=2 else "#16a34a"
    st.markdown(
        f'<div style="background:{vc}15;border:2px solid {vc};border-radius:10px;'
        f'padding:14px 20px;margin:10px 0;text-align:center">'
        f'<b style="color:{vc};font-size:1.1rem">Book Consensus: {verdict}</b>'
        f'<div style="font-size:.82rem;color:#475569;margin-top:4px">'
        f'{close_v} books say close/act · {hold_v} books say hold</div></div>',
        unsafe_allow_html=True,
    )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE C — COVERED CALL OPTIMISER  ███
    # ═══════════════════════════════════════════════════════════════
