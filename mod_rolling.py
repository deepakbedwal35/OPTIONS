"""
Module: 🔄  Rolling Engine
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
    st.title("🔄 Rolling Engine — McMillan + Cordier Rules")

    explain(
        "Rolling is the #1 defence for threatened option positions. "
        "McMillan: 'Roll before you're in trouble, not after.' "
        "Cordier: 'A trade that can be managed is better than a trade that can't.' "
        "This engine tells you exactly WHEN to roll and WHERE to roll based on your current position.",
        "mcmillan",
    )

    pos_type = st.selectbox("Current position", [
        "Short Call (threatened)",
        "Short Put (threatened)",
        "Short Strangle (one side breached)",
        "Iron Condor (short strike breached)",
        "Covered Call (ITM)",
    ])
    dte_left  = st.slider("DTE remaining", 1, 45, 15)
    pnl_pct   = st.slider("Current P&L vs entry premium (%)", -200, 100, -30,
                          help="-100% = lost all premium. +50% = made 50% of premium.")
    delta_now = st.slider("Current short option delta (abs)", 0.05, 0.80, 0.35,
                          help="Start = 0.15. Crept up = position is threatened.")

    st.markdown("---")

    # Decision logic
    explain(
        f"<b>Current delta: {delta_now:.2f}</b> — "
        f"{'✅ still in control zone (< 0.25)' if delta_now < 0.25 else '⚠️ delta expanding — monitor closely (0.25–0.40)' if delta_now < 0.40 else '🚨 HIGH DELTA — rolling action required now (> 0.40)'}",
        "safe" if delta_now < 0.25 else "warning" if delta_now < 0.40 else "danger",
    )

    if pnl_pct >= 50:
        explain(
            "✅ <b>50% profit rule triggered (McMillan + Cordier).</b> "
            "CLOSE THE POSITION NOW. Both McMillan and Cordier independently discovered that "
            "closing at 50% profit locks in gains while eliminating the risk of a reversal. "
            "The remaining 50% is not worth the risk of holding to expiry.",
            "safe",
        )
    elif pnl_pct <= -100:
        explain(
            "🚨 <b>DOUBLE-PREMIUM STOP LOSS HIT.</b> Cordier's hard rule: if you've lost 2× the premium received, close immediately. "
            "No rolling from here — rolling a deep-loss position adds more risk. Take the loss, reassess.",
            "danger",
        )
    elif delta_now > 0.40 or pnl_pct < -50:
        explain(
            "⚠️ <b>Rolling action recommended.</b> "
            "McMillan roll sequence: (1) buy back the threatened option, (2) sell same option at next expiry + move strike further OTM. "
            "Aim for net credit on the roll if possible. If you can only roll at a debit, make sure the new strike is at least 1.5σ away.",
            "warning",
        )
    else:
        explain(
            "🔵 <b>Monitor mode.</b> No action needed yet. Set a delta alert at 0.30 for this position. "
            "McMillan: don't roll for the sake of rolling — only roll when the position is genuinely threatened.",
            "explain",
        )

    # Roll option table
    st.subheader("Roll Options Analysis")
    roll_options = pd.DataFrame({
        "Roll Type":   ["Roll Out (same strike)", "Roll Out + Up/Down", "Roll Out + Widen", "Close (no roll)"],
        "Action":      [
            "Buy current, sell same strike next expiry",
            "Buy current, sell further OTM next expiry",
            "Buy current, sell wider spread next expiry",
            "Buy current, take the loss",
        ],
        "When to use": [
            "DTE > 15 + mild threat (delta 0.25–0.35)",
            "Delta > 0.35 + position directionally threatened",
            "Condor/spread: one side breached, widen the spread",
            "Loss > 2× premium OR earnings event approaching",
        ],
        "Credit/Debit": [
            "Usually net credit (time value difference)",
            "Usually small debit (moving strike costs)",
            "Usually net debit (wider protection costs)",
            "Net debit = full loss of remaining value",
        ],
        "Risk change":  [
            "Same risk, more time",
            "Less risk, more time",
            "Less risk, more time, defined max loss",
            "Risk eliminated",
        ],
    })
    st.dataframe(roll_options, use_container_width=True, hide_index=True)

    explain(
        "Cordier's rule on rolling frequency: 'Roll once, maybe twice. Never three times.' "
        "Each roll is an admission that the original trade was wrong. "
        "Rolling repeatedly compounds losses. Better to take a small loss early than roll into a larger disaster. "
        "McMillan adds: always check that the new position still passes your original entry criteria.",
        "cordier",
    )

    # Rolling P&L visualisation
    st.subheader("P&L Trajectory — Current vs Rolled Position")
    entry_prem = 200
    current_pnl = entry_prem * (pnl_pct / 100)
    dte_range   = np.linspace(dte_left, 0, 50)
    # Simulate: current position theta decay if not rolled
    current_vals = current_pnl + np.linspace(0, -current_pnl * 0.2, 50)
    # Rolled position: reset to -small debit, then new theta decay
    roll_start   = current_pnl - 20   # roll debit
    roll_vals    = roll_start + np.linspace(0, abs(roll_start) * 1.2, 50)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=dte_range, y=current_vals, name="Current (no roll)",
                                   line=dict(color="#ef4444", width=2, dash="dash")))
    fig_roll.add_trace(go.Scatter(x=dte_range, y=roll_vals, name="After roll",
                                   line=dict(color="#22c55e", width=2)))
    fig_roll.add_hline(y=0, line=dict(color="#94a3b8", dash="dot"))
    fig_roll.update_layout(xaxis_title="DTE remaining", yaxis_title="P&L (₹)",
                            xaxis=dict(autorange="reversed"),
                            template="plotly_white", height=280)
    st.plotly_chart(fig_roll, use_container_width=True)



    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 11 — INDEX BIG-MOVE CALENDAR  ███
    # ═══════════════════════════════════════════════════════════════
