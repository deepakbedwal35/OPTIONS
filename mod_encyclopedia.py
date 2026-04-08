"""
Module: 📚  Strategy Encyclopedia
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
    st.title("📚 Strategy Encyclopedia — 50+ Strategies")

    explain(
        "Quick-reference guide to all major strategies synthesised across McMillan, Natenberg, "
        "Passarelli, and Cordier. Each entry links to the book chapter and explains the trade "
        "in plain English. Use the filters to find the right strategy for your current market view.",
        "explain",
    )

    STRATS = [
        ("Long Call",           "Basic",    "Bullish",   "Limited",    "McMillan Ch.1",
         "Buy 1 call. Pay premium for right to buy at strike. Profit = max(0,S−K)−premium. Best when IV is LOW and expecting big up move."),
        ("Long Put",            "Basic",    "Bearish",   "Limited",    "McMillan Ch.1",
         "Buy 1 put. Profit = max(0,K−S)−premium. Insurance or directional bearish. Buy when IV is LOW."),
        ("Covered Call",        "Basic",    "Neutral",   "Stock risk", "McMillan Ch.2 + Cordier",
         "Own stock + sell OTM call. Cordier: sell 0.20 delta. Income strategy. Caps upside. Reduces cost basis."),
        ("Cash-Secured Put",    "Basic",    "Bullish",   "Large",      "Cordier Ch.5",
         "Sell put + hold cash. Want to buy stock at lower price. Cordier's primary income strategy. Sell 0.15–0.20 delta."),
        ("Protective Put",      "Basic",    "Bullish",   "Limited",    "McMillan Ch.2",
         "Long stock + buy put. Insurance. Max loss defined. Buy when IV is LOW (cheap insurance)."),
        ("Collar",              "Basic",    "Neutral",   "Limited",    "McMillan Ch.2",
         "Long stock + sell call + buy put. Zero-cost possible. Caps both upside and downside."),
        ("Bull Call Spread",    "Spread",   "Bullish",   "Limited",    "McMillan Ch.3",
         "Buy lower call + sell higher call. Reduce cost of directional trade. Net debit."),
        ("Bear Put Spread",     "Spread",   "Bearish",   "Limited",    "McMillan Ch.3",
         "Buy higher put + sell lower put. Cheap bearish trade. Net debit."),
        ("Bull Put Spread",     "Spread",   "Bullish",   "Limited",    "McMillan Ch.3 + Cordier",
         "Sell higher put + buy lower put. Net credit. Most common credit spread. Defined risk."),
        ("Bear Call Spread",    "Spread",   "Bearish",   "Limited",    "McMillan Ch.3",
         "Sell lower call + buy higher call. Net credit. Bearish to neutral."),
        ("Long Straddle",       "Neutral",  "Volatile",  "Limited",    "McMillan Ch.11 + Natenberg",
         "Buy ATM call + put. Profit from big move either way. Enemy = theta. Buy when IV is VERY LOW."),
        ("Short Straddle",      "Selling",  "Neutral",   "Unlimited",  "Cordier (use strangle instead)",
         "Sell ATM call + put. Max theta decay. Very dangerous — unlimited risk. Cordier prefers strangle."),
        ("Long Strangle",       "Neutral",  "Volatile",  "Limited",    "McMillan Ch.11",
         "Buy OTM call + put. Cheaper straddle. Needs even bigger move. Better for event plays."),
        ("Short Strangle",      "Selling",  "Neutral",   "Unlimited",  "Cordier Ch.6 PRIMARY",
         "Sell OTM call + put. Cordier's main strategy. 0.15 delta. Buy back at 50% profit. IVR > 50%."),
        ("Iron Condor",         "Neutral",  "Neutral",   "Limited",    "McMillan Ch.14",
         "Bull put spread + bear call spread. Defined risk strangle. Collect ≥ 1/3 of width. Most popular NSE strategy."),
        ("Iron Butterfly",      "Neutral",  "Neutral",   "Limited",    "McMillan Ch.14",
         "Sell ATM straddle + buy wings. Higher premium than condor, tighter profit zone."),
        ("Calendar Spread",     "Advanced", "Neutral",   "Limited",    "McMillan Ch.15 + Natenberg Ch.11",
         "Sell near-month + buy far-month (same strike). Profit from differential theta decay. Also long vega."),
        ("Diagonal Spread",     "Advanced", "Mildly dir","Limited",    "McMillan Ch.15",
         "Calendar + vertical. Different strikes AND expiries. Poor man's covered call."),
        ("Ratio Spread",        "Advanced", "Mildly dir","Unlimited",  "McMillan Ch.16 + Natenberg",
         "Buy 1 + sell 2 at higher strike. Can be credit. Extra short leg is uncovered."),
        ("Backspread",          "Advanced", "Volatile",  "Limited dn", "Natenberg Ch.12",
         "Sell 1 + buy 2 higher. Long gamma + vega. Natenberg favourite when IV is cheap."),
        ("Jade Lizard",         "Advanced", "Bullish",   "Put risk",   "McMillan Advanced",
         "Short put + short call spread. No upside risk if credit ≥ spread width. Creative premium structure."),
        ("Broken Wing Butterfly","Advanced","Neutral",   "Limited",    "McMillan Advanced",
         "Skip-strike butterfly. Often done for credit. Popular in weekly NIFTY options."),
        ("Synthetic Long Stock","Synthetic","Bullish",   "Large",      "McMillan Ch.20",
         "Buy ATM call + sell ATM put. Replicates stock with less capital. Put-call parity."),
        ("Risk Reversal",       "Synthetic","Bullish",   "Put risk",   "Natenberg Ch.14",
         "Sell OTM put + buy OTM call. Exploits put skew. Often net credit on high-skew stocks."),
        ("Conversion",          "Synthetic","Neutral",   "Locked",     "Natenberg Workbook",
         "Long stock + long put + short call. Risk-free arbitrage when options mispriced."),
    ]

    col_f1, col_f2 = st.columns(2)
    cat_f   = col_f1.selectbox("Category", ["All","Basic","Spread","Neutral","Selling","Advanced","Synthetic"])
    out_f   = col_f2.selectbox("Outlook",  ["All","Bullish","Bearish","Neutral","Volatile"])

    filtered = STRATS
    if cat_f != "All":
        filtered = [s for s in filtered if s[1] == cat_f]
    if out_f != "All":
        filtered = [s for s in filtered if out_f in s[2]]

    st.caption(f"Showing {len(filtered)} strategies")
    for name, cat, outlook, risk, book, desc in filtered:
        outlook_color = {"Bullish":"#dcfce7","Bearish":"#fee2e2","Neutral":"#dbeafe",
                         "Volatile":"#fef9c3"}.get(outlook.split("/")[0], "#f1f5f9")
        with st.expander(f"**{name}** — {outlook} | {cat} | {book}"):
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"**Category:** {cat}")
            c2.markdown(f'**Outlook:** <span style="background:{outlook_color};padding:2px 8px;border-radius:10px;font-size:.82rem">{outlook}</span>', unsafe_allow_html=True)
            c3.markdown(f"**Max Risk:** {risk}")
            st.markdown(f"**Source:** `{book}`")
            explain(desc, "explain")
