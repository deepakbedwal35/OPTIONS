"""
Module: 📖  Book Strategy Matrix
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
    st.title("📖 15-Book Strategy Matrix — Cross-Reference Guide")

    explain(
        "Every book recommends different strategies for different market conditions. "
        "This matrix shows WHICH strategy WHICH book recommends for WHICH condition — "
        "with exact entry criteria, strike rules, and exit rules from each author. "
        "Filter by condition to find which strategies ALL books agree on.",
        "explain",
    )

    BOOK_MATRIX = [
        ("McMillan",       "Bull Call Spread",     "Bullish",         "Buy ATM, sell 5–10% OTM",          "IV low, trend confirmed",            "Close at 80% max profit",           "Low"),
        ("McMillan",       "Iron Condor",           "Neutral",         "Sell 1σ OTM both sides",            "IVR > 40%, no events",               "Close at 50% profit",               "Medium"),
        ("McMillan",       "Long Straddle",         "Pre-event",       "Buy ATM call+put",                  "IV < 30th percentile",               "Close 1 day before event",          "Medium"),
        ("McMillan",       "Calendar Spread",       "Neutral low IV",  "Sell near, buy far (same K)",       "Near IV < Far IV",                   "Close when near expires",           "Low"),
        ("McMillan",       "Covered Call",          "Neutral/Bullish", "Sell 2–5% OTM call",                "Own stock, no events",               "20/10 rule",                        "Low"),
        ("Natenberg",      "Backspread",            "Big move",        "Buy 2×, sell 1× lower",             "IV < HV, low IVR",                   "Close when IV spikes 30%+",         "Medium"),
        ("Natenberg",      "Risk Reversal",         "Bullish + skew",  "Sell OTM put, buy OTM call",        "Put skew > 3% vs call",              "Close when skew normalises",        "Medium"),
        ("Natenberg",      "Ratio Spread",          "Mildly bullish",  "Buy 1, sell 2 higher",              "IV overpriced vs HV",                "Close at 50% credit",               "High"),
        ("Passarelli",     "Delta-neutral Strangle","Neutral",         "Sell 0.15–0.20 delta both sides",   "IVR > 50%, delta < 0.20",            "Close when delta > 0.30",           "High"),
        ("Passarelli",     "Gamma Scalping",        "Volatile",        "Buy ATM straddle",                  "IV cheap, big move expected",        "Delta hedge daily",                 "High"),
        ("Guy Cohen",      "Income Spread",         "Neutral/Bullish", "Bull put spread, 5% OTM short",     "Stock above 50DMA, IV > 20%",        "Exit at 70% profit or −100%",       "Medium"),
        ("Guy Cohen",      "Diagonal",              "Mildly Bullish",  "Buy 90d+ ITM, sell 30d ATM",        "Trend confirmed, moderate IV",       "Re-sell near-month monthly",        "Low"),
        ("Hull",           "Protective Put",        "Hedge long stock","Buy 5% OTM put",                    "Own stock, uncertain outlook",       "Roll monthly",                      "Low"),
        ("Hull",           "Synthetic Forward",     "Strong direction","Buy call, sell put (same K)",        "Strong trend, low cost of carry",    "Close at target or stop",           "High"),
        ("Ellman",         "OTM Covered Call",      "Mildly bullish",  "Sell 2–5% OTM call",                "Earnings ≥ 6 weeks away",            "20/10 rule + monthly reset",        "Low"),
        ("Ellman",         "ITM Covered Call",      "Uncertain",       "Sell 2–5% ITM call",                "Want max downside protection",       "Close if stock drops 7%+",          "Low"),
        ("Pezim",          "Bull Put Spread",       "Bullish",         "Sell ATM put, buy 5% OTM put",      "Stock > 20DMA, sector strong",       "Stop at 2× premium",                "Low"),
        ("Pezim",          "Momentum Call",         "Strong uptrend",  "Buy ATM call 30–45 DTE",            "RSI 50–70, above 50+200 DMA",        "2× premium OR −50% stop",           "Medium"),
        ("Vine",           "Portfolio Strangle",    "Neutral",         "Sell 2σ OTM, diversified",          "Portfolio delta neutral, IVR > 50%", "Close when theta < 0",              "High"),
        ("Vine",           "Collar (portfolio)",    "Defensive",       "Sell OTM call, buy OTM put",        "Bull market, near highs, nervous",   "Hold through cycle",                "Low"),
        ("Duarte",         "Simple Iron Condor",    "Sideways",        "Sell 5% OTM both sides",            "Index rangebound 20 days, IVR > 30%","Close at 50% or 2× loss",           "Medium"),
        ("Duarte",         "Long Call momentum",    "Bullish breakout","Buy ATM or slight OTM",             "Breaks 52-week high",                "+100% take / −50% stop",            "Medium"),
        ("Abraham",        "Weekly Income Condor",  "Neutral",         "Sell weekly 3% OTM both sides",     "IVR > 40%, no events this week",     "Close Thursday regardless",         "Medium"),
        ("Abraham",        "Monthly Cash Put",      "Bullish quality", "Sell 10% OTM put monthly",          "Quality stock, want to own",         "50% profit OR take assignment",     "Medium"),
        ("Kaushik",        "NSE BankNifty IC",      "Rangebound index","Sell ±3% OTM weekly IC",            "BankNifty between key levels",       "Close Friday morning",              "Medium"),
        ("Kaushik",        "NSE Earnings Strangle", "Pre-earnings",    "Sell 15% OTM both sides",           "IV spike before results",            "Close day after (IV crush)",        "High"),
        ("Sincere",        "Covered Call",          "Neutral income",  "Sell ATM call monthly",             "Own shares, no events",              "If stock drops 10% re-evaluate",    "Low"),
        ("Sincere",        "Cash-Secured Put",      "Want stock cheaper","Sell ATM−5% put",                 "Willing to own at strike",           "Let expire or take stock",          "Low"),
        ("Carter",         "Weekly Theta Sprint",   "Neutral weekly",  "Sell 2σ OTM strangle 7 DTE",        "IVR > 45%, no event this week",      "Close Thursday at 50%+",            "High"),
        ("Carter",         "Weekly IC",             "Neutral weekly",  "Sell 1.5σ OTM IC 7 DTE",           "Index in channel, low catalyst",     "Close Thursday",                    "Medium"),
        ("Sebastian/Chen", "Portfolio Theta Engine","Neutral portfolio","Delta-neutral mixed book",          "Portfolio delta < ±50, theta > 0",   "Rebalance when delta ±50 breached", "High"),
        ("Sebastian/Chen", "Vega-Neutral Spread",   "Neutral vol",     "Match vega of long+short",          "VIX at midpoint of range",           "Exit when vega imbalance > 20%",    "Medium"),
    ]

    col_mf1, col_mf2, col_mf3 = st.columns(3)
    bk_f = col_mf1.selectbox("Book", ["All"]+sorted(set(r[0] for r in BOOK_MATRIX)))
    cd_f = col_mf2.selectbox("Condition", ["All"]+sorted(set(r[2] for r in BOOK_MATRIX)))
    rk_f = col_mf3.selectbox("Risk", ["All","Low","Medium","High"])

    fm = BOOK_MATRIX
    if bk_f != "All": fm = [r for r in fm if r[0]==bk_f]
    if cd_f != "All": fm = [r for r in fm if r[2]==cd_f]
    if rk_f != "All": fm = [r for r in fm if r[6]==rk_f]

    st.caption(f"{len(fm)} entries from {len(set(r[0] for r in fm))} books")
    mdf = pd.DataFrame(fm, columns=["Book","Strategy","Condition","Strike Rule","Entry Criteria","Exit Rule","Risk"])

    def sm_row(row):
        rc = {"Low":"#f0fdf4","Medium":"#fef9c3","High":"#fee2e2"}.get(row["Risk"],"")
        tc = {"Low":"#166534","Medium":"#854d0e","High":"#991b1b"}.get(row["Risk"],"")
        return ["" if c!="Risk" else f"background-color:{rc};color:{tc};font-weight:600" for c in row.index]

    st.dataframe(mdf.style.apply(sm_row,axis=1), use_container_width=True, height=540, hide_index=True)
    explain(
        "Filter by <b>Condition</b> to see what every book recommends for your current market. "
        "When 4+ books agree on the same strategy for the same condition, that's a high-conviction setup. "
        "The Strike Rule column gives the exact entry criterion from each author.",
        "explain",
    )

    st.markdown("---")
    st.subheader("🤝 Consensus Finder — Which Strategy Do Most Books Agree On?")
    cond_c = st.selectbox("Your market condition", sorted(set(r[2] for r in BOOK_MATRIX)), key="cc_sel2")
    matching = [r for r in BOOK_MATRIX if r[2]==cond_c]
    sc = {}
    for r in matching: sc[r[1]] = sc.get(r[1],0)+1
    cdf = pd.DataFrame([{"Strategy":k,"Books agree":v,"Confidence":"⭐"*v} for k,v in sorted(sc.items(),key=lambda x:-x[1])])
    if not cdf.empty:
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        top_c = cdf.iloc[0]
        explain(
            f"For <b>{cond_c}</b> markets: <b>{top_c['Strategy']}</b> has highest consensus "
            f"({top_c['Books agree']} books). This is the safest choice when multiple authors agree.",
            "safe",
        )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 12 — STRATEGY ENCYCLOPEDIA  ███
    # ═══════════════════════════════════════════════════════════════
