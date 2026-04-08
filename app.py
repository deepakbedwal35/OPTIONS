"""
╔══════════════════════════════════════════════════════════════════╗
║         OPTIONS ALPHA PLATFORM v5.0 — NSE Edition               ║
║  15 Books · Real NSE data · 15-yr win-rates · Modular arch       ║
╚══════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
Deps: pip install streamlit yfinance pandas numpy scipy plotly
"""

import streamlit as st

# ── page config must be FIRST ─────────────────────────────────────
st.set_page_config(
    page_title="Options Alpha v6.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── inject global CSS ─────────────────────────────────────────────
from utils import GLOBAL_CSS, explain, NSE_FNO, SECTORS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── sidebar navigation ────────────────────────────────────────────
MODULES = [
    ("🔭", "NSE Scanner + Win-Rate"),
    ("📊", "Option Chain Viewer"),
    ("📈", "Payoff Builder"),
    ("🔢", "Greeks Dashboard"),
    ("⚙️", "BS Pricer + Edge"),
    ("🌊", "Volatility Lab"),
    ("💰", "Option Selling Engine"),
    ("🎯", "Smart Strike Finder"),
    ("🚪", "Exit Strategy Engine"),
    ("📋", "Covered Call Optimiser"),
    ("⚡", "Weekly Options Lab"),
    ("🛡️", "Hedge Fund Dashboard"),
    ("📖", "Book Strategy Matrix"),
    ("📅", "Big-Move Calendar"),
    ("🔬", "Strategy Backtester"),
    ("🎲", "Probability Calculator"),
    ("🔄", "Rolling Engine"),
    ("📚", "Strategy Encyclopedia"),
    ("⚡", "Live Feed + Auto Execution"),
    ("🟢", "Buy Signal Engine"),
    ("📓", "Trade Journal"),
    ("📱", "Telegram Alerts"),
]

with st.sidebar:
    st.markdown(
        '<div style="color:#f1f5f9;font-size:1.05rem;font-weight:700;padding:4px 0">📊 Options Alpha v6.0</div>',
        unsafe_allow_html=True,
    )
    st.caption("15 Books · NSE F&O · 15-yr history")
    st.markdown("---")
    labels = [f"{e}  {n}" for e, n in MODULES]
    sel = st.radio("Select Option", labels, label_visibility="collapsed")
    module = sel.split("  ", 1)[1].strip()
    st.markdown("---")
    st.markdown('<div style="color:#94a3b8;font-size:.72rem;line-height:1.6">'
                '📗 McMillan &nbsp;📘 Natenberg &nbsp;📙 Passarelli<br>'
                '🟠 Cohen(×2) &nbsp;🔵 Hull &nbsp;🟣 Ellman<br>'
                '📕 Pezim · Vine · Duarte · Abraham<br>'
                'Kaushik · Sincere · Carter · Sebastian/Chen'
                '</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:.7rem;margin-top:8px">'
                '⚡ Price: 5 min · Win-rates: 1 hr · History: 15 yrs</div>',
                unsafe_allow_html=True)

# ── route to module ───────────────────────────────────────────────
if module == "NSE Scanner + Win-Rate":
    import mod_scanner;          mod_scanner.render()
elif module == "Option Chain Viewer":
    import mod_option_chain;     mod_option_chain.render()
elif module == "Payoff Builder":
    import mod_payoff;           mod_payoff.render()
elif module == "Greeks Dashboard":
    import mod_greeks;           mod_greeks.render()
elif module == "BS Pricer + Edge":
    import mod_bs_pricer;        mod_bs_pricer.render()
elif module == "Volatility Lab":
    import mod_vol_lab;          mod_vol_lab.render()
elif module == "Option Selling Engine":
    import mod_selling_engine;   mod_selling_engine.render()
elif module == "Smart Strike Finder":
    import mod_strike_finder;    mod_strike_finder.render()
elif module == "Exit Strategy Engine":
    import mod_exit_engine;      mod_exit_engine.render()
elif module == "Covered Call Optimiser":
    import mod_covered_call;     mod_covered_call.render()
elif module == "Weekly Options Lab":
    import mod_weekly;           mod_weekly.render()
elif module == "Hedge Fund Dashboard":
    import mod_hedge_fund;       mod_hedge_fund.render()
elif module == "Book Strategy Matrix":
    import mod_book_matrix;      mod_book_matrix.render()
elif module == "Big-Move Calendar":
    import mod_big_move_calendar; mod_big_move_calendar.render()
elif module == "Probability Calculator":
    import mod_probability;      mod_probability.render()
elif module == "Rolling Engine":
    import mod_rolling;          mod_rolling.render()
elif module == "Strategy Encyclopedia":
    import mod_encyclopedia;     mod_encyclopedia.render()
elif module == "Strategy Backtester":
    import mod_backtest;         mod_backtest.render()

elif module == "Buy Signal Engine":
    import mod_buy_engine;       mod_buy_engine.render()
elif module == "Live Feed + Auto Execution":
    import mod_live_feed;        mod_live_feed.render()
elif module == "Trade Journal":
    import alerts_journal;       alerts_journal.render_journal_tab()
elif module == "Telegram Alerts":
    import mod_telegram_config;  mod_telegram_config.render()
else:
    st.info("Select a module from the sidebar.")