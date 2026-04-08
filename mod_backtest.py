"""
Module: 📅  Historical Strategy Backtester
Options Alpha Platform v6.0

Simulates NSE options strategies across up to 25 years of real price history.
Uses Black-Scholes + rolling HV as IV proxy (no historical chain data needed).
Inspired by AlgoTest.in — free, local, no API keys required.

Strategies supported:
  • Short Strangle  — sell OTM call + OTM put every monthly expiry
  • Short Straddle  — sell ATM call + ATM put every monthly expiry
  • Cash-Secured Put (CSP) — sell OTM put every month
  • Covered Call    — sell OTM call every month
  • Iron Condor     — short strangle + long wings for defined risk

Exit rules (Cordier / McMillan framework):
  • Close at 50% profit (collect half the premium, move on)
  • Stop-loss at 2× premium collected (limit blowup)
  • Hard close at expiry (whatever P&L remains)

Books referenced:
  Cordier — "The Complete Guide to Option Selling"
  McMillan — "Options as Strategic Investment"
  Natenberg — "Option Volatility & Pricing"
  Carter — "Teach Me All About Options"
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from utils import (
    explain, NSE_FNO, SECTORS, INDICES_ONLY,
    bs_price, bs_greeks,
    fetch_history,
)
from config import (
    RISK_FREE_RATE, TRADING_DAYS_YEAR,
    NSE_STRIKE_STEP, DEFAULT_STRIKE_STEP,
    BACKTEST_DEFAULT_YEARS, BACKTEST_MAX_YEARS,
    BACKTEST_DELTA_TARGET, BACKTEST_PROFIT_TARGET, BACKTEST_LOSS_LIMIT,
    NSE_LOT_SIZES, DEFAULT_LOT_SIZE,
)


# ─────────────────────────────────────────────────────────────────
# Core backtest engine
# ─────────────────────────────────────────────────────────────────

def _hv_at(returns: pd.Series, date, window: int = 20) -> float:
    """Rolling HV at a given date using last `window` returns."""
    idx = returns.index.searchsorted(date)
    if idx < window:
        idx = window
    slice_ = returns.iloc[max(0, idx - window): idx]
    if len(slice_) < 5:
        return 0.20        # fallback 20% if not enough data
    return float(slice_.std() * np.sqrt(TRADING_DAYS_YEAR))


def _snap_strike(spot: float, target: float, step: int) -> int:
    """Snap target strike to nearest valid NSE strike grid."""
    return int(round(target / step) * step)


def _get_monthly_expiry_dates(closes: pd.Series) -> list:
    """
    Return list of last-Thursday-of-every-month dates present in closes.
    NSE monthly F&O expiry = last Thursday of month.
    If that Thursday is missing (holiday), step back to Wednesday etc.
    """
    expiries = []
    dates = closes.index
    # Group by year-month and find last Thursday
    df_dates = pd.DataFrame({"date": dates})
    df_dates["ym"] = df_dates["date"].dt.to_period("M")
    df_dates["dow"] = df_dates["date"].dt.dayofweek   # 0=Mon, 3=Thu

    for ym, group in df_dates.groupby("ym"):
        thursdays = group[group["dow"] == 3]["date"]
        if len(thursdays) == 0:
            # No Thursday that month — take last trading day
            thursdays = group["date"]
        expiries.append(thursdays.iloc[-1])
    return sorted(expiries)


def run_backtest(
    ticker: str,
    sym: str,
    strategy: str,
    years: int,
    delta_target: float,
    profit_target_pct: float,
    loss_limit_mult: float,
    hv_window: int,
    iv_premium_pct: float,    # IV = HV * (1 + iv_premium_pct/100)
    wing_width_pct: float,    # for Iron Condor — wing distance as % OTM beyond short
) -> dict:
    """
    Main backtest loop. Returns dict with trade_log DataFrame + summary stats.
    """
    # ── Fetch history ──────────────────────────────────────────
    hist = fetch_history(ticker, period="max")
    if hist is None or len(hist) < 252:
        return {"error": "Insufficient history data from yfinance."}

    closes = hist["Close"].dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)

    # Trim to requested years
    cutoff = closes.index[-1] - pd.DateOffset(years=years)
    closes = closes[closes.index >= cutoff]
    if len(closes) < 252:
        return {"error": f"Only {len(closes)} trading days in range — need at least 252."}

    log_returns = np.log(closes / closes.shift(1)).dropna()

    # NSE step + lot size
    step     = NSE_STRIKE_STEP.get(sym, DEFAULT_STRIKE_STEP)
    lot_size = NSE_LOT_SIZES.get(sym, DEFAULT_LOT_SIZE)
    r        = RISK_FREE_RATE

    # ── Get monthly expiry dates ───────────────────────────────
    expiries = _get_monthly_expiry_dates(closes)

    # ── Trade loop ────────────────────────────────────────────
    trades = []
    equity_curve = []   # (date, cumulative_pnl)
    cum_pnl = 0.0

    for i in range(len(expiries) - 1):
        entry_date  = expiries[i]
        exit_date   = expiries[i + 1]   # next monthly expiry = target close date

        # Find actual trading dates (closes might skip weekends/holidays)
        entry_dates_avail = closes.index[closes.index >= entry_date]
        exit_dates_avail  = closes.index[closes.index <= exit_date]
        if len(entry_dates_avail) == 0 or len(exit_dates_avail) == 0:
            continue

        actual_entry = entry_dates_avail[0]
        actual_exit  = exit_dates_avail[-1]

        S_entry = float(closes.loc[actual_entry])
        S_exit  = float(closes.loc[actual_exit])
        T_days  = max((actual_exit - actual_entry).days, 1)
        T       = T_days / 365.0

        # IV estimate at entry
        hv_entry = _hv_at(log_returns, actual_entry, hv_window)
        iv_entry = hv_entry * (1 + iv_premium_pct / 100)
        if iv_entry < 0.05:
            iv_entry = 0.10   # floor

        # ── Strike selection by strategy ──────────────────────
        if strategy in ("Short Strangle", "Iron Condor"):
            # Short call: find strike where delta ≈ delta_target (OTM call)
            call_target = S_entry * np.exp(norm.ppf(1 - delta_target) * iv_entry * np.sqrt(T))
            K_call = _snap_strike(S_entry, call_target, step)
            # Short put: find strike where |delta| ≈ delta_target (OTM put)
            put_target  = S_entry * np.exp(norm.ppf(delta_target) * iv_entry * np.sqrt(T))
            K_put  = _snap_strike(S_entry, put_target, step)

        elif strategy == "Short Straddle":
            K_call = K_put = _snap_strike(S_entry, S_entry, step)

        elif strategy == "Cash-Secured Put":
            put_target = S_entry * np.exp(norm.ppf(delta_target) * iv_entry * np.sqrt(T))
            K_put  = _snap_strike(S_entry, put_target, step)
            K_call = None

        elif strategy == "Covered Call":
            call_target = S_entry * np.exp(norm.ppf(1 - delta_target) * iv_entry * np.sqrt(T))
            K_call = _snap_strike(S_entry, call_target, step)
            K_put  = None

        # ── Premium at entry ──────────────────────────────────
        premium_call = bs_price(S_entry, K_call, T, r, iv_entry, "call") if K_call else 0.0
        premium_put  = bs_price(S_entry, K_put,  T, r, iv_entry, "put")  if K_put  else 0.0

        # Iron Condor: buy wings
        wing_prem_call = wing_prem_put = 0.0
        K_long_call = K_long_put = None
        if strategy == "Iron Condor":
            K_long_call = _snap_strike(S_entry, K_call * (1 + wing_width_pct / 100), step)
            K_long_put  = _snap_strike(S_entry, K_put  * (1 - wing_width_pct / 100), step)
            wing_prem_call = bs_price(S_entry, K_long_call, T, r, iv_entry, "call")
            wing_prem_put  = bs_price(S_entry, K_long_put,  T, r, iv_entry, "put")

        credit_collected = (premium_call + premium_put
                            - wing_prem_call - wing_prem_put)

        if credit_collected < 0.50:
            continue   # skip trades with negligible premium

        profit_target_price = credit_collected * (1 - profit_target_pct)
        stop_loss_price     = credit_collected * (1 + loss_limit_mult)

        # ── Mid-trade check (using closes between entry and exit) ──
        daily_dates = closes.index[(closes.index > actual_entry) & (closes.index <= actual_exit)]
        exit_reason = "expiry"
        exit_day    = actual_exit
        S_exit_used = S_exit

        for chk_date in daily_dates:
            S_chk = float(closes.loc[chk_date])
            T_rem = max((actual_exit - chk_date).days / 365.0, 0.001)
            hv_chk = _hv_at(log_returns, chk_date, hv_window)
            iv_chk = hv_chk * (1 + iv_premium_pct / 100)
            if iv_chk < 0.05:
                iv_chk = 0.10

            current_val  = 0.0
            if K_call:
                current_val += bs_price(S_chk, K_call, T_rem, r, iv_chk, "call")
            if K_put:
                current_val += bs_price(S_chk, K_put,  T_rem, r, iv_chk, "put")
            if K_long_call:
                current_val -= bs_price(S_chk, K_long_call, T_rem, r, iv_chk, "call")
            if K_long_put:
                current_val -= bs_price(S_chk, K_long_put,  T_rem, r, iv_chk, "put")

            if current_val <= profit_target_price:
                exit_reason = "profit_target"
                exit_day    = chk_date
                S_exit_used = S_chk
                break
            if current_val >= stop_loss_price:
                exit_reason = "stop_loss"
                exit_day    = chk_date
                S_exit_used = S_chk
                break

        # ── Final P&L at exit ──────────────────────────────────
        T_exit = max((actual_exit - exit_day).days / 365.0, 0.0) if exit_reason == "expiry" else 0.001
        hv_exit = _hv_at(log_returns, exit_day, hv_window)
        iv_exit = hv_exit * (1 + iv_premium_pct / 100)
        if iv_exit < 0.05:
            iv_exit = 0.10

        exit_val = 0.0
        if exit_reason == "expiry":
            # At expiry: intrinsic value only
            if K_call: exit_val += max(S_exit_used - K_call, 0)
            if K_put:  exit_val += max(K_put - S_exit_used, 0)
            if K_long_call: exit_val -= max(S_exit_used - K_long_call, 0)
            if K_long_put:  exit_val -= max(K_long_put - S_exit_used, 0)
        else:
            if K_call: exit_val += bs_price(S_exit_used, K_call, T_exit, r, iv_exit, "call")
            if K_put:  exit_val += bs_price(S_exit_used, K_put,  T_exit, r, iv_exit, "put")
            if K_long_call: exit_val -= bs_price(S_exit_used, K_long_call, T_exit, r, iv_exit, "call")
            if K_long_put:  exit_val -= bs_price(S_exit_used, K_long_put,  T_exit, r, iv_exit, "put")

        pnl_points = credit_collected - exit_val
        pnl_rupees = pnl_points * lot_size

        cum_pnl += pnl_rupees
        equity_curve.append({"date": exit_day, "cum_pnl": cum_pnl, "trade_pnl": pnl_rupees})

        dist_call = round((K_call / S_entry - 1) * 100, 1) if K_call else None
        dist_put  = round((K_put  / S_entry - 1) * 100, 1) if K_put  else None

        trades.append({
            "Entry Date":     actual_entry.strftime("%Y-%m-%d"),
            "Exit Date":      exit_day.strftime("%Y-%m-%d"),
            "DTE":            T_days,
            "Spot Entry":     round(S_entry, 1),
            "Spot Exit":      round(S_exit_used, 1),
            "IV (%)":         round(iv_entry * 100, 1),
            "K Call":         K_call,
            "K Put":          K_put,
            "Dist Call (%)":  dist_call,
            "Dist Put (%)":   dist_put,
            "Credit (₹)":     round(credit_collected, 2),
            "Exit Val (₹)":   round(exit_val, 2),
            "P&L pts":        round(pnl_points, 2),
            "P&L (₹)":        round(pnl_rupees, 0),
            "Exit Reason":    exit_reason,
            "Win":            pnl_points > 0,
        })

    if not trades:
        return {"error": "No valid trades found in the date range."}

    trade_df   = pd.DataFrame(trades)
    equity_df  = pd.DataFrame(equity_curve)

    # ── Summary statistics ─────────────────────────────────────
    n_trades    = len(trade_df)
    n_wins      = trade_df["Win"].sum()
    win_rate    = round(n_wins / n_trades * 100, 1)
    avg_pnl     = round(trade_df["P&L (₹)"].mean(), 0)
    total_pnl   = round(trade_df["P&L (₹)"].sum(), 0)
    best_trade  = round(trade_df["P&L (₹)"].max(), 0)
    worst_trade = round(trade_df["P&L (₹)"].min(), 0)
    avg_credit  = round(trade_df["Credit (₹)"].mean(), 2)
    avg_iv      = round(trade_df["IV (%)"].mean(), 1)

    # Max drawdown from equity curve
    eq = equity_df["cum_pnl"].values
    peak = np.maximum.accumulate(eq)
    dd   = eq - peak
    max_dd = round(float(dd.min()), 0)

    # Sharpe (monthly returns)
    monthly_pnl = trade_df["P&L (₹)"]
    sharpe = round(monthly_pnl.mean() / max(monthly_pnl.std(), 1) * np.sqrt(12), 2)

    # Profit factor
    gross_profit = trade_df[trade_df["P&L (₹)"] > 0]["P&L (₹)"].sum()
    gross_loss   = abs(trade_df[trade_df["P&L (₹)"] < 0]["P&L (₹)"].sum())
    profit_factor = round(gross_profit / max(gross_loss, 1), 2)

    # Exit reason breakdown
    exit_counts = trade_df["Exit Reason"].value_counts().to_dict()

    # Yearly breakdown
    trade_df["Year"] = pd.to_datetime(trade_df["Exit Date"]).dt.year
    yearly = trade_df.groupby("Year").agg(
        Trades=("Win", "count"),
        Wins=("Win", "sum"),
        PnL=("P&L (₹)", "sum"),
    ).reset_index()
    yearly["Win %"] = (yearly["Wins"] / yearly["Trades"] * 100).round(1)
    yearly["PnL"]   = yearly["PnL"].round(0)

    return {
        "trade_df":     trade_df,
        "equity_df":    equity_df,
        "yearly_df":    yearly,
        "n_trades":     n_trades,
        "win_rate":     win_rate,
        "avg_pnl":      avg_pnl,
        "total_pnl":    total_pnl,
        "best_trade":   best_trade,
        "worst_trade":  worst_trade,
        "max_drawdown": max_dd,
        "sharpe":       sharpe,
        "profit_factor":profit_factor,
        "avg_credit":   avg_credit,
        "avg_iv":       avg_iv,
        "lot_size":     lot_size,
        "exit_counts":  exit_counts,
        "years_tested": years,
        "actual_years": round((pd.to_datetime(trade_df["Exit Date"].iloc[-1])
                               - pd.to_datetime(trade_df["Entry Date"].iloc[0])).days / 365.25, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Streamlit render
# ─────────────────────────────────────────────────────────────────

def render():
    st.title("📅 Historical Strategy Backtester — Up to 25 Years")

    explain(
        "Simulates NSE options strategies on <b>real price history</b> (up to 25 years via yfinance). "
        "Since historical NSE option chains aren't freely available, <b>IV is estimated as rolling HV × a premium multiplier</b> "
        "(the typical IV/HV ratio for NSE stocks is 1.10–1.25×). "
        "Every trade enters on the last Thursday of the month and exits via profit target, stop-loss, or expiry — "
        "exactly as Cordier and McMillan prescribe. "
        "<b>This is directionally accurate</b>, not tick-exact — use it for strategy selection, not live sizing.",
        "explain",
    )

    # ── Sidebar controls ──────────────────────────────────────
    st.markdown("### ⚙️ Backtest Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        sym_bt = st.selectbox("Stock / Index", list(NSE_FNO.keys()), key="bt_sym")
        strategy = st.selectbox(
            "Strategy",
            ["Short Strangle", "Short Straddle", "Cash-Secured Put", "Covered Call", "Iron Condor"],
            key="bt_strat",
        )
        years = st.slider(
            "History (years)", 5, BACKTEST_MAX_YEARS, BACKTEST_DEFAULT_YEARS, key="bt_years"
        )

    with col2:
        delta_target = st.slider(
            "Target Delta (short strikes)", 0.05, 0.35, BACKTEST_DELTA_TARGET,
            step=0.01, key="bt_delta",
            help="McMillan: 0.10–0.20 for conservative selling. Higher delta = more premium but more risk.",
        )
        profit_target_pct = st.slider(
            "Profit Target (%)", 25, 75, int(BACKTEST_PROFIT_TARGET * 100),
            key="bt_pt",
            help="Cordier: close at 50% of max profit. Don't be greedy.",
        )
        loss_limit_mult = st.slider(
            "Stop-Loss (× credit)", 1.5, 4.0, BACKTEST_LOSS_LIMIT,
            step=0.5, key="bt_sl",
            help="Exit if current loss = N× premium collected. Cordier uses 2×.",
        )

    with col3:
        hv_window = st.selectbox(
            "HV Window (days)", [10, 20, 30, 60], index=1, key="bt_hv",
            help="Rolling window to estimate historical volatility at each entry date.",
        )
        iv_premium_pct = st.slider(
            "IV Premium over HV (%)", 0, 40, 18, key="bt_ivp",
            help="NSE IV is typically 15–25% above HV. This is the IV/HV spread baked into pricing.",
        )
        wing_width_pct = st.slider(
            "Iron Condor Wing Width (%)", 2, 10, 4, key="bt_wing",
            help="Long wing distance as % beyond the short strike. Only used for Iron Condor.",
        )

    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure parameters above and click **▶ Run Backtest** to start.")
        explain(
            "<b>How it works:</b> "
            "On every last Thursday of each month, the engine looks up the real closing price, "
            "estimates IV using rolling HV, finds the correct OTM strikes using Black-Scholes delta, "
            "and 'sells' a paper position. Each day until the next expiry, it checks if "
            "the 50% profit target or 2× stop-loss is hit. If neither — it closes at expiry using intrinsic value. "
            "P&L is accumulated per lot.",
            "explain",
        )
        return

    # ── Run the engine ────────────────────────────────────────
    ticker = NSE_FNO[sym_bt]
    with st.spinner(f"Running {years}-year backtest for {sym_bt} — {strategy}…"):
        result = run_backtest(
            ticker=ticker,
            sym=sym_bt,
            strategy=strategy,
            years=years,
            delta_target=delta_target,
            profit_target_pct=profit_target_pct / 100,
            loss_limit_mult=loss_limit_mult,
            hv_window=hv_window,
            iv_premium_pct=iv_premium_pct,
            wing_width_pct=wing_width_pct,
        )

    if "error" in result:
        st.error(f"Backtest failed: {result['error']}")
        return

    # ── Summary metrics ───────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📊 Results — {sym_bt} | {strategy} | {result['actual_years']:.1f} years | {result['n_trades']} trades")

    wr_color  = "normal" if result["win_rate"] >= 60 else "inverse"
    sh_color  = "normal" if result["sharpe"]   >= 0.5 else "inverse"
    dd_color  = "inverse" if result["max_drawdown"] < -50000 else "normal"

    m = st.columns(8)
    m[0].metric("Win Rate",       f"{result['win_rate']}%",       delta="Good" if result["win_rate"] >= 60 else "Below target")
    m[1].metric("Total P&L",      f"₹{result['total_pnl']:,.0f}")
    m[2].metric("Avg P&L/Trade",  f"₹{result['avg_pnl']:,.0f}")
    m[3].metric("Best Trade",     f"₹{result['best_trade']:,.0f}")
    m[4].metric("Worst Trade",    f"₹{result['worst_trade']:,.0f}")
    m[5].metric("Max Drawdown",   f"₹{result['max_drawdown']:,.0f}")
    m[6].metric("Sharpe",         str(result["sharpe"]))
    m[7].metric("Profit Factor",  str(result["profit_factor"]))

    # Exit reason breakdown
    ec = result["exit_counts"]
    ec_cols = st.columns(3)
    ec_cols[0].metric("Profit Target Exits", ec.get("profit_target", 0),
                      help="Closed early at 50% profit — the ideal outcome")
    ec_cols[1].metric("Stop-Loss Exits",     ec.get("stop_loss", 0),
                      help="Closed at 2× loss — the Cordier discipline kicking in")
    ec_cols[2].metric("Held to Expiry",      ec.get("expiry", 0),
                      help="Neither target nor stop hit — closed at intrinsic value")

    # ── Book commentary ───────────────────────────────────────
    if result["win_rate"] >= 65 and result["sharpe"] >= 0.5:
        explain(
            f"<b>Cordier verdict:</b> Win rate {result['win_rate']}% with Sharpe {result['sharpe']} — "
            f"this strategy has a solid edge on {sym_bt}. "
            f"Average IV at entry was {result['avg_iv']}%. "
            f"McMillan's 70% probability-of-profit threshold is {'✅ met' if result['win_rate'] >= 70 else '⚠️ close'}. "
            f"Profit factor {result['profit_factor']}× means you collected {result['profit_factor']}× more in winners than you lost in losers.",
            "safe",
        )
    elif result["win_rate"] >= 55:
        explain(
            f"<b>Natenberg verdict:</b> Win rate {result['win_rate']}% is acceptable but not exceptional. "
            f"Consider raising delta target from {delta_target:.2f} to 0.20+ to widen strikes, "
            f"or increase the IV premium slider if the IV/HV spread is typically higher for {sym_bt}.",
            "warning",
        )
    else:
        explain(
            f"<b>McMillan warning:</b> Win rate {result['win_rate']}% is below the 60% minimum for premium selling. "
            f"This symbol may have too many large moves. Try Iron Condor (defined risk) "
            f"or reduce position size. Max drawdown ₹{result['max_drawdown']:,.0f} confirms high risk.",
            "danger",
        )

    # ── Equity Curve ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Equity Curve")

    eq_df = result["equity_df"]
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq_df["date"], y=eq_df["cum_pnl"],
        mode="lines", name="Cumulative P&L",
        line=dict(color="#4f46e5", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.07)",
    ))
    # Highlight drawdown zones
    peak_vals = np.maximum.accumulate(eq_df["cum_pnl"].values)
    dd_vals   = eq_df["cum_pnl"].values - peak_vals
    fig_eq.add_trace(go.Scatter(
        x=eq_df["date"], y=peak_vals,
        mode="lines", name="Peak P&L",
        line=dict(color="#22c55e", width=1, dash="dot"),
    ))
    fig_eq.add_hline(y=0, line_width=1, line_color="#94a3b8")
    fig_eq.update_layout(
        template="plotly_white", height=380,
        yaxis_title="Cumulative P&L (₹)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Year-by-year breakdown ────────────────────────────────
    st.subheader("📆 Year-by-Year Performance")
    yearly_df = result["yearly_df"]

    fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in yearly_df["PnL"]]
    fig_yr.add_trace(go.Bar(
        x=yearly_df["Year"], y=yearly_df["PnL"],
        name="Annual P&L (₹)", marker_color=bar_colors,
    ), secondary_y=False)
    fig_yr.add_trace(go.Scatter(
        x=yearly_df["Year"], y=yearly_df["Win %"],
        mode="lines+markers", name="Win Rate (%)",
        line=dict(color="#f97316", width=2),
        marker=dict(size=6),
    ), secondary_y=True)
    fig_yr.add_hline(y=0, line_width=1, line_color="#94a3b8")
    fig_yr.update_yaxes(title_text="Annual P&L (₹)", secondary_y=False)
    fig_yr.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0, 100])
    fig_yr.update_layout(
        template="plotly_white", height=340,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_yr, use_container_width=True)

    # Yearly table
    yearly_disp = yearly_df.copy()
    yearly_disp["PnL"] = yearly_disp["PnL"].apply(lambda v: f"₹{v:,.0f}")
    yearly_disp.columns = ["Year", "Trades", "Wins", "P&L", "Win %"]

    def color_pnl(v):
        try:
            num = float(str(v).replace("₹","").replace(",",""))
            return "color:#15803d;font-weight:600" if num >= 0 else "color:#dc2626;font-weight:600"
        except:
            return ""

    st.dataframe(
        yearly_disp.style.applymap(color_pnl, subset=["P&L"])
                         .applymap(lambda v: "color:#15803d" if isinstance(v,(int,float)) and v>=60
                                   else ("color:#dc2626" if isinstance(v,(int,float)) and v<50 else ""),
                                   subset=["Win %"]),
        use_container_width=True, hide_index=True, height=min(40 * len(yearly_disp) + 40, 500),
    )

    # ── P&L Distribution histogram ────────────────────────────
    st.markdown("---")
    st.subheader("📊 P&L Distribution")
    col_hist, col_scatter = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            result["trade_df"], x="P&L (₹)", nbins=40,
            color_discrete_sequence=["#4f46e5"],
            title="P&L per Trade Distribution",
        )
        fig_hist.add_vline(x=0, line_width=1.5, line_color="#ef4444")
        fig_hist.add_vline(x=result["avg_pnl"], line_width=1.5,
                           line_color="#22c55e", annotation_text="Mean",
                           annotation_position="top right")
        fig_hist.update_layout(template="plotly_white", height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_scatter:
        trade_df_plot = result["trade_df"].copy()
        trade_df_plot["Color"] = trade_df_plot["Exit Reason"].map({
            "profit_target": "#22c55e",
            "stop_loss": "#ef4444",
            "expiry": "#f97316",
        })
        fig_sc = go.Figure()
        for reason, color, label in [
            ("profit_target", "#22c55e", "Profit Target"),
            ("stop_loss",     "#ef4444", "Stop Loss"),
            ("expiry",        "#f97316", "Expiry"),
        ]:
            sub = trade_df_plot[trade_df_plot["Exit Reason"] == reason]
            if len(sub) == 0: continue
            fig_sc.add_trace(go.Scatter(
                x=pd.to_datetime(sub["Exit Date"]),
                y=sub["P&L (₹)"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.75),
            ))
        fig_sc.add_hline(y=0, line_width=1, line_color="#94a3b8")
        fig_sc.update_layout(
            title="Trade P&L over Time (by Exit Type)",
            template="plotly_white", height=300,
            margin=dict(t=40, b=20),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Full trade log ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Trade Log")

    log_cols = ["Entry Date", "Exit Date", "DTE", "Spot Entry", "K Call", "K Put",
                "Dist Call (%)", "Dist Put (%)", "IV (%)", "Credit (₹)",
                "Exit Val (₹)", "P&L pts", "P&L (₹)", "Exit Reason"]

    def color_row_pnl(v):
        if not isinstance(v, (int, float)): return ""
        if v > 0: return "background-color:#f0fdf4;color:#166534"
        if v < 0: return "background-color:#fff1f2;color:#991b1b"
        return ""

    def color_exit(v):
        if v == "profit_target": return "background-color:#dcfce7;color:#166534;font-weight:600"
        if v == "stop_loss":     return "background-color:#fee2e2;color:#991b1b;font-weight:600"
        return "background-color:#fef9c3;color:#854d0e"

    st.dataframe(
        result["trade_df"][log_cols].style
            .applymap(color_row_pnl, subset=["P&L (₹)", "P&L pts"])
            .applymap(color_exit, subset=["Exit Reason"])
            .format({"Spot Entry": "{:.1f}", "Credit (₹)": "₹{:.2f}",
                     "Exit Val (₹)": "₹{:.2f}", "P&L pts": "{:.2f}",
                     "P&L (₹)": "₹{:,.0f}"}),
        use_container_width=True, height=500, hide_index=True,
    )

    explain(
        "<b>How to read this table:</b> "
        "'Credit' = premium collected at entry. 'Exit Val' = cost to close. "
        "'P&L pts' = Credit − Exit Val (positive = profit). "
        "'P&L ₹' = P&L pts × lot size. "
        "Green rows = winners. Red rows = stop-loss exits. Yellow rows = held to expiry. "
        f"Lot size used: {result['lot_size']} shares/lot for {sym_bt}.",
        "explain",
    )
