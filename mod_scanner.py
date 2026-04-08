"""
Module: 🔭  NSE Scanner + Win-Rate
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
    st.title("🔭 NSE F&O Scanner — Historical Win-Rate Analysis")

    explain(
        "This scanner fetches <b>real market data</b> for every NSE F&O stock and computes "
        "<b>how often each stock moved up/down/flat over a 1-month horizon</b>, "
        "using up to <b>15 years</b> of daily price history (~3,750 rolling windows per stock). "
        "The coloured columns are the most important: if a stock is flat <5% in 60%+ of months, "
        "it's an ideal <b>short-strangle or iron condor</b> candidate (Cordier framework). "
        "If it moves >10% often, avoid selling naked — use defined-risk structures (McMillan).",
        "explain",
    )

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        sector_sel = st.selectbox("Sector / Group", list(SECTORS.keys()))
    with col_f2:
        min_ivr = st.slider("Min IVR filter (%)", 0, 100, 0)
    with col_f3:
        sig_only = st.checkbox("Sell-Premium signals only", value=False)

    syms_to_scan = SECTORS[sector_sel]

    # ── Parallel fetch using ThreadPoolExecutor (~5x faster) ──
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(sym):
        try:
            r = build_scanner_row(sym)
            if r:
                try:
                    from utils import compute_buy_score
                    bs = compute_buy_score(sym)
                    r["Buy Score"] = bs["score"] if bs else None
                    r["Buy Grade"] = bs["grade"] if bs else None
                except Exception:
                    r["Buy Score"] = None
                    r["Buy Grade"] = None
            return sym, r
        except Exception:
            return sym, None

    progress_bar = st.progress(0, text="Fetching data in parallel…")
    rows = []
    completed = 0
    total = len(syms_to_scan)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in syms_to_scan}
        for future in as_completed(futures):
            sym, r = future.result()
            if r:
                rows.append(r)
            completed += 1
            progress_bar.progress(completed / total, text=f"Loaded {completed}/{total} — {sym}")

    progress_bar.empty()
    # Sort rows to match original sector order
    order = {sym: i for i, sym in enumerate(syms_to_scan)}
    rows.sort(key=lambda r: order.get(r.get("Symbol", ""), 9999))

    if not rows:
        st.error("Could not fetch data. Check internet connection.")
        return

    df = pd.DataFrame(rows)
    if min_ivr > 0:
        df = df[df["IVR (%)"] >= min_ivr]
    if sig_only:
        df = df[df["Signal"] == "SELL PREM"]

    # ── colour coding helper ──────────────────────────────────
    def colour_flat(v):
        if pd.isna(v):   return ""
        if v >= 60:      return "background-color:#dcfce7;color:#166534;font-weight:600"
        if v >= 45:      return "background-color:#fef9c3;color:#854d0e"
        return "background-color:#fee2e2;color:#991b1b"

    def colour_move(v):
        if pd.isna(v):   return ""
        if v >= 40:      return "background-color:#fee2e2;color:#991b1b;font-weight:600"
        if v >= 25:      return "background-color:#fef9c3;color:#854d0e"
        return "background-color:#dcfce7;color:#166534"

    def colour_ivr(v):
        if pd.isna(v):   return ""
        if v >= 60:      return "background-color:#dcfce7;color:#166534;font-weight:600"
        if v >= 40:      return "background-color:#fef9c3;color:#854d0e"
        return "background-color:#fee2e2;color:#991b1b"

    def colour_signal(v):
        if v == "SELL PREM": return "background-color:#dcfce7;color:#166534;font-weight:700"
        return "background-color:#f1f5f9;color:#64748b"

    def colour_buy_score(v):
        if not isinstance(v, (int, float)): return ""
        if v >= 26: return "background-color:#dbeafe;color:#1e3a8a;font-weight:700"
        if v >= 22: return "background-color:#eff6ff;color:#1e3a8a"
        if v >= 18: return "background-color:#fef9c3;color:#854d0e"
        return ""

    def colour_buy_grade(v):
        if v == "A+": return "background-color:#dbeafe;color:#1e3a8a;font-weight:700"
        if v == "A":  return "background-color:#eff6ff;color:#1e3a8a"
        return ""

   # ── SAFE styling (FIXED FOR STREAMLIT CLOUD) ─────────────────────

    if df is None or df.empty:
        st.warning("⚠️ No data available. API may be blocked or failed.")
        st.stop()

   
    buy_score_cols = ["Buy Score"] if "Buy Score" in df.columns else []
    buy_grade_cols = ["Buy Grade"] if "Buy Grade" in df.columns else []
    flat_cols      = ["Flat <5% (%)"] if "Flat <5% (%)" in df.columns else []
    move_cols      = [c for c in ["Up >5% (%)","Down >5% (%)","Up >10% (%)","Down >10% (%)"] if c in df.columns]
    ivr_cols       = ["IVR (%)"] if "IVR (%)" in df.columns else []
    signal_cols    = ["Signal"] if "Signal" in df.columns else []

    styled = df.style

    if buy_score_cols:
        styled = styled.applymap(colour_buy_score, subset=buy_score_cols)

    if buy_grade_cols:
        styled = styled.applymap(colour_buy_grade, subset=buy_grade_cols)

    if flat_cols:
        styled = styled.applymap(colour_flat, subset=flat_cols)

    if move_cols:
        styled = styled.applymap(colour_move, subset=move_cols)

    if ivr_cols:
        styled = styled.applymap(colour_ivr, subset=ivr_cols)

    if signal_cols:
        styled = styled.applymap(colour_signal, subset=signal_cols)

    styled = styled.format({
        "Spot (₹)":          "₹{:,.2f}",
        "HV20 (%)":          "{:.1f}%",
        "IV approx (%)":     "{:.1f}%",
        "IVR (%)":           "{:.0f}%",
        "Mean 30d move (%)": "{:+.2f}%",
        "Up >5% (%)":        "{:.1f}%",
        "Down >5% (%)":      "{:.1f}%",
        "Flat <5% (%)":      "{:.1f}%",
        "Up >10% (%)":       "{:.1f}%",
        "Down >10% (%)":     "{:.1f}%",
    }, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=520)

    # ── legend ───────────────────────────────────────────────
    st.markdown("""
    **Column guide:**
    - **Flat <5%** — % of 30-day windows where price stayed within ±5% of entry → 🟢 ≥60% ideal for strangle/condor selling
    - **Up/Down >5%** — % of windows with a >5% directional move → 🔴 ≥40% = high mover, avoid naked selling
    - **IVR** — IV Rank: how expensive options are vs 1-yr range → 🟢 ≥60% = sell premium now
    - **Signal** — 🟢 SELL PREM = high IVR + historically rangebound (Cordier criteria)
    - **Data yrs** — years of history used for win-rate calculation
    """)

    explain(
        "How to use this table: find stocks with <b>Flat <5% ≥ 60%</b> AND <b>IVR ≥ 50%</b>. "
        "These are Cordier's ideal candidates — historically rangebound AND options currently expensive. "
        "Stocks with <b>Up/Down >10% ≥ 30%</b> are movers — only trade defined-risk structures (spreads, condors) on those. "
        "The 'Mean 30d move' column tells you the average expected drift — a strong upward drift means "
        "put strikes are safer to sell than call strikes.",
        "cordier",
    )

    # ── drill-down: click a stock ─────────────────────────────
    st.markdown("---")
    st.subheader("📊 Stock Drill-Down — 30-Day Move Distribution")
    drill_sym = st.selectbox("Pick a stock to analyse", [r["Symbol"] for r in rows])

    wr = compute_monthly_winrate(NSE_FNO[drill_sym])
    if wr:
        s = wr["summary"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Observations",  f"{s['n']:,}")
        c2.metric("Mean 30d move", f"{s['mean']:+.2f}%")
        c3.metric("Any Up %",      f"{s['up_rate']:.1f}%")
        c4.metric("Flat <5%",      f"{s['flat_5pct']:.1f}%",
                  delta="Good for selling" if s["flat_5pct"]>50 else "Caution",
                  delta_color="normal")
        c5.metric("Data years",    f"{wr['years']} yrs")

        explain(
            f"Over {wr['years']} years of data ({s['n']:,} rolling 30-day windows), "
            f"<b>{drill_sym}</b> stayed within ±5% of the entry price in "
            f"<b>{s['flat_5pct']:.1f}%</b> of all cases. "
            f"It moved up >5% in {s['up_5pct']:.1f}% of cases and down >5% in {s['dn_5pct']:.1f}% of cases. "
            f"The mean 30-day drift is {s['mean']:+.2f}%, "
            f"meaning {'the stock has a slight upward bias — put selling is safer than call selling' if s['mean']>0 else 'slight downward bias — call selling is relatively safer'}. "
            f"This data directly tells you which strikes have historically been safe to sell.",
            "natenberg",
        )

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            # Distribution histogram
            fig_hist = go.Figure()
            bins  = wr["hist_bins"]
            cnts  = wr["hist_counts"]
            mids  = (bins[:-1] + bins[1:]) / 2
            colours = ["#ef4444" if m < -5 else "#22c55e" if m > 5 else "#60a5fa" for m in mids]
            fig_hist.add_trace(go.Bar(x=mids, y=cnts, marker_color=colours, name="Frequency"))
            fig_hist.add_vline(x=0,  line=dict(color="black",  dash="dot"), annotation_text="0%")
            fig_hist.add_vline(x=5,  line=dict(color="#22c55e", dash="dash"), annotation_text="+5%")
            fig_hist.add_vline(x=-5, line=dict(color="#ef4444", dash="dash"), annotation_text="-5%")
            fig_hist.update_layout(
                title=f"{drill_sym} — Distribution of 30-Day Returns ({wr['years']} yrs)",
                xaxis_title="30-Day Price Change (%)",
                yaxis_title="Frequency (# of windows)",
                template="plotly_white", height=340,
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            explain(
                "Blue bars = flat zone (±5%) — if stock closes here, short strangles & condors profit. "
                "Red bars = big down moves — dangerous for short puts. "
                "Green bars = big up moves — dangerous for short calls. "
                "The wider the blue zone vs red/green, the better the stock is for premium selling.",
                "explain",
            )

        with col_d2:
            # Monthly seasonality
            bm = wr["by_month"]
            fig_season = go.Figure()
            fig_season.add_trace(go.Bar(
                x=bm["month_name"], y=bm["mean_move"],
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in bm["mean_move"]],
                name="Mean move",
            ))
            fig_season.add_trace(go.Scatter(
                x=bm["month_name"], y=bm["flat_5"],
                name="Flat <5% rate (%)", yaxis="y2",
                line=dict(color="#4f46e5", width=2), mode="lines+markers",
            ))
            fig_season.update_layout(
                title=f"{drill_sym} — Monthly Seasonality",
                yaxis=dict(title="Mean 30d Move (%)", side="left"),
                yaxis2=dict(title="Flat <5% Rate (%)", overlaying="y", side="right"),
                template="plotly_white", height=340,
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_season, use_container_width=True)
            explain(
                "Seasonality shows which months the stock historically trends vs stays flat. "
                "The purple line (right axis) shows the flat-rate per month — "
                "months with a high flat rate are the <b>best calendar months to sell options</b> on this stock. "
                "Cordier calls this 'seasonal premium selling windows.'",
                "cordier",
            )

        # Threshold summary table
        st.subheader("Threshold Probability Table")
        thresh_df = pd.DataFrame({
            "Threshold":    ["±5%", "±10%", "±15%", "±20%"],
            "Flat (%)":     [s["flat_5pct"],  s["flat_10pct"],  s["flat_15pct"],  s["flat_20pct"]],
            "Up move (%)":  [s["up_5pct"],    s["up_10pct"],    s["up_15pct"],    s["up_20pct"]],
            "Down move (%)": [s["dn_5pct"],   s["dn_10pct"],    s["dn_15pct"],    s["dn_20pct"]],
            "Beyond (%)":   [s["beyond_5pct"],s["beyond_10pct"],s["beyond_15pct"],s["beyond_20pct"]],
        })
        st.dataframe(
            thresh_df.style
            .applymap(lambda v: "background-color:#dcfce7;color:#166534;font-weight:600" if v >= 60 else
                                "background-color:#fef9c3;color:#854d0e"                 if v >= 40 else
                                "background-color:#fee2e2;color:#991b1b",
                      subset=["Flat (%)"])
            .format("{:.1f}%", subset=["Flat (%)","Up move (%)","Down move (%)","Beyond (%)"]),
            use_container_width=True, hide_index=True,
        )
        explain(
            "Read this as: 'If I sell a <b>±5% strangle</b> on this stock, historically "
            f"it would have stayed inside those strikes in <b>{s['flat_5pct']:.1f}%</b> of cases — "
            f"that's my <b>probability of full profit</b>. "
            f"The remaining {s['beyond_5pct']:.1f}% of cases, one side would have been breached.' "
            "Use this table to pick your strike distance. Cordier's minimum: sell strikes where "
            "the flat-rate (prob of full profit) is ≥ 70%.",
            "cordier",
        )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 2 — OPTION CHAIN VIEWER  ███
    # ═══════════════════════════════════════════════════════════════
