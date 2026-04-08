"""
mod_big_move_calendar.py
Big-Move Calendar for ALL NSE F&O stocks + indices (15-year history)
Sources: Natenberg (IV regime), Cordier (seasonal selling windows),
         Carter (event avoidance), McMillan (historical context)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from utils import (
    explain, NSE_FNO, SECTORS, INDICES_ONLY, ALL_STOCKS_ONLY,
    fetch_history, fetch_spot_iv, compute_monthly_winrate, get_yf
)
from datetime import datetime


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

@st.cache_data(ttl=3600, show_spinner=False)
def get_monthly_returns(ticker: str):
    yf = get_yf()
    try:
        hist = yf.Ticker(ticker).history(period="15y", auto_adjust=True)
        if hist is None or hist.empty: return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        monthly = hist["Close"].resample("ME").last()
        mret    = ((monthly/monthly.shift(1))-1)*100
        return mret.dropna()
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def big_move_summary(ticker: str, threshold: float = 10.0):
    """Return summary stats for one ticker."""
    mret = get_monthly_returns(ticker)
    if mret is None or len(mret) < 12: return None
    n   = len(mret)
    up  = int((mret >= threshold).sum())
    dn  = int((mret <= -threshold).sum())
    flt = n - up - dn
    return {
        "n": n, "up": up, "dn": dn, "flat": flt,
        "up_pct":  round(up/n*100,1),
        "dn_pct":  round(dn/n*100,1),
        "flat_pct":round(flt/n*100,1),
        "mean":    round(float(mret.mean()),2),
        "best":    round(float(mret.max()),2),
        "worst":   round(float(mret.min()),2),
        "std":     round(float(mret.std()),2),
        "years":   max(1, n//12),
        "mret":    mret,
    }


def render():
    st.title("📅 Big-Move Calendar — All NSE F&O Stocks & Indices")

    explain(
        "This module analyses <b>15 years of monthly returns</b> for every NSE F&O stock and index. "
        "It shows which <b>month + year</b> had moves beyond your threshold, "
        "seasonal patterns (which calendar months are historically dangerous), "
        "and a full year × month heatmap. "
        "<b>Cordier:</b> 'Knowing your seasonal history is the option seller's hidden edge.' "
        "<b>Carter:</b> 'Never sell options in months with high historical volatility for that stock.'",
        "explain",
    )

    # ── controls ─────────────────────────────────────────────
    col_c1, col_c2, col_c3 = st.columns([2,2,1])
    with col_c1:
        view_mode = st.radio("View", ["Single Stock/Index", "All-Stock Scanner"], horizontal=True)
    with col_c2:
        threshold = st.slider("Big-move threshold (%)", 5, 25, 10,
                              help="Months where price moved more than this % are flagged")
    with col_c3:
        asset_type = st.radio("Asset class", ["All","Indices","Stocks"], horizontal=True)

    if view_mode == "Single Stock/Index":
        _render_single(threshold, asset_type)
    else:
        _render_scanner(threshold, asset_type)


# ─────────────────────────────────────────────────────────────────
# SINGLE STOCK DEEP DIVE
# ─────────────────────────────────────────────────────────────────
def _render_single(threshold, asset_type):
    if asset_type == "Indices":
        choices = INDICES_ONLY
    elif asset_type == "Stocks":
        choices = ALL_STOCKS_ONLY
    else:
        choices = list(NSE_FNO.keys())

    sym  = st.selectbox("Select stock / index", choices, key="bmc_sym")
    mret = get_monthly_returns(NSE_FNO[sym])

    if mret is None or len(mret) < 12:
        st.error("Insufficient data for this symbol.")
        return

    summ = big_move_summary(NSE_FNO[sym], threshold)

    # ── KPI row ───────────────────────────────────────────────
    st.markdown("---")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total months",       f"{summ['n']}")
    k2.metric("Data years",         f"{summ['years']} yrs")
    k3.metric(f"Up >{threshold}%",  f"{summ['up']}",  delta=f"{summ['up_pct']}% of months")
    k4.metric(f"Down >{threshold}%",f"{summ['dn']}",  delta=f"-{summ['dn_pct']}%", delta_color="inverse")
    k5.metric("Flat (inside)",       f"{summ['flat']}",delta=f"{summ['flat_pct']}%")
    k6.metric("Best / Worst",        f"{summ['best']:+.1f}% / {summ['worst']:+.1f}%")

    if summ["dn"] > 0:
        explain(
            f"Over {summ['years']} years, <b>{sym}</b> crashed more than {threshold}% in "
            f"<b>{summ['dn']} months</b> ({summ['dn_pct']}%). "
            f"The worst single month was <b>{summ['worst']:+.1f}%</b>. "
            f"Option sellers who held naked positions through these months faced potentially unlimited losses. "
            f"Cordier: iron condors would have capped losses in {summ['dn']} of these events.",
            "cordier",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 All Big Moves",
        "📊 Waterfall Chart",
        "🗓️ Seasonality",
        "🔥 Year×Month Heatmap",
        "⚡ Event Log",
    ])

    # ── TAB 1: Chronological table ────────────────────────────
    with tab1:
        st.subheader(f"All months where {sym} moved > ±{threshold}%")
        big = mret[mret.abs() >= threshold].copy().sort_index()
        if len(big) == 0:
            st.info(f"No months exceed ±{threshold}%. Try lowering the threshold.")
        else:
            bdf = pd.DataFrame({
                "Month-Year": big.index.strftime("%b %Y"),
                "Return (%)": big.values.round(2),
                "Direction":  ["🟢 RALLY" if v>=0 else "🔴 CRASH" for v in big.values],
                "Magnitude":  [f"{abs(v):.1f}%" for v in big.values],
                "Year":       big.index.year,
                "Month":      big.index.strftime("%b"),
                "Pct from 0": [f"{abs(v)/threshold:.1f}× threshold" for v in big.values],
            })

            def _style_ret(v):
                if not isinstance(v,(int,float)): return ""
                if v >=  threshold: return "background-color:#dcfce7;color:#14532d;font-weight:700"
                if v <= -threshold: return "background-color:#fee2e2;color:#7f1d1d;font-weight:700"
                return ""

            st.dataframe(
                bdf[["Month-Year","Return (%)","Direction","Magnitude","Pct from 0"]]
                .style.applymap(_style_ret, subset=["Return (%)"]),
                use_container_width=True, height=480, hide_index=True,
            )

            c1, c2 = st.columns(2)
            c1.success(f"🟢 Biggest rally: {bdf.loc[bdf['Return (%)'].idxmax(),'Month-Year']} ({bdf['Return (%)'].max():+.1f}%)")
            c2.error(  f"🔴 Biggest crash: {bdf.loc[bdf['Return (%)'].idxmin(),'Month-Year']} ({bdf['Return (%)'].min():+.1f}%)")

            explain(
                "Green rows = rallies beyond threshold (short calls threatened). "
                "Red rows = crashes beyond threshold (short puts threatened). "
                "<b>Pct from 0</b> shows how many multiples of the threshold this move was — "
                "a 2× threshold move in a strangle is often unrecoverable.",
                "explain",
            )

    # ── TAB 2: Waterfall bar chart ────────────────────────────
    with tab2:
        st.subheader(f"{sym} — 15-Year Monthly Return Waterfall")
        bar_colors = [
            "#15803d" if v >= threshold else
            "#dc2626" if v <= -threshold else
            "#60a5fa" if v >= 0 else
            "#f97316"
            for v in mret.values
        ]
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            x=[d.strftime("%b %Y") for d in mret.index],
            y=mret.values,
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Return: %{y:+.2f}%<extra></extra>",
        ))
        fig_wf.add_hline(y= threshold, line=dict(color="#15803d",dash="dash",width=1.5),
                         annotation_text=f"+{threshold}%", annotation_font_color="#15803d")
        fig_wf.add_hline(y=-threshold, line=dict(color="#dc2626",dash="dash",width=1.5),
                         annotation_text=f"-{threshold}%", annotation_font_color="#dc2626")
        fig_wf.add_hline(y=0, line=dict(color="#94a3b8",width=1))
        fig_wf.update_layout(
            title=f"{sym} — Monthly Returns ({summ['years']} years)",
            xaxis_title="Month", yaxis_title="Return (%)",
            template="plotly_white", height=420,
            xaxis=dict(tickangle=-60, nticks=40), bargap=0.15,
            font=dict(color="#1e293b"),
        )
        st.plotly_chart(fig_wf, use_container_width=True)
        explain(
            "🟢 Dark green = rally > threshold. 🔴 Dark red = crash > threshold. "
            "Blue = moderate up. Orange = moderate down. "
            "Clusters of red (2008, 2020) are exactly when naked option sellers blew up. "
            "Iron condors with defined width survived these; naked strangles did not.",
            "danger",
        )

    # ── TAB 3: Monthly seasonality ────────────────────────────
    with tab3:
        st.subheader(f"{sym} — Calendar Month Seasonality")
        mdf = mret.to_frame("ret")
        mdf["month"] = mdf.index.month
        season = mdf.groupby("month")["ret"].agg(
            avg_return="mean",
            up_big=lambda x: (x >= threshold).sum(),
            dn_big=lambda x: (x <= -threshold).sum(),
            up_any=lambda x: (x > 0).mean()*100,
            flat_pct=lambda x: (x.abs() < threshold).mean()*100,
            count="count",
        ).reset_index()
        season["month_name"] = season["month"].apply(lambda m: MONTH_NAMES[m-1])

        col_s1, col_s2 = st.columns([1,1])
        with col_s1:
            fig_sea = go.Figure()
            fig_sea.add_trace(go.Bar(
                x=season["month_name"], y=season["avg_return"],
                marker_color=["#15803d" if v>=0 else "#dc2626" for v in season["avg_return"]],
                text=[f"{v:+.1f}%" for v in season["avg_return"]],
                textposition="outside",
                textfont=dict(color="#1e293b", size=11),
            ))
            fig_sea.update_layout(
                title=f"Average Monthly Return by Calendar Month",
                yaxis_title="Avg Return (%)",
                template="plotly_white", height=320,
                font=dict(color="#1e293b"),
            )
            st.plotly_chart(fig_sea, use_container_width=True)

        with col_s2:
            tbl = season[["month_name","avg_return","up_big","dn_big","flat_pct","count"]].copy()
            tbl.columns = ["Month",f"Avg Ret (%)",f"Up>{threshold}% (n)",
                           f"Down>{threshold}% (n)",f"Flat<{threshold}% (%)","Total months"]

            def _style_season(row):
                out = []
                for c in row.index:
                    if c == "Avg Ret (%)":
                        out.append("color:#15803d;font-weight:700" if row[c]>=1 else
                                   "color:#dc2626;font-weight:700" if row[c]<=-1 else
                                   "color:#334155")
                    elif c == f"Flat<{threshold}% (%)":
                        out.append("background-color:#dcfce7;color:#14532d;font-weight:700" if row[c]>=70 else
                                   "background-color:#fef9c3;color:#78350f" if row[c]>=55 else
                                   "background-color:#fee2e2;color:#7f1d1d")
                    elif c in [f"Down>{threshold}% (n)"]:
                        out.append("color:#dc2626;font-weight:700" if row[c]>=3 else "color:#1e293b")
                    elif c in [f"Up>{threshold}% (n)"]:
                        out.append("color:#15803d;font-weight:700" if row[c]>=3 else "color:#1e293b")
                    else:
                        out.append("color:#1e293b")
                return out

            st.dataframe(
                tbl.style.apply(_style_season, axis=1)
                .format({"Avg Ret (%)":"{:+.2f}%", f"Flat<{threshold}% (%)":"{:.1f}%"}),
                use_container_width=True, hide_index=True, height=340,
            )

        explain(
            f"The <b>Flat<{threshold}%</b> column is your key metric. "
            f"Green (≥70%) = safe to sell options that month. "
            f"Red (<55%) = high chance of a big move — avoid naked selling. "
            f"Cordier: build your option selling calendar around high-flat months. "
            f"Carter: skip the worst 2–3 months every year and you dramatically improve your win-rate.",
            "cordier",
        )

    # ── TAB 4: Year × Month heatmap ───────────────────────────
    with tab4:
        st.subheader(f"{sym} — Year × Month Return Heatmap (15 yrs)")
        mdf2 = mret.to_frame("ret")
        mdf2["year"]  = mdf2.index.year
        mdf2["month"] = mdf2.index.month
        pivot = mdf2.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
        pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]
        pivot = pivot.sort_index(ascending=False)

        z_max = max(25, float(mret.abs().max()))
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[str(yr) for yr in pivot.index.tolist()],
            colorscale=[
                [0.0,  "#7f1d1d"],
                [0.25, "#ef4444"],
                [0.42, "#fca5a5"],
                [0.5,  "#f8fafc"],
                [0.58, "#86efac"],
                [0.75, "#22c55e"],
                [1.0,  "#14532d"],
            ],
            zmid=0, zmin=-z_max, zmax=z_max,
            text=[[f"{v:+.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size":9, "color":"#1e293b"},
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:+.2f}%<extra></extra>",
            colorbar=dict(title="Ret %", tickformat="+.0f",
                          tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
        ))
        fig_hm.update_layout(
            title=f"{sym} — Monthly Return Heatmap",
            xaxis_title="Month", yaxis_title="Year",
            template="plotly_white",
            height=max(400, len(pivot)*26),
            font=dict(color="#1e293b"),
            xaxis=dict(tickfont=dict(color="#1e293b",size=11)),
            yaxis=dict(tickfont=dict(color="#1e293b",size=11)),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        explain(
            "Deep red = crash >20%. Light red = moderate fall. White = flat. Light green = moderate gain. Deep green = rally >20%. "
            "Look for <b>row patterns</b> (crisis years like 2008, 2020 = full red rows). "
            "Look for <b>column patterns</b> (Jan, Mar, Oct often show seasonal tendencies). "
            "Cells with both red columns AND red rows = the historically most dangerous month for this stock.",
            "natenberg",
        )

    # ── TAB 5: Consecutive event log ─────────────────────────
    with tab5:
        st.subheader(f"{sym} — Consecutive Big-Move Sequences")
        explain(
            "Consecutive big months = extended trends or crisis periods. "
            "These are when option sellers are most at risk — a bad month is followed by another. "
            "Natenberg: during multi-month trends, IV stays elevated — wait for normalisation before selling.",
            "natenberg",
        )
        vals  = mret.values
        dates = mret.index
        seqs  = []
        i = 0
        while i < len(vals):
            if abs(vals[i]) >= threshold:
                s = i
                while i < len(vals) and abs(vals[i]) >= threshold:
                    i += 1
                e = i - 1
                sv = vals[s:e+1]
                cum = (np.prod([1+v/100 for v in sv])-1)*100
                seqs.append({
                    "Start":          dates[s].strftime("%b %Y"),
                    "End":            dates[e].strftime("%b %Y"),
                    "Consecutive months": e-s+1,
                    "Cumulative (%)": round(float(cum),2),
                    "Peak month (%)": round(float(max(sv,key=abs)),2),
                    "Type":           "CRASH 🔴" if sum(sv)<0 else "RALLY 🟢",
                })
            else:
                i += 1

        if seqs:
            sdf = pd.DataFrame(seqs)
            st.dataframe(
                sdf.style
                .applymap(lambda v: "background-color:#fee2e2;color:#7f1d1d;font-weight:700"
                          if "CRASH" in str(v) else
                          "background-color:#dcfce7;color:#14532d;font-weight:700"
                          if "RALLY" in str(v) else "color:#1e293b",
                          subset=["Type"])
                .format({"Cumulative (%)":"{:+.2f}%","Peak month (%)":"{:+.2f}%"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No consecutive big-move sequences found at this threshold.")


# ─────────────────────────────────────────────────────────────────
# ALL-STOCK SCANNER VIEW
# ─────────────────────────────────────────────────────────────────
def _render_scanner(threshold, asset_type):
    st.subheader(f"📊 All F&O Stocks — Big-Move Summary (>{threshold}% threshold)")

    explain(
        "This scanner computes the big-move statistics for <b>every NSE F&O stock</b> in one table. "
        "Sort by <b>Crash% or Rally%</b> to find the most volatile stocks (avoid naked selling). "
        "Sort by <b>Flat%</b> to find the most rangebound stocks (ideal for premium selling). "
        "The <b>Best Month to Sell</b> column shows which calendar month historically has the "
        "highest flat rate — Cordier's seasonal selling window.",
        "cordier",
    )

    if asset_type == "Indices":
        scan_syms = INDICES_ONLY
    elif asset_type == "Stocks":
        scan_syms = ALL_STOCKS_ONLY[:40]  # limit for speed
        st.caption("⚡ Showing first 40 stocks for speed. Use Single Stock view for full analysis.")
    else:
        scan_syms = INDICES_ONLY + ALL_STOCKS_ONLY[:35]
        st.caption("⚡ Showing indices + top 35 stocks. Use Single Stock view for full analysis.")

    progress = st.progress(0, text="Loading historical data…")
    scan_rows = []
    for idx, sym in enumerate(scan_syms):
        summ = big_move_summary(NSE_FNO[sym], threshold)
        if summ:
            # Find best month to sell (highest flat rate)
            mdf = summ["mret"].to_frame("ret")
            mdf["month"] = mdf.index.month
            month_flat = mdf.groupby("month")["ret"].apply(lambda x: (x.abs()<threshold).mean()*100)
            best_month_num = int(month_flat.idxmax()) if not month_flat.empty else 0
            best_month_name = MONTH_NAMES[best_month_num-1] if best_month_num else "—"
            best_month_flat = round(float(month_flat.max()),1) if not month_flat.empty else 0
            worst_month_num = int(month_flat.idxmin()) if not month_flat.empty else 0
            worst_month_name = MONTH_NAMES[worst_month_num-1] if worst_month_num else "—"

            scan_rows.append({
                "Symbol":          sym,
                "Years":           summ["years"],
                "Months":          summ["n"],
                f"Up>{threshold}% (%)":  summ["up_pct"],
                f"Down>{threshold}% (%)":summ["dn_pct"],
                f"Flat<{threshold}% (%)":summ["flat_pct"],
                "Mean Ret (%)":    summ["mean"],
                "Best Month (%)":  summ["best"],
                "Worst Month (%)": summ["worst"],
                "Sell Month":      f"{best_month_name} ({best_month_flat:.0f}%)",
                "Avoid Month":     worst_month_name,
                "Sell Score":      round(summ["flat_pct"] - summ["dn_pct"] - summ["up_pct"]*0.5, 1),
            })
        progress.progress((idx+1)/len(scan_syms), text=f"Loading {sym}…")
        time.sleep(0.01)

    progress.empty()

    if not scan_rows:
        st.error("No data loaded.")
        return

    sdf = pd.DataFrame(scan_rows).sort_values("Sell Score", ascending=False)

    flat_col   = f"Flat<{threshold}% (%)"
    up_col     = f"Up>{threshold}% (%)"
    down_col   = f"Down>{threshold}% (%)"

    def _style_scanner(row):
        out = []
        for c in row.index:
            if c == flat_col:
                out.append("background-color:#dcfce7;color:#14532d;font-weight:700" if row[c]>=70 else
                           "background-color:#fef9c3;color:#78350f" if row[c]>=55 else
                           "background-color:#fee2e2;color:#7f1d1d;font-weight:700")
            elif c in [up_col, down_col]:
                out.append("background-color:#fee2e2;color:#7f1d1d;font-weight:700" if row[c]>=30 else
                           "background-color:#fef9c3;color:#78350f" if row[c]>=20 else
                           "color:#14532d")
            elif c == "Sell Score":
                out.append("background-color:#dcfce7;color:#14532d;font-weight:700" if row[c]>=30 else
                           "background-color:#fef9c3;color:#78350f" if row[c]>=10 else
                           "background-color:#fee2e2;color:#7f1d1d" if row[c]<0 else
                           "color:#1e293b")
            elif c == "Worst Month (%)":
                out.append("color:#dc2626;font-weight:700" if row[c] <= -threshold else "color:#1e293b")
            elif c == "Best Month (%)":
                out.append("color:#15803d;font-weight:700" if row[c] >= threshold else "color:#1e293b")
            else:
                out.append("color:#1e293b")
        return out

    st.dataframe(
        sdf.style.apply(_style_scanner, axis=1)
        .format({
            flat_col:         "{:.1f}%",
            up_col:           "{:.1f}%",
            down_col:         "{:.1f}%",
            "Mean Ret (%)":   "{:+.2f}%",
            "Best Month (%)": "{:+.1f}%",
            "Worst Month (%)":"{:+.1f}%",
            "Sell Score":     "{:.1f}",
        }),
        use_container_width=True, height=560, hide_index=True,
    )

    st.markdown("---")

    # Legend
    st.markdown("""
    <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:.82rem;color:#1e293b">
      <span style="background:#dcfce7;color:#14532d;padding:3px 10px;border-radius:8px;font-weight:700">🟢 Flat ≥70% — safest for premium selling</span>
      <span style="background:#fef9c3;color:#78350f;padding:3px 10px;border-radius:8px">🟡 Flat 55–69% — moderate risk</span>
      <span style="background:#fee2e2;color:#7f1d1d;padding:3px 10px;border-radius:8px;font-weight:700">🔴 Flat <55% or move ≥30% — avoid naked selling</span>
    </div>
    <div style="margin-top:8px;font-size:.82rem;color:#475569">
      <b>Sell Score</b> = Flat% − Down% − Up%×0.5 — higher is better for premium selling<br>
      <b>Sell Month</b> = historically safest calendar month to sell options (highest flat rate)<br>
      <b>Avoid Month</b> = historically most volatile calendar month
    </div>
    """, unsafe_allow_html=True)

    # Summary charts
    st.markdown("---")
    st.subheader("📊 Comparative Charts")
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        # Flat rate comparison
        top20 = sdf.head(20)
        fig_flat = go.Figure(go.Bar(
            x=top20["Symbol"], y=top20[flat_col],
            marker_color=["#15803d" if v>=70 else "#ca8a04" if v>=55 else "#dc2626" for v in top20[flat_col]],
            text=[f"{v:.0f}%" for v in top20[flat_col]],
            textposition="outside",
            textfont=dict(color="#1e293b", size=10),
        ))
        fig_flat.add_hline(y=70, line=dict(color="#15803d",dash="dash",width=1.5))
        fig_flat.update_layout(
            title=f"Top 20 Stocks — Flat<{threshold}% Rate (Best for Selling)",
            yaxis_title="Flat Rate (%)", template="plotly_white", height=340,
            xaxis_tickangle=-45, font=dict(color="#1e293b"),
        )
        st.plotly_chart(fig_flat, use_container_width=True)

    with col_ch2:
        # Crash rate comparison — sorted by most dangerous
        worst20 = sdf.sort_values(down_col, ascending=False).head(20)
        fig_dn = go.Figure(go.Bar(
            x=worst20["Symbol"], y=worst20[down_col],
            marker_color=["#dc2626" if v>=30 else "#f97316" if v>=20 else "#fbbf24" for v in worst20[down_col]],
            text=[f"{v:.0f}%" for v in worst20[down_col]],
            textposition="outside",
            textfont=dict(color="#1e293b", size=10),
        ))
        fig_dn.update_layout(
            title=f"Most Dangerous — Down>{threshold}% Rate (Avoid Naked Puts)",
            yaxis_title="Crash Rate (%)", template="plotly_white", height=340,
            xaxis_tickangle=-45, font=dict(color="#1e293b"),
        )
        st.plotly_chart(fig_dn, use_container_width=True)

    explain(
        "Left chart: stocks with the highest flat rate are the best candidates for short strangles. "
        "Right chart: stocks with the highest crash rate are the most dangerous for short puts. "
        "Cordier: 'Never sell naked puts on stocks that appear in the right chart.' "
        "For those stocks, always use bull put spreads (defined risk) instead.",
        "cordier",
    )
