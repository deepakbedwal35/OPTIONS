"""
Module: 📊  Option Chain Viewer
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
    st.title("📊 Live Option Chain Viewer")

    explain(
        "Real NSE option chains fetched via yfinance. "
        "IV is computed from the market mid-price using Black-Scholes inverse (Newton's method). "
        "The <b>Top-5 OI Strikes</b> and <b>Top-5 Volume Strikes</b> sections show the "
        "most active strikes with their exact distance from current spot — "
        "these are the market's key support/resistance levels and the best strikes for premium selling.",
        "natenberg",
    )

    sym_c   = st.selectbox("Select F&O stock / index", list(NSE_FNO.keys()))
    stats_c = fetch_spot_iv(NSE_FNO[sym_c])
    chain, expiry = fetch_option_chain(NSE_FNO[sym_c])

    if stats_c:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spot",          f"₹{stats_c['spot']:,.2f}")
        c2.metric("HV20",          f"{stats_c['hv20']:.1f}%")
        c3.metric("IV (approx)",   f"{stats_c['iv']:.1f}%")
        c4.metric("IVR",           f"{stats_c['ivr']:.0f}%")

    if chain is None:
        st.warning(
            "Option chain not available for this symbol via yfinance. "
            "Indices (NIFTY/BANKNIFTY) have limited free data — try a stock like RELIANCE or TCS."
        )
        return

    st.caption(f"📅 Expiry loaded: **{expiry}**")
    S   = stats_c["spot"] if stats_c else 1000.0
    r_c = 0.065
    T_c = max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365, 0.001)

    # ── enrich chain with Greeks + distance ──────────────────
    def enrich_chain(df, opt_type):
        rows = []
        for _, row in df.iterrows():
            K      = float(row["strike"])
            bid    = float(row.get("bid", 0) or 0)
            ask    = float(row.get("ask", 0) or 0)
            last   = float(row.get("lastPrice", 0) or 0)
            mid    = (bid + ask) / 2 if (bid + ask) > 0 else last
            oi     = int(row.get("openInterest", 0) or 0)
            vol    = int(row.get("volume", 0) or 0)
            mkt_iv = float(row.get("impliedVolatility", 0) or 0) * 100
            if mkt_iv == 0 and mid > 0.5:
                mkt_iv = implied_vol(S, K, T_c, r_c, mid, opt_type) * 100
            iv_use = max(mkt_iv, 1.0) / 100
            g      = bs_greeks(S, K, T_c, r_c, iv_use, opt_type)
            dist_pct  = (K / S - 1) * 100          # +ve = above spot, -ve = below
            dist_pts  = K - S                       # absolute ₹ distance
            rows.append({
                "Strike":           K,
                "Dist from Spot %": round(dist_pct, 2),
                "Dist from Spot ₹": round(dist_pts, 0),
                "Mid Price":        round(mid, 2),
                "IV (%)":           round(mkt_iv, 2),
                "Delta":            round(g["delta"], 3),
                "Theta/day":        round(g["theta"], 3),
                "Vega/%IV":         round(g["vega"],  3),
                "Prob ITM (%)":     round(g["prob_itm"] * 100, 1),
                "Open Interest":    oi,
                "Volume":           vol,
            })
        return pd.DataFrame(rows)

    calls_df = enrich_chain(chain.calls, "call")
    puts_df  = enrich_chain(chain.puts,  "put")

    # ─────────────────────────────────────────────────────────
    # TOP-5 STRIKES BY OI  (calls above spot, puts below spot)
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Top 5 Strikes by Open Interest")
    explain(
        "<b>Open Interest (OI)</b> = number of outstanding contracts at a strike. "
        "High OI means large institutional presence. The strike with the highest OI acts as a "
        "<b>magnet for price</b> near expiry — market makers hedge there, creating gravity. "
        "Call OI above spot = resistance (call wall). Put OI below spot = support (put wall). "
        "Cordier: sell the strangle OUTSIDE the max-OI strikes — you want both walls on your side.",
        "cordier",
    )

    col_oi_c, col_oi_p = st.columns(2)

    def make_top5_table(df, side, by_col, spot):
        """Return styled Top-5 df filtered to the correct side of spot."""
        if df.empty or by_col not in df.columns:
            return pd.DataFrame()
        if side == "call":
            filtered = df[df["Strike"] >= spot * 0.98].copy()
        else:
            filtered = df[df["Strike"] <= spot * 1.02].copy()
        top5 = filtered.nlargest(5, by_col).copy()
        top5 = top5[[
            "Strike", "Dist from Spot %", "Dist from Spot ₹",
            "Mid Price", "IV (%)", "Delta", by_col,
        ]].reset_index(drop=True)
        top5.index = top5.index + 1   # rank 1–5
        # Direction label
        top5["Direction"] = top5["Dist from Spot %"].apply(
            lambda v: f"⬆ {abs(v):.1f}% above" if v >= 0 else f"⬇ {abs(v):.1f}% below"
        )
        return top5

    with col_oi_c:
        st.markdown("**📞 Top 5 CALL strikes (OI)**")
        top5_call_oi = make_top5_table(calls_df, "call", "Open Interest", S)
        if not top5_call_oi.empty:
            st.dataframe(
                top5_call_oi.style
                .applymap(lambda v: "background-color:#dbeafe;font-weight:600"
                          if isinstance(v, (int, float)) and v == top5_call_oi["Open Interest"].max()
                          else "", subset=["Open Interest"])
                .format({
                    "Strike":           "₹{:,.0f}",
                    "Dist from Spot %": "{:+.2f}%",
                    "Dist from Spot ₹": "₹{:+,.0f}",
                    "Mid Price":        "₹{:.2f}",
                    "IV (%)":           "{:.1f}%",
                    "Delta":            "{:.3f}",
                    "Open Interest":    "{:,.0f}",
                }),
                use_container_width=True,
            )
            best = top5_call_oi.iloc[0]
            explain(
                f"🏆 <b>Highest Call OI: Strike ₹{best['Strike']:,.0f}</b> "
                f"({best['Direction']}, IV={best['IV (%)']:.1f}%, Δ={best['Delta']:.3f}). "
                f"This is the <b>call wall</b> — major resistance. "
                f"If selling a strangle, place your short call AT or ABOVE this strike.",
                "mcmillan",
            )

    with col_oi_p:
        st.markdown("**📟 Top 5 PUT strikes (OI)**")
        top5_put_oi = make_top5_table(puts_df, "put", "Open Interest", S)
        if not top5_put_oi.empty:
            st.dataframe(
                top5_put_oi.style
                .applymap(lambda v: "background-color:#fee2e2;font-weight:600"
                          if isinstance(v, (int, float)) and v == top5_put_oi["Open Interest"].max()
                          else "", subset=["Open Interest"])
                .format({
                    "Strike":           "₹{:,.0f}",
                    "Dist from Spot %": "{:+.2f}%",
                    "Dist from Spot ₹": "₹{:+,.0f}",
                    "Mid Price":        "₹{:.2f}",
                    "IV (%)":           "{:.1f}%",
                    "Delta":            "{:.3f}",
                    "Open Interest":    "{:,.0f}",
                }),
                use_container_width=True,
            )
            best = top5_put_oi.iloc[0]
            explain(
                f"🏆 <b>Highest Put OI: Strike ₹{best['Strike']:,.0f}</b> "
                f"({best['Direction']}, IV={best['IV (%)']:.1f}%, Δ={best['Delta']:.3f}). "
                f"This is the <b>put wall</b> — major support. "
                f"If selling a strangle, place your short put AT or BELOW this strike.",
                "mcmillan",
            )

    # ─────────────────────────────────────────────────────────
    # TOP-5 STRIKES BY VOLUME
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Top 5 Strikes by Volume (Today's Activity)")
    explain(
        "<b>Volume</b> = contracts traded TODAY. High volume = smart money is active at this strike RIGHT NOW. "
        "Volume spikes at specific strikes signal institutional positioning — "
        "they may be hedging a large stock position (puts) or writing covered calls. "
        "Natenberg: unusually high volume at an OTM strike often precedes a directional move toward that strike.",
        "natenberg",
    )

    col_vol_c, col_vol_p = st.columns(2)

    with col_vol_c:
        st.markdown("**📞 Top 5 CALL strikes (Volume)**")
        top5_call_vol = make_top5_table(calls_df, "call", "Volume", S)
        if not top5_call_vol.empty:
            st.dataframe(
                top5_call_vol.style
                .applymap(lambda v: "background-color:#dcfce7;font-weight:600"
                          if isinstance(v, (int, float)) and v == top5_call_vol["Volume"].max()
                          else "", subset=["Volume"])
                .format({
                    "Strike":           "₹{:,.0f}",
                    "Dist from Spot %": "{:+.2f}%",
                    "Dist from Spot ₹": "₹{:+,.0f}",
                    "Mid Price":        "₹{:.2f}",
                    "IV (%)":           "{:.1f}%",
                    "Delta":            "{:.3f}",
                    "Volume":           "{:,.0f}",
                }),
                use_container_width=True,
            )

    with col_vol_p:
        st.markdown("**📟 Top 5 PUT strikes (Volume)**")
        top5_put_vol = make_top5_table(puts_df, "put", "Volume", S)
        if not top5_put_vol.empty:
            st.dataframe(
                top5_put_vol.style
                .applymap(lambda v: "background-color:#dcfce7;font-weight:600"
                          if isinstance(v, (int, float)) and v == top5_put_vol["Volume"].max()
                          else "", subset=["Volume"])
                .format({
                    "Strike":           "₹{:,.0f}",
                    "Dist from Spot %": "{:+.2f}%",
                    "Dist from Spot ₹": "₹{:+,.0f}",
                    "Mid Price":        "₹{:.2f}",
                    "IV (%)":           "{:.1f}%",
                    "Delta":            "{:.3f}",
                    "Volume":           "{:,.0f}",
                }),
                use_container_width=True,
            )

    # ─────────────────────────────────────────────────────────
    # OI + VOLUME COMBINED BAR CHART
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 OI + Volume Distribution")

    if not calls_df.empty and not puts_df.empty:
        fig_oi = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Call OI by Strike", "Put OI by Strike",
                "Call Volume by Strike", "Put Volume by Strike",
            ],
        )
        # OI bars
        fig_oi.add_trace(go.Bar(
            x=calls_df["Strike"], y=calls_df["Open Interest"],
            marker_color="#60a5fa", name="Call OI",
        ), row=1, col=1)
        fig_oi.add_trace(go.Bar(
            x=puts_df["Strike"], y=puts_df["Open Interest"],
            marker_color="#f87171", name="Put OI",
        ), row=1, col=2)
        # Volume bars
        fig_oi.add_trace(go.Bar(
            x=calls_df["Strike"], y=calls_df["Volume"],
            marker_color="#34d399", name="Call Vol",
        ), row=2, col=1)
        fig_oi.add_trace(go.Bar(
            x=puts_df["Strike"], y=puts_df["Volume"],
            marker_color="#fb923c", name="Put Vol",
        ), row=2, col=2)
        # Spot lines on all subplots
        for r, c in [(1,1),(1,2),(2,1),(2,2)]:
            fig_oi.add_vline(
                x=S, line=dict(color="black", dash="dash", width=1.5),
                row=r, col=c,
            )
        fig_oi.update_layout(
            template="plotly_white", height=500,
            showlegend=False,
            title_text=f"{sym_c} — OI & Volume Distribution (Expiry: {expiry})",
        )
        st.plotly_chart(fig_oi, use_container_width=True)
        explain(
            "Black dashed line = current spot. "
            "The tallest call OI bar = <b>call wall (resistance)</b>. "
            "The tallest put OI bar = <b>put wall (support)</b>. "
            "High volume bars (bottom row) show TODAY'S institutional activity — "
            "these strikes are being actively hedged or traded right now. "
            "Ideal iron condor: short call ≥ call wall, short put ≤ put wall.",
            "cordier",
        )

    # ─────────────────────────────────────────────────────────
    # FULL CHAIN TABLES
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📞 Full Call Chain", "📟 Full Put Chain", "📈 IV Skew"])

    def highlight_atm(df):
        def row_color(row):
            mono = abs(row["Dist from Spot %"])
            if mono < 1.5: return ["background-color:#dbeafe"] * len(row)
            if mono < 4:   return ["background-color:#f0fdf4"] * len(row)
            return [""] * len(row)
        return df.style.apply(row_color, axis=1).format({
            "Strike":           "₹{:,.0f}",
            "Dist from Spot %": "{:+.2f}%",
            "Dist from Spot ₹": "₹{:+,.0f}",
            "Mid Price":        "₹{:.2f}",
            "IV (%)":           "{:.1f}%",
            "Delta":            "{:.3f}",
            "Theta/day":        "₹{:.3f}",
            "Vega/%IV":         "₹{:.3f}",
            "Prob ITM (%)":     "{:.1f}%",
            "Open Interest":    "{:,.0f}",
            "Volume":           "{:,.0f}",
        })

    with tab1:
        explain(
            "Blue rows = ATM (±1.5%). Green rows = near-ATM (±4%). "
            "<b>Dist from Spot %</b> tells you exactly how far above/below spot each strike is. "
            "<b>Cordier rule:</b> only sell call strikes where Dist > +5% AND Delta < 0.20.",
            "cordier",
        )
        st.dataframe(highlight_atm(calls_df), use_container_width=True, height=460)

    with tab2:
        explain(
            "OTM puts (Dist from Spot < −5%) always show higher IV than equivalent OTM calls — "
            "this is the put skew. Natenberg: when OTM put IV > OTM call IV by >3%, sell puts. "
            "The <b>Dist from Spot ₹</b> column shows the exact cushion in rupees before your short put is threatened.",
            "natenberg",
        )
        st.dataframe(highlight_atm(puts_df), use_container_width=True, height=460)

    with tab3:
        common = set(calls_df["Strike"].tolist()) & set(puts_df["Strike"].tolist())
        if common:
            c_iv   = calls_df[calls_df["Strike"].isin(common)].set_index("Strike")["IV (%)"]
            p_iv   = puts_df [puts_df ["Strike"].isin(common)].set_index("Strike")["IV (%)"]
            sk_df  = pd.DataFrame({"Call IV": c_iv, "Put IV": p_iv}).sort_index()
            sk_df["Skew (Put − Call)"] = sk_df["Put IV"] - sk_df["Call IV"]
            fig_sk = go.Figure()
            fig_sk.add_trace(go.Scatter(x=sk_df.index, y=sk_df["Call IV"],
                                        name="Call IV", line=dict(color="#60a5fa", width=2)))
            fig_sk.add_trace(go.Scatter(x=sk_df.index, y=sk_df["Put IV"],
                                        name="Put IV",  line=dict(color="#f87171", width=2)))
            fig_sk.add_trace(go.Bar(x=sk_df.index, y=sk_df["Skew (Put − Call)"],
                                    name="Skew (P−C)", marker_color="#a78bfa", opacity=0.5,
                                    yaxis="y2"))
            fig_sk.add_vline(x=S, line=dict(color="black", dash="dash"), annotation_text="Spot")
            fig_sk.update_layout(
                title="Volatility Smile + Skew",
                xaxis_title="Strike (₹)", yaxis_title="IV (%)",
                yaxis2=dict(title="Skew (P−C) %", overlaying="y", side="right"),
                template="plotly_white", height=360,
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_sk, use_container_width=True)
            explain(
                "Purple bars (right axis) show the skew at each strike. "
                "When bars are tall on the left (OTM puts), the market is paying a fear premium. "
                "Natenberg: skew > 3% at −10% OTM put = strong signal to sell that put. "
                "Risk reversal trade: sell the expensive OTM put, buy the cheap OTM call.",
                "natenberg",
            )


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 3 — PAYOFF BUILDER  ███
    # ═══════════════════════════════════════════════════════════════
