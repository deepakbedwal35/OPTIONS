"""
Module: 🌊  Volatility Lab
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
from config import RISK_FREE_RATE, NSE_STRIKE_STEP, DEFAULT_STRIKE_STEP


def render():
    st.title("🌊 Volatility Lab — Natenberg Framework")

    explain(
        "Natenberg: 'You are not trading the stock — you are trading volatility.' "
        "IV measures the market's expectation of future moves. "
        "HV measures what the stock actually did. "
        "When IV >> HV: sell options (market overpaying). When IV << HV: buy options (market underpaying). "
        "This lab gives you every tool Natenberg uses.",
        "natenberg",
    )

    sym_vl = st.selectbox("Stock", list(NSE_FNO.keys()), key="vl_sym")
    with st.spinner("Loading..."):
        stats_vl  = fetch_spot_iv(NSE_FNO[sym_vl])
        hist_vl   = fetch_history(NSE_FNO[sym_vl], period="2y")

    if not stats_vl:
        st.error("No data")
        return

    m = st.columns(5)
    m[0].metric("Spot",       f"₹{stats_vl['spot']:,.2f}")
    m[1].metric("HV10",       f"{stats_vl['hv10']:.1f}%")
    m[2].metric("HV20",       f"{stats_vl['hv20']:.1f}%")
    m[3].metric("IV approx",  f"{stats_vl['iv']:.1f}%")
    m[4].metric("IVR",        f"{stats_vl['ivr']:.0f}%",
                delta="Sell signal" if stats_vl["ivr"]>50 else "Wait",
                delta_color="normal")

    iv_hv_ratio = stats_vl["iv"] / max(stats_vl["hv20"], 1)
    if iv_hv_ratio > 1.3:
        explain(
            f"IV/HV ratio = <b>{iv_hv_ratio:.2f}x</b> — options are pricing {(iv_hv_ratio-1)*100:.0f}% more vol than realised. "
            f"This is Natenberg's primary sell signal. The options market is overestimating future moves. "
            f"Collect this premium: sell strangles or iron condors.",
            "safe",
        )
    elif iv_hv_ratio < 0.85:
        explain(
            f"IV/HV ratio = <b>{iv_hv_ratio:.2f}x</b> — options are CHEAP vs realised vol. "
            f"Natenberg: buy options when IV < HV. Long straddle or backspread makes sense.",
            "warning",
        )
    else:
        explain(
            f"IV/HV ratio = <b>{iv_hv_ratio:.2f}x</b> — fairly priced. "
            f"No strong edge on either side. Wait for a better opportunity.",
            "explain",
        )

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Vol cone
        hv_vals = [stats_vl["hv10"], stats_vl["hv20"], stats_vl["hv30"],
                   stats_vl["hv60"], stats_vl["hv90"]]
        labels  = ["HV10","HV20","HV30","HV60","HV90"]
        fig_vc  = go.Figure()
        fig_vc.add_trace(go.Bar(x=labels, y=hv_vals, name="Historical Vol",
                                marker_color="#60a5fa", opacity=0.75))
        fig_vc.add_hline(y=stats_vl["iv"], line=dict(color="#ef4444", width=2, dash="dash"),
                         annotation_text=f"IV≈{stats_vl['iv']:.1f}%")
        fig_vc.update_layout(title="Vol Cone — HV at Different Lookbacks",
                             yaxis_title="Vol (%)", template="plotly_white", height=280)
        st.plotly_chart(fig_vc, use_container_width=True)
        explain(
            "If the red dashed IV line is <b>above all bars</b> → options are expensive vs every lookback → sell. "
            "If IV is below HV10 → options are very cheap, even recent history shows more vol → buy. "
            "Natenberg recommends comparing IV to 30-day HV as the primary reference.",
            "natenberg",
        )

    with col_v2:
        if hist_vl is not None and len(hist_vl) > 25:
            ret_vl   = np.log(hist_vl["Close"]/hist_vl["Close"].shift(1)).dropna()
            roll_hv  = ret_vl.rolling(20).std() * np.sqrt(252) * 100
            roll_hv  = roll_hv.dropna().tail(252)
            fig_rv   = go.Figure()
            fig_rv.add_trace(go.Scatter(x=roll_hv.index, y=roll_hv.values,
                                        name="Rolling HV20", line=dict(color="#22c55e", width=1.8)))
            fig_rv.add_hline(y=stats_vl["iv"], line=dict(color="#ef4444", dash="dash"),
                             annotation_text="IV")
            fig_rv.update_layout(title="Realised Vol — Rolling 20d (1 year)",
                                 yaxis_title="HV (%)", template="plotly_white", height=280)
            st.plotly_chart(fig_rv, use_container_width=True)
            explain(
                "The green line is realised HV over time. High spikes = past events (earnings, macro). "
                "If the red IV line is ABOVE most of the green line, the market is pricing in more fear than history justifies. "
                "Natenberg: track HV over time — mean reversion is the vol trader's edge.",
                "natenberg",
            )

    if hist_vl is not None:
        st.subheader("Price Chart with ±1σ / ±2σ Expected Move Bands")
        price_d  = hist_vl["Close"].tail(120)
        ret_d    = np.log(hist_vl["Close"]/hist_vl["Close"].shift(1)).dropna()
        daily_sd = ret_d.tail(20).std()
        bands    = {"+2σ": price_d*(1+2*daily_sd), "+1σ": price_d*(1+daily_sd),
                    "-1σ": price_d*(1-daily_sd),   "-2σ": price_d*(1-2*daily_sd)}
        fig_bb   = go.Figure()
        fig_bb.add_trace(go.Scatter(x=price_d.index, y=price_d.values,
                                    name="Price", line=dict(color="#1e293b", width=2)))
        colors = {"#f87171":"rgba(248,113,113,0.15)", "#fbbf24":"rgba(251,191,36,0.1)"}
        for (label, series), (c_line, c_fill) in zip(
            [("+2σ", bands["+2σ"]), ("-2σ", bands["-2σ"])],
            [(("#ef4444","rgba(239,68,68,0.08)")), (("#ef4444","rgba(239,68,68,0.08)"))],
        ):
            fig_bb.add_trace(go.Scatter(x=series.index, y=series.values, name=label,
                                        line=dict(color="#ef4444", dash="dash", width=1)))
        for label, series in [("+1σ", bands["+1σ"]), ("-1σ", bands["-1σ"])]:
            fig_bb.add_trace(go.Scatter(x=series.index, y=series.values, name=label,
                                        line=dict(color="#f59e0b", dash="dot", width=1)))
        fig_bb.update_layout(title=f"{sym_vl} — Price with Daily ±1σ/±2σ Bands",
                             template="plotly_white", height=320)
        st.plotly_chart(fig_bb, use_container_width=True)
        explain(
            "The ±2σ bands (red dashed) represent the daily expected move. "
            "When price is near the ±2σ band, it often mean-reverts (Natenberg: overextension). "
            "Cordier uses the 2σ distance as his minimum strike-selection criterion: "
            "sell options only at or beyond the 2σ level from current price.",
            "cordier",
        )



    # ─────────────────────────────────────────────────────────────
    # IV SMILE / SKEW CHART  —  Natenberg Chapter 14
    # ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("😊 IV Smile & Skew — Natenberg Chapter 14")

    explain(
        "Natenberg's key insight: <b>IV is not flat across strikes</b>. "
        "Real markets show a 'smile' (OTM options are pricier than ATM) or a 'skew' (puts costlier than calls). "
        "NSE equity options show a <b>negative skew</b> — OTM puts are always more expensive than OTM calls "
        "because institutions buy puts for portfolio protection. "
        "The skew tells you: (1) which side the market fears more, "
        "(2) which options are rich to sell, and (3) where your BS theoretical price is wrong.",
        "natenberg",
    )

    with st.spinner("Computing IV smile from live chain..."):
        chain_vl, expiry_vl = fetch_option_chain(NSE_FNO[sym_vl])

    S_vl   = stats_vl["spot"]
    iv_vl  = stats_vl["iv"] / 100
    r_vl   = RISK_FREE_RATE
    step_vl = NSE_STRIKE_STEP.get(sym_vl, DEFAULT_STRIKE_STEP)

    # ── Try to build smile from live chain ────────────────────
    call_smile_rows = []
    put_smile_rows  = []
    chain_ok = False

    if chain_vl is not None:
        try:
            today = datetime.today()
            T_vl  = max((datetime.strptime(expiry_vl, "%Y-%m-%d") - today).days / 365.0, 0.001)

            calls_df = chain_vl.calls[["strike", "lastPrice", "bid", "ask", "volume", "openInterest"]].copy()
            puts_df  = chain_vl.puts[["strike",  "lastPrice", "bid", "ask", "volume", "openInterest"]].copy()

            # Use mid price where bid/ask available, else lastPrice
            calls_df["mid"] = np.where(
                (calls_df["bid"] > 0) & (calls_df["ask"] > 0),
                (calls_df["bid"] + calls_df["ask"]) / 2,
                calls_df["lastPrice"],
            )
            puts_df["mid"] = np.where(
                (puts_df["bid"] > 0) & (puts_df["ask"] > 0),
                (puts_df["bid"] + puts_df["ask"]) / 2,
                puts_df["lastPrice"],
            )

            # Compute IV for each strike using bisection (implied_vol from utils)
            for _, row in calls_df.iterrows():
                K = float(row["strike"])
                mkt = float(row["mid"])
                if mkt < 0.10 or K <= 0: continue
                moneyness = (K - S_vl) / S_vl * 100
                if abs(moneyness) > 25: continue          # skip very deep ITM/OTM
                try:
                    iv_c = implied_vol(S_vl, K, T_vl, r_vl, mkt, "call")
                    if 0.01 < iv_c < 5.0:
                        call_smile_rows.append({
                            "Strike": int(K),
                            "Moneyness (%)": round(moneyness, 1),
                            "IV (%)": round(iv_c * 100, 2),
                            "Mid (₹)": round(mkt, 2),
                            "Volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                            "OI": int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0,
                            "Type": "Call",
                        })
                except Exception:
                    pass

            for _, row in puts_df.iterrows():
                K = float(row["strike"])
                mkt = float(row["mid"])
                if mkt < 0.10 or K <= 0: continue
                moneyness = (K - S_vl) / S_vl * 100
                if abs(moneyness) > 25: continue
                try:
                    iv_p = implied_vol(S_vl, K, T_vl, r_vl, mkt, "put")
                    if 0.01 < iv_p < 5.0:
                        put_smile_rows.append({
                            "Strike": int(K),
                            "Moneyness (%)": round(moneyness, 1),
                            "IV (%)": round(iv_p * 100, 2),
                            "Mid (₹)": round(mkt, 2),
                            "Volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                            "OI": int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0,
                            "Type": "Put",
                        })
                except Exception:
                    pass

            chain_ok = len(call_smile_rows) >= 3 and len(put_smile_rows) >= 3

        except Exception:
            chain_ok = False

    # ── Fallback: synthetic smile using SVI-like parametric model ──
    if not chain_ok:
        st.caption("⚠️ Live chain unavailable — showing **synthetic IV smile** using SVI parametric model (Gatheral). Directionally correct for NSE equity skew patterns.")
        T_vl  = 30 / 365.0
        # SVI params calibrated to typical NSE equity skew
        # v(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
        # a=ATM var, b=wings slope, rho=skew(-0.3 typical NSE), m=0, sigma=smoothing
        a     = iv_vl**2
        b     = 0.40
        rho   = -0.30     # negative skew: puts more expensive
        m     = 0.0
        sigma = 0.15
        moneyness_range = np.linspace(-0.20, 0.20, 40)   # -20% to +20%
        for pct in moneyness_range:
            K = S_vl * (1 + pct)
            k = np.log(K / S_vl)            # log-moneyness
            # SVI total variance
            w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            w = max(w, 0.0001)
            iv_syn = np.sqrt(w / T_vl) * 100
            if 1 < iv_syn < 200:
                row = {
                    "Strike": int(round(K / step_vl) * step_vl),
                    "Moneyness (%)": round(pct * 100, 1),
                    "IV (%)": round(iv_syn, 2),
                    "Mid (₹)": round(bs_price(S_vl, K, T_vl, r_vl, np.sqrt(w/T_vl), "call" if pct >= 0 else "put"), 2),
                    "Volume": 0,
                    "OI": 0,
                    "Type": "Call" if pct >= 0 else "Put",
                }
                if pct >= 0:
                    call_smile_rows.append(row)
                else:
                    put_smile_rows.append(row)

    # ── Chart 1 — IV Smile (calls + puts on same moneyness axis) ──
    call_df_sm = pd.DataFrame(call_smile_rows).sort_values("Moneyness (%)")
    put_df_sm  = pd.DataFrame(put_smile_rows).sort_values("Moneyness (%)")

    fig_smile = go.Figure()

    # ATM flat line (what BS assumes — constant IV)
    atm_iv = stats_vl["iv"]
    x_range = sorted(set(
        list(call_df_sm["Moneyness (%)"].values) +
        list(put_df_sm["Moneyness (%)"].values)
    ))
    fig_smile.add_trace(go.Scatter(
        x=x_range, y=[atm_iv] * len(x_range),
        mode="lines", name="Flat IV (BS assumes this)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    ))

    # Put IV curve
    if len(put_df_sm):
        fig_smile.add_trace(go.Scatter(
            x=put_df_sm["Moneyness (%)"], y=put_df_sm["IV (%)"],
            mode="lines+markers", name="Put IV",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="<b>Put</b><br>Strike: ₹%{customdata[0]:,}<br>Moneyness: %{x:.1f}%<br>IV: %{y:.1f}%<br>Mid: ₹%{customdata[1]}<extra></extra>",
            customdata=put_df_sm[["Strike","Mid (₹)"]].values,
        ))

    # Call IV curve
    if len(call_df_sm):
        fig_smile.add_trace(go.Scatter(
            x=call_df_sm["Moneyness (%)"], y=call_df_sm["IV (%)"],
            mode="lines+markers", name="Call IV",
            line=dict(color="#4f46e5", width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="<b>Call</b><br>Strike: ₹%{customdata[0]:,}<br>Moneyness: %{x:.1f}%<br>IV: ₹%{customdata[1]}<extra></extra>",
            customdata=call_df_sm[["Strike","Mid (₹)"]].values,
        ))

    # ATM vertical line
    fig_smile.add_vline(x=0, line_width=1.5, line_color="#22c55e",
                        annotation_text=f"ATM ₹{S_vl:,.0f}", annotation_position="top")

    fig_smile.update_layout(
        title=f"{sym_vl} — IV Smile ({expiry_vl if chain_ok else '~30d synthetic'})",
        xaxis_title="Moneyness (% from spot)",
        yaxis_title="Implied Volatility (%)",
        template="plotly_white", height=380,
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    # Shade OTM put region (negative skew premium)
    fig_smile.add_vrect(x0=-25, x1=0, fillcolor="rgba(239,68,68,0.04)",
                        line_width=0, annotation_text="Put skew zone",
                        annotation_position="top left",
                        annotation_font=dict(size=11, color="#ef4444"))
    fig_smile.add_vrect(x0=0, x1=25, fillcolor="rgba(79,70,229,0.04)",
                        line_width=0, annotation_text="Call wing",
                        annotation_position="top right",
                        annotation_font=dict(size=11, color="#4f46e5"))

    st.plotly_chart(fig_smile, use_container_width=True)

    # ── Skew metrics ──────────────────────────────────────────
    sk1, sk2, sk3, sk4 = st.columns(4)

    # 25-delta skew: put25d IV - call25d IV
    try:
        put25_iv  = float(put_df_sm[put_df_sm["Moneyness (%)"].between(-12, -3)]["IV (%)"].mean())
        call25_iv = float(call_df_sm[call_df_sm["Moneyness (%)"].between(3, 12)]["IV (%)"].mean())
        skew_25d  = round(put25_iv - call25_iv, 2)
        atm_ref   = stats_vl["iv"]
        put_wing_premium  = round(put25_iv  - atm_ref, 2)
        call_wing_premium = round(call25_iv - atm_ref, 2)
    except Exception:
        skew_25d = put_wing_premium = call_wing_premium = None
        put25_iv = call25_iv = atm_ref = stats_vl["iv"]

    sk1.metric("25Δ Skew (P-C)", f"{skew_25d:+.1f}%" if skew_25d is not None else "—",
               help="Put IV minus Call IV at ~25-delta. Negative = normal (puts more expensive). Strongly negative = fear mode.")
    sk2.metric("ATM IV", f"{atm_ref:.1f}%",
               help="Implied vol at the money — the Natenberg reference point.")
    sk3.metric("OTM Put Premium", f"{put_wing_premium:+.1f}%" if put_wing_premium is not None else "—",
               help="How much more OTM puts cost vs ATM IV. This is the skew premium you collect when selling puts.")
    sk4.metric("OTM Call Premium", f"{call_wing_premium:+.1f}%" if call_wing_premium is not None else "—",
               help="How much more OTM calls cost vs ATM IV. Usually lower than put premium on NSE equity.")

    # Skew interpretation
    if skew_25d is not None:
        if skew_25d > 5:
            explain(
                f"<b>Strong negative skew: {skew_25d:+.1f}%.</b> "
                f"OTM puts are significantly more expensive than OTM calls. "
                f"Market is paying a <b>fear premium</b> for downside protection. "
                f"Natenberg: this is a <b>put-selling opportunity</b> — you collect rich IV on the put side. "
                f"But skew also signals the market expects a bigger down-move than up-move. "
                f"Use defined risk (spreads) rather than naked puts.",
                "warning",
            )
        elif skew_25d > 2:
            explain(
                f"<b>Moderate skew: {skew_25d:+.1f}%.</b> "
                f"Normal NSE equity market conditions. "
                f"Puts trade at a small premium to calls — institutional hedging demand. "
                f"Natenberg: skew is priced in — neither side is obviously mispriced. "
                f"Sell the higher-IV side (puts) but size conservatively.",
                "explain",
            )
        else:
            explain(
                f"<b>Flat skew: {skew_25d:+.1f}%.</b> "
                f"Calls and puts are nearly equally priced. "
                f"This is unusual for NSE equity — could mean a binary event (earnings, macro) is expected "
                f"where the direction is uncertain but a move is likely. "
                f"Natenberg: flat skew + high ATM IV = straddle environment, not strangle.",
                "safe",
            )

    # ── Chart 2 — Term Structure (IV across DTEs) ─────────────
    st.markdown("---")
    st.subheader("📅 IV Term Structure — Near vs Far Expiry")

    explain(
        "Natenberg's second smile: <b>IV varies across expiries</b>, not just strikes. "
        "Normal term structure: near-term IV > far-term IV (events are closer in time). "
        "Inverted: far IV > near IV — market expects a bigger future event (budget, election, earnings). "
        "Sell the expiry with the <b>highest IV</b> — that's where premium is richest.",
        "natenberg",
    )

    # Build synthetic term structure using HV term structure
    dtes    = [7, 15, 21, 30, 45, 60, 90]
    hv_data = {
        7:  stats_vl.get("hv10", stats_vl["iv"] * 0.9),
        15: stats_vl.get("hv10", stats_vl["iv"] * 0.92),
        21: stats_vl.get("hv20", stats_vl["iv"] * 0.95),
        30: stats_vl.get("hv20", stats_vl["iv"]),
        45: stats_vl.get("hv30", stats_vl["iv"] * 1.02),
        60: stats_vl.get("hv60", stats_vl["iv"] * 1.03) if stats_vl.get("hv60") else stats_vl["iv"] * 1.03,
        90: stats_vl.get("hv90", stats_vl["iv"] * 1.04) if stats_vl.get("hv90") else stats_vl["iv"] * 1.04,
    }
    # IV term struct: near-term gets event-vol spike (1.25x), far-term mean reverts
    iv_term = []
    for dte in dtes:
        hv_ref  = hv_data[dte]
        # Near expiries: IV includes event premium (earnings, weekly event)
        event_factor = 1.25 if dte <= 15 else 1.18 if dte <= 30 else 1.12 if dte <= 60 else 1.08
        iv_term.append(round(hv_ref * event_factor, 1))

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=dtes, y=iv_term,
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in iv_term],
        textposition="top center",
        name="IV (approx)",
        line=dict(color="#4f46e5", width=2.5),
        marker=dict(size=9, color=["#ef4444" if v == max(iv_term) else "#4f46e5" for v in iv_term]),
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.06)",
    ))
    fig_ts.add_trace(go.Scatter(
        x=dtes,
        y=[hv_data[d] for d in dtes],
        mode="lines",
        name="HV (realised)",
        line=dict(color="#22c55e", width=1.5, dash="dot"),
    ))
    # Highlight max IV expiry
    max_iv_dte = dtes[iv_term.index(max(iv_term))]
    fig_ts.add_vline(x=max_iv_dte, line_width=1.5, line_color="#f97316",
                     annotation_text=f"Richest IV ({max_iv_dte}d)",
                     annotation_position="top right")
    fig_ts.update_layout(
        title=f"{sym_vl} — IV Term Structure (approx)",
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility (%)",
        template="plotly_white", height=320,
        legend=dict(x=0.6, y=0.99),
        margin=dict(t=50, b=30),
        xaxis=dict(tickvals=dtes, ticktext=[f"{d}d" for d in dtes]),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    richest_dte = dtes[iv_term.index(max(iv_term))]
    explain(
        f"<b>Richest expiry: {richest_dte}-day options</b> (highest IV). "
        f"Natenberg: sell the expiry where IV is highest relative to realised vol. "
        f"The green dotted line is HV — the gap between IV and HV is your theoretical edge per expiry. "
        f"Note: this term structure uses rolling HV as a proxy; actual chain-derived term structure "
        f"will differ. Enable a broker API (Angel One/Fyers) in the Live Feed module for real data.",
        "natenberg",
    )

    # ── Chart 3 — OI distribution (from chain if available) ───
    if chain_ok and (len(call_df_sm) > 0 or len(put_df_sm) > 0):
        st.markdown("---")
        st.subheader("📊 Open Interest Distribution (Pain Point Analysis)")

        explain(
            "Max Pain theory: the market tends to expire near the strike where <b>total option buyer loss is maximised</b>. "
            "This is where OI is highest for both calls and puts combined. "
            "Knowing the max-pain strike helps you understand where smart money (option sellers) want expiry. "
            "This is the 'gravitational pull' that traders reference in the last week before expiry.",
            "explain",
        )

        oi_calls = call_df_sm[["Strike","OI"]].rename(columns={"OI":"Call OI"})
        oi_puts  = put_df_sm[["Strike","OI"]].rename(columns={"OI":"Put OI"})
        oi_merge = pd.merge(oi_calls, oi_puts, on="Strike", how="outer").fillna(0)
        oi_merge = oi_merge[oi_merge["Strike"] > 0].sort_values("Strike")
        oi_merge["Total OI"] = oi_merge["Call OI"] + oi_merge["Put OI"]

        if len(oi_merge) > 0:
            # Max pain calculation
            max_pain_strike = None
            min_pain = float("inf")
            for K_test in oi_merge["Strike"].values:
                # Total pain = sum of intrinsic losses for all buyers at this expiry price
                call_pain = ((oi_merge["Strike"] < K_test).astype(float) *
                             (K_test - oi_merge["Strike"]) * oi_merge["Call OI"]).sum()
                put_pain  = ((oi_merge["Strike"] > K_test).astype(float) *
                             (oi_merge["Strike"] - K_test) * oi_merge["Put OI"]).sum()
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = K_test

            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(
                x=oi_merge["Strike"], y=oi_merge["Call OI"],
                name="Call OI", marker_color="#4f46e5", opacity=0.8,
            ))
            fig_oi.add_trace(go.Bar(
                x=oi_merge["Strike"], y=oi_merge["Put OI"],
                name="Put OI", marker_color="#ef4444", opacity=0.8,
            ))
            fig_oi.add_vline(x=S_vl, line_color="#22c55e", line_width=2,
                             annotation_text=f"Spot ₹{S_vl:,.0f}", annotation_position="top left")
            if max_pain_strike:
                fig_oi.add_vline(x=max_pain_strike, line_color="#f97316",
                                 line_dash="dash", line_width=2,
                                 annotation_text=f"Max Pain ₹{max_pain_strike:,}",
                                 annotation_position="top right")
            fig_oi.update_layout(
                title=f"{sym_vl} — OI Distribution & Max Pain ({expiry_vl})",
                xaxis_title="Strike",
                yaxis_title="Open Interest (contracts)",
                barmode="group",
                template="plotly_white", height=340,
                legend=dict(x=0.01, y=0.99),
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_oi, use_container_width=True)

            pain_dist = round((max_pain_strike - S_vl) / S_vl * 100, 2) if max_pain_strike else 0
            explain(
                f"<b>Max Pain Strike: ₹{max_pain_strike:,}</b> "
                f"({'above' if pain_dist > 0 else 'below'} spot by {abs(pain_dist):.1f}%). "
                f"Option sellers benefit if {sym_vl} expires near ₹{max_pain_strike:,}. "
                f"Resistance from call sellers is highest at the strike with peak Call OI (blue). "
                f"Support from put sellers is strongest at peak Put OI (red). "
                f"These levels act as magnets in the final week before expiry.",
                "explain",
            )

    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 7 — OPTION SELLING ENGINE  ███
    # ═══════════════════════════════════════════════════════════════
