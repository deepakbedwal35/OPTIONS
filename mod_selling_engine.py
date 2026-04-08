"""
Module: 💰  Option Selling Engine
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
from config import (
    RISK_FREE_RATE, NSE_STRIKE_STEP, DEFAULT_STRIKE_STEP,
    NSE_LOT_SIZES, DEFAULT_LOT_SIZE, MARGIN_ESTIMATE_PCT,
)


def render():
    st.title("💰 Option Selling Engine — Cordier & Gross Framework")

    explain(
        "Cordier's philosophy: <b>'Time is your ally.'</b> Every day that passes, "
        "options lose time value. Sellers collect this decay. "
        "The key rules: (1) sell 0.10–0.20 delta only, "
        "(2) collect at least 1/3 of margin as premium (3:1 rule), "
        "(3) IVR must be > 50%, (4) always pass the funeral test (survive a 2σ move), "
        "(5) exit at 50% profit — don't get greedy.",
        "cordier",
    )

    sym_ose = st.selectbox("Stock", list(NSE_FNO.keys()), key="ose_sym")
    dte_ose = st.slider("Target DTE (days)", 15, 60, 30)

    with st.spinner("Loading live data..."):
        stats_ose = fetch_spot_iv(NSE_FNO[sym_ose])
        chain_ose, expiry_ose = fetch_option_chain(NSE_FNO[sym_ose])
        wr_ose = compute_monthly_winrate(NSE_FNO[sym_ose])

    if not stats_ose:
        st.error("No data.")
        return

    S_ose = stats_ose["spot"]
    iv_ose = stats_ose["iv"] / 100
    T_ose  = max(dte_ose / 365, 0.001)
    r_ose  = RISK_FREE_RATE

    m = st.columns(5)
    m[0].metric("Spot",      f"₹{S_ose:,.2f}")
    m[1].metric("IV",        f"{stats_ose['iv']:.1f}%")
    m[2].metric("IVR",       f"{stats_ose['ivr']:.0f}%")
    m[3].metric("HV20",      f"{stats_ose['hv20']:.1f}%")
    m[4].metric("Flat <5% (hist)",
                f"{wr_ose['summary']['flat_5pct']:.1f}%" if wr_ose else "—",
                delta="Good" if wr_ose and wr_ose['summary']['flat_5pct']>55 else "Caution",
                delta_color="normal")

    st.markdown("---")
    # Find strikes at target deltas
    target_deltas = [0.10, 0.15, 0.20]
    call_strikes, put_strikes = [], []

    for td in target_deltas:
        # Binary search for call strike at delta = td
        lo_k, hi_k = S_ose * 0.80, S_ose * 1.50
        for _ in range(60):
            mid_k = (lo_k + hi_k) / 2
            d = bs_greeks(S_ose, mid_k, T_ose, r_ose, iv_ose, "call")["delta"]
            if d > td: lo_k = mid_k
            else:      hi_k = mid_k
        _step  = NSE_STRIKE_STEP.get(sym_ose, DEFAULT_STRIKE_STEP)
        call_k = int(round(mid_k / _step) * _step)
        call_g = bs_greeks(S_ose, call_k, T_ose, r_ose, iv_ose, "call")
        call_p = bs_price(S_ose, call_k, T_ose, r_ose, iv_ose, "call")

        # Put strike
        lo_k, hi_k = S_ose * 0.50, S_ose * 1.20
        for _ in range(60):
            mid_k = (lo_k + hi_k) / 2
            d = abs(bs_greeks(S_ose, mid_k, T_ose, r_ose, iv_ose, "put")["delta"])
            if d < td: hi_k = mid_k
            else:      lo_k = mid_k
        put_k  = int(round(mid_k / _step) * _step)
        put_g = bs_greeks(S_ose, put_k, T_ose, r_ose, iv_ose, "put")
        put_p = bs_price(S_ose, put_k, T_ose, r_ose, iv_ose, "put")

        call_strikes.append({
            "Target Δ": f"{td:.2f}", "Type": "Call Sell",
            "Strike": call_k,
            "Premium": round(call_p, 2),
            "Delta": round(call_g["delta"], 3),
            "Theta/d": round(call_g["theta"], 3),
            "Prob OTM (%)": round((1 - call_g["prob_itm"]) * 100, 1),
            "σ distance": round(np.log(call_k / S_ose) / (iv_ose * np.sqrt(T_ose)), 2),
        })
        put_strikes.append({
            "Target Δ": f"{td:.2f}", "Type": "Put Sell",
            "Strike": put_k,
            "Premium": round(put_p, 2),
            "Delta": round(put_g["delta"], 3),
            "Theta/d": round(put_g["theta"], 3),
            "Prob OTM (%)": round((1 - put_g["prob_itm"]) * 100, 1),
            "σ distance": round(np.log(S_ose / put_k) / (iv_ose * np.sqrt(T_ose)), 2),
        })

    col_c, col_p = st.columns(2)
    with col_c:
        st.subheader("Recommended Call Strikes (to sell)")
        call_df = pd.DataFrame(call_strikes)
        st.dataframe(
            call_df.style.applymap(
                lambda v: "background-color:#dcfce7" if isinstance(v,float) and v >= 80 else "",
                subset=["Prob OTM (%)"]
            ).format({"Strike":"₹{:,.0f}","Premium":"₹{:.2f}","Theta/d":"₹{:.3f}","Prob OTM (%)":"{:.1f}%"}),
            use_container_width=True, hide_index=True,
        )
    with col_p:
        st.subheader("Recommended Put Strikes (to sell)")
        put_df = pd.DataFrame(put_strikes)
        st.dataframe(
            put_df.style.applymap(
                lambda v: "background-color:#dcfce7" if isinstance(v,float) and v >= 80 else "",
                subset=["Prob OTM (%)"]
            ).format({"Strike":"₹{:,.0f}","Premium":"₹{:.2f}","Theta/d":"₹{:.3f}","Prob OTM (%)":"{:.1f}%"}),
            use_container_width=True, hide_index=True,
        )

    explain(
        "<b>σ distance</b>: how many standard deviations the strike is from current price. "
        "Cordier's hard rule: σ distance must be ≥ 1.5 (preferably ≥ 2.0) for naked selling. "
        "The <b>Prob OTM</b> column tells you the BS-model probability of this option expiring worthless — "
        "this is your theoretical win rate per trade. Compare it to the historical win rate from the Scanner.",
        "cordier",
    )

    # Strangle risk checks
    st.markdown("---")
    st.subheader("Strangle Risk Checks (0.15-delta)")
    best_call = call_strikes[1]  # 0.15 delta
    best_put  = put_strikes[1]
    total_prem = best_call["Premium"] + best_put["Premium"]
    margin_est = S_ose * 0.08
    three_one  = total_prem >= margin_est / 3
    ivr_ok     = stats_ose["ivr"] >= 50
    sigma_ok   = min(best_call["σ distance"], best_put["σ distance"]) >= 1.5
    flat_ok    = wr_ose and wr_ose["summary"]["flat_5pct"] >= 55 if wr_ose else False

    checks = [
        ("Delta 0.10–0.20 ✓",             True),
        (f"IVR ≥ 50% (current: {stats_ose['ivr']:.0f}%)", ivr_ok),
        (f"3:1 rule — premium ₹{total_prem:.0f} vs margin ₹{margin_est:.0f}", three_one),
        (f"Sigma distance ≥ 1.5 (min: {min(best_call['σ distance'],best_put['σ distance']):.2f})", sigma_ok),
        (f"Historical flat <5% ≥ 55% ({wr_ose['summary']['flat_5pct']:.1f}%)" if wr_ose else "Historical data N/A", flat_ok),
        (f"DTE 15–60 days (current: {dte_ose}d)",  15 <= dte_ose <= 60),
    ]
    for label, ok in checks:
        icon = "✅" if ok else "❌"
        colour = "#dcfce7" if ok else "#fee2e2"
        st.markdown(f'<div style="background:{colour};padding:6px 12px;border-radius:6px;margin:3px 0;font-size:.85rem">{icon} {label}</div>', unsafe_allow_html=True)

    all_ok = all(ok for _, ok in checks)
    if all_ok:
        explain(
            f"✅ ALL CHECKS PASS. This is a Cordier-grade strangle setup. "
            f"Sell the {best_call['Strike']} call + {best_put['Strike']} put for total premium ₹{total_prem:.0f}. "
            f"Target exit: buy back when premium drops to ₹{total_prem*0.5:.0f} (50% profit rule). "
            f"Stop loss: if position doubles against you, close immediately.",
            "safe",
        )
    else:
        failed = [lbl for lbl, ok in checks if not ok]
        explain(
            f"⚠️ {len(failed)} check(s) failed: {', '.join(failed[:3])}. "
            f"Do NOT enter a naked strangle until all Cordier criteria are met. "
            f"Consider an iron condor (defined risk) instead if you want to trade now.",
            "warning",
        )



    # ─────────────────────────────────────────────────────────────
    # MARGIN CALCULATOR & RETURN ON MARGIN
    # ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💹 Margin Calculator — Return on Margin (RoM)")

    explain(
        "The <b>real metric for option sellers is Return on Margin (RoM)</b>, not just premium collected. "
        "A ₹500 premium on ₹50,000 margin = 1% RoM. Annualised over 12 monthly trades = 12%. "
        "Cordier's 3:1 rule targets ≥33% annual RoM. "
        "SPAN margin is exchange-set; we use an 8% of notional proxy (conservative for naked selling). "
        "For covered calls and CSP, the capital at risk is the stock/cash position.",
        "cordier",
    )

    lot_size_ose = NSE_LOT_SIZES.get(sym_ose, DEFAULT_LOT_SIZE)
    step_ose     = NSE_STRIKE_STEP.get(sym_ose, DEFAULT_STRIKE_STEP)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        n_lots = st.number_input("Number of Lots", min_value=1, max_value=100, value=1, step=1, key="ose_lots")
    with mc2:
        margin_method = st.radio("Margin Method", ["SPAN Proxy (8% notional)", "Custom Amount"], key="ose_margin_method", horizontal=True)
    with mc3:
        custom_margin = st.number_input("Custom Margin per lot (₹)", min_value=1000, value=int(S_ose * lot_size_ose * 0.08), step=1000, key="ose_custom_margin") if margin_method == "Custom Amount" else None

    # Build strangle rows for 3 delta levels
    strat_rows = []
    for i, td in enumerate([0.10, 0.15, 0.20]):
        c_data = call_strikes[i]
        p_data = put_strikes[i]
        total_prem_pts  = c_data["Premium"] + p_data["Premium"]
        total_prem_rs   = total_prem_pts * lot_size_ose * n_lots

        # Margin estimate
        if margin_method == "SPAN Proxy (8% notional)":
            # SEBI SPAN: max(call side margin, put side margin) + premium of other leg
            call_margin = S_ose * lot_size_ose * MARGIN_ESTIMATE_PCT
            put_margin  = S_ose * lot_size_ose * MARGIN_ESTIMATE_PCT
            span_margin = (max(call_margin, put_margin) + min(call_margin, put_margin) * 0.50) * n_lots
        else:
            span_margin = custom_margin * n_lots

        # Return on Margin
        rom_per_trade   = total_prem_rs / span_margin * 100 if span_margin > 0 else 0
        # Annualised: assume 12 monthly trades, not compounded
        rom_annual      = rom_per_trade * (30 / max(dte_ose, 1)) * 12

        # Break-even moves
        upper_be = c_data["Strike"] + total_prem_pts
        lower_be = p_data["Strike"] - total_prem_pts
        upper_be_pct = (upper_be / S_ose - 1) * 100
        lower_be_pct = (lower_be / S_ose - 1) * 100

        # Max loss estimate (hard stop at 3× premium)
        max_loss_rs = total_prem_pts * 3 * lot_size_ose * n_lots

        strat_rows.append({
            "Target Δ":         f"0.{td*100:.0f}",
            "Call Strike":      f"₹{c_data['Strike']:,}",
            "Put Strike":       f"₹{p_data['Strike']:,}",
            "Total Premium":    f"₹{total_prem_pts:.1f} pts",
            "Premium (₹)":      round(total_prem_rs, 0),
            "SPAN Margin (₹)":  round(span_margin, 0),
            "RoM / Trade (%)":  round(rom_per_trade, 2),
            "RoM Annual (%)":   round(rom_annual, 1),
            "Upper BE":         f"₹{upper_be:,.0f} (+{upper_be_pct:.1f}%)",
            "Lower BE":         f"₹{lower_be:,.0f} ({lower_be_pct:.1f}%)",
            "Max Loss Est (₹)": round(max_loss_rs, 0),
            "3:1 Rule ✓":       "✅" if total_prem_rs >= span_margin / 3 else "❌",
        })

    rom_df = pd.DataFrame(strat_rows)

    def color_rom(v):
        if not isinstance(v, (int, float)): return ""
        if v >= 20: return "background-color:#dcfce7;color:#166534;font-weight:600"
        if v >= 10: return "background-color:#fef9c3;color:#854d0e"
        return "background-color:#fee2e2;color:#991b1b"

    def color_maxloss(v):
        if not isinstance(v, (int, float)): return ""
        return "color:#dc2626;font-weight:600"

    st.dataframe(
        rom_df.style
            .applymap(color_rom,     subset=["RoM / Trade (%)", "RoM Annual (%)"])
            .applymap(color_maxloss, subset=["Max Loss Est (₹)"])
            .format({
                "Premium (₹)":      "₹{:,.0f}",
                "SPAN Margin (₹)":  "₹{:,.0f}",
                "RoM / Trade (%)":  "{:.2f}%",
                "RoM Annual (%)":   "{:.1f}%",
                "Max Loss Est (₹)": "₹{:,.0f}",
            }),
        use_container_width=True, hide_index=True,
    )

    # Best row highlight
    best_rom_idx = rom_df["RoM Annual (%)"].astype(float).idxmax()
    best_rom_row = strat_rows[best_rom_idx]
    best_rom_val = best_rom_row["RoM Annual (%)"]
    three_one_ok = best_rom_row["3:1 Rule ✓"] == "✅"

    if best_rom_val >= 20 and three_one_ok:
        explain(
            f"<b>Best RoM: {best_rom_val:.1f}% annualised</b> at Δ={best_rom_row['Target Δ']} "
            f"({best_rom_row['Call Strike']} call / {best_rom_row['Put Strike']} put). "
            f"Premium ₹{best_rom_row['Premium (₹)']:,.0f} vs margin ₹{best_rom_row['SPAN Margin (₹)']:,.0f}. "
            f"Cordier 3:1 rule: ✅ met. "
            f"Upper break-even: {best_rom_row['Upper BE']}. Lower: {best_rom_row['Lower BE']}. "
            f"If this trade runs 12× per year, expected annual return = {best_rom_val:.0f}% on margin deployed.",
            "safe",
        )
    elif best_rom_val >= 10:
        explain(
            f"<b>RoM {best_rom_val:.1f}%/year</b> is moderate. "
            f"Cordier targets 30–40%+ annual RoM. Consider moving to a higher IV environment or "
            f"increasing DTE slightly to collect more premium. "
            f"Current lot size for {sym_ose}: {lot_size_ose} shares.",
            "warning",
        )
    else:
        explain(
            f"<b>RoM {best_rom_val:.1f}%/year is too low</b> — IV is not high enough relative to margin requirements. "
            f"Wait for IVR ≥ 50% before selling. Current IVR: {stats_ose['ivr']:.0f}%. "
            f"Natenberg's rule: only sell when IV > HV (IV/HV > 1.0).",
            "danger",
        )

    # Capital efficiency chart
    mc_chart_col1, mc_chart_col2 = st.columns(2)

    with mc_chart_col1:
        # RoM comparison bar chart
        fig_rom = go.Figure(go.Bar(
            x=[r["Target Δ"] for r in strat_rows],
            y=[r["RoM Annual (%)"] for r in strat_rows],
            marker_color=["#22c55e" if r["RoM Annual (%)"] >= 20
                          else "#f97316" if r["RoM Annual (%)"] >= 10
                          else "#ef4444" for r in strat_rows],
            text=[f"{r['RoM Annual (%)']:.1f}%" for r in strat_rows],
            textposition="outside",
        ))
        fig_rom.add_hline(y=20, line_dash="dot", line_color="#16a34a",
                          annotation_text="Cordier target (20%)", annotation_position="right")
        fig_rom.update_layout(
            title="Annual RoM by Delta Target",
            xaxis_title="Target Delta",
            yaxis_title="Annual RoM (%)",
            template="plotly_white", height=300,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_rom, use_container_width=True)

    with mc_chart_col2:
        # Premium vs Margin waterfall
        fig_pm = go.Figure()
        deltas = [r["Target Δ"] for r in strat_rows]
        prems  = [r["Premium (₹)"] for r in strat_rows]
        margins = [r["SPAN Margin (₹)"] for r in strat_rows]
        fig_pm.add_trace(go.Bar(name="Premium Collected (₹)", x=deltas, y=prems,
                                marker_color="#4f46e5"))
        fig_pm.add_trace(go.Bar(name="SPAN Margin Required (₹)", x=deltas, y=margins,
                                marker_color="#e2e8f0", opacity=0.7))
        fig_pm.update_layout(
            title="Premium vs Margin (per setup)",
            barmode="overlay",
            template="plotly_white", height=300,
            margin=dict(t=40, b=20),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_pm, use_container_width=True)

    explain(
        f"<b>Lot size for {sym_ose}:</b> {lot_size_ose} shares. "
        f"<b>SPAN margin proxy:</b> max(call margin, put margin) + 50% of smaller leg = approx 12–14% of notional for strangles. "
        f"Actual SPAN margin varies daily — check your broker's margin calculator before trading. "
        f"For <b>covered calls</b>: margin = stock purchase cost (much higher but no SPAN required). "
        f"For <b>CSP</b>: margin = put strike × lot size (cash secured).",
        "explain",
    )

    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE 8 — STRIKE RECOMMENDER  ███
    # ═══════════════════════════════════════════════════════════════
