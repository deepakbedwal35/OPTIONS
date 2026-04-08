"""
Module: 🎯  Smart Strike Finder
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
    st.title("🎯 Smart Strike Finder — 15-Book Consensus Score")

    explain(
        "Every book approaches strike selection differently. "
        "McMillan uses probability of profit. Natenberg uses theoretical edge. "
        "Passarelli targets specific delta ranges. Hull derives strikes from put-call parity. "
        "Ellman scores covered call strikes by moneyness. Vine scores by risk-reward ratio. "
        "This module runs ALL their criteria simultaneously and gives each strike a "
        "<b>composite consensus score out of 100</b> — the higher the score, "
        "the more books agree this is the right strike to trade.",
        "explain",
    )

    sym_sf = st.selectbox("Stock / Index", list(NSE_FNO.keys()), key="sf_sym")
    with st.spinner("Loading live data..."):
        stats_sf = fetch_spot_iv(NSE_FNO[sym_sf])
        wr_sf    = compute_monthly_winrate(NSE_FNO[sym_sf])

    if not stats_sf:
        st.error("No data.")
        return

    S_sf   = stats_sf["spot"]
    iv_sf  = stats_sf["iv"] / 100
    hv_sf  = stats_sf["hv20"] / 100
    ivr_sf = stats_sf["ivr"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Spot",   f"₹{S_sf:,.2f}")
    c2.metric("IV",     f"{stats_sf['iv']:.1f}%")
    c3.metric("HV20",   f"{stats_sf['hv20']:.1f}%")
    c4.metric("IVR",    f"{ivr_sf:.0f}%")
    c5.metric("IV/HV",  f"{iv_sf/max(hv_sf,0.01):.2f}x")

    col_params, col_out = st.columns([1, 2])
    with col_params:
        strategy_sf = st.radio("Strategy", ["Sell Call","Sell Put","Sell Strangle","Buy Call","Buy Put"], key="sf_strat")
        dte_sf  = st.selectbox("DTE", [7,15,21,30,45,60], index=3, key="sf_dte")
        from config import RISK_FREE_RATE
        r_sf    = RISK_FREE_RATE
        T_sf    = max(dte_sf / 365, 0.001)
        st.markdown("**Book weights (0=ignore, 10=full weight):**")
        w_mcm  = st.slider("McMillan (Probability)",    0,10,10,key="w1")
        w_nat  = st.slider("Natenberg (IV Edge)",       0,10,10,key="w2")
        w_pass = st.slider("Passarelli (Delta target)", 0,10, 9,key="w3")
        w_hull = st.slider("Hull (Moneyness)",          0,10, 8,key="w4")
        w_ellm = st.slider("Ellman (CC Score)",         0,10, 8,key="w5")
        w_vine = st.slider("Vine (Risk-Reward)",        0,10, 9,key="w6")
        w_hist = st.slider("Historical Win-Rate",       0,10,10,key="w7")
        w_cart = st.slider("Carter (Weekly momentum)",  0,10, 7,key="w8")

    with col_out:
        is_sell    = "Sell" in strategy_sf
        opt_type_sf= "call" if "Call" in strategy_sf else "put"

        # ── Fetch real NSE option chain strikes ──────────────────
        chain_sf, expiry_sf = fetch_option_chain(NSE_FNO[sym_sf])
        real_strikes = []
        chain_source = "theoretical"

        if chain_sf is not None:
            try:
                call_strikes = list(chain_sf.calls["strike"].dropna().astype(int))
                put_strikes  = list(chain_sf.puts["strike"].dropna().astype(int))
                real_strikes = sorted(set(call_strikes + put_strikes))
                chain_source = f"live chain ({expiry_sf})"
            except Exception:
                real_strikes = []

        if real_strikes:
            # Filter to strikes relevant to the strategy direction
            if strategy_sf == "Sell Strangle":
                put_side  = [k for k in real_strikes if S_sf * 0.70 <= k <= S_sf * 0.99]
                call_side = [k for k in real_strikes if S_sf * 1.01 <= k <= S_sf * 1.30]
                cand_strikes = sorted(set(put_side + call_side))
            elif is_sell and "Call" in strategy_sf:
                cand_strikes = [k for k in real_strikes if S_sf * 1.01 <= k <= S_sf * 1.30]
            elif is_sell:
                cand_strikes = [k for k in real_strikes if S_sf * 0.70 <= k <= S_sf * 0.99]
            elif "Call" in strategy_sf:
                cand_strikes = [k for k in real_strikes if S_sf * 0.93 <= k <= S_sf * 1.12]
            else:
                cand_strikes = [k for k in real_strikes if S_sf * 0.88 <= k <= S_sf * 1.07]

            st.caption(f"Using real strikes from {chain_source} — {len(cand_strikes)} candidates")
        else:
            # Fallback: generate synthetic grid snapped to typical NSE steps
            from config import NSE_STRIKE_STEP, DEFAULT_STRIKE_STEP
            step = NSE_STRIKE_STEP.get(sym_sf, DEFAULT_STRIKE_STEP)
            if strategy_sf == "Sell Strangle":
                raw = list(np.arange(S_sf * 0.70, S_sf * 0.99, step)) + list(np.arange(S_sf * 1.01, S_sf * 1.30, step))
            elif is_sell and "Call" in strategy_sf:
                raw = list(np.arange(S_sf * 1.01, S_sf * 1.30, step))
            elif is_sell:
                raw = list(np.arange(S_sf * 0.70, S_sf * 0.99, step))
            elif "Call" in strategy_sf:
                raw = list(np.arange(S_sf * 0.93, S_sf * 1.12, step))
            else:
                raw = list(np.arange(S_sf * 0.88, S_sf * 1.07, step))
            cand_strikes = sorted(set(int(round(k / step) * step) for k in raw))
            st.caption(f"Live chain unavailable — showing theoretical strikes (step Rs.{step}). Prices are BS estimates.")

        score_rows = []
        for K in cand_strikes:
            K = int(K)
            g = bs_greeks(S_sf, K, T_sf, r_sf, iv_sf, opt_type_sf)
            p = bs_price(S_sf, K, T_sf, r_sf, iv_sf, opt_type_sf)
            if p < 0.5: continue
            delta_abs  = abs(g["delta"])
            prob_otm   = (1 - g["prob_itm"]) if is_sell else g["prob_itm"]
            dist_pct   = abs((K / S_sf - 1) * 100)
            sigma_dist = abs(np.log(S_sf / K)) / (iv_sf * np.sqrt(T_sf))
            iv_hv_r    = iv_sf / max(hv_sf, 0.01)
            theta_abs  = abs(g["theta"])
            margin_est = S_sf * 0.08

            s_mcm  = min(100, prob_otm * 100 / 70 * 100) if is_sell else min(100, g["prob_itm"] * 100 / 50 * 100)
            s_nat  = min(100, (iv_hv_r - 1) * 200) if (iv_hv_r > 1 and is_sell) else \
                     min(100, (1 - iv_hv_r) * 200) if (iv_hv_r < 1 and not is_sell) else 0
            if is_sell:
                s_pass = 100 if 0.10 <= delta_abs <= 0.20 else max(0, 100 - abs(delta_abs - 0.15) * 600)
            else:
                s_pass = 100 if 0.40 <= delta_abs <= 0.55 else max(0, 100 - abs(delta_abs - 0.50) * 400)
            s_hull = (min(100, dist_pct * 12) if dist_pct >= 5 else dist_pct * 8) if is_sell else max(0, 100 - dist_pct * 15)
            s_ellm = 100 if 2 <= dist_pct <= 5 else max(0, 100 - abs(dist_pct - 3.5) * 20)
            s_vine = min(100, (theta_abs * 365 / max(margin_est * 0.001, 0.001)) * 50)
            flat_rate = 0
            if wr_sf:
                closest = min([5,10,15,20], key=lambda x: abs(x - dist_pct))
                flat_rate = wr_sf["summary"].get(f"flat_{closest}pct", 50)
            s_hist = flat_rate
            s_cart = min(100, sigma_dist / 2.0 * 100) if is_sell else min(100, (1/max(sigma_dist,0.1)) * 50)

            total_w = w_mcm+w_nat+w_pass+w_hull+w_ellm+w_vine+w_hist+w_cart
            composite = (s_mcm*w_mcm + s_nat*w_nat + s_pass*w_pass + s_hull*w_hull +
                         s_ellm*w_ellm + s_vine*w_vine + s_hist*w_hist + s_cart*w_cart) / max(total_w,1)
            grade = "A+" if composite>=85 else "A" if composite>=75 else "B" if composite>=65 else "C" if composite>=50 else "D"

            score_rows.append({
                "Strike": int(K),
                "Dist %": f"{'+' if K>S_sf else ''}{(K/S_sf-1)*100:.1f}%",
                "Price (₹)": round(p,2), "Delta": round(g["delta"],3),
                "Prob OTM (%)": round(prob_otm*100,1), "Hist Flat (%)": round(flat_rate,1),
                "σ dist": round(sigma_dist,2), "Theta/d": round(g["theta"],3),
                "Score": round(composite,1), "Grade": grade,
                "McMillan": round(s_mcm,0), "Natenberg": round(s_nat,0),
                "Passarelli": round(s_pass,0), "Hull": round(s_hull,0),
                "Ellman": round(s_ellm,0), "Vine": round(s_vine,0),
                "Hist WR": round(s_hist,0), "Carter": round(s_cart,0),
            })

        if not score_rows:
            st.warning("No candidate strikes. Adjust parameters.")
            return

        score_df = pd.DataFrame(score_rows).sort_values("Score", ascending=False)
        best = score_df.iloc[0]
        grade_color = {"A+":"#15803d","A":"#16a34a","B":"#d97706","C":"#ea580c","D":"#dc2626"}.get(best["Grade"],"#64748b")

        st.markdown(f"""
        <div class="score-card">
          <div class="score-title">🏆 Top Recommended Strike</div>
          <div style="display:flex;gap:24px;align-items:center;margin-top:6px;flex-wrap:wrap">
            <div><div style="font-size:.75rem;color:#64748b">Strike</div>
                 <div class="score-val">₹{best['Strike']:,}</div></div>
            <div><div style="font-size:.75rem;color:#64748b">Distance</div>
                 <div class="score-val">{best['Dist %']}</div></div>
            <div><div style="font-size:.75rem;color:#64748b">Premium</div>
                 <div class="score-val">₹{best['Price (₹)']:.2f}</div></div>
            <div><div style="font-size:.75rem;color:#64748b">Consensus Score</div>
                 <div class="score-val" style="color:{grade_color}">{best['Score']}/100 {best['Grade']}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        explain(
            f"Strike ₹{best['Strike']:,} earns <b>{best['Score']:.0f}/100</b> across 8 book frameworks. "
            f"McMillan: {best['McMillan']:.0f}/100 (prob OTM={best['Prob OTM (%)']:.1f}%). "
            f"Natenberg: {best['Natenberg']:.0f}/100 (IV/HV edge). "
            f"Passarelli: {best['Passarelli']:.0f}/100 (Δ={best['Delta']:.3f}, target 0.10–0.20). "
            f"Historical win-rate at this distance: {best['Hist Flat (%)']:.1f}%. "
            f"Carter sigma distance: {best['σ dist']:.2f}σ "
            f"({'✅ Safe' if best['σ dist']>=1.5 else '⚠️ Close — use defined risk'}).",
            "safe" if best["Score"] >= 70 else "warning",
        )

        # Radar chart
        categories = ["McMillan","Natenberg","Passarelli","Hull","Ellman","Vine","Hist WR","Carter"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[best[c] for c in categories]+[best[categories[0]]],
            theta=categories+[categories[0]],
            fill="toself", fillcolor="rgba(79,70,229,0.15)",
            line=dict(color="#4f46e5",width=2), name=f"₹{best['Strike']:,}",
        ))
        if len(score_df) > 1:
            sec = score_df.iloc[1]
            fig_radar.add_trace(go.Scatterpolar(
                r=[sec[c] for c in categories]+[sec[categories[0]]],
                theta=categories+[categories[0]],
                fill="toself", fillcolor="rgba(249,115,22,0.10)",
                line=dict(color="#f97316",width=1.5,dash="dash"), name=f"₹{sec['Strike']:,}",
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                                template="plotly_white", height=360, legend=dict(x=0.01,y=0.99))
        st.plotly_chart(fig_radar, use_container_width=True)

        explain(
            "Each axis = one book's scoring framework. A perfect strike fills the entire radar. "
            "Hull low = wrong moneyness for the strategy. Ellman low = not ideal for covered calls. "
            "Carter low = too close for weekly selling. When multiple axes are weak, choose the next-best strike.",
            "explain",
        )

        st.subheader("All Candidate Strikes — Full Scoring Table")
        disp_cols = ["Strike","Dist %","Price (₹)","Delta","Prob OTM (%)","Hist Flat (%)","σ dist","Score","Grade",
                     "McMillan","Natenberg","Passarelli","Hull","Ellman","Vine","Hist WR","Carter"]

        def colour_score(v):
            if not isinstance(v,(int,float)): return ""
            if v>=85: return "background-color:#dcfce7;color:#166534;font-weight:700"
            if v>=70: return "background-color:#f0fdf4;color:#166534"
            if v>=55: return "background-color:#fef9c3;color:#854d0e"
            return "background-color:#fee2e2;color:#991b1b"

        st.dataframe(
            score_df[disp_cols].style
            .applymap(colour_score, subset=["Score","McMillan","Natenberg","Passarelli","Hull","Ellman","Vine","Hist WR","Carter"])
            .applymap(lambda v: "font-weight:700;color:#15803d" if v=="A+" else
                               "font-weight:700;color:#16a34a" if v=="A" else
                               "color:#d97706" if v=="B" else "color:#dc2626" if v in ("C","D") else "",
                      subset=["Grade"])
            .format({"Price (₹)":"₹{:.2f}","Prob OTM (%)":"{:.1f}%","Hist Flat (%)":"{:.1f}%",
                     "σ dist":"{:.2f}","Score":"{:.1f}"}),
            use_container_width=True, height=480, hide_index=True,
        )

        st.markdown("---")
        st.subheader("📚 What Each Book Says")
        for book_name, book_text, book_style in [
            ("McMillan (Options as Strategic Investment)",
             f"Prob OTM = {best['Prob OTM (%)']:.1f}%. Minimum 70% for selling. "
             f"{'✅ Meets criterion.' if best['Prob OTM (%)']>=70 else '❌ Below McMillan 70% minimum.'}",
             "safe" if best['Prob OTM (%)']>=70 else "danger"),
            ("Natenberg (Option Volatility & Pricing)",
             f"IV/HV = {iv_sf/max(hv_sf,0.01):.2f}x. "
             f"{'Options overpriced → SELL (IV > HV).' if iv_sf>hv_sf else 'Options underpriced → BUY (IV < HV).'}",
             "safe" if (iv_sf>hv_sf and is_sell) else "warning"),
            ("Passarelli (Trading Options Greeks)",
             f"Delta = {best['Delta']:.3f}. Target sell zone: 0.10–0.20. "
             f"Theta = ₹{best['Theta/d']:.3f}/day (daily income). "
             f"{'✅ In Passarelli sell zone.' if 0.09<=abs(best['Delta'])<=0.22 else '⚠️ Outside ideal delta range.'}",
             "safe" if 0.09<=abs(best['Delta'])<=0.22 else "warning"),
            ("Guy Cohen (Bible of Options Strategies)",
             f"OTM distance = {abs((best['Strike']/S_sf-1)*100):.1f}%. Cohen 5–10% OTM for income. "
             f"{'✅ In Cohen range.' if 5<=abs((best['Strike']/S_sf-1)*100)<=10 else '⚠️ Outside Cohen 5–10%.'}",
             "safe" if 5<=abs((best['Strike']/S_sf-1)*100)<=10 else "warning"),
            ("Hull (Fundamentals of Futures and Options)",
             f"Moneyness K/S = {best['Strike']/S_sf:.3f}. Put-call parity: "
             f"C=₹{bs_price(S_sf,best['Strike'],T_sf,r_sf,iv_sf,'call'):.2f}, "
             f"P=₹{bs_price(S_sf,best['Strike'],T_sf,r_sf,iv_sf,'put'):.2f}. Parity verified.",
             "explain"),
            ("Ellman (Exit Strategies for Covered Call Writing)",
             f"OTM score: {best['Ellman']:.0f}/100. Sweet spot 2–5% OTM (current: {abs((best['Strike']/S_sf-1)*100):.1f}%). "
             f"{'✅ Ellman zone.' if 2<=abs((best['Strike']/S_sf-1)*100)<=5 else 'Consider closer strike for max CC income.'}",
             "safe" if 2<=abs((best['Strike']/S_sf-1)*100)<=5 else "explain"),
            ("Carter (Weekly Options Trading Strategies)",
             f"σ distance = {best['σ dist']:.2f}. Carter weekly minimum: 1.0σ. "
             f"{'✅ Safe for weekly selling.' if best['σ dist']>=1.0 else '❌ Too close — use monthly expiry.'}",
             "safe" if best['σ dist']>=1.0 else "danger"),
            ("Sebastian/Chen (The Option Trader's Hedge Fund)",
             f"This strike contributes Δ={best['Delta']:.3f} to portfolio. "
             f"Sebastian: keep total portfolio delta < ±0.30. Vega = ₹{bs_greeks(S_sf,best['Strike'],T_sf,r_sf,iv_sf,opt_type_sf)['vega']:.3f}.",
             "explain"),
        ]:
            explain(f"<b>{book_name}:</b> {book_text}", book_style)


    # ═══════════════════════════════════════════════════════════════
    # ███  MODULE B — EXIT STRATEGY ENGINE  ███
    # ═══════════════════════════════════════════════════════════════
