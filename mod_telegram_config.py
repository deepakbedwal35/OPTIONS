"""
Module: 📱  Telegram Alerts — Config & Test
Options Alpha Platform v5.0

Central place to configure the Telegram bot, test connectivity,
and send manual alerts or batch summaries.
"""

import streamlit as st
import time
from datetime import datetime

from utils import explain, NSE_FNO, fetch_spot_iv, compute_monthly_winrate
from alerts_journal import (
    send_telegram, send_dual_alert,
    format_quick_signal, format_full_analysis,
    format_options_alert, format_scan_summary,
)


def render():
    st.title("📱 Telegram Alerts — Configuration & Test Panel")

    explain(
        "Configure your Telegram bot once here and test all alert types. "
        "To create a bot: message <b>@BotFather</b> on Telegram → /newbot → copy the token. "
        "To get your chat ID: message <b>@userinfobot</b> on Telegram. "
        "For a group/channel: add your bot as admin, send a message, then use "
        "<b>@JsonDumpBot</b> to get the chat ID (will be a negative number like -1001234567).",
        "explain",
    )

    # ── Credentials ──────────────────────────────────────────────────────────
    st.subheader("🔑 Bot Credentials")
    col1, col2 = st.columns(2)
    with col1:
        token = st.text_input(
            "Bot Token",
            type="password",
            key="tg_cfg_token",
            placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            help="From @BotFather → /newbot",
        )
    with col2:
        chat_id = st.text_input(
            "Chat / Channel ID",
            key="tg_cfg_chat",
            placeholder="-1001234567890 or @yourchannel",
            help="From @userinfobot or @JsonDumpBot",
        )

    # Save to session state for use across modules
    if token:
        st.session_state["global_tg_token"] = token
    if chat_id:
        st.session_state["global_tg_chat"] = chat_id

    # ── Connection test ───────────────────────────────────────────────────────
    st.subheader("🔌 Connection Test")
    if st.button("🧪 Send Test Message", key="tg_test_btn"):
        if not token or not chat_id:
            st.error("Enter both token and chat ID first.")
        else:
            test_msg = (
                f"<b>✅ Options Alpha Platform — Connected!</b>\n\n"
                f"Bot is configured correctly.\n"
                f"Alerts will be delivered to this chat.\n\n"
                f"<i>Test sent: {datetime.now().strftime('%d %b %Y %H:%M IST')}</i>"
            )
            ok = send_telegram(token, chat_id, test_msg)
            if ok:
                st.success("✅ Test message delivered! Check your Telegram.")
            else:
                st.error("❌ Failed. Verify token and chat ID — ensure bot is admin in group/channel.")

    st.markdown("---")

    # ── Alert type previews ───────────────────────────────────────────────────
    st.subheader("👁️ Alert Previews")

    preview_sym = st.selectbox("Preview symbol", list(NSE_FNO.keys())[:20], key="tg_preview_sym")
    preview_tab = st.radio(
        "Alert type",
        ["Quick Signal (Alert 1)", "Full Analysis (Alert 2)", "Options Chain Alert"],
        horizontal=True,
        key="tg_preview_tab",
    )

    # Sample data for preview
    sample = {
        "symbol": preview_sym, "price": 2954.50, "score": 26, "grade": "A+",
        "entry": 2955.00, "sl": 2890.00, "t1": 3085.00, "t2": 3200.00,
        "dow": "UPTREND", "rsi": 62.4,
        "strike": 3000.0, "option_type": "CE", "expiry": "27-Mar-2025",
        "call_wall": 3000.0, "put_wall": 2900.0, "max_pain": 2950.0,
        "pcr": 0.85, "iv": 18.5,
        "fii": "FII net buyers ₹1,240 Cr", "sector": "Energy",
    }

    if preview_tab == "Quick Signal (Alert 1)":
        msg = format_quick_signal(
            symbol=sample["symbol"], price=sample["price"],
            score=sample["score"], grade=sample["grade"],
            entry=sample["entry"], sl=sample["sl"],
            t1=sample["t1"], t2=sample["t2"],
            dow_signal=sample["dow"], rsi=sample["rsi"],
            strike_price=sample["strike"], option_type=sample["option_type"],
            expiry=sample["expiry"], call_wall=sample["call_wall"],
            put_wall=sample["put_wall"], max_pain=sample["max_pain"],
            pcr=sample["pcr"], atm_iv=sample["iv"],
            fii_signal=sample["fii"], sector_rank=sample["sector"],
        )
    elif preview_tab == "Full Analysis (Alert 2)":
        msg = format_full_analysis(
            symbol=sample["symbol"], price=sample["price"],
            score=sample["score"], grade=sample["grade"],
            entry=sample["entry"], sl=sample["sl"],
            t1=sample["t1"], t2=sample["t2"],
            dow_signal=sample["dow"], rsi=sample["rsi"],
            strike_price=sample["strike"], option_type=sample["option_type"],
            expiry=sample["expiry"],
            macd_signal="Bullish crossover on daily chart",
            volume_signal="Volume 2.3× average — strong accumulation",
            pattern="Breakout from 6-week consolidation",
            support_level=2890.0, resistance_level=3000.0, atr=42.0,
            risk_reward_t1=2.0,
            call_wall=sample["call_wall"], put_wall=sample["put_wall"],
            max_pain=sample["max_pain"], pcr=sample["pcr"], atm_iv=sample["iv"],
            fii_signal=sample["fii"], sector_rank=sample["sector"],
        )
    else:
        msg = format_options_alert(
            symbol=sample["symbol"], price=sample["price"],
            expiry=sample["expiry"], dte=6,
            dominant="UPTREND", up_pct=60, dn_pct=20, sd_pct=20,
            sell_rec="Sell Put — put wall strong support",
            call_wall=sample["call_wall"], put_wall=sample["put_wall"],
            max_pain=sample["max_pain"], pcr=sample["pcr"], atm_iv=sample["iv"],
            best_sell_strike=2900.0, sell_option_type="PE",
            fii_signal=sample["fii"],
        )

    # Show preview with formatted text
    st.markdown("**Message preview** (HTML tags shown — will render in Telegram):")
    st.code(msg, language=None)

    col_send, col_char = st.columns([1, 3])
    with col_send:
        if st.button("📤 Send This Preview", key="tg_send_preview"):
            if not token or not chat_id:
                st.error("Configure credentials above first.")
            else:
                ok = send_telegram(token, chat_id, msg)
                st.success("✅ Sent!" if ok else "❌ Failed")
    with col_char:
        st.caption(f"Message length: {len(msg)} characters (Telegram max 4096)")

    st.markdown("---")

    # ── Live alert for any stock ──────────────────────────────────────────────
    st.subheader("🚀 Send Live Signal Alert")
    explain(
        "Pick any stock, and the platform will fetch live data, score the signal, "
        "and fire all three Telegram alerts (quick card + full analysis + chart). "
        "This is the same function used by the Buy Signal Engine and Selling Engine.",
        "explain",
    )

    la_col1, la_col2, la_col3 = st.columns(3)
    with la_col1:
        live_sym = st.selectbox("Stock", list(NSE_FNO.keys()), key="tg_live_sym")
        live_grade = st.selectbox("Override grade", ["A+", "A", "B", "C", "D"], key="tg_live_grade")
    with la_col2:
        live_entry = st.number_input("Entry ₹", min_value=0.0, key="tg_live_entry")
        live_sl    = st.number_input("Stop Loss ₹", min_value=0.0, key="tg_live_sl")
    with la_col3:
        live_t1   = st.number_input("Target 1 ₹", min_value=0.0, key="tg_live_t1")
        live_t2   = st.number_input("Target 2 ₹", min_value=0.0, key="tg_live_t2")

    la_col4, la_col5 = st.columns(2)
    with la_col4:
        live_strike      = st.number_input("Strike ₹ (options only, 0 = equity)", min_value=0.0, key="tg_live_strike")
        live_option_type = st.selectbox("Option type", ["", "CE", "PE"], key="tg_live_otype")
    with la_col5:
        live_expiry      = st.text_input("Expiry date (YYYY-MM-DD)", key="tg_live_expiry", placeholder="2025-03-27")
        live_send_chart  = st.checkbox("Include chart", value=True, key="tg_live_chart")

    if st.button("🔔 Send Full Alert (All 3 Messages)", type="primary", key="tg_live_send"):
        if not token or not chat_id:
            st.error("Configure credentials above first.")
        elif not live_entry or not live_sl or not live_t1:
            st.error("Fill Entry, SL, and T1.")
        else:
            with st.spinner("Fetching data and sending alerts…"):
                from utils import fetch_spot_iv
                stats = fetch_spot_iv(NSE_FNO[live_sym]) or {}
                spot  = stats.get("spot", live_entry)

                results = send_dual_alert(
                    bot_token    = token,
                    chat_id      = chat_id,
                    symbol       = live_sym,
                    price        = spot,
                    score        = {"A+":26,"A":22,"B":18,"C":14,"D":10}.get(live_grade, 18),
                    grade        = live_grade,
                    entry        = live_entry or spot,
                    sl           = live_sl,
                    t1           = live_t1,
                    t2           = live_t2 or live_t1,
                    dow_signal   = "UPTREND",
                    rsi          = 55.0,
                    strike_price = live_strike,
                    option_type  = live_option_type,
                    expiry       = live_expiry,
                    sector_rank  = stats.get("sector", ""),
                    send_chart   = live_send_chart,
                )

            for key, label in [("alert1","Quick signal"), ("alert2","Full analysis"), ("chart","Chart")]:
                icon = "✅" if results.get(key) else "⚠️"
                st.write(f"{icon} {label}: {'sent' if results.get(key) else 'failed/skipped'}")

    st.markdown("---")

    # ── How to set up guide ──────────────────────────────────────────────────
    st.subheader("📖 Setup Guide")
    with st.expander("Step-by-step: Create Telegram Bot + Get Chat ID"):
        st.markdown("""
**Step 1 — Create a bot:**
1. Open Telegram, search for **@BotFather**
2. Send `/newbot`
3. Give it a name (e.g. "My NSE Alerts")
4. Give it a username ending in `bot` (e.g. `my_nse_alerts_bot`)
5. Copy the **API token** — looks like `1234567890:ABCdef...`

**Step 2 — Get your personal Chat ID:**
1. Search for **@userinfobot** on Telegram
2. Send `/start` — it replies with your user ID (e.g. `987654321`)
3. Use this as the Chat ID for personal alerts

**Step 3 — Group/Channel alerts:**
1. Create a group or channel in Telegram
2. Add your bot as an **administrator**
3. Send any message in the group
4. Search **@JsonDumpBot**, forward that message to it
5. The `chat.id` field in the JSON response is your Chat ID (negative number like `-1001234567`)

**Step 4 — Test:**
- Paste token + chat ID above → click "Send Test Message"
- If you see the message in Telegram, you're all set!

**Troubleshooting:**
- `Forbidden`: Bot is not admin in the group/channel
- `Chat not found`: Wrong chat ID format
- `Unauthorized`: Wrong bot token
        """)
