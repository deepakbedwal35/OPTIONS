"""
alerts_journal.py  —  Enhanced Edition
========================================
11. TELEGRAM ALERTS  — dual-message system
    • Alert 1 : Quick signal  (Entry / SL / T1 / T2 / Strike)
    • Alert 2 : Full analysis (Why it triggered — all scoring factors)
    • Alert 3 : Chart image   (candlestick + indicators via matplotlib)

12. TRADE JOURNAL    — paper trade tracking, accuracy report

NEW vs original
───────────────
• send_dual_alert()          → fires Alert-1 then Alert-2 in sequence
• format_full_analysis()     → rich "why it triggered" breakdown
• generate_chart_image()     → saves PNG candlestick chart, returns filepath
• send_chart_to_telegram()   → uploads the PNG via sendPhoto API
• Options strike price logic injected into every formatter
• Sector / FII context woven into the analysis block
"""

import io, json, os, math, time, base64, tempfile
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM CONFIGURATION  — edit these 4 lines, everything else auto-uses them
# ═══════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN  = os.environ.get("TG_BOT_TOKEN",  "8690413631:AAHMyTfl85zEVWx4lRNT29veyBhic2SkimA")
TELEGRAM_CHAT_ID    = os.environ.get("TG_CHAT_ID",    "6367416705")
TELEGRAM_MIN_GRADE  = os.environ.get("TG_MIN_GRADE",  "C")   # "A+", "A", or "B"
TELEGRAM_ENABLED    = True

# Grade priority order (used by grade filter below)
_GRADE_ORDER = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4}

def _grade_passes_filter(grade: str) -> bool:
    """Return True if this grade meets or exceeds TELEGRAM_MIN_GRADE."""
    return _GRADE_ORDER.get(grade, 9) <= _GRADE_ORDER.get(TELEGRAM_MIN_GRADE, 9)



# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _tg_post(bot_token: str, method: str, **kwargs) -> dict:
    """Low-level Telegram API call. Returns JSON response dict."""
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    try:
        r = requests.post(url, timeout=12, **kwargs)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_telegram(bot_token: str, chat_id: str, message: str,
                  disable_preview: bool = True) -> bool:
    """Send a text message. Returns True on success."""
    if not bot_token or not chat_id:
        return False
    res = _tg_post(
        bot_token, "sendMessage",
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_preview,
        },
    )
    return res.get("ok", False)


def send_chart_to_telegram(bot_token: str, chat_id: str,
                            image_path: str, caption: str = "") -> bool:
    """Upload a chart PNG via Telegram sendPhoto. Returns True on success."""
    if not bot_token or not chat_id or not image_path:
        return False
    if not Path(image_path).exists():
        return False
    with open(image_path, "rb") as f:
        res = _tg_post(
            bot_token, "sendPhoto",
            files={"photo": ("chart.png", f, "image/png")},
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
        )
    return res.get("ok", False)


def send_dual_alert(
    # ── required fields ────────────────────────────────
    symbol: str,
    price: float,
    score: int,
    grade: str,
    entry: float,
    sl: float,
    t1: float,
    t2: float,
    dow_signal: str,
    rsi: float,
    # ── credentials (auto-filled from global config) ───
    bot_token: str = "",
    chat_id: str = "",
    # ── options / strike ───────────────────────────────
    strike_price: float = 0.0,
    option_type: str = "",          # "CE" | "PE" | ""
    expiry: str = "",
    call_wall: float = 0.0,
    put_wall: float = 0.0,
    max_pain: float = 0.0,
    pcr: float = 0.0,
    atm_iv: float = 0.0,
    # ── scoring breakdown ──────────────────────────────
    macd_signal: str = "",
    volume_signal: str = "",
    pattern: str = "",
    support_level: float = 0.0,
    resistance_level: float = 0.0,
    atr: float = 0.0,
    risk_reward_t1: float = 0.0,
    # ── context ────────────────────────────────────────
    fii_signal: str = "",
    sector_rank: str = "",
    earnings_warning: str = "",
    # ── chart ──────────────────────────────────────────
    send_chart: bool = True,
) -> dict:
    """
    Fire three sequential Telegram messages for one signal:
      1. Quick signal card   (entry/SL/T1/T2 + strike if options)
      2. Full analysis card  (all scoring factors + reasoning)
      3. Chart image         (candlestick + EMA20/50 + RSI panel)

    Returns dict with keys 'alert1', 'alert2', 'chart' (bool each).
    """
    # Fall back to global config if caller didn't pass credentials
    if not bot_token:
        bot_token = TELEGRAM_BOT_TOKEN
    if not chat_id:
        chat_id = TELEGRAM_CHAT_ID
    if not TELEGRAM_ENABLED:
        return {"alert1": False, "alert2": False, "chart": False,
                "skipped": "TELEGRAM_ENABLED is False"}
    if not _grade_passes_filter(grade):
        return {"alert1": False, "alert2": False, "chart": False,
                "skipped": f"Grade {grade} below filter {TELEGRAM_MIN_GRADE}"}

    results = {}

    # ── Alert 1: Quick signal ──────────────────────────────────────────────────
    msg1 = format_quick_signal(
        symbol=symbol, price=price, score=score, grade=grade,
        entry=entry, sl=sl, t1=t1, t2=t2,
        dow_signal=dow_signal, rsi=rsi,
        strike_price=strike_price, option_type=option_type, expiry=expiry,
        call_wall=call_wall, put_wall=put_wall, max_pain=max_pain,
        pcr=pcr, atm_iv=atm_iv,
        fii_signal=fii_signal, sector_rank=sector_rank,
        earnings_warning=earnings_warning,
    )
    results["alert1"] = send_telegram(bot_token, chat_id, msg1)
    time.sleep(0.5)   # avoid Telegram rate limit

    # ── Alert 2: Full analysis ─────────────────────────────────────────────────
    msg2 = format_full_analysis(
        symbol=symbol, price=price, score=score, grade=grade,
        entry=entry, sl=sl, t1=t1, t2=t2,
        dow_signal=dow_signal, rsi=rsi,
        strike_price=strike_price, option_type=option_type, expiry=expiry,
        macd_signal=macd_signal, volume_signal=volume_signal,
        pattern=pattern, support_level=support_level,
        resistance_level=resistance_level, atr=atr,
        risk_reward_t1=risk_reward_t1,
        call_wall=call_wall, put_wall=put_wall, max_pain=max_pain,
        pcr=pcr, atm_iv=atm_iv,
        fii_signal=fii_signal, sector_rank=sector_rank,
        earnings_warning=earnings_warning,
    )
    results["alert2"] = send_telegram(bot_token, chat_id, msg2)

    # ── Alert 3: Chart ─────────────────────────────────────────────────────────
    results["chart"] = False
    if send_chart:
        try:
            chart_path = generate_chart_image(
                symbol=symbol,
                entry=entry, sl=sl, t1=t1, t2=t2,
                grade=grade,
            )
            if chart_path:
                cap = (
                    f"<b>{symbol}</b>  ₹{price:.2f}  |  {grade} Signal\n"
                    f"Entry ₹{entry:.2f}  SL ₹{sl:.2f}  T1 ₹{t1:.2f}  T2 ₹{t2:.2f}"
                )
                results["chart"] = send_chart_to_telegram(
                    bot_token, chat_id, chart_path, caption=cap
                )
                Path(chart_path).unlink(missing_ok=True)  # clean up temp file
        except Exception as e:
            results["chart_error"] = str(e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTER 1 — QUICK SIGNAL  (Alert 1)
# ═══════════════════════════════════════════════════════════════════════════════

def format_quick_signal(
    symbol: str, price: float, score: int, grade: str,
    entry: float, sl: float, t1: float, t2: float,
    dow_signal: str, rsi: float,
    strike_price: float = 0.0, option_type: str = "", expiry: str = "",
    call_wall: float = 0.0, put_wall: float = 0.0,
    max_pain: float = 0.0, pcr: float = 0.0, atm_iv: float = 0.0,
    fii_signal: str = "", sector_rank: str = "", earnings_warning: str = "",
) -> str:
    """
    Compact Telegram card — fits on one screen.
    Shows strike price for options trades.
    """
    grade_emoji = {"A+": "🚀", "A": "✅", "B": "🟡", "C": "🟠", "D": "🔴"}.get(grade, "📊")

    lines = [
        f"<b>{grade_emoji} NSE Pro Scanner — {grade} Signal</b>",
        "",
        f"<b>{symbol}</b>  ₹{price:.2f}",
        f"Score: {score}/30  |  Dow: {dow_signal}  |  RSI: {rsi:.0f}",
    ]

    # Options strike block (only if provided)
    if strike_price and option_type:
        lines += [
            "",
            f"📌 Strike: <b>₹{strike_price:,.0f} {option_type}</b>",
        ]
        if expiry:
            lines.append(f"📅 Expiry: {expiry}")
        if atm_iv:
            lines.append(f"🌀 ATM IV: {atm_iv:.1f}%")

    lines += [
        "",
        f"🎯 Entry:  ₹{entry:.2f}",
        f"🛑 SL:     ₹{sl:.2f}  ({abs((sl/entry-1)*100):.1f}% risk)",
        f"✅ T1:     ₹{t1:.2f}",
        f"🏆 T2:     ₹{t2:.2f}",
    ]

    # Options walls
    if call_wall and put_wall:
        lines += [
            "",
            f"🔴 Call wall: ₹{call_wall:,.0f}  |  🟢 Put wall: ₹{put_wall:,.0f}",
        ]
        if max_pain:
            lines.append(f"📍 Max pain: ₹{max_pain:,.0f}  |  PCR: {pcr:.2f}")

    if fii_signal:
        lines.append(f"🏦 FII: {fii_signal}")
    if sector_rank:
        lines.append(f"📊 Sector: {sector_rank}")
    if earnings_warning:
        lines.append(f"⚠️ {earnings_warning}")

    lines += ["", f"<i>⏰ {datetime.now().strftime('%d %b %Y %H:%M IST')}</i>"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTER 2 — FULL ANALYSIS  (Alert 2)  ← NEW
# ═══════════════════════════════════════════════════════════════════════════════

def format_full_analysis(
    symbol: str, price: float, score: int, grade: str,
    entry: float, sl: float, t1: float, t2: float,
    dow_signal: str, rsi: float,
    strike_price: float = 0.0, option_type: str = "", expiry: str = "",
    macd_signal: str = "", volume_signal: str = "",
    pattern: str = "", support_level: float = 0.0,
    resistance_level: float = 0.0, atr: float = 0.0,
    risk_reward_t1: float = 0.0,
    call_wall: float = 0.0, put_wall: float = 0.0,
    max_pain: float = 0.0, pcr: float = 0.0, atm_iv: float = 0.0,
    fii_signal: str = "", sector_rank: str = "", earnings_warning: str = "",
) -> str:
    """
    Detailed 'why it triggered' analysis message.
    Explains every scoring dimension so the trader understands the signal.
    """
    risk_pct  = abs((sl / entry - 1) * 100)
    rr_t1     = risk_reward_t1 if risk_reward_t1 else round(abs(t1 - entry) / max(abs(entry - sl), 0.01), 1)
    rr_t2     = round(abs(t2 - entry) / max(abs(entry - sl), 0.01), 1)

    # ── Dow theory narrative ───────────────────────────────────────────────────
    dow_map = {
        "UPTREND":   ("📈", "Higher highs and higher lows confirmed. Primary trend is bullish."),
        "DOWNTREND": ("📉", "Lower highs and lower lows in place. Trend is bearish — caution."),
        "SIDEWAYS":  ("➡️", "Price consolidating. No clear directional bias."),
    }
    dow_emoji, dow_desc = dow_map.get(dow_signal, ("📊", "Trend not determined."))

    # ── RSI interpretation ────────────────────────────────────────────────────
    if rsi >= 70:
        rsi_desc = f"RSI {rsi:.0f} — overbought zone. Watch for mean reversion."
    elif rsi >= 55:
        rsi_desc = f"RSI {rsi:.0f} — bullish momentum, room to run."
    elif rsi >= 40:
        rsi_desc = f"RSI {rsi:.0f} — neutral. No extreme reading."
    else:
        rsi_desc = f"RSI {rsi:.0f} — oversold. Potential bounce setup."

    # ── Options context ───────────────────────────────────────────────────────
    options_block = []
    if strike_price and option_type:
        options_block.append("")
        options_block.append(f"<b>🎰 Options trade details</b>")
        options_block.append(f"  Strike: ₹{strike_price:,.0f} {option_type}  |  Expiry: {expiry}")
        if atm_iv:
            iv_comment = "elevated — premium is rich" if atm_iv > 25 else "moderate" if atm_iv > 15 else "low — cheap premium"
            options_block.append(f"  ATM IV: {atm_iv:.1f}% ({iv_comment})")
        if call_wall and put_wall:
            dist_call = abs(call_wall - price) / price * 100
            dist_put  = abs(price - put_wall)  / price * 100
            options_block.append(f"  Call wall ₹{call_wall:,.0f} is {dist_call:.1f}% away (resistance)")
            options_block.append(f"  Put wall ₹{put_wall:,.0f} is {dist_put:.1f}% away (support)")
        if max_pain:
            pain_bias = "above" if price > max_pain else "below"
            options_block.append(
                f"  Max pain ₹{max_pain:,.0f}: spot is {pain_bias} pain — "
                f"{'sellers may defend it' if price > max_pain else 'magnet effect upward possible'}"
            )
        if pcr:
            pcr_desc = "bearish sentiment (CE heavy)" if pcr < 0.7 else "bullish (PE heavy)" if pcr > 1.2 else "neutral"
            options_block.append(f"  PCR {pcr:.2f} → {pcr_desc}")

    lines = [
        f"<b>🔬 Full Signal Analysis — {symbol}</b>",
        f"<i>Why this stock triggered a {grade} signal</i>",
        "",
        f"<b>📊 Score: {score}/30 — Grade {grade}</b>",
        f"CMP ₹{price:.2f}  |  Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",
        "",
        "─" * 28,
        "",
        f"<b>1. Trend analysis (Dow theory)</b>",
        f"  {dow_emoji} {dow_desc}",
    ]

    if support_level:
        lines.append(f"  🟢 Key support:    ₹{support_level:.2f}")
    if resistance_level:
        lines.append(f"  🔴 Key resistance: ₹{resistance_level:.2f}")

    lines += [
        "",
        f"<b>2. Momentum</b>",
        f"  • {rsi_desc}",
    ]
    if macd_signal:
        lines.append(f"  • MACD: {macd_signal}")

    lines += [
        "",
        f"<b>3. Volume</b>",
    ]
    if volume_signal:
        lines.append(f"  • {volume_signal}")
    else:
        lines.append("  • Volume data not supplied")

    lines += [
        "",
        f"<b>4. Pattern / structure</b>",
        f"  • {pattern if pattern else 'No specific pattern tagged'}",
    ]

    if atr:
        lines += [
            "",
            f"<b>5. Volatility (ATR)</b>",
            f"  • ATR: ₹{atr:.2f}  ({atr/price*100:.1f}% of price)",
            f"  • SL is {abs(entry-sl)/atr:.1f}× ATR below entry — {'tight' if abs(entry-sl)/atr < 1.2 else 'normal' if abs(entry-sl)/atr < 2 else 'wide'} stop",
        ]

    lines += [
        "",
        f"<b>6. Risk / reward</b>",
        f"  • Risk:      ₹{abs(entry-sl):.2f}  ({risk_pct:.1f}% of capital)",
        f"  • Reward T1: ₹{abs(t1-entry):.2f}  (R:R 1:{rr_t1})",
        f"  • Reward T2: ₹{abs(t2-entry):.2f}  (R:R 1:{rr_t2})",
    ]

    lines += options_block

    if fii_signal or sector_rank:
        lines += ["", f"<b>7. Market context</b>"]
        if fii_signal:
            lines.append(f"  🏦 FII/DII: {fii_signal}")
        if sector_rank:
            lines.append(f"  📊 Sector:  {sector_rank}")

    if earnings_warning:
        lines += ["", f"<b>⚠️ Earnings risk</b>", f"  {earnings_warning}"]

    lines += [
        "",
        "─" * 28,
        f"<b>Trade plan</b>",
        f"  Entry: ₹{entry:.2f}  →  SL: ₹{sl:.2f}  →  T1: ₹{t1:.2f}  →  T2: ₹{t2:.2f}",
        "",
        "<i>This is a paper-trade signal. Always verify with your own analysis.</i>",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTER — OPTIONS CHAIN ALERT  (unchanged + strike enriched)
# ═══════════════════════════════════════════════════════════════════════════════

def format_options_alert(
    symbol: str, price: float, expiry: str, dte: int,
    dominant: str, up_pct: float, dn_pct: float, sd_pct: float,
    sell_rec: str, call_wall: float, put_wall: float,
    max_pain: float, pcr: float, atm_iv: float,
    best_sell_strike: float = 0.0,   # ← NEW: specific strike recommendation
    sell_option_type: str = "",       # ← NEW: "CE" or "PE"
    fii_signal: str = "",
) -> str:
    dom_emoji = {"UPTREND": "📈", "DOWNTREND": "📉", "SIDEWAYS": "➡️"}.get(dominant, "➡️")
    lines = [
        f"<b>📊 Options Chain — {symbol}</b>",
        "",
        f"<b>{symbol}</b>  ₹{price:,.0f}  |  Expiry: {expiry} ({dte}d)",
        "",
        f"Trend by expiry:",
        f"  📈 Up: {up_pct:.0f}%  |  📉 Down: {dn_pct:.0f}%  |  ➡️ Side: {sd_pct:.0f}%",
        f"  {dom_emoji} <b>Dominant: {dominant}</b>",
        "",
        f"🎯 <b>{sell_rec}</b>",
    ]

    if best_sell_strike and sell_option_type:
        lines += [
            "",
            f"📌 Recommended strike: <b>₹{best_sell_strike:,.0f} {sell_option_type}</b>",
            f"   ({"above call wall — sell CE" if sell_option_type == "CE" else "below put wall — sell PE"})",
        ]

    lines += [
        "",
        f"Key levels:",
        f"  🔴 Call wall (CE): ₹{call_wall:,.0f}",
        f"  🟢 Put wall  (PE): ₹{put_wall:,.0f}",
        f"  📍 Max pain:       ₹{max_pain:,.0f}",
        f"  PCR: {pcr:.2f}  |  IV: {atm_iv:.1f}%",
    ]
    if fii_signal:
        lines.append(f"🏦 FII: {fii_signal}")
    lines += ["", f"<i>Generated: {datetime.now().strftime('%d %b %Y %H:%M')}</i>"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION  (Alert 3)  ← NEW
# ═══════════════════════════════════════════════════════════════════════════════

def generate_chart_image(
    symbol: str,
    entry: float, sl: float, t1: float, t2: float,
    grade: str,
    period: str = "3mo",
    figsize: tuple = (12, 8),
) -> str | None:
    """
    Download OHLCV from yfinance and plot:
      • Candlestick chart (green/red bars)
      • EMA 20 (blue) and EMA 50 (orange)
      • Entry / SL / T1 / T2 horizontal lines
      • RSI panel below

    Returns path to a temporary PNG, or None on failure.
    Caller is responsible for deleting the file after use.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None   # yfinance not installed

    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period=period)
        if df.empty or len(df) < 20:
            return None

        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # ── Indicators ────────────────────────────────────────────────────────
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        # RSI
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta).clip(lower=0).rolling(14).mean()
        rs     = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # ── Plot ──────────────────────────────────────────────────────────────
        style  = "dark_background"
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor="#0d0d0d",
        )
        for ax in (ax1, ax2):
            ax.set_facecolor("#141414")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.spines[:].set_color("#2a2a2a")

        # Candlesticks (manual bars)
        for idx, (ts, row) in enumerate(df.tail(60).iterrows()):
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            color = "#26a69a" if c >= o else "#ef5350"
            ax1.plot([idx, idx], [l, h], color=color, linewidth=0.8, alpha=0.9)
            ax1.bar(idx, abs(c - o), bottom=min(o, c),
                    color=color, width=0.6, alpha=0.95)

        n = min(60, len(df))
        x_range = range(n)
        ax1.plot(x_range, df["EMA20"].tail(n).values,
                 color="#2196F3", linewidth=1.2, label="EMA 20")
        ax1.plot(x_range, df["EMA50"].tail(n).values,
                 color="#FF9800", linewidth=1.2, label="EMA 50")

        # Entry / SL / Target lines
        level_cfg = [
            (entry, "#FFD700",  f"Entry ₹{entry:.2f}",  "--"),
            (sl,    "#f44336",  f"SL    ₹{sl:.2f}",     ":"),
            (t1,    "#4CAF50",  f"T1    ₹{t1:.2f}",     "--"),
            (t2,    "#81C784",  f"T2    ₹{t2:.2f}",     "--"),
        ]
        for level, color, label, ls in level_cfg:
            ax1.axhline(level, color=color, linestyle=ls, linewidth=1.1, alpha=0.85)
            ax1.text(n * 1.01, level, label,
                     va="center", ha="left", fontsize=7,
                     color=color, fontweight="bold")

        grade_color = {"A+": "#FFD700", "A": "#4CAF50", "B": "#FFA726"}.get(grade, "#aaaaaa")
        ax1.set_title(
            f"{symbol}  —  {grade} Signal  |  Score indicator",
            color=grade_color, fontsize=11, fontweight="bold", pad=8,
        )
        ax1.legend(loc="upper left", fontsize=7,
                   facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")
        ax1.set_xlim(-1, n + 8)
        ax1.set_ylabel("Price (₹)", color="#888", fontsize=8)

        # RSI panel
        rsi_vals = df["RSI"].tail(n).values
        ax2.plot(x_range, rsi_vals, color="#CE93D8", linewidth=1.1)
        ax2.axhline(70, color="#f44336", linestyle="--", linewidth=0.7, alpha=0.6)
        ax2.axhline(30, color="#4CAF50", linestyle="--", linewidth=0.7, alpha=0.6)
        ax2.fill_between(x_range, rsi_vals, 70,
                         where=(rsi_vals >= 70), alpha=0.15, color="#f44336")
        ax2.fill_between(x_range, rsi_vals, 30,
                         where=(rsi_vals <= 30), alpha=0.15, color="#4CAF50")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color="#888", fontsize=8)
        ax2.set_xlim(-1, n + 8)

        # Footer
        fig.text(
            0.5, 0.01,
            f"Generated by NSE Pro Scanner  •  {datetime.now().strftime('%d %b %Y %H:%M IST')}",
            ha="center", color="#555", fontsize=7,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", prefix=f"chart_{symbol}_"
        )
        plt.savefig(tmp.name, dpi=130, bbox_inches="tight",
                    facecolor="#0d0d0d", edgecolor="none")
        plt.close(fig)
        return tmp.name

    except Exception as e:
        try:
            plt.close("all")
        except:
            pass
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN SUMMARY (batch)
# ═══════════════════════════════════════════════════════════════════════════════



def format_buy_alert(
    symbol: str, price: float, score: int, grade: str,
    entry: float, sl: float, t1: float, t2: float,
    dow_signal: str, rsi: float,
    strike_price: float = 0.0, option_type: str = '',
    expiry: str = '', opt_premium: float = 0.0,
    opt_sl: float = 0.0, opt_t1: float = 0.0, opt_t2: float = 0.0,
    macd_text: str = '', vol_text: str = '',
    ema_stack: bool = True,
    iv: float = 0.0, hv: float = 0.0,
    fii_signal: str = '', sector: str = '',
    earnings_warning: str = '',
) -> str:
    """
    Dedicated BUY signal Telegram alert.
    Shows equity levels PLUS option levels if a strike is given.
    """
    grade_emoji = {'A+': '🚀', 'A': '✅', 'B': '🟡', 'C': '🟠', 'D': '🔴'}.get(grade, '📊')
    lines = [
        f'<b>{grade_emoji} BUY Signal — {grade}  |  NSE Alpha</b>',
        '',
        f'<b>{symbol}</b>  ₹{price:.2f}',
        f'Score: {score}/30  |  Dow: {dow_signal}  |  RSI: {rsi:.0f}',
        '',
        '─── Equity Trade ───',
        f'🎯 Entry:  ₹{entry:.2f}',
        f'🛑 SL:     ₹{sl:.2f}  ({abs((sl/entry-1)*100):.1f}% risk)',
        f'✅ T1:     ₹{t1:.2f}',
        f'🏆 T2:     ₹{t2:.2f}',
    ]
    if strike_price and option_type and opt_premium:
        lines += [
            '',
            f'─── Options Trade ───',
            f'📌 Buy {symbol} {int(strike_price)} {option_type}  |  Expiry: {expiry}',
            f'💰 Premium:  ₹{opt_premium:.2f}',
            f'🛑 Option SL: ₹{opt_sl:.2f}  (−40%)',
            f'✅ Option T1: ₹{opt_t1:.2f}  (+60%)',
            f'🏆 Option T2: ₹{opt_t2:.2f}  (+120%)',
        ]
    lines += ['']
    if ema_stack:
        lines.append('📈 EMA stack bullish (Price > EMA20 > EMA50) ✅')
    if macd_text:
        lines.append(f'🔀 {macd_text}')
    if vol_text:
        lines.append(f'📊 {vol_text}')
    if iv and hv:
        iv_status = 'Cheap (IV < HV) ✅' if iv < hv else 'Rich (IV > HV)'
        lines.append(f'🌊 IV {iv:.1f}% vs HV {hv:.1f}% — {iv_status}')
    if fii_signal:
        lines.append(f'🏦 FII: {fii_signal}')
    if sector:
        lines.append(f'📂 Sector: {sector}')
    if earnings_warning:
        lines.append(f'⚠️ {earnings_warning}')
    lines += ['', f'<i>⏰ {datetime.now().strftime("%d %b %Y %H:%M IST")}</i>']
    return "\n".join(lines)

def format_scan_summary(results: list, fii_signal: str = "") -> str:
    """Format a batch Telegram summary of all scan results."""
    grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    top = sorted(results, key=lambda x: grade_order.get(x.get("grade", "D"), 4))[:10]
    lines = [
        f"<b>📊 NSE Pro Scan Complete</b>",
        f"{len(results)} signals found",
        "",
    ]
    if fii_signal:
        lines += [f"🏦 Market: {fii_signal}", ""]
    for r in top:
        g    = r.get("grade", "?")
        chg  = r.get("change", 0)
        sign = "+" if chg >= 0 else ""
        lines.append(
            f"{'🚀' if g=='A+' else '✅' if g=='A' else '🟡'} "
            f"<b>{r['symbol']}</b> ₹{r['price']} ({sign}{chg}%) "
            f"| Score {r.get('score',0)}/30 | {g} | {r['dow']['signal']}"
        )
    lines += ["", f"<i>Scanned: {datetime.now().strftime('%d %b %Y %H:%M')}</i>"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE JOURNAL  (unchanged + minor improvements)
# ═══════════════════════════════════════════════════════════════════════════════

JOURNAL_FILE = "trade_journal.json"


def _load_journal() -> list:
    if JOURNAL_FILE in st.session_state:
        return st.session_state[JOURNAL_FILE]
    try:
        if Path(JOURNAL_FILE).exists():
            with open(JOURNAL_FILE) as f:
                data = json.load(f)
                st.session_state[JOURNAL_FILE] = data
                return data
    except:
        pass
    return []


def _save_journal(trades: list):
    st.session_state[JOURNAL_FILE] = trades
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    except:
        pass


def add_paper_trade(
    symbol: str, entry: float, sl: float,
    t1: float, t2: float, score: int,
    grade: str, setup_type: str = "",
    strike_price: float = 0.0, option_type: str = "",
) -> dict:
    """Record a new paper trade (now also stores options strike info)."""
    trades = _load_journal()
    trade  = {
        "id":           len(trades) + 1,
        "symbol":       symbol,
        "entry":        entry,
        "sl":           sl,
        "t1":           t1,
        "t2":           t2,
        "score":        score,
        "grade":        grade,
        "setup_type":   setup_type,
        "strike_price": strike_price,
        "option_type":  option_type,
        "date_entry":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status":       "OPEN",
        "outcome":      None,
        "exit_price":   None,
        "return_pct":   None,
        "date_exit":    None,
        "notes":        "",
    }
    trades.append(trade)
    _save_journal(trades)
    return trade


def update_trade_prices(trades: list) -> list:
    """Auto-update open trades with current prices from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return trades

    updated = []
    for t in trades:
        if t["status"] != "OPEN":
            updated.append(t); continue
        try:
            df = yf.Ticker(t["symbol"] + ".NS").history(period="2d")
            if not df.empty:
                cmp  = float(df["Close"].iloc[-1])
                high = float(df["High"].iloc[-1])
                low  = float(df["Low"].iloc[-1])
                if low <= t["sl"]:
                    t.update({"status": "CLOSED", "outcome": "SL_HIT",
                               "exit_price": t["sl"], "date_exit": datetime.now().strftime("%Y-%m-%d"),
                               "return_pct": round((t["sl"] / t["entry"] - 1) * 100, 2)})
                elif high >= t["t2"]:
                    t.update({"status": "CLOSED", "outcome": "T2_HIT",
                               "exit_price": t["t2"], "date_exit": datetime.now().strftime("%Y-%m-%d"),
                               "return_pct": round((t["t2"] / t["entry"] - 1) * 100, 2)})
                elif high >= t["t1"]:
                    t["outcome"] = "T1_HIT"
                t["current_price"]   = round(cmp, 2)
                t["unrealised_pct"]  = round((cmp / t["entry"] - 1) * 100, 2)
        except:
            pass
        updated.append(t)
    _save_journal(updated)
    return updated


def get_journal_stats(trades: list) -> dict:
    if not trades:
        return {
            "total": 0, "open": 0, "closed": 0,
            "win_rate": 0, "t1_rate": 0,
            "avg_return": 0, "best_trade": 0, "worst_trade": 0,
            "total_pnl_pct": 0,
            "grade_stats": {},
            "open_trades": [],      # ← was missing — caused KeyError
            "recent": [],           # ← was missing
        }
    closed = [t for t in trades if t["status"] == "CLOSED"]
    open_t = [t for t in trades if t["status"] == "OPEN"]
    rets   = [t["return_pct"] for t in closed if t.get("return_pct") is not None]
    wins   = [r for r in rets if r > 0]
    t1_hits = [t for t in closed if t.get("outcome") in ("T1_HIT", "T2_HIT")]

    grade_stats = {}
    for grade in ["A+", "A", "B", "C", "D"]:
        g_trades = [t for t in closed if t.get("grade") == grade]
        g_rets   = [t["return_pct"] for t in g_trades if t.get("return_pct") is not None]
        g_wins   = [r for r in g_rets if r > 0]
        if g_trades:
            grade_stats[grade] = {
                "total":    len(g_trades),
                "win_rate": round(len(g_wins) / len(g_rets) * 100, 1) if g_rets else 0,
                "avg_ret":  round(np.mean(g_rets), 2) if g_rets else 0,
            }

    return {
        "total":         len(trades),
        "open":          len(open_t),
        "closed":        len(closed),
        "win_rate":      round(len(wins) / len(rets) * 100, 1) if rets else 0,
        "t1_rate":       round(len(t1_hits) / len(closed) * 100, 1) if closed else 0,
        "avg_return":    round(np.mean(rets), 2) if rets else 0,
        "best_trade":    max(rets) if rets else 0,
        "worst_trade":   min(rets) if rets else 0,
        "total_pnl_pct": round(sum(rets), 2) if rets else 0,
        "grade_stats":   grade_stats,
        "open_trades":   open_t,
        "recent":        sorted(closed, key=lambda x: x.get("date_exit", ""), reverse=True)[:5],
    }


def render_journal_tab():
    """Render the full trade journal UI as a Streamlit section."""
    st.markdown('<div class="section-title">📓 Trade Journal — Paper Trading Tracker</div>',
                unsafe_allow_html=True)

    trades = _load_journal()

    if any(t["status"] == "OPEN" for t in trades):
        if st.button("🔄 Update Open Trades", key="journal_refresh"):
            trades = update_trade_prices(trades)
            st.success("Prices updated!")

    stats = get_journal_stats(trades)

    if stats["closed"] > 0:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", stats["total"])
        c2.metric("Win Rate",     f"{stats['win_rate']}%",
                  delta="Good" if stats["win_rate"] >= 60 else "Needs work")
        c3.metric("T1 Hit Rate",  f"{stats['t1_rate']}%")
        c4.metric("Avg Return",   f"{stats['avg_return']:+.2f}%")
        c5.metric("Open Trades",  stats["open"])

        if stats["grade_stats"]:
            st.markdown("**Grade accuracy:**")
            gcols = st.columns(len(stats["grade_stats"]))
            for i, (g, gdata) in enumerate(stats["grade_stats"].items()):
                gc  = "#3dd68c" if gdata["win_rate"] >= 60 else "#f5a623" if gdata["win_rate"] >= 45 else "#f75f5f"
                gcols[i].markdown(f"""
                <div class="card" style="text-align:center;padding:10px;border-top:3px solid {gc};">
                  <div style="font-size:20px;font-weight:800;color:{gc};">{g}</div>
                  <div style="font-size:11px;color:#aaaaaa;">{gdata['total']} trades</div>
                  <div style="font-size:14px;font-weight:700;color:{gc};">{gdata['win_rate']}% wins</div>
                  <div style="font-size:11px;color:#6b6b80;">avg {gdata['avg_ret']:+.1f}%</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No closed trades yet. Paper trade signals using the button below each stock analysis.")

    if stats["open_trades"]:
        st.markdown("**📂 Open Trades:**")
        for t in stats["open_trades"]:
            cmp   = t.get("current_price", t["entry"])
            unr   = t.get("unrealised_pct", 0)
            unr_c = "#3dd68c" if unr >= 0 else "#f75f5f"
            out   = t.get("outcome", "")
            strike_info = (
                f"  Strike ₹{t['strike_price']:,.0f} {t['option_type']}"
                if t.get("strike_price") and t.get("option_type") else ""
            )
            st.markdown(f"""
            <div class="card" style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:10px 14px;">
              <div style="min-width:80px;">
                <div style="font-size:13px;font-weight:700;color:#f0f0f5;">{t['symbol']}</div>
                <div style="font-size:9px;color:#6b6b80;">{t.get('date_entry','')[:10]}</div>
              </div>
              <div style="font-size:11px;color:#aaaaaa;">
                Entry ₹{t['entry']:.2f} | SL ₹{t['sl']:.2f} | T1 ₹{t['t1']:.2f}{strike_info}
              </div>
              <div style="font-size:13px;font-weight:700;color:{unr_c};">{unr:+.1f}%</div>
              {f'<span style="background:rgba(61,214,140,0.15);color:#3dd68c;padding:2px 8px;border-radius:10px;font-size:10px;">T1 HIT ✅</span>' if out=="T1_HIT" else ''}
              <div style="font-size:11px;color:#6b6b80;">Score {t['score']} · {t['grade']}</div>
            </div>""", unsafe_allow_html=True)

    if stats["recent"]:
        st.markdown("**📋 Recent Closed Trades:**")
        for t in stats["recent"]:
            ret_c     = "#3dd68c" if (t.get("return_pct") or 0) > 0 else "#f75f5f"
            out_emoji = {"T2_HIT": "🏆", "T1_HIT": "✅", "SL_HIT": "❌", "MANUAL": "📋"}.get(t.get("outcome", ""), "📋")
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;
                 border-bottom:1px solid #2a2a32;font-size:12px;">
              <span style="color:#f0f0f5;font-weight:600;">{out_emoji} {t['symbol']}</span>
              <span style="color:#6b6b80;">{t.get('date_exit','')}</span>
              <span style="color:#aaaaaa;">Entry ₹{t['entry']:.2f} → ₹{t.get('exit_price',0):.2f}</span>
              <span style="color:{ret_c};font-weight:700;">{t.get('return_pct',0):+.1f}%</span>
              <span style="color:#7c6af7;">{t.get('grade','')}</span>
            </div>""", unsafe_allow_html=True)

    open_syms = [t["symbol"] for t in trades if t["status"] == "OPEN"]
    if open_syms:
        st.markdown("---")
        mc1, mc2, mc3 = st.columns(3)
        close_sym    = mc1.selectbox("Close trade:", open_syms, key="journal_close_sym")
        close_price  = mc2.number_input("Exit price ₹:", min_value=0.0, value=0.0, key="journal_close_px")
        close_reason = mc3.selectbox("Reason:", ["T1_HIT", "T2_HIT", "SL_HIT", "MANUAL"], key="journal_close_reason")
        if st.button("✅ Close Trade", key="journal_close_btn") and close_price > 0:
            for t in trades:
                if t["symbol"] == close_sym and t["status"] == "OPEN":
                    t.update({
                        "status":     "CLOSED",
                        "outcome":    close_reason,
                        "exit_price": close_price,
                        "return_pct": round((close_price / t["entry"] - 1) * 100, 2),
                        "date_exit":  datetime.now().strftime("%Y-%m-%d"),
                    })
                    break
            _save_journal(trades)
            st.success(f"Trade {close_sym} closed!")
            st.rerun()

    if trades:
        df_exp = pd.DataFrame(trades)
        st.download_button(
            "⬇️ Export Journal CSV", df_exp.to_csv(index=False),
            "trade_journal.csv", "text/csv", key="journal_export",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK-USE EXAMPLE  (run this file directly to test)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Fill these in ──────────────────────────────────────────────────────────
    BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
    CHAT_ID   = os.environ.get("TG_CHAT_ID", "")

    if not BOT_TOKEN or not CHAT_ID:
        print("Set TG_BOT_TOKEN and TG_CHAT_ID environment variables to test.")
    else:
        results = send_dual_alert(
            bot_token    = BOT_TOKEN,
            chat_id      = CHAT_ID,
            symbol       = "RELIANCE",
            price        = 2954.50,
            score        = 26,
            grade        = "A+",
            entry        = 2955.00,
            sl           = 2890.00,
            t1           = 3085.00,
            t2           = 3200.00,
            dow_signal   = "UPTREND",
            rsi          = 62.4,
            # Options fields
            strike_price = 3000.0,
            option_type  = "CE",
            expiry       = "27-Mar-2025",
            call_wall    = 3000.0,
            put_wall     = 2900.0,
            max_pain     = 2950.0,
            pcr          = 0.85,
            atm_iv       = 18.5,
            # Scoring breakdown
            macd_signal  = "Bullish crossover on daily chart",
            volume_signal= "Volume 2.3× average — strong accumulation",
            pattern      = "Breakout from 6-week consolidation range",
            support_level= 2890.0,
            resistance_level = 3000.0,
            atr          = 42.0,
            risk_reward_t1 = 2.0,
            # Context
            fii_signal   = "FII net buyers ₹1,240 Cr today",
            sector_rank  = "Oil & Gas — rank 2/22 sectors",
            earnings_warning = "",
            send_chart   = True,
        )
        print("Alert results:", results)
