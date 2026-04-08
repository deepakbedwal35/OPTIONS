"""
config.py — Shared constants for Options Alpha Platform v6.0

Single source of truth for all platform-wide settings.
Import anywhere with:  from config import RISK_FREE_RATE, NSE_LOT_SIZES, ...
"""

# ─────────────────────────────────────────────────────────────────
# Market constants
# ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE      = 0.065          # RBI repo rate approximation (6.5%)
TRADING_DAYS_YEAR   = 252            # NSE trading days per year
CALENDAR_DAYS_YEAR  = 365            # for T = DTE / 365 in BS

# ─────────────────────────────────────────────────────────────────
# NSE option strike intervals (points between consecutive strikes)
# ─────────────────────────────────────────────────────────────────
NSE_STRIKE_STEP = {
    "NIFTY":       50,
    "BANKNIFTY":   100,
    "FINNIFTY":    50,
    "MIDCPNIFTY":  25,
    "SENSEX":      100,
    "RELIANCE":    50,
    "TCS":         50,
    "INFY":        20,
    "HDFCBANK":    10,
    "ICICIBANK":   10,
    "SBIN":        5,
    "AXISBANK":    10,
    "KOTAKBANK":   20,
    "LT":          20,
    "WIPRO":       5,
    "HINDUNILVR":  20,
    "BAJFINANCE":  50,
    "MARUTI":      100,
    "TITAN":       20,
    "ULTRACEMCO":  50,
}
DEFAULT_STRIKE_STEP = 50             # fallback if symbol not in above dict

# ─────────────────────────────────────────────────────────────────
# NSE lot sizes (number of shares per 1 lot)
# ─────────────────────────────────────────────────────────────────
NSE_LOT_SIZES = {
    "NIFTY":       50,
    "BANKNIFTY":   15,
    "FINNIFTY":    40,
    "MIDCPNIFTY":  75,
    "RELIANCE":    250,
    "TCS":         150,
    "INFY":        300,
    "HDFCBANK":    550,
    "ICICIBANK":   700,
    "SBIN":        1500,
    "AXISBANK":    625,
    "KOTAKBANK":   400,
    "LT":          375,
    "WIPRO":       1500,
    "BAJFINANCE":  125,
    "HINDUNILVR":  300,
    "MARUTI":      100,
    "TITAN":       375,
    "ULTRACEMCO":  100,
}
DEFAULT_LOT_SIZE = 500               # fallback for unlisted symbols

# ─────────────────────────────────────────────────────────────────
# Option selling rules (Cordier framework defaults)
# ─────────────────────────────────────────────────────────────────
MIN_IVR_FOR_SELLING     = 50
MIN_PROB_OTM_SELL       = 70
TARGET_DELTA_SELL_LOW   = 0.10
TARGET_DELTA_SELL_HIGH  = 0.20
MIN_SIGMA_DIST_WEEKLY   = 1.0
MIN_SIGMA_DIST_MONTHLY  = 1.5
PROFIT_TARGET_PCT       = 50
LOSS_LIMIT_MULTIPLIER   = 2.0
MARGIN_ESTIMATE_PCT     = 0.08

# ─────────────────────────────────────────────────────────────────
# Scanner defaults
# ─────────────────────────────────────────────────────────────────
SCANNER_MIN_FLAT_PCT    = 50
SCANNER_CACHE_TTL       = 300
HISTORY_CACHE_TTL       = 3600
WINRATE_CACHE_TTL       = 3600

# ─────────────────────────────────────────────────────────────────
# Backtester defaults
# ─────────────────────────────────────────────────────────────────
BACKTEST_DEFAULT_YEARS  = 15
BACKTEST_MAX_YEARS      = 25
BACKTEST_DTE_ENTRY      = 30
BACKTEST_DELTA_TARGET   = 0.15
BACKTEST_PROFIT_TARGET  = 0.50
BACKTEST_LOSS_LIMIT     = 2.0

# ─────────────────────────────────────────────────────────────────
# UI / display
# ─────────────────────────────────────────────────────────────────
PLATFORM_NAME    = "Options Alpha v6.0"
PLATFORM_TAGLINE = "15 Books · NSE F&O · 15-yr history"
