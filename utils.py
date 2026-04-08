"""
utils.py — Shared utilities for Options Alpha Platform v5.0
Contains: CSS, explain(), BS engine, data fetchers, NSE universe
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import importlib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# yfinance lazy loader
# ─────────────────────────────────────────────────────────────────
def get_yf():
    return importlib.import_module("yfinance")


# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS  (inject once from app.py)
# ═══════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
/* ── layout ── */
.main .block-container { padding-top: 1rem; max-width: 1440px; }

/* ── typography ── */
h1 { font-size: 1.65rem !important; font-weight: 700 !important;
     color: #0f172a !important; letter-spacing: -0.3px; }
h2 { font-size: 1.2rem !important;  font-weight: 600 !important; color: #1e293b !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #334155 !important; }

/* ── explain boxes — rich colour text ── */
.box-explain   { background:#eef2ff; border-left:4px solid #4f46e5;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#1e1b4b; line-height:1.55; }
.box-explain b  { color:#3730a3; }

.box-cordier   { background:#fff4ed; border-left:4px solid #ea580c;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#431407; line-height:1.55; }
.box-cordier b  { color:#c2410c; }

.box-natenberg { background:#f0fdf4; border-left:4px solid #16a34a;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#14532d; line-height:1.55; }
.box-natenberg b { color:#15803d; }

.box-mcmillan  { background:#fffbeb; border-left:4px solid #d97706;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#451a03; line-height:1.55; }
.box-mcmillan b { color:#b45309; }

.box-warning   { background:#fefce8; border-left:4px solid #ca8a04;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#713f12; line-height:1.55; }
.box-warning b  { color:#92400e; }

.box-safe      { background:#f0fdf4; border-left:4px solid #22c55e;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#14532d; line-height:1.55; }
.box-safe b     { color:#15803d; }

.box-danger    { background:#fff1f2; border-left:4px solid #ef4444;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#4c0519; line-height:1.55; }
.box-danger b   { color:#b91c1c; }

.box-hull      { background:#f0f9ff; border-left:4px solid #0284c7;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#0c4a6e; line-height:1.55; }
.box-hull b     { color:#0369a1; }

.box-ellman    { background:#fdf4ff; border-left:4px solid #a855f7;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#3b0764; line-height:1.55; }
.box-ellman b   { color:#7e22ce; }

.box-cohen     { background:#fff7ed; border-left:4px solid #f97316;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#431407; line-height:1.55; }
.box-cohen b    { color:#c2410c; }

.box-carter    { background:#ecfdf5; border-left:4px solid #10b981;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#064e3b; line-height:1.55; }
.box-carter b   { color:#059669; }

.box-sebastian { background:#fff1f2; border-left:4px solid #be123c;
                 padding:11px 16px; border-radius:0 10px 10px 0; margin:8px 0;
                 font-size:.84rem; color:#4c0519; line-height:1.55; }
.box-sebastian b { color:#9f1239; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background:#f8fafc; border-radius:10px; padding:8px 12px;
    border:1px solid #e2e8f0;
}
div[data-testid="metric-container"] label {
    color:#475569 !important; font-size:.78rem !important; font-weight:600 !important;
    text-transform:uppercase; letter-spacing:.4px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#0f172a !important; font-size:1.35rem !important; font-weight:700 !important;
}

/* ── dataframe / table text ── */
[data-testid="stDataFrame"] td {
    color: #1e293b !important;
    font-size: .82rem !important;
}
[data-testid="stDataFrame"] th {
    color: #0f172a !important;
    font-size: .78rem !important;
    font-weight: 700 !important;
    background: #f1f5f9 !important;
    text-transform: uppercase;
    letter-spacing: .3px;
}
[data-testid="stDataFrame"] tr:hover td {
    background: #f8fafc !important;
}

/* ── score card ── */
.score-card  { border:1px solid #e2e8f0; border-radius:12px; padding:16px 20px;
               margin:8px 0; background:#f8fafc; }
.score-title { font-size:.92rem; font-weight:700; color:#0f172a; margin-bottom:6px; }
.score-val   { font-size:1.65rem; font-weight:800; color:#1e293b; }

/* ── pill badges ── */
.pill-green { background:#dcfce7; color:#14532d; border-radius:12px;
              padding:2px 10px; font-size:.78rem; font-weight:700; }
.pill-red   { background:#fee2e2; color:#7f1d1d; border-radius:12px;
              padding:2px 10px; font-size:.78rem; font-weight:700; }
.pill-amber { background:#fef9c3; color:#78350f; border-radius:12px;
              padding:2px 10px; font-size:.78rem; font-weight:700; }
.pill-blue  { background:#dbeafe; color:#1e3a8a; border-radius:12px;
              padding:2px 10px; font-size:.78rem; font-weight:700; }

/* ── sidebar ── */
[data-testid="stSidebarContent"] { background:#0f172a; }
[data-testid="stSidebarContent"] * { color:#e2e8f0 !important; }
[data-testid="stSidebarContent"] .stRadio label {
    font-size:.81rem !important; color:#cbd5e1 !important; padding:3px 0;
}
[data-testid="stSidebarContent"] .stRadio label[data-baseweb="radio"]:hover {
    color:#ffffff !important;
}

/* ── section divider ── */
.section-sep { border:none; border-top:2px solid #e2e8f0; margin:16px 0; }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE  — coloured, book-branded boxes
# ═══════════════════════════════════════════════════════════════
BOX_META = {
    "explain":   ("box-explain",   "💡",           "#3730a3"),
    "cordier":   ("box-cordier",   "📕 Cordier",   "#c2410c"),
    "natenberg": ("box-natenberg", "📘 Natenberg", "#15803d"),
    "mcmillan":  ("box-mcmillan",  "📗 McMillan",  "#b45309"),
    "warning":   ("box-warning",   "⚠️ Note",      "#92400e"),
    "safe":      ("box-safe",      "✅",           "#15803d"),
    "danger":    ("box-danger",    "🚨 Risk",      "#b91c1c"),
    "hull":      ("box-hull",      "📘 Hull",      "#0369a1"),
    "ellman":    ("box-ellman",    "📗 Ellman",    "#7e22ce"),
    "cohen":     ("box-cohen",     "🟠 Cohen",     "#c2410c"),
    "carter":    ("box-carter",    "⚡ Carter",    "#059669"),
    "sebastian": ("box-sebastian", "🛡️ Sebastian/Chen", "#9f1239"),
}

def explain(text: str, style: str = "explain"):
    cls, icon, _ = BOX_META.get(style, BOX_META["explain"])
    st.markdown(
        f'<div class="{cls}"><span style="font-weight:700">{icon}</span>&nbsp; {text}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# NSE F&O UNIVERSE  (~95 stocks + 4 indices)
# ═══════════════════════════════════════════════════════════════
NSE_FNO = {
    "NIFTY":      "^NSEI",      "BANKNIFTY":  "^NSEBANK",
    "FINNIFTY":   "NIFTY_FIN_SERVICE.NS",      "MIDCPNIFTY": "^NSEMDCP50",
    "RELIANCE":   "RELIANCE.NS","TCS":         "TCS.NS",
    "INFY":       "INFY.NS",    "HDFCBANK":    "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS","SBIN":        "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS","KOTAKBANK":  "KOTAKBANK.NS",
    "LT":         "LT.NS",      "AXISBANK":    "AXISBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS","ITC":        "ITC.NS",
    "BAJFINANCE": "BAJFINANCE.NS","BAJAJFINSV": "BAJAJFINSV.NS",
    "ASIANPAINT": "ASIANPAINT.NS","MARUTI":     "MARUTI.NS",
    "TATAMOTORS": "TATAMOTORS.NS","TATASTEEL":  "TATASTEEL.NS",
    "ADANIPORTS": "ADANIPORTS.NS","NTPC":       "NTPC.NS",
    "POWERGRID":  "POWERGRID.NS","ONGC":        "ONGC.NS",
    "COALINDIA":  "COALINDIA.NS","JSWSTEEL":   "JSWSTEEL.NS",
    "SUNPHARMA":  "SUNPHARMA.NS","DRREDDY":    "DRREDDY.NS",
    "CIPLA":      "CIPLA.NS",   "DIVISLAB":    "DIVISLAB.NS",
    "WIPRO":      "WIPRO.NS",   "HCLTECH":     "HCLTECH.NS",
    "TECHM":      "TECHM.NS",   "ULTRACEMCO":  "ULTRACEMCO.NS",
    "GRASIM":     "GRASIM.NS",  "BRITANNIA":   "BRITANNIA.NS",
    "NESTLEIND":  "NESTLEIND.NS","INDUSINDBK":  "INDUSINDBK.NS",
    "BPCL":       "BPCL.NS",    "EICHERMOT":   "EICHERMOT.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS","APOLLOHOSP": "APOLLOHOSP.NS",
    "BEL":        "BEL.NS",     "HAL":         "HAL.NS",
    "BHEL":       "BHEL.NS",    "IRCTC":       "IRCTC.NS",
    "IRFC":       "IRFC.NS",    "PFC":         "PFC.NS",
    "RECLTD":     "RECLTD.NS",  "CANBK":       "CANBK.NS",
    "BANKBARODA": "BANKBARODA.NS","PNB":        "PNB.NS",
    "ZOMATO":     "ZOMATO.NS",  "NYKAA":       "NYKAA.NS",
    "MUTHOOTFIN": "MUTHOOTFIN.NS","CHOLAFIN":  "CHOLAFIN.NS",
    "LTIM":       "LTIM.NS",    "PERSISTENT":  "PERSISTENT.NS",
    "MPHASIS":    "MPHASIS.NS", "COFORGE":     "COFORGE.NS",
    "AUROPHARMA": "AUROPHARMA.NS","TORNTPHARM":"TORNTPHARM.NS",
    "LUPIN":      "LUPIN.NS",   "BIOCON":      "BIOCON.NS",
    "GLENMARK":   "GLENMARK.NS","TATACHEM":    "TATACHEM.NS",
    "UPL":        "UPL.NS",     "PIIND":       "PIIND.NS",
    "DEEPAKNTR":  "DEEPAKNTR.NS","SRF":        "SRF.NS",
    "VOLTAS":     "VOLTAS.NS",  "HAVELLS":     "HAVELLS.NS",
    "DIXON":      "DIXON.NS",   "ABCAPITAL":   "ABCAPITAL.NS",
    "MANAPPURAM": "MANAPPURAM.NS","M&M":        "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS","ASHOKLEY":  "ASHOKLEY.NS",
    "MOTHERSON":  "MOTHERSON.NS","MRF":         "MRF.NS",
    "BALKRISIND": "BALKRISIND.NS","BOSCHLTD":  "BOSCHLTD.NS",
    "GODREJCP":   "GODREJCP.NS","MARICO":      "MARICO.NS",
    "DABUR":      "DABUR.NS",   "TITAN":       "TITAN.NS",
    "TRENT":      "TRENT.NS",   "DLF":         "DLF.NS",
    "GODREJPROP": "GODREJPROP.NS","PRESTIGE":  "PRESTIGE.NS",
    "OBEROIRLTY": "OBEROIRLTY.NS",
}

SECTORS = {
    "All F&O":      list(NSE_FNO.keys()),
    "Indices":      ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"],
    "IT":           ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT","MPHASIS","COFORGE"],
    "Banking":      ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","CANBK","BANKBARODA","PNB"],
    "NBFC":         ["BAJFINANCE","BAJAJFINSV","MUTHOOTFIN","CHOLAFIN","MANAPPURAM","ABCAPITAL"],
    "Pharma":       ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","LUPIN","AUROPHARMA","TORNTPHARM","BIOCON","GLENMARK"],
    "PSU/Defence":  ["BEL","HAL","BHEL","NTPC","POWERGRID","ONGC","COALINDIA","PFC","RECLTD","IRCTC","IRFC"],
    "Auto":         ["TATAMOTORS","MARUTI","M&M","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","ASHOKLEY","MRF","BALKRISIND","BOSCHLTD","MOTHERSON"],
    "FMCG":         ["HINDUNILVR","ITC","BRITANNIA","NESTLEIND","MARICO","DABUR","GODREJCP"],
    "Metals":       ["TATASTEEL","JSWSTEEL","COALINDIA"],
    "Realty":       ["DLF","GODREJPROP","PRESTIGE","OBEROIRLTY"],
    "New Age":      ["ZOMATO","NYKAA"],
    "Chemicals":    ["UPL","PIIND","DEEPAKNTR","SRF","TATACHEM"],
    "Consumer Elec":["VOLTAS","HAVELLS","DIXON"],
}

INDICES_ONLY = ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"]
ALL_STOCKS_ONLY = [s for s in NSE_FNO.keys() if s not in INDICES_ONLY]


# ═══════════════════════════════════════════════════════════════
# BLACK-SCHOLES ENGINE
# ═══════════════════════════════════════════════════════════════
def bs_d1d2(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 1e-6:
        return max(S-K, 0) if opt=="call" else max(K-S, 0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    if opt == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, opt="call"):
    empty = {k: 0.0 for k in ["delta","gamma","theta","vega","rho","vanna","charm","volga","prob_itm","d1","d2"]}
    if T <= 1e-6 or sigma <= 1e-6:
        return empty
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    pdf1 = norm.pdf(d1)
    delta = norm.cdf(d1) if opt=="call" else norm.cdf(d1)-1
    gamma = pdf1 / (S*sigma*np.sqrt(T))
    vega  = S*pdf1*np.sqrt(T)/100
    if opt=="call":
        theta = (-(S*pdf1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        theta = (-(S*pdf1*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    vanna = -pdf1*d2/sigma
    charm = -pdf1*(2*r*T - d2*sigma*np.sqrt(T))/(2*T*sigma*np.sqrt(T))/365
    volga = vega*d1*d2/sigma
    prob_itm = norm.cdf(d2) if opt=="call" else norm.cdf(-d2)
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
                vanna=vanna, charm=charm, volga=volga, prob_itm=prob_itm, d1=d1, d2=d2)

def implied_vol(S, K, T, r, mkt, opt="call", tol=0.001):
    lo, hi = 0.001, 10.0
    for _ in range(200):
        mid = (lo+hi)/2
        p   = bs_price(S, K, T, r, mid, opt)
        if abs(p-mkt) < tol: return mid
        if p > mkt: hi = mid
        else:       lo = mid
    return mid


# ═══════════════════════════════════════════════════════════════
# DATA FETCHERS  (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str = "15y"):
    yf = get_yf()
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty: return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except:
        return None

@st.cache_data(ttl=180, show_spinner=False)
def fetch_spot_iv(ticker: str):
    yf = get_yf()
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1y", auto_adjust=True)
        if hist.empty: return None
        spot = float(hist["Close"].iloc[-1])
        ret  = np.log(hist["Close"]/hist["Close"].shift(1)).dropna()
        hv   = {
            "hv10": float(ret.tail(10).std()*np.sqrt(252)*100),
            "hv20": float(ret.tail(20).std()*np.sqrt(252)*100),
            "hv30": float(ret.tail(30).std()*np.sqrt(252)*100),
            "hv60": float(ret.tail(60).std()*np.sqrt(252)*100) if len(ret)>=60 else 0,
            "hv90": float(ret.tail(90).std()*np.sqrt(252)*100) if len(ret)>=90 else 0,
        }
        iv_approx = hv["hv20"]*1.18
        hi52 = float(hist["Close"].tail(252).max())
        lo52 = float(hist["Close"].tail(252).min())
        iv_hi = hv["hv90"]*1.4 if hv["hv90"] else iv_approx*1.4
        iv_lo = hv["hv10"]*0.8
        ivr   = max(0.0, min(100.0, (iv_approx-iv_lo)/max(iv_hi-iv_lo,0.1)*100))
        info  = tk.info
        return {"spot":spot,"hi52":hi52,"lo52":lo52,"iv":iv_approx,"ivr":ivr,
                "name":info.get("longName",ticker),"sector":info.get("sector","—"),**hv}
    except:
        return None

@st.cache_data(ttl=180, show_spinner=False)
def fetch_option_chain(ticker: str):
    yf = get_yf()
    try:
        tk  = yf.Ticker(ticker)
        exp = tk.options
        if not exp: return None, None
        today = datetime.today()
        valid = [e for e in exp if (datetime.strptime(e,"%Y-%m-%d")-today).days >= 15]
        if not valid: valid = list(exp)
        chain = tk.option_chain(valid[0])
        return chain, valid[0]
    except:
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def compute_monthly_winrate(ticker: str):
    hist = fetch_history(ticker)
    if hist is None or len(hist) < 60: return None
    closes = hist["Close"].dropna()
    fwd    = closes.shift(-21)
    pct    = ((fwd-closes)/closes*100).dropna()
    rows   = pd.DataFrame({"pct_change": pct.values}, index=pct.index)
    summary = {
        "n":      len(rows),
        "mean":   float(rows["pct_change"].mean()),
        "median": float(rows["pct_change"].median()),
        "std":    float(rows["pct_change"].std()),
        "max":    float(rows["pct_change"].max()),
        "min":    float(rows["pct_change"].min()),
        "up_rate":float((rows["pct_change"]>0).mean()*100),
        "dn_rate":float((rows["pct_change"]<0).mean()*100),
    }
    for t in [5,10,15,20]:
        summary[f"up_{t}pct"]     = float((rows["pct_change"]>= t).mean()*100)
        summary[f"dn_{t}pct"]     = float((rows["pct_change"]<=-t).mean()*100)
        summary[f"flat_{t}pct"]   = float((rows["pct_change"].abs()<t).mean()*100)
        summary[f"beyond_{t}pct"] = float((rows["pct_change"].abs()>=t).mean()*100)
    rows["month"] = rows.index.month
    by_month = rows.groupby("month")["pct_change"].agg(
        mean_move="mean",
        up_rate=lambda x: (x>0).mean()*100,
        flat_5=lambda x: (x.abs()<5).mean()*100,
    ).reset_index()
    by_month["month_name"] = by_month["month"].apply(
        lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
    )
    hc, hb = np.histogram(rows["pct_change"].values, bins=40, range=(-50,50))
    return {"rows":rows,"summary":summary,"by_month":by_month,
            "hist_counts":hc,"hist_bins":hb,"years":max(1,len(rows)//252)}

@st.cache_data(ttl=300, show_spinner=False)
def build_scanner_row(sym: str):
    ticker = NSE_FNO.get(sym)
    if not ticker: return None
    stats = fetch_spot_iv(ticker)
    if not stats: return None
    wr    = compute_monthly_winrate(ticker)
    return {
        "Symbol":            sym,
        "Sector":            stats.get("sector","—")[:18],
        "Spot (₹)":          round(stats["spot"],2),
        "HV20 (%)":          round(stats["hv20"],1),
        "IV approx (%)":     round(stats["iv"],1),
        "IVR (%)":           round(stats["ivr"],1),
        "Mean 30d (%)":      round(wr["summary"]["mean"],2) if wr else None,
        "Up >5% (%)":        round(wr["summary"]["up_5pct"],1) if wr else None,
        "Down >5% (%)":      round(wr["summary"]["dn_5pct"],1) if wr else None,
        "Flat <5% (%)":      round(wr["summary"]["flat_5pct"],1) if wr else None,
        "Up >10% (%)":       round(wr["summary"]["up_10pct"],1) if wr else None,
        "Down >10% (%)":     round(wr["summary"]["dn_10pct"],1) if wr else None,
        "Signal":            ("SELL PREM" if stats["ivr"]>50 and (wr["summary"]["flat_5pct"] if wr else 0)>50 else "WAIT"),
        "Data yrs":          wr["years"] if wr else None,
    }


# =================================================================
# BUY SIGNAL UTILITY  (called from mod_buy_engine)
# =================================================================

def compute_buy_score(sym: str) -> dict:
    """Lightweight buy-score stub — full logic lives in mod_buy_engine."""
    return None  # placeholder; mod_buy_engine has the full implementation

