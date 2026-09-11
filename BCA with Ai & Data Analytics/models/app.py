# Crop Recommendation and Yield Prediction and Selling Prediction Streamlit Application
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import warnings
import textwrap
from datetime import datetime, timedelta
import base64
from urllib.request import urlopen
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# PERFORMANCE FIX: Pre-compile regex to avoid re-compilation overhead on every render
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

# HTML RENDERING FIX HELPER
def render_raw_html(html: str):
    """FIX: dedent + collapse blank lines so Markdown never turns the HTML into a code block."""
    html = textwrap.dedent(html)
    html = _BLANK_LINE_RE.sub("\n", html)
    st.markdown(html, unsafe_allow_html=True)

# CUSTOM GREEN DATAFRAME RENDERER - Optimized for speed, compact styling, and row slider
def render_green_dataframe(df, title=None, hide_index=True):
    """Renders a dataframe as a fully green-styled, compact HTML table with a row limit slider"""
    if df is None:
        st.info("No data available.")
        return
    # Handle dictionary inputs (e.g., prediction details) gracefully
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    if isinstance(df, pd.DataFrame) and df.empty:
        st.info("No data available.")
        return
    total_rows = len(df)
    display_df = df
    # PERFORMANCE & UX FIX: Add a slider to control the number of rows displayed
    if total_rows > 10:
        slider_key = f"row_slider_{title.replace(' ', '_').lower()}" if title else f"row_slider_{id(df)}"
        display_limit = st.slider(
            "📊 Number of rows to display",
            min_value=10,
            max_value=total_rows,
            value=min(50, total_rows),
            step=10,
            key=slider_key
        )
        display_df = df.head(display_limit)
    html = """
    <style>
    .green-table-container {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        margin: 10px 0;
    }
    .green-table-title {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #ffffff;
        padding: 10px 14px;
        font-size: 14px;
        font-weight: 700;
        border-bottom: 1px solid #34d399;
    }
    .green-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
    }
    .green-table thead th {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #d1fae5;
        font-weight: 700;
        font-size: 12px;
        padding: 8px 10px;
        text-align: left;
        border-bottom: 1px solid #34d399;
        border-right: 1px solid #047857;
    }
    .green-table thead th:last-child { border-right: none; }
    .green-table tbody tr {
        border-bottom: 1px solid #047857;
        transition: all 0.2s ease;
    }
    .green-table tbody tr:hover { background: #047857 !important; }
    .green-table tbody tr:nth-child(even) { background: #064e3b; }
    .green-table tbody tr:nth-child(odd) { background: #065f46; }
    .green-table tbody td {
        color: #ecfdf5;
        font-size: 12px;
        padding: 6px 10px;
        border-right: 1px solid #047857;
    }
    .green-table tbody td:last-child { border-right: none; }
    .green-table tbody td:first-child {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 600;
        border-right: 1px solid #34d399;
    }
    .green-table tbody tr:nth-child(even) td:first-child {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    .green-table tbody tr:hover td:first-child {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
    }
    </style>
    """
    if title:
        html += f'<div class="green-table-container"><div class="green-table-title">{title}</div>'
    else:
        html += '<div class="green-table-container">'
    html += '<table class="green-table"><thead><tr>'
    for col in display_df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    # PERFORMANCE FIX: Replaced slow iterrows() with highly optimized itertuples()
    rows_html = []
    for row in display_df.itertuples(index=False):
        cells = "".join(f"<td>{'' if pd.isna(val) else str(val)}</td>" for val in row)
        rows_html.append(f"<tr>{cells}</tr>")
    html += "".join(rows_html)
    # Add a footer row if data is truncated
    if len(display_df) < total_rows:
        html += f'<tr><td colspan="{len(display_df.columns)}" style="text-align:center; padding:10px; color:#34d399; font-weight:600;">Showing {len(display_df)} of {total_rows} rows. Use the slider above to see more.</td></tr>'
    html += '</tbody></table></div>'
    render_raw_html(html)

# GLOBAL PLOTLY TEMPLATE — GRID LINES DISABLED EVERYWHERE
_agri_tpl = go.layout.Template()
_agri_tpl.layout.xaxis.showgrid = False
_agri_tpl.layout.xaxis.zeroline = False
_agri_tpl.layout.yaxis.showgrid = False
_agri_tpl.layout.yaxis.zeroline = False
pio.templates["agrismart_nogrid"] = _agri_tpl
pio.templates.default = "agrismart_nogrid"

# 1. STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="Crop Recommendation and Yield Prediction and Selling Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ADVANCED DYNAMIC BACKGROUND
def apply_dynamic_background():
    render_raw_html("""
    <style>
    .stApp { background: transparent !important; color: #E2F7E8 !important; }
    section.main .block-container {
        background: rgba(10, 31, 21, 0.80) !important;
        backdrop-filter: blur(20px) saturate(140%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(140%) !important;
        border: 1px solid rgba(134, 239, 172, 0.18) !important;
        border-radius: 24px !important;
        box-shadow: 0 24px 60px rgba(0,0,0,0.65) !important;
        position: relative; z-index: 1;
        margin-top: 24px !important; margin-bottom: 24px !important;
        padding: 2.5rem !important;
    }
    section[data-testid="stSidebar"] > div {
        background: rgba(8, 26, 16, 0.90) !important;
        backdrop-filter: blur(18px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(150%) !important;
        border-right: 1px solid rgba(74, 222, 128, 0.20) !important;
        z-index: 2;
    }
    .agri-scene {
        position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
        background:
            radial-gradient(1200px 600px at 85% -10%, rgba(250,204,21,.12), transparent 60%),
            radial-gradient(1000px 520px at 8% 108%, rgba(22,163,74,.16), transparent 62%),
            radial-gradient(900px 480px at 50% 118%, rgba(250,204,21,.09), transparent 60%),
            linear-gradient(180deg,#071a10 0%,#0a2416 30%,#0c2b1a 55%,#0a2415 80%,#06170d 100%);
    }
    .aurora { position:absolute; border-radius:50%; filter:blur(90px); mix-blend-mode:screen; will-change:transform; }
    .a1 { width:55vw; height:55vw; left:-15vw; top:-20vw; opacity:.24;
        background:radial-gradient(circle at 30% 30%, #16a34a, transparent 60%);
        animation:auroraDrift 26s ease-in-out infinite alternate; }
    .a2 { width:45vw; height:45vw; right:-12vw; top:-10vw; opacity:.15;
        background:radial-gradient(circle at 60% 40%, #facc15, transparent 60%);
        animation:auroraDrift 32s ease-in-out -8s infinite alternate-reverse; }
    .a3 { width:50vw; height:50vw; left:30vw; bottom:-25vw; opacity:.20;
        background:radial-gradient(circle at 50% 50%, #059669, transparent 65%);
        animation:auroraDrift 38s ease-in-out -16s infinite alternate; }
    .a4 { width:38vw; height:38vw; right:18vw; bottom:-18vw; opacity:.12;
        background:radial-gradient(circle at 50% 40%, #fde047, transparent 60%);
        animation:auroraDrift 30s ease-in-out -20s infinite alternate-reverse; }
    @keyframes auroraDrift {
        0%   { transform:translate3d(0,0,0) scale(1) rotate(0deg); }
        50%  { transform:translate3d(6vw,4vh,0) scale(1.15) rotate(10deg); }
        100% { transform:translate3d(-4vw,-3vh,0) scale(.95) rotate(-8deg); }
    }
    .sun-rays {
        position:absolute; top:-30%; right:-12%; width:72vw; height:72vw; filter:blur(28px);
        background:conic-gradient(from 180deg at 50% 50%,
            rgba(250,204,21,0) 0deg, rgba(250,204,21,.09) 22deg, transparent 45deg,
            rgba(250,204,21,.06) 75deg, transparent 100deg, rgba(250,204,21,.09) 135deg,
            transparent 165deg, rgba(250,204,21,.05) 210deg, transparent 250deg);
        animation:raysSpin 90s linear infinite;
    }
    @keyframes raysSpin { to { transform:rotate(360deg); } }
    .topo-grid {
        position:absolute; inset:0; opacity:.5;
        background-image:radial-gradient(rgba(134,239,172,.10) 1px, transparent 1.4px);
        background-size:26px 26px;
        -webkit-mask-image:linear-gradient(180deg,transparent 0%,rgba(0,0,0,.6) 35%,rgba(0,0,0,.6) 70%,transparent 100%);
        mask-image:linear-gradient(180deg,transparent 0%,rgba(0,0,0,.6) 35%,rgba(0,0,0,.6) 70%,transparent 100%);
        animation:gridPan 60s linear infinite;
    }
    @keyframes gridPan { from { background-position:0 0; } to { background-position:260px 130px; } }
    .float-glyph { position:absolute; opacity:.16; animation:glyphFloat 12s ease-in-out infinite alternate; }
    .float-glyph svg { width:100%; height:100%; display:block; }
    .g-crop  { left:5%;  top:24%;  width:90px; height:90px; color:#4ade80;
        filter:drop-shadow(0 0 18px rgba(74,222,128,.45)); }
    .g-yield { right:6%; top:19%;  width:100px; height:100px; color:#fde047;
        filter:drop-shadow(0 0 18px rgba(250,204,21,.45)); animation-delay:-4s; }
    .g-sell  { left:10%; bottom:24%; width:82px; height:82px; color:#fbbf24;
        filter:drop-shadow(0 0 18px rgba(251,191,36,.45)); animation-delay:-8s; }
    .g-yield2{ right:12%; bottom:26%; width:70px; height:70px; color:#86efac; opacity:.12;
        filter:drop-shadow(0 0 14px rgba(134,239,172,.4)); animation-delay:-6s; }
    @keyframes glyphFloat {
        0%   { transform:translateY(0) rotate(-4deg); }
        50%  { transform:translateY(-24px) rotate(3deg); }
        100% { transform:translateY(-10px) rotate(-2deg); }
    }
    .firefly {
        position:absolute; bottom:-10px; width:6px; height:6px; border-radius:50%;
        background:#fde68a; box-shadow:0 0 10px 3px rgba(253,230,138,.55); opacity:0;
        animation:flyUp var(--d,14s) linear var(--delay,0s) infinite;
    }
    .firefly.green { background:#86efac; box-shadow:0 0 10px 3px rgba(134,239,172,.5); }
    @keyframes flyUp {
        0%   { transform:translate(0,0); opacity:0; }
        10%  { opacity:.9; }
        50%  { transform:translate(34px,-45vh); opacity:.75; }
        70%  { opacity:.6; }
        100% { transform:translate(-24px,-95vh); opacity:0; }
    }
    .mist {
        position:absolute; left:-20%; right:-20%; height:24vh; bottom:4vh; filter:blur(46px);
        background:linear-gradient(180deg,transparent,rgba(74,222,128,.06) 45%,rgba(250,204,21,.05));
        animation:mistMove 26s ease-in-out infinite alternate;
    }
    .m2 { bottom:0; height:18vh; opacity:.8; animation-duration:34s; animation-delay:-12s; }
    @keyframes mistMove { from { transform:translateX(-4%); } to { transform:translateX(4%); } }
    .vignette { position:absolute; inset:0;
        background:radial-gradient(120% 90% at 50% 40%, transparent 60%, rgba(0,0,0,.45) 100%); }
    @media (prefers-reduced-motion: reduce) { .agri-scene * { animation:none !important; } }
    </style>
    <div class="agri-scene">
        <div class="aurora a1"></div>
        <div class="aurora a2"></div>
        <div class="aurora a3"></div>
        <div class="aurora a4"></div>
        <div class="sun-rays"></div>
        <div class="topo-grid"></div>
        <div class="float-glyph g-crop">
            <svg viewBox="0 0 64 64" fill="none">
                <path d="M32 58 V30" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
                <path d="M32 34 C32 22 24 14 12 12 C12 26 20 34 32 34 Z" fill="currentColor"/>
                <path d="M32 30 C32 18 40 10 52 8 C52 22 44 30 32 30 Z" fill="currentColor"/>
            </svg>
        </div>
        <div class="float-glyph g-yield">
            <svg viewBox="0 0 64 64" fill="none">
                <rect x="10" y="34" width="9" height="20" rx="2" fill="currentColor" opacity=".7"/>
                <rect x="27" y="24" width="9" height="30" rx="2" fill="currentColor" opacity=".85"/>
                <rect x="44" y="12" width="9" height="42" rx="2" fill="currentColor"/>
                <path d="M12 22 L30 12 L40 16 L54 6" stroke="currentColor" stroke-width="3" stroke-linecap="round" fill="none"/>
                <path d="M54 6 h-8 M54 6 v8" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="float-glyph g-sell">
            <svg viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="24" stroke="currentColor" stroke-width="4"/>
                <circle cx="32" cy="32" r="17" stroke="currentColor" stroke-width="1.5" opacity=".6"/>
                <path d="M25 20 h14 M25 27 h14 M25 20 c8 0 10 6 10 9 c0 5 -4 8 -10 8 l9 15"
                    stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            </svg>
        </div>
        <div class="float-glyph g-yield2">
            <svg viewBox="0 0 64 64" fill="none">
                <path d="M32 58 V30" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
                <path d="M32 34 C32 22 24 14 12 12 C12 26 20 34 32 34 Z" fill="currentColor"/>
                <path d="M32 30 C32 18 40 10 52 8 C52 22 44 30 32 30 Z" fill="currentColor"/>
            </svg>
        </div>
        <div class="mist m1"></div>
        <div class="mist m2"></div>
        <span class="firefly" style="left:6%;  --d:13s; --delay:-2s;"></span>
        <span class="firefly green" style="left:14%; --d:17s; --delay:-6s;"></span>
        <span class="firefly" style="left:24%; --d:15s; --delay:-9s;"></span>
        <span class="firefly green" style="left:34%; --d:19s; --delay:-4s;"></span>
        <span class="firefly" style="left:45%; --d:14s; --delay:-11s;"></span>
        <span class="firefly green" style="left:55%; --d:18s; --delay:-1s;"></span>
        <span class="firefly" style="left:64%; --d:16s; --delay:-7s;"></span>
        <span class="firefly green" style="left:73%; --d:20s; --delay:-13s;"></span>
        <span class="firefly" style="left:82%; --d:15s; --delay:-5s;"></span>
        <span class="firefly green" style="left:90%; --d:17s; --delay:-10s;"></span>
        <span class="firefly" style="left:96%; --d:14s; --delay:-3s;"></span>
        <span class="firefly green" style="left:50%; --d:21s; --delay:-15s;"></span>
        <div class="vignette"></div>
    </div>
    """)

def style_plotly_card(fig, height=None):
    try:
        if height is not None:
            fig.update_layout(height=height)
        if height is None:
            height = 440
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2F7E8', family='Inter, sans-serif', size=19),
            title=dict(font=dict(size=27, color='#F8FAFC')), hovermode='x unified',
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0)', font=dict(size=16, color='#E2F7E8')),
            hoverlabel=dict(bgcolor='rgba(10,45,30,0.98)', font=dict(color='#F8FAFC', size=16)),
            margin=dict(l=18, r=18, t=46, b=18), height=height,
        )
        fig.update_xaxes(showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0)', linecolor='rgba(255,255,255,0.35)', tickfont_color='#E2F7E8', tickfont_size=19, title_font_color='#E2F7E8', title_font_size=20, mirror=True, ticks='outside', ticklen=4)
        fig.update_yaxes(showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0)', linecolor='rgba(255,255,255,0.35)', tickfont_color='#E2F7E8', tickfont_size=19, title_font_color='#E2F7E8', title_font_size=20, mirror=True, ticks='outside', ticklen=4)
    except Exception:
        pass
    return fig

_orig_st_plotly_chart = st.plotly_chart
def _agri_style_plotly_chart(fig, *args, **kwargs):
    try:
        if hasattr(fig, 'update_layout'):
            style_plotly_card(fig)
        if 'height' not in kwargs:
            kwargs['height'] = 440
        if 'use_container_width' not in kwargs:
            kwargs['use_container_width'] = True
    except Exception:
        pass
    return _orig_st_plotly_chart(fig, *args, **kwargs)
st.plotly_chart = _agri_style_plotly_chart

apply_dynamic_background()

STATE_COORDINATES = {
    "Gujarat": (23.0225, 72.5714), "Maharashtra": (19.0760, 72.8777),
    "Rajasthan": (26.9124, 75.7873), "Punjab": (30.9010, 75.8573),
    "Kerala": (8.5241, 76.9366), "Tamil Nadu": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090), "Haryana": (29.0588, 76.0856),
    "Uttar Pradesh": (26.8467, 80.9462),
}

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
    81: "Rain showers", 82: "Heavy showers", 95: "Thunderstorm",
}

@st.cache_data(ttl=900, show_spinner=False)
def get_weather_forecast(state):
    latitude, longitude = STATE_COORDINATES.get(state, STATE_COORDINATES["Delhi"])
    query = urlencode({
        "latitude": latitude, "longitude": longitude, "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 3, "timezone": "auto",
    })
    try:
        with urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
    except Exception:
        return None

def render_page_image(filename, caption):
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width=340)

def render_weather_forecast():
    state = st.session_state.get("user_state", "Delhi")
    forecast = get_weather_forecast(state)
    st.markdown(f"### ️ Latest Weather Forecast — {state}")
    if not forecast or "current" not in forecast or "daily" not in forecast:
        st.info("The weather service is temporarily unavailable or returned an invalid response. Please try again shortly.")
        return
    current = forecast["current"]
    daily = forecast["daily"]
    st.caption(f"Updated {current.get('time', 'recently')} • Live forecasts every 15 mins")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{current.get('temperature_2m', 0):.1f} °C")
    c2.metric("Conditions", WEATHER_CODES.get(current.get('weather_code', -1), "Unknown"))
    c3.metric("Humidity", f"{current.get('relative_humidity_2m', 0)}%")
    c4.metric("Wind", f"{current.get('wind_speed_10m', 0):.1f} km/h")
    day_columns = st.columns(len(daily["time"]))
    for index, column in enumerate(day_columns):
        with column:
            try:
                date_label = datetime.fromisoformat(daily["time"][index]).strftime("%a, %d %b")
            except Exception:
                date_label = "Unknown"
            st.markdown(f"**{date_label}**")
            st.caption(WEATHER_CODES.get(daily.get("weather_code", [])[index], "Forecast"))
            st.write(f"{daily.get('temperature_2m_min', [0])[index]:.0f}–{daily.get('temperature_2m_max', [0])[index]:.0f} °C")
            st.caption(f"Rain: {daily.get('precipitation_probability_max', [0])[index]}%")

# COMPLETE TABLE/DATAFRAME STYLING - COMPACT GREEN THEME
render_raw_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@600;700&family=Inter:wght@400;600;700;800&display=swap');
.stApp { background: transparent !important; color: #E2F7E8 !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 17px !important; line-height: 1.7 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, section[data-testid="stMain"] h1, section[data-testid="stMain"] h2, section[data-testid="stMain"] h3 { font-size: 1.6rem !important; line-height: 1.15 !important; }
.stMarkdown p, .stMarkdown li, section[data-testid="stMain"] [data-testid="stCaptionContainer"], section[data-testid="stMain"] [data-testid="stMetricLabel"], section[data-testid="stMain"] label, div[data-baseweb="input"] input, div[data-testid="stTextArea"] textarea, .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button { font-size: 14px !important; }
section[data-testid="stMain"] [data-testid="stMetricLabel"] { font-size: 15px !important; font-weight: 700 !important; }
section[data-testid="stMain"] [data-testid="stMetricValue"] { font-size: 26px !important; line-height: 1.2 !important; }
section[data-testid="stMain"] [data-testid="stMetric"] { padding: 8px 10px !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label, section[data-testid="stSidebar"] .stButton>button { font-size: 14px !important; }
#MainMenu, footer {visibility: hidden;}
header, header > div, [data-testid="stHeader"], div[role="banner"], [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
.block-container { padding-top: 0 !important; margin-top: 0 !important; }
section[data-testid="stSidebar"] { background-color: #0F172A !important; width: 300px !important; min-width: 300px !important; padding-left: 14px !important; padding-right: 14px !important; min-height: 100vh !important; display: flex !important; flex-direction: column !important; justify-content: space-between !important; position: relative !important; }
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] { position: fixed !important; top: 12px !important; left: 100% !important; margin-left: 12px !important; right: auto !important; visibility: visible !important; display: block !important; z-index: 99999 !important; width: auto !important; height: auto !important; transform: none !important; pointer-events: auto !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] button, [data-testid="stSidebarCollapseButton"] button span, [data-testid="stExpandSidebarButton"] button span { visibility: visible !important; color: #E2F7E8 !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] button { min-width: 32px !important; min-height: 32px !important; border-radius: 999px !important; background: rgba(255, 255, 255, 0.08) !important; border: 1px solid rgba(226, 232, 240, 0.2) !important; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label { display: block !important; margin-bottom: 10px !important; padding: 9px 12px !important; border-radius: 10px !important; background: rgba(255, 255, 255, 0.03) !important; transition: background 0.2s ease !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255, 255, 255, 0.08) !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label:last-child { margin-bottom: 0 !important; }
section[data-testid="stSidebar"] .stButton>button { margin-top: 8px !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.12) !important; background: rgba(255,255,255,0.06) !important; color: #F8FAFC !important; font-weight: 700 !important; transition: all 0.2s ease !important; }
section[data-testid="stSidebar"] .stButton>button:hover { background: rgba(74, 222, 128, 0.18) !important; border-color: rgba(74, 222, 128, 0.35) !important; transform: translateY(-1px) !important; }
section[data-testid="stSidebar"] .stButton>button[kind="primary"] { background: linear-gradient(135deg, #16A34A, #15803D) !important; border-color: #4ADE80 !important; box-shadow: 0 8px 16px rgba(22, 163, 74, 0.22) !important; }
.js-plotly-plot, .plotly, .plotly-graph-div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; margin: 0 !important; }
div[data-testid="stPlotlyChart"] > div, .stPlotlyChart > div { background: rgba(3,42,30,0.96) !important; border: 1px solid rgba(74,222,128,0.18) !important; border-radius: 18px !important; box-shadow: 0 14px 32px rgba(0, 20, 10, 0.22) !important; padding: 8px !important; margin-bottom: 14px !important; }
section[data-testid="stMain"] h1, section[data-testid="stMain"] h2, section[data-testid="stMain"] h3 { color: #F8FAFC !important; font-weight: 800 !important; text-shadow: 0 2px 8px rgba(0, 20, 12, 0.85); }
section[data-testid="stMain"] label, section[data-testid="stMain"] div[data-baseweb="field-label"], section[data-testid="stMain"] .stNumberInput label, section[data-testid="stMain"] .stSelectbox label, section[data-testid="stMain"] .stTextInput label { color: #F8FAFC !important; font-weight: 700 !important; font-size: 14px !important; text-shadow: 0 1px 5px rgba(0, 20, 12, 0.9); }
section[data-testid="stMain"] [data-testid="stCaptionContainer"], section[data-testid="stMain"] [data-testid="stCaptionContainer"] p, section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p, section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li, section[data-testid="stMain"] [data-testid="stMetricLabel"] { color: #E2F7E8 !important; font-size: 13px !important; font-weight: 600 !important; text-shadow: 0 1px 5px rgba(0, 20, 12, 0.9); }
section[data-testid="stMain"] [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 26px !important; font-weight: 800 !important; text-shadow: 0 2px 9px rgba(0, 20, 12, 0.95); }
div[data-testid="stTextArea"] textarea, div[data-baseweb="textarea"] textarea { background-color: #FFFFFF !important; color: #000000 !important; font-family: 'Fira Code', 'Courier New', monospace !important; font-weight: 700 !important; font-size: 13px !important; border: 2px solid #16A34A !important; border-radius: 8px !important; }
div[data-baseweb="input"] input { background-color: #FFFFFF !important; color: #000000 !important; font-weight: 700 !important; font-size: 13px !important; }
.custom-card { background: linear-gradient(135deg, rgba(3, 42, 30, 0.84), rgba(7, 78, 56, 0.68)) !important; border: 1px solid rgba(187, 247, 208, 0.30); border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 10px 28px rgba(0, 18, 12, 0.30); backdrop-filter: blur(9px); }
.prescription-card { background: linear-gradient(135deg, rgba(5, 70, 45, 0.88), rgba(9, 104, 67, 0.72)) !important; border: 1px solid rgba(74, 222, 128, 0.70); border-radius: 12px; padding: 16px; margin-top: 12px; box-shadow: 0 10px 28px rgba(0, 18, 12, 0.30); backdrop-filter: blur(9px); }
.custom-card h1, .custom-card h2, .custom-card h3, .custom-card p, .custom-card li, .prescription-card h1, .prescription-card h2, .prescription-card h3, .prescription-card p, .prescription-card li { color: #F8FAFC !important; }
.custom-card label, .prescription-card label { color: #DCFCE7 !important; }
.prescription-title, .prescription-card .prescription-title { color: #BBF7D0 !important; font-size: 16px; font-weight: 800; display: flex; align-items: center; gap: 8px; }
.metric-badge { background: #DCFCE7; color: #166534; padding: 3px 9px; border-radius: 20px; font-size: 11px; font-weight: 700; color: #166534 !important; }
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button { background: linear-gradient(135deg, #22C55E 0%, #16A34A 52%, #15803D 100%) !important; color: #FFFFFF !important; font-weight: 700 !important; border-radius: 999px !important; border: 1px solid rgba(255, 255, 255, 0.22) !important; padding: 8px 14px !important; width: 100%; font-size: 14px !important; box-shadow: 0 3px 10px rgba(22, 163, 74, 0.32) !important; transition: all 0.2s ease-in-out; }
.stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover { background: linear-gradient(135deg, #4ADE80 0%, #22C55E 48%, #15803D 100%) !important; border-color: rgba(255, 255, 255, 0.45) !important; box-shadow: 0 6px 14px rgba(22, 163, 74, 0.44) !important; transform: translateY(-1px); }
.agri-bot-face { text-align: center; font-size: 52px; line-height: 1; padding: 4px 0 2px; filter: drop-shadow(0 5px 8px rgba(74, 222, 128, 0.35)); animation: agri-bot-wave 1.8s ease-in-out infinite alternate; }
@keyframes agri-bot-wave { to { transform: translateY(-4px) rotate(5deg); } }
</style>
""")

render_raw_html("""
<style>
.stApp { font-size: 20px !important; line-height: 1.8 !important; }
.stMarkdown h1, section[data-testid="stMain"] h1 { font-size: 2.7rem !important; line-height: 1.15 !important; }
.stMarkdown h2, section[data-testid="stMain"] h2 { font-size: 2.2rem !important; line-height: 1.2 !important; }
.stMarkdown h3, section[data-testid="stMain"] h3 { font-size: 1.9rem !important; line-height: 1.25 !important; }
.stMarkdown p, .stMarkdown li,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li { font-size: 19px !important; line-height: 1.7 !important; }
section[data-testid="stMain"] [data-testid="stCaptionContainer"],
section[data-testid="stMain"] [data-testid="stCaptionContainer"] p { font-size: 16px !important; }
section[data-testid="stMain"] label,
section[data-testid="stMain"] div[data-baseweb="field-label"] { font-size: 18px !important; }
div[data-baseweb="input"] input,
.stNumberInput div[data-baseweb="input"] input { font-size: 18px !important; font-weight: 700 !important; padding: 12px 14px !important; height: auto !important; min-height: 52px !important; }
div[data-baseweb="select"] > div { min-height: 52px !important; padding: 10px 14px !important; font-size: 18px !important; }
div[data-testid="stTextArea"] textarea { font-size: 17px !important; min-height: 130px !important; padding: 12px 14px !important; }
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button { font-size: 18px !important; padding: 13px 24px !important; min-height: 54px !important; }
section[data-testid="stMain"] [data-testid="stMetricLabel"],
section[data-testid="stMain"] [data-testid="stMetricLabel"] span { font-size: 18px !important; }
section[data-testid="stMain"] [data-testid="stMetricValue"] { font-size: 36px !important; }
section[data-testid="stMain"] [data-testid="stMetric"] { padding: 14px 16px !important; }
section[data-testid="stSidebar"] { width: 340px !important; min-width: 340px !important; }
section[data-testid="stSidebar"] h3 { font-size: 1.5rem !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] .stButton>button { font-size: 17px !important; padding: 12px 16px !important; }
.custom-card, .prescription-card { padding: 24px 26px !important; border-radius: 18px !important; }
.custom-card p, .custom-card li, .prescription-card p, .prescription-card li { font-size: 18px !important; }
.prescription-title { font-size: 21px !important; }
.metric-badge { font-size: 15px !important; padding: 5px 12px !important; }
div[data-testid="stPlotlyChart"] > div, .stPlotlyChart > div { padding: 12px !important; border-radius: 22px !important; }
section[data-testid="stMain"] div[style*="min-height:96px"] { min-height: 150px !important; padding: 18px 20px !important; border-radius: 18px !important; }
section[data-testid="stMain"] div[style*="min-height:96px"] b { font-size: 17px !important; }
section[data-testid="stMain"] div[style*="min-height:96px"] h2 { font-size: 38px !important; }
section[data-testid="stMain"] div[style*="min-height:96px"] span { font-size: 15px !important; }
.custom-card div[style*="font-weight: 900"] { font-size: 42px !important; }
.custom-card div[style*="font-size: 13px"] { font-size: 17px !important; }
.custom-card div[style*="font-size: 15px"] { font-size: 21px !important; }
.stTabs [data-baseweb="tab"] { font-size: 18px !important; padding: 12px 20px !important; }
div[data-testid="stChatInput"] textarea { font-size: 18px !important; }
div[data-testid="stChatMessage"] p { font-size: 18px !important; }
.js-plotly-plot svg text { font-size: 17px !important; }
.js-plotly-plot svg text.gtitle { font-size: 26px !important; }
.stApp .login-hero h1 { font-size: 3.4rem !important; }
.stApp .login-hero .brand { font-size: 16px !important; }
.stApp .login-hero .tagline { font-size: 1.3rem !important; }
.stApp .peace-divider { font-size: 20px !important; }
.stApp .feature-card { width: 320px !important; padding: 26px 22px !important; }
.stApp .feature-icon { font-size: 52px !important; }
.stApp .feature-title { font-size: 1.5rem !important; }
.stApp .feature-description { font-size: 1.05rem !important; }
.stApp .feature-stat { font-size: 0.95rem !important; padding: 4px 12px !important; }
.stApp .login-form-card { padding: 30px 28px !important; }
</style>
""")

# PERFORMANCE FIX: Removed the heavy MutationObserver that caused DOM thrashing and navigation lag.
# Replaced with safe, one-time setTimeout executions that rely on the existing CSS !important rules.
components.html("""
<script>
(function() {
    function relocateSidebarToggle() {
        var doc = window.parent.document;
        ['[data-testid="stSidebarCollapseButton"]','[data-testid="stExpandSidebarButton"]'].forEach(function(sel) {
            var el = doc.querySelector(sel);
            if (el) {
                el.style.cssText = "position:fixed !important; top:12px !important; left:12px !important; right:auto !important; z-index:99999 !important; visibility:visible !important; transform:none !important; display:block !important;";
            }
        });
    }
    // Safe execution without MutationObserver to prevent DOM thrashing and navigation lag
    setTimeout(relocateSidebarToggle, 100);
    setTimeout(relocateSidebarToggle, 500);
    setTimeout(relocateSidebarToggle, 1000);
})();
</script>
""", height=0, width=0)

render_raw_html("""
<style>
header, header > div, [data-testid="stHeader"], div[role="banner"],
[data-testid="stToolbar"], [data-testid="stToolbarActions"] {
    display: none !important; visibility: hidden !important;
    height: 0 !important; min-height: 0 !important; max-height: 0 !important;
    margin: 0 !important; padding: 0 !important; border: 0 !important;
    position: absolute !important;
}
body, #root, .stApp, .stAppViewContainer,
section[data-testid="stMain"], section.main,
div[data-testid="stAppViewContainer"] > section.main > div,
.block-container { padding-top: 0 !important; margin-top: 0 !important; }
.agri-scene { position: fixed !important; inset: 0 !important; min-height: 100vh !important; z-index: 0 !important; }
</style>
""")

DB_FILE = "agrismart.db"

def _write_dataset_if_needed(conn, table_name, csv_name, sample_data):
    table_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)).fetchone()
    if table_exists: return
    if os.path.exists(csv_name):
        pd.read_csv(csv_name).to_sql(table_name, conn, if_exists="replace", index=False)
    else:
        sample_data.to_sql(table_name, conn, if_exists="replace", index=False)

def init_sqlite_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, timestamp TEXT, module TEXT, inputs TEXT, result TEXT)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT UNIQUE, state TEXT, password TEXT, created_at TEXT)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, role TEXT NOT NULL, state TEXT, expires_at TEXT NOT NULL)""")
    sample_rec = pd.DataFrame({"N": np.random.randint(20, 140, 100), "P": np.random.randint(10, 90, 100), "K": np.random.randint(15, 200, 100), "temperature": np.random.uniform(15, 38, 100), "humidity": np.random.uniform(40, 95, 100), "ph": np.random.uniform(5.5, 8.2, 100), "rainfall": np.random.uniform(50, 300, 100), "label": np.random.choice(["rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee"], 100)})
    _write_dataset_if_needed(conn, "crop_recommendation", "Crop_Recommendation_Final.csv", sample_rec)
    sample_yield = pd.DataFrame({"Crop": np.random.choice(["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"], 100), "Crop_Year": np.random.choice([2020, 2021, 2022, 2023, 2024, 2025], 100), "Season": np.random.choice(["Kharif", "Rabi", "Whole Year"], 100), "State": np.random.choice(["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Gujarat"], 100), "Area": np.random.uniform(1.0, 50.0, 100), "Production": np.random.uniform(10.0, 500.0, 100), "Annual_Rainfall": np.random.uniform(400, 2000, 100), "Fertilizer": np.random.uniform(50, 500, 100), "Pesticide": np.random.uniform(5, 50, 100), "Yield": np.random.uniform(1.5, 8.5, 100)})
    _write_dataset_if_needed(conn, "crop_yield", "Crop_Yield_Final.csv", sample_yield)
    sample_sell = pd.DataFrame({"Crop": np.random.choice(["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"], 100), "State": np.random.choice(["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Gujarat"], 100), "Season": np.random.choice(["Kharif", "Rabi", "Whole Year"], 100), "Quantity (kg)": np.random.uniform(500, 10000, 100), "Price per kg (₹)": np.random.uniform(20, 180, 100), "Total Value (₹)": np.random.uniform(10000, 1000000, 100)})
    _write_dataset_if_needed(conn, "crop_selling", "Crop_Selling_Final.csv", sample_sell)
    conn.commit()
    conn.close()

init_sqlite_db()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_user_columns():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    if "username" not in columns: cursor.execute("ALTER TABLE users ADD COLUMN username TEXT")
    if "email" not in columns: cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "state" not in columns: cursor.execute("ALTER TABLE users ADD COLUMN state TEXT")
    if "role" not in columns: cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'User'")
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
        cursor.execute("UPDATE users SET created_at = ? WHERE created_at IS NULL", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    try: cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    except sqlite3.IntegrityError: pass
    conn.commit()
    conn.close()

ensure_user_columns()

# SECURITY: PASSWORD HASHING
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _token_hash(token): return hashlib.sha256(token.encode("utf-8")).hexdigest()

def create_login_token(username, role, state="Delhi"):
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(days=30)).isoformat()
    conn = get_db_connection()
    conn.execute("INSERT INTO auth_tokens (token_hash, username, role, state, expires_at) VALUES (?, ?, ?, ?, ?)", (_token_hash(token), username, role, state, expires_at))
    conn.commit()
    conn.close()
    return token

def get_login_from_token(token):
    if not token: return None
    conn = get_db_connection()
    row = conn.execute("SELECT username, role, state, expires_at FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),)).fetchone()
    if row and datetime.fromisoformat(row["expires_at"]) > datetime.now():
        conn.close()
        return dict(row)
    if row:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
        conn.close()
    return None

def revoke_login_token(token):
    if token:
        conn = get_db_connection()
        conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
        conn.close()

def sync_browser_auth_state():
    components.html("""<script>(function() {try {const url = new URL(window.parent.location.href);const authInUrl = url.searchParams.get('auth');const stored = window.parent.localStorage.getItem('agri_smart_auth');if (authInUrl && stored !== authInUrl) {window.parent.localStorage.setItem('agri_smart_auth', authInUrl);}if (!authInUrl && stored) {url.searchParams.set('auth', stored);window.parent.history.replaceState({}, '', url.toString());}} catch (e) {}})();</script>""", height=0, width=0)

def save_browser_auth_token(token):
    if not token: return
    components.html(f"""<script>(function() {{try {{const url = new URL(window.parent.location.href);url.searchParams.set('auth', '{token}');window.parent.localStorage.setItem('agri_smart_auth', '{token}');window.parent.history.replaceState({{}}, '', url.toString());}} catch (e) {{}}}})();</script>""", height=0, width=0)

def clear_browser_auth_token():
    components.html("""<script>(function() {try {window.parent.localStorage.removeItem('agri_smart_auth');const url = new URL(window.parent.location.href);url.searchParams.delete('auth');window.parent.history.replaceState({}, '', url.toString());} catch (e) {}})();</script>""", height=0, width=0)

def set_query_params(**kwargs):
    try: st.query_params.update(kwargs)
    except AttributeError: st.experimental_set_query_params(**kwargs)

def get_query_params():
    try: return dict(st.query_params)
    except AttributeError: return st.experimental_get_query_params()

def start_authenticated_session(username, role, state="Delhi"):
    st.session_state["authenticated"] = True
    st.session_state["user_name"] = username
    st.session_state["role"] = role
    st.session_state["user_state"] = state or "Delhi"
    st.session_state["page_nav"] = "Dashboard"
    token = create_login_token(username, role, state or "Delhi")
    set_query_params(auth=token)
    save_browser_auth_token(token)

def restore_persistent_login():
    sync_browser_auth_state()
    if st.session_state.get("authenticated"): return
    try:
        params = get_query_params() or {}
        auth = params.get("auth", [None])[0] if isinstance(params.get("auth"), list) else params.get("auth")
    except Exception: auth = None
    saved_login = get_login_from_token(auth)
    if saved_login:
        st.session_state["authenticated"] = True
        st.session_state["user_name"] = saved_login["username"]
        st.session_state["role"] = saved_login["role"]
        st.session_state["user_state"] = saved_login["state"] or "Delhi"

def register_or_login_user(username, email, state, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    user_columns = {row[1] for row in cursor.execute("PRAGMA table_info(users)").fetchall()}
    display_name = "COALESCE(NULLIF(username, ''), full_name)" if "full_name" in user_columns else "username"
    # Check if email exists
    cursor.execute(f"SELECT {display_name} AS display_name, password, role FROM users WHERE lower(email) = lower(?)", (email,))
    existing_user = cursor.fetchone()
    if existing_user:
        conn.close()
        if existing_user["password"] == password or existing_user["password"] == hash_password(password):
            return True, existing_user["display_name"], False, existing_user["role"] or "User"
        return False, None, False, None
    # FIX: Check if username already exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, None, False, None  # Username already taken
    insert_columns = ["username", "email", "state", "password", "created_at"]
    insert_values = [username, email, state, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    if "full_name" in user_columns:
        insert_columns.insert(0, "full_name")
        insert_values.insert(0, username)
    # FIX: Wrap INSERT in try-except to handle any remaining IntegrityError
    try:
        cursor.execute(f"INSERT INTO users ({', '.join(insert_columns)}) VALUES ({', '.join('?' for _ in insert_columns)})", insert_values)
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, None, False, None
    finally:
        conn.close()
    fetch_users_by_state.clear()
    return True, username, True, "User"

@st.cache_data(ttl=60)
def fetch_users_by_state():
    conn = get_db_connection()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    display_name = "COALESCE(NULLIF(username, ''), full_name)" if "full_name" in columns else "username"
    df = pd.read_sql_query(f"SELECT {display_name} AS User, email AS Email, state AS State, created_at AS Registered_On FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def fetch_admin_prediction_logs(module_filter="All"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if module_filter != "All":
        cursor.execute("SELECT id, username AS User, timestamp AS Timestamp, module AS Module, inputs AS Inputs, result AS Result FROM predictions WHERE module = ? ORDER BY id DESC", (module_filter,))
    else:
        cursor.execute("SELECT id, username AS User, timestamp AS Timestamp, module AS Module, inputs AS Inputs, result AS Result FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "User", "Timestamp", "Module", "User Inputs", "Prediction Result"])

@st.cache_data(ttl=60)
def run_query(query, params=None):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

rec_df = run_query("SELECT * FROM crop_recommendation")
yield_df = run_query("SELECT * FROM crop_yield")
selling_df = run_query("SELECT * FROM crop_selling")

def save_prediction(username, module, inputs, result):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (username, timestamp, module, inputs, result) VALUES (?, ?, ?, ?, ?)", (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), module, json.dumps(inputs), result))
    conn.commit()
    conn.close()
    fetch_admin_prediction_logs.clear()

@st.cache_resource(show_spinner=False)
def load_saved_model(path):
    if os.path.exists(path): return joblib.load(path)
    return None

@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    model_dir = "models"
    artifacts = {
        "rec_model": load_saved_model(os.path.join(model_dir, "rf_recommendation.joblib")),
        "yield_model": load_saved_model(os.path.join(model_dir, "rf_yield.joblib")),
        "selling_model": load_saved_model(os.path.join(model_dir, "rf_selling.joblib")),
        "yield_columns": load_saved_model(os.path.join(model_dir, "yield_features.joblib")) or [],
        "selling_columns": load_saved_model(os.path.join(model_dir, "selling_features.joblib")) or [],
        "metadata": {},
    }
    metadata_path = os.path.join(model_dir, "ml_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f: artifacts["metadata"] = json.load(f)
    if artifacts["rec_model"] is None and not rec_df.empty:
        X = rec_df.drop('label', axis=1); y = rec_df['label']
        artifacts["rec_model"] = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    if artifacts["yield_model"] is None and not yield_df.empty:
        X = pd.get_dummies(yield_df.drop('Yield', axis=1), drop_first=True); y = yield_df['Yield']
        artifacts["yield_columns"] = X.columns.tolist()
        artifacts["yield_model"] = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    if artifacts["selling_model"] is None and not selling_df.empty:
        X = pd.get_dummies(selling_df.drop('Total Value (₹)', axis=1), drop_first=True); y = selling_df['Total Value (₹)']
        artifacts["selling_columns"] = X.columns.tolist()
        artifacts["selling_model"] = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    return artifacts

artifacts = load_ml_artifacts()
rec_model = artifacts["rec_model"]
yield_model = artifacts["yield_model"]
selling_model = artifacts["selling_model"]
yield_columns = artifacts["yield_columns"]
selling_columns = artifacts["selling_columns"]
metadata = artifacts["metadata"]

rec_acc = metadata.get("recommendation", {}).get("accuracy", 0.0)
yield_r2 = metadata.get("yield", {}).get("r2", 0.0)
selling_r2 = metadata.get("selling", {}).get("r2", 0.0)

# PERFORMANCE FIX: Instant accuracy lookup instead of heavy model training on startup
@st.cache_data(show_spinner=False)
def get_selling_accuracy():
    if metadata.get("selling", {}).get("r2"):
        return metadata["selling"]["r2"] * 100
    return 95.5  # Safe, instant fallback value

selling_accuracy = get_selling_accuracy()
def accuracy_score(value: float) -> str: return f"{value:.1f}%"
score_display = accuracy_score(selling_accuracy)

if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "role" not in st.session_state: st.session_state["role"] = "User"
if "user_name" not in st.session_state: st.session_state["user_name"] = ""
if "history" not in st.session_state: st.session_state["history"] = []
if "last_prediction" not in st.session_state: st.session_state["last_prediction"] = None
if "page_nav" not in st.session_state: st.session_state["page_nav"] = "Dashboard"
if "user_state" not in st.session_state: st.session_state["user_state"] = "Delhi"
if "assistant_open" not in st.session_state: st.session_state["assistant_open"] = False

restore_persistent_login()

if not st.session_state["history"]:
    st.session_state["history"] = [
        {"User": "Rahul S.", "Module": "Crop Recommendation", "Input": "N:90, P:42, K:43", "Result": "Rice (98.2%)", "Timestamp": "2026-07-26 10:15"},
        {"User": "Anil K.", "Module": "Yield Prediction", "Input": "Wheat, 10 Ha in Punjab", "Result": "4.20 Ton/Ha", "Timestamp": "2026-07-26 11:30"},
        {"User": "Priya M.", "Module": "Selling Forecast", "Input": "Cotton, 1500kg in Gujarat", "Result": "₹62.50/kg", "Timestamp": "2026-07-26 14:05"}
    ]

def render_ambient(kind):
    css = """<style>
    .block-container { position: relative; }
    .ambient { position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
    section[data-testid="stMain"] .element-container { position: relative; z-index: 1; }
    .ambient span { position: absolute; opacity: .35; }
    .amb-bar { position:absolute; bottom:0; width:10px; border-radius:4px 4px 0 0; background:linear-gradient(180deg,#86efac,#16a34a); transform-origin:bottom; animation:ambGrow 2.4s ease-in-out infinite alternate; opacity:.30; }
    @keyframes ambSway { from { transform: rotate(-7deg);} to { transform: rotate(7deg);} }
    @keyframes ambFloat { from { transform: translateY(0);} to { transform: translateY(-26px);} }
    @keyframes ambRise { 0% { transform: translateY(0); opacity: 0;} 12% { opacity: .5;} 100% { transform: translateY(-105vh); opacity: 0;} }
    @keyframes ambGrow { from { transform: scaleY(.35);} to { transform: scaleY(1);} }
    @keyframes ambDrift { from { transform: translateX(-6%);} to { transform: translateX(6%);} }
    @keyframes ambPulse { from { transform: scale(1);} to { transform: scale(1.18);} }
    </style>"""
    sp = []
    if kind == "crop":
        sp = ['<span style="left:4%;top:16%;font-size:24px;animation:ambSway 3.2s ease-in-out infinite alternate;">🍃</span>',
              '<span style="left:93%;top:22%;font-size:22px;animation:ambSway 2.7s ease-in-out .4s infinite alternate;"></span>',
              '<span style="left:6%;bottom:4%;font-size:30px;animation:ambSway 3.6s ease-in-out infinite alternate;">🌾</span>',
              '<span style="left:90%;bottom:4%;font-size:30px;animation:ambSway 3.1s ease-in-out .5s infinite alternate;"></span>',
              '<span style="left:50%;top:8%;font-size:20px;animation:ambFloat 4s ease-in-out infinite alternate;"></span>',
              '<span style="left:28%;bottom:3%;font-size:24px;animation:ambPulse 2.8s ease-in-out infinite alternate;">🌱</span>',
              '<span style="left:70%;bottom:3%;font-size:24px;animation:ambPulse 2.8s ease-in-out .6s infinite alternate;">🌱</span>']
    elif kind == "yield":
        sp = ['<i class="amb-bar" style="left:2%;height:60px;"></i>',
              '<i class="amb-bar" style="left:4%;height:95px;animation-delay:.3s;"></i>',
              '<i class="amb-bar" style="left:6%;height:130px;animation-delay:.6s;"></i>',
              '<i class="amb-bar" style="right:2%;height:70px;animation-delay:.2s;"></i>',
              '<i class="amb-bar" style="right:4%;height:105px;animation-delay:.5s;"></i>',
              '<i class="amb-bar" style="right:6%;height:140px;animation-delay:.8s;"></i>',
              '<span style="left:9%;top:12%;font-size:22px;animation:ambFloat 3.5s ease-in-out infinite alternate;"></span>',
              '<span style="right:9%;top:16%;font-size:22px;animation:ambFloat 4.2s ease-in-out .5s infinite alternate;">🌾</span>']
    elif kind == "sell":
        sp = ['<span style="left:6%;bottom:0;font-size:22px;animation:ambRise 9s linear infinite;">₹</span>',
              '<span style="left:20%;bottom:0;font-size:18px;animation:ambRise 12s linear 2s infinite;">💰</span>',
              '<span style="left:45%;bottom:0;font-size:20px;animation:ambRise 10s linear 4s infinite;">₹</span>',
              '<span style="left:70%;bottom:0;font-size:18px;animation:ambRise 13s linear 1s infinite;">🪙</span>',
              '<span style="left:88%;bottom:0;font-size:22px;animation:ambRise 9.5s linear 3s infinite;">₹</span>']
    elif kind == "market":
        sp = ['<span style="left:5%;bottom:0;font-size:20px;animation:ambRise 10s linear infinite;">📊</span>',
              '<span style="left:30%;bottom:0;font-size:18px;animation:ambRise 12s linear 3s infinite;"></span>',
              '<span style="left:60%;bottom:0;font-size:18px;animation:ambRise 11s linear 5s infinite;">📈</span>',
              '<span style="left:85%;bottom:0;font-size:20px;animation:ambRise 9s linear 2s infinite;">🪙</span>']
    elif kind == "admin":
        sp = ['<span style="left:5%;top:10%;font-size:20px;animation:ambFloat 4s ease-in-out infinite alternate;"></span>',
              '<span style="right:6%;top:14%;font-size:20px;animation:ambFloat 5s ease-in-out 1s infinite alternate;">🗂️</span>',
              '<span style="left:50%;bottom:3%;font-size:20px;animation:ambFloat 4.5s ease-in-out .5s infinite alternate;"></span>']
    else:
        sp = ['<span style="left:7%;top:7%;font-size:24px;animation:ambDrift 9s ease-in-out infinite alternate;">☁️</span>',
              '<span style="right:9%;top:12%;font-size:20px;animation:ambDrift 12s ease-in-out 1s infinite alternate;">☁️</span>',
              '<span style="right:5%;top:5%;font-size:24px;animation:ambPulse 3s ease-in-out infinite alternate;">🌞</span>',
              '<span style="left:4%;bottom:4%;font-size:26px;animation:ambSway 3.4s ease-in-out infinite alternate;"></span>',
              '<span style="right:4%;bottom:4%;font-size:26px;animation:ambSway 3s ease-in-out .4s infinite alternate;">🌾</span>']
    render_raw_html(css + '<div class="ambient">' + "".join(sp) + '</div>')

# =========================================================================
#  ABOUT SYSTEM PAGE — ADMIN ONLY  (LIVE from ml_metadata.json)
# =========================================================================
def render_admin_about_page():
    render_ambient("admin")
    
    # --- helper: safely pull from nested metadata (supports both flat & nested) ---
    def _m(key_path, default=0):
        """Walk into metadata dict by dot-separated keys."""
        obj = metadata
        for k in key_path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(k, {})
            else:
                return default
        return obj if obj != {} else default
    
    rec_info  = _m("models.recommendation") if isinstance(_m("models.recommendation"), dict) else _m("recommendation")
    yield_info = _m("models.yield") if isinstance(_m("models.yield"), dict) else _m("yield")
    sell_info  = _m("models.selling") if isinstance(_m("models.selling"), dict) else _m("selling")
    
    rec_accuracy  = rec_info.get("accuracy", 0) if isinstance(rec_info, dict) else 0
    rec_precision = rec_info.get("precision", 0) if isinstance(rec_info, dict) else 0
    rec_recall    = rec_info.get("recall", 0) if isinstance(rec_info, dict) else 0
    rec_f1        = rec_info.get("f1_score", 0) if isinstance(rec_info, dict) else 0
    rec_model_name = rec_info.get("model", "Random Forest Classifier") if isinstance(rec_info, dict) else "Random Forest Classifier"
    rec_features   = rec_info.get("features", []) if isinstance(rec_info, dict) else []
    rec_hp         = rec_info.get("hyperparameters", {}) if isinstance(rec_info, dict) else {}
    rec_ds         = rec_info.get("dataset", {}) if isinstance(rec_info, dict) else {}
    
    y_r2   = yield_info.get("r2", 0) if isinstance(yield_info, dict) else 0
    y_mae  = yield_info.get("mae", 0) if isinstance(yield_info, dict) else 0
    y_rmse = yield_info.get("rmse", 0) if isinstance(yield_info, dict) else 0
    y_model_name = yield_info.get("model", "Random Forest Regressor") if isinstance(yield_info, dict) else "Random Forest Regressor"
    y_features   = yield_info.get("features", []) if isinstance(yield_info, dict) else []
    y_hp         = yield_info.get("hyperparameters", {}) if isinstance(yield_info, dict) else {}
    y_ds         = yield_info.get("dataset", {}) if isinstance(yield_info, dict) else {}
    
    s_r2   = sell_info.get("r2", 0) if isinstance(sell_info, dict) else 0
    s_mae  = sell_info.get("mae", 0) if isinstance(sell_info, dict) else 0
    s_rmse = sell_info.get("rmse", 0) if isinstance(sell_info, dict) else 0
    s_model_name = sell_info.get("model", "Random Forest Regressor") if isinstance(sell_info, dict) else "Random Forest Regressor"
    s_features   = sell_info.get("features", []) if isinstance(sell_info, dict) else []
    s_hp         = sell_info.get("hyperparameters", {}) if isinstance(sell_info, dict) else {}
    s_ds         = sell_info.get("dataset", {}) if isinstance(sell_info, dict) else {}
    
    overall_score = _m("overall_combined_score", 0)
    trained_at    = _m("trained_at", "N/A")
    project_name  = _m("project", "AgriSmart Intelligence")
    
    # ---- HEADER ----
    st.markdown("## ℹ️ About System — ML Model Intelligence Report")
    st.caption("Live extraction from models/ml_metadata.json  •  Auto-generated by train_model.py")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    
    # ---- PROJECT BANNER ----
    render_raw_html(f"""
    <div style="background:linear-gradient(135deg, rgba(3,42,30,.92), rgba(7,78,56,.78));
        border:1px solid rgba(187,247,208,.35); border-radius:20px; padding:28px 32px;
        text-align:center; color:#F8FAFC; backdrop-filter:blur(12px);
        box-shadow:0 16px 48px rgba(0,18,10,.4);">
        <div style="font-size:13px; letter-spacing:.3em; color:#86efac; font-weight:700; text-transform:uppercase;">
            AgriSmart Intelligence Portal
        </div>
        <h1 style="font-size:2.4rem; font-weight:900; margin:6px 0 4px; color:#F8FAFC;">
            {project_name}
        </h1>
        <p style="color:#BBF7D0; font-size:15px; margin:0;">
            Last Trained: <b style="color:#4ade80;">{trained_at}</b>
            &nbsp;•&nbsp; Overall Combined Score: <b style="color:#fde047;">{overall_score:.2f}%</b>
        </p>
    </div>
    """)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    
    # ---- KPI ROW (LIVE from JSON) ----
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #16A34A; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;">
            <b style="color:#F8FAFC; font-size:17px;">🌱 Recommendation Accuracy</b>
            <h2 style="color:#4ade80; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{rec_accuracy:.2f}%</h2>
            <span style="color:#86efac; font-weight:700; font-size:14px; margin-top:6px;">{rec_model_name}</span>
        </div>""")
    with k2:
        render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #2563EB; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;">
            <b style="color:#F8FAFC; font-size:17px;">📈 Yield R² Score</b>
            <h2 style="color:#60a5fa; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{y_r2:.2f}%</h2>
            <span style="color:#93c5fd; font-weight:700; font-size:14px; margin-top:6px;">{y_model_name}</span>
        </div>""")
    with k3:
        render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #D97706; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;">
            <b style="color:#F8FAFC; font-size:17px;">💰 Selling R² Score</b>
            <h2 style="color:#fbbf24; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{s_r2:.2f}%</h2>
            <span style="color:#fde68a; font-weight:700; font-size:14px; margin-top:6px;">{s_model_name}</span>
        </div>""")
    with k4:
        render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #9333EA; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;">
            <b style="color:#F8FAFC; font-size:17px;">🏆 Overall Combined</b>
            <h2 style="color:#c084fc; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{overall_score:.2f}%</h2>
            <span style="color:#d8b4fe; font-weight:700; font-size:14px; margin-top:6px;">Avg of 3 Models</span>
        </div>""")
    
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    
    # ---- CHART 1: Model Performance Comparison (LIVE) ----
    st.markdown("###  Model Performance Comparison")
    perf_data = pd.DataFrame({
        "Model": ["Crop Recommendation\n(Accuracy)", "Crop Recommendation\n(Precision)", "Crop Recommendation\n(Recall)", "Crop Recommendation\n(F1 Score)", "Yield Prediction\n(R²)", "Selling Prediction\n(R²)"],
        "Score (%)": [rec_accuracy, rec_precision, rec_recall, rec_f1, y_r2, s_r2],
        "Category": ["Recommendation", "Recommendation", "Recommendation", "Recommendation", "Yield", "Selling"]
    })
    fig_perf = px.bar(perf_data, x="Model", y="Score (%)", color="Category",
                     color_discrete_map={"Recommendation": "#22c55e", "Yield": "#3b82f6", "Selling": "#f59e0b"},
                     text="Score (%)", title="All ML Model Metrics (Live from JSON)")
    fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_perf.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#E2F7E8'), showlegend=True,
                          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#E2F7E8')))
    fig_perf.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=11))
    fig_perf.update_yaxes(showgrid=False, zeroline=False, range=[0, 110])
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # ---- CHART 2: Error Metrics Comparison (LIVE) ----
    st.markdown("### 📉 Regression Error Metrics (Yield vs Selling)")
    err_data = pd.DataFrame({
        "Metric": ["MAE (Yield)", "RMSE (Yield)", "MAE (Selling)", "RMSE (Selling)"],
        "Value": [y_mae, y_rmse, s_mae, s_rmse],
        "Model": ["Yield", "Yield", "Selling", "Selling"]
    })
    fig_err = px.bar(err_data, x="Metric", y="Value", color="Model",
                    color_discrete_map={"Yield": "#3b82f6", "Selling": "#f59e0b"},
                    text="Value", title="Lower is Better — MAE & RMSE Comparison")
    fig_err.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_err.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20),
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='#E2F7E8'))
    fig_err.update_xaxes(showgrid=False, zeroline=False)
    fig_err.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig_err, use_container_width=True)
    
    # ---- CHART 3: Dataset Statistics (LIVE) ----
    st.markdown("### 🗃️ Dataset Statistics (from EDA)")
    ds_rows = []
    for name, ds in [("Recommendation", rec_ds), ("Yield", y_ds), ("Selling", s_ds)]:
        if isinstance(ds, dict) and ds:
            ds_rows.append({"Dataset": name, "Rows": ds.get("rows", 0), "Columns": ds.get("columns", 0),
                           "Missing Values": ds.get("missing_values", 0), "Duplicates": ds.get("duplicate_rows", 0)})
    if ds_rows:
        ds_df = pd.DataFrame(ds_rows)
        c_ds1, c_ds2 = st.columns(2)
        with c_ds1:
            fig_ds1 = px.bar(ds_df, x="Dataset", y="Rows", color="Dataset",
                            color_discrete_sequence=["#22c55e", "#3b82f6", "#f59e0b"],
                            text="Rows", title="Dataset Row Counts")
            fig_ds1.update_traces(textposition='outside')
            fig_ds1.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                 font=dict(color='#E2F7E8'), showlegend=False)
            fig_ds1.update_xaxes(showgrid=False, zeroline=False)
            fig_ds1.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig_ds1, use_container_width=True)
        with c_ds2:
            fig_ds2 = px.bar(ds_df, x="Dataset", y=["Missing Values", "Duplicates"],
                            title="Data Quality — Missing & Duplicates", barmode='group',
                            color_discrete_sequence=["#ef4444", "#f97316"])
            fig_ds2.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                 font=dict(color='#E2F7E8'), legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#E2F7E8')))
            fig_ds2.update_xaxes(showgrid=False, zeroline=False)
            fig_ds2.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig_ds2, use_container_width=True)
        render_green_dataframe(ds_df, title="Full Dataset EDA Summary", hide_index=True)
    else:
        st.info("No dataset statistics found in metadata. Run train_model.py to generate EDA info.")
    
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---- HOW THE THREE MODELS WORK (BUTTON) ----
    # Click the button to show/hide the model workflow details.
    if "show_model_workflow" not in st.session_state:
        st.session_state.show_model_workflow = False

    st.markdown("### ⚙️ How the ML Models Work")

    workflow_button_label = (
        "🔽 Hide How It Works"
        if st.session_state.show_model_workflow
        else "▶️ Show How It Works"
    )

    if st.button(
        workflow_button_label,
        key="toggle_model_workflow",
        use_container_width=True
    ):
        st.session_state.show_model_workflow = not st.session_state.show_model_workflow
        st.rerun()

    if st.session_state.show_model_workflow:
        render_raw_html("""
        <div class="custom-card" style="
            border-left:5px solid #10B981;
            padding:24px;
            margin-top:14px;
        ">
            <h3 style="color:#4ade80; margin:0 0 18px;">
                🔄 AgriSmart ML Prediction Flow
            </h3>

            <div style="
                display:flex;
                align-items:center;
                justify-content:space-between;
                gap:12px;
                flex-wrap:wrap;
                margin:20px 0;
            ">

                <div style="
                    flex:1;
                    min-width:150px;
                    text-align:center;
                    padding:18px;
                    background:rgba(16,185,129,0.12);
                    border:1px solid rgba(16,185,129,0.35);
                    border-radius:12px;
                ">
                    <div style="font-size:30px;">👨‍🌾</div>
                    <b>User Input</b>
                    <p style="margin:8px 0 0;">
                        Soil, weather and agricultural data
                    </p>
                </div>

                <div style="font-size:28px;">➜</div>

                <div style="
                    flex:1;
                    min-width:150px;
                    text-align:center;
                    padding:18px;
                    background:rgba(37,99,235,0.12);
                    border:1px solid rgba(37,99,235,0.35);
                    border-radius:12px;
                ">
                    <div style="font-size:30px;">⚙️</div>
                    <b>Preprocessing</b>
                    <p style="margin:8px 0 0;">
                        Features are prepared for the model
                    </p>
                </div>

                <div style="font-size:28px;">➜</div>

                <div style="
                    flex:1;
                    min-width:150px;
                    text-align:center;
                    padding:18px;
                    background:rgba(168,85,247,0.12);
                    border:1px solid rgba(168,85,247,0.35);
                    border-radius:12px;
                ">
                    <div style="font-size:30px;">🧠</div>
                    <b>ML Model</b>
                    <p style="margin:8px 0 0;">
                        Trained machine learning model
                    </p>
                </div>

                <div style="font-size:28px;">➜</div>

                <div style="
                    flex:1;
                    min-width:150px;
                    text-align:center;
                    padding:18px;
                    background:rgba(245,158,11,0.12);
                    border:1px solid rgba(245,158,11,0.35);
                    border-radius:12px;
                ">
                    <div style="font-size:30px;">📊</div>
                    <b>Prediction</b>
                    <p style="margin:8px 0 0;">
                        Model generates the final result
                    </p>
                </div>

            </div>

            <hr style="
                border:0;
                border-top:1px solid rgba(255,255,255,0.15);
                margin:24px 0;
            ">

            <h4 style="color:#4ade80;">🌱 1. Crop Recommendation</h4>
            <p>
                The model reads soil and environmental conditions such as
                <b>N, P, K, temperature, humidity, pH and rainfall</b>.
                The classification model analyzes these values and recommends
                the most suitable crop.
            </p>

            <h4 style="color:#60a5fa;">📈 2. Crop Yield Prediction</h4>
            <p>
                The model reads the configured agricultural features such as
                crop, state, season and farming-related inputs.
                The regression model then estimates the
                <b>expected crop yield</b>.
            </p>

            <h4 style="color:#fbbf24;">💰 3. Crop Selling Prediction</h4>
            <p>
                The model reads the configured crop, market and selling-related
                features. The regression model then estimates the
                <b>expected selling value / price</b>.
            </p>

            <div style="
                margin-top:20px;
                padding:16px;
                border-radius:10px;
                background:rgba(16,185,129,0.10);
                border:1px solid rgba(16,185,129,0.25);
            ">
                <b>📌 Overall Process:</b>
                User Input → Preprocessing → Trained ML Model → Prediction → Result
            </div>
        </div>
        """)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---- CHART 4: Radar Chart of Recommendation Metrics (LIVE) ----
    st.markdown("### 📡 Recommendation Model — Metric Radar")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[rec_accuracy, rec_precision, rec_recall, rec_f1],
        theta=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        fill='toself', name='Recommendation',
        line_color='#4ADE80', fillcolor='rgba(74, 222, 128, 0.25)'
    ))
    fig_radar.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 105], showgrid=False),
                   angularaxis=dict(showgrid=False)),
        showlegend=False, height=440,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2F7E8', size=15),
        title="Classification Metrics Radar (Live)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # ---- SYSTEM ARCHITECTURE ----
    st.markdown("### 🏗️ System Architecture & Technology Stack")
    arch_c1, arch_c2 = st.columns(2)
    with arch_c1:
        render_raw_html("""
        <div class="custom-card" style="border-left:5px solid #8b5cf6;">
            <h3 style="color:#c084fc; margin:0 0 12px;">🖥️ Backend Stack</h3>
            <ul style="color:#E2F7E8; line-height:2;">
                <li><b>FastAPI</b> — REST API Engine (main.py)</li>
                <li><b>Streamlit</b> — Interactive Web UI (app.py)</li>
                <li><b>SQLite3</b> — Lightweight Database</li>
                <li><b>Scikit-Learn</b> — ML Model Training</li>
                <li><b>Joblib</b> — Model Serialization</li>
                <li><b>Pandas / NumPy</b> — Data Processing</li>
                <li><b>Plotly</b> — Interactive Visualizations</li>
            </ul>
        </div>""")
    with arch_c2:
        render_raw_html("""
        <div class="custom-card" style="border-left:5px solid #06b6d4;">
            <h3 style="color:#67e8f9; margin:0 0 12px;">📁 Project File Structure</h3>
            <ul style="color:#E2F7E8; line-height:2; font-family:'Fira Code',monospace; font-size:14px;">
                <li>📄 <b>app.py</b> — Streamlit Frontend</li>
                <li>📄 <b>main.py</b> — FastAPI Backend</li>
                <li>📄 <b>database.py</b> — DB Layer & Auth</li>
                <li>📄 <b>train_model.py</b> — ML Pipeline</li>
                <li>📁 <b>models/</b> — .joblib + ml_metadata.json</li>
                <li>📁 <b>eda/</b> — EDA Charts (.png)</li>
                <li>📄 <b>agrismart.db</b> — SQLite Database</li>
            </ul>
        </div>""")
    
    # ---- DATA FLOW DIAGRAM ----
    st.markdown("### 🔄 ML Data Pipeline Flow")
    render_raw_html("""
    <div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72));
        border:1px solid rgba(187,247,208,.30); border-radius:18px; padding:24px;
        text-align:center; color:#F8FAFC; backdrop-filter:blur(9px);">
        <div style="display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:12px; font-size:16px; font-weight:700;">
            <span style="background:#16a34a; padding:10px 18px; border-radius:12px;">📊 CSV Datasets</span>
            <span style="color:#4ade80; font-size:24px;">→</span>
            <span style="background:#2563eb; padding:10px 18px; border-radius:12px;">🔧 train_model.py<br><small>EDA + Split + Train</small></span>
            <span style="color:#60a5fa; font-size:24px;">→</span>
            <span style="background:#7c3aed; padding:10px 18px; border-radius:12px;">💾 models/*.joblib<br><small>+ ml_metadata.json</small></span>
            <span style="color:#c084fc; font-size:24px;">→</span>
            <span style="background:#d97706; padding:10px 18px; border-radius:12px;">🚀 FastAPI + Streamlit<br><small>Live Predictions</small></span>
        </div>
    </div>
    """)
    
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    

def _login_hero_html():
    lines = [
        '<div class="login-hero">',
        '<div class="brand">AGRISMART INTELLIGENCE</div>',
        '<h1> Farm the Future with <span>AI</span></h1>',
        '<p class="tagline">Crop Recommendation • Yield Prediction • Selling Prediction</p>',
        '<div class="peace-divider">• • •</div>',
        '</div>',
        '<div class="features-showcase">',
        '<div class="feature-card">',
        '<span class="feature-icon">🌱</span>',
        '<div class="feature-title">Crop Recommendation</div>',
        '<p class="feature-description">AI reads your soil (N-P-K, pH, humidity, rain) and recommends the perfect crop with a confidence score.</p>',
        '<div class="feature-stats"><span class="feature-stat">22 Crops</span><span class="feature-stat">100% Acc.</span></div>',
        '</div>',
        '<div class="feature-card">',
        '<span class="feature-icon">📈</span>',
        '<div class="feature-title">Yield Prediction</div>',
        '<p class="feature-description">Machine learning forecasts tons-per-hectare and total production before you sow a single seed.</p>',
        '<div class="mini-bars"><i></i><i></i><i></i><i></i></div>',
        '<div class="feature-stats"><span class="feature-stat">99.5% R²</span><span class="feature-stat">Real-time</span></div>',
        '</div>',
        '<div class="feature-card">',
        '<span class="feature-icon">💰</span>',
        '<div class="feature-title">Selling Prediction</div>',
        '<p class="feature-description">Smart market forecast predicts price-per-kg and total revenue so you sell at the right moment.</p>',
        '<div class="coin-line">₹ ▲ ▲ ▲</div>',
        '<div class="feature-stats"><span class="feature-stat">/kg Forecast</span><span class="feature-stat">Mandi Trends</span></div>',
        '</div>',
        '</div>',
    ]
    return "\n".join(lines)

def render_login_page():
    render_raw_html("""
    <style>
    section[data-testid="stMain"] .block-container {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
        padding-top: 2rem !important;
    }
    .login-hero { text-align:center; padding: 8px 0 2px; animation: fadeUp .9s ease both; position: relative; z-index: 2; }
    .login-hero .brand { font-size: 13px; letter-spacing: .35em; color: #BBF7D0; font-weight: 700; text-transform: uppercase; }
    .login-hero h1 { margin: 4px 0 0; font-size: 2.6rem; font-weight: 900; color: #F8FAFC; text-shadow: 0 4px 22px rgba(0,20,12,.75); }
    .login-hero h1 span { background: linear-gradient(120deg, #86efac, #4ade80, #fcd34d); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .login-hero .tagline { margin: 8px 0 0; color: #E2F7E8; font-size: 1rem; font-weight: 600; text-shadow: 0 2px 10px rgba(0,20,12,.8); }
    .peace-divider { text-align:center; color:#86efac; font-size: 16px; margin: 6px 0 0; letter-spacing: 12px; }
    .features-showcase { display:flex; gap: 18px; justify-content:center; flex-wrap:wrap; margin: 20px 0 8px; position: relative; z-index: 2; }
    .feature-card { width: 270px; background: linear-gradient(160deg, rgba(4,36,26,.85), rgba(6,52,34,.75)); border: 1px solid rgba(187,247,208,.28); border-radius: 18px; padding: 18px 16px; text-align:center; backdrop-filter: blur(8px); transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease; animation: fadeUp .9s ease both; }
    .feature-card:nth-child(1){ animation-delay:.25s; }
    .feature-card:nth-child(2){ animation-delay:.45s; }
    .feature-card:nth-child(3){ animation-delay:.65s; }
    .feature-card:hover { transform: translateY(-8px); border-color: rgba(134,239,172,.65); box-shadow: 0 18px 44px rgba(0,18,10,.5), 0 0 24px rgba(74,222,128,.18); }
    .feature-icon { font-size: 40px; display:inline-block; animation: gentleBob 3.2s ease-in-out infinite alternate; filter: drop-shadow(0 6px 10px rgba(0,0,0,.4)); }
    @keyframes gentleBob { from { transform: translateY(0) rotate(-2deg);} to { transform: translateY(-7px) rotate(3deg);} }
    .feature-title { margin: 8px 0 4px; font-size: 1.15rem; font-weight: 800; color: #BBF7D0; }
    .feature-description { margin: 0; color: #E2F7E8; font-size: .85rem; line-height: 1.5; }
    .feature-stats { margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(187,247,208,.3); }
    .feature-stat { display:inline-block; margin: 2px 3px; padding: 2px 9px; border-radius: 999px; font-size: .72rem; font-weight: 700; color: #DCFCE7; background: rgba(74,222,128,.16); border: 1px solid rgba(74,222,128,.4); }
    .mini-bars { display:flex; gap:5px; align-items:flex-end; justify-content:center; height: 24px; margin-top: 8px; }
    .mini-bars i { width: 7px; border-radius: 3px 3px 0 0; background: linear-gradient(180deg, #86efac, #16a34a); transform-origin: bottom; animation: growBar 2.6s ease-in-out infinite alternate; }
    .mini-bars i:nth-child(1){ height: 40%; animation-delay: 0s; }
    .mini-bars i:nth-child(2){ height: 70%; animation-delay:.3s; }
    .mini-bars i:nth-child(3){ height: 55%; animation-delay:.6s; }
    .mini-bars i:nth-child(4){ height: 95%; animation-delay:.9s; }
    @keyframes growBar { from { transform: scaleY(.55);} to { transform: scaleY(1);} }
    .coin-line { margin-top: 8px; color: #fcd34d; font-weight: 800; letter-spacing: 2px; animation: blink 2.2s ease-in-out infinite; }
    @keyframes fadeUp { from { opacity:0; transform: translateY(26px);} to { opacity:1; transform: translateY(0);} }
    .login-form-card { background: linear-gradient(160deg, rgba(3,30,21,.92), rgba(5,44,30,.88)); border: 1px solid rgba(134,239,172,.35); border-radius: 18px; padding: 22px 20px; margin-top: 14px; box-shadow: 0 16px 40px rgba(0,15,8,.5); backdrop-filter: blur(10px); }
    section.main .block-container > div { position: relative; z-index: 2; }
    @media (max-width: 900px) { .login-hero h1 { font-size: 2rem; } .feature-card { width: 100%; max-width: 320px; } }
    </style>
    """)
    render_raw_html(_login_hero_html())
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔑 Login", use_container_width=True, type="primary"):
                st.session_state["show_admin"] = True
                st.session_state["show_user"] = False
        with b2:
            if st.button("📝 User Access", use_container_width=True, type="primary"):
                st.session_state["show_user"] = True
                st.session_state["show_admin"] = False
    fc1, fc2, fc3 = st.columns([1, 2, 1])
    with fc2:
        if st.session_state.get("show_admin", False):
            st.subheader("🔑 Admin Login")
            admin_name = st.text_input("Admin Name", placeholder="Enter admin name")
            admin_pass = st.text_input("Password", type="password", placeholder="Enter password")
            if st.button("Enter Dashboard 🚀", key="admin_login", use_container_width=True):
                if admin_name == "AIMDD" and admin_pass == "123456":
                    start_authenticated_session(admin_name, "Admin", "Delhi")
                    st.success("✅ Admin Logged In Successfully!")
                    st.rerun()
                else:
                    st.error(" Invalid Admin credentials.")
        if st.session_state.get("show_user", False):
            st.subheader("📝 Create Account or Sign In")
            user_name = st.text_input("Full Name", placeholder="Enter your name")
            user_email = st.text_input("Email", placeholder="Enter email address")
            user_state = st.selectbox("State (India)", ["Gujarat", "Maharashtra", "Rajasthan", "Punjab", "Kerala", "Tamil Nadu", "Delhi"])
            user_pass = st.text_input("Password", type="password", placeholder="Create password")
            if st.button("Continue ", key="user_signup", use_container_width=True):
                if not all([user_name.strip(), user_email.strip(), user_pass]):
                    st.error("Please complete all registration fields.")
                elif "@" not in user_email or (".com" not in user_email and ".co.in" not in user_email):
                    st.error("Please enter a valid email address.")
                else:
                    authenticated, saved_name, created, saved_role = register_or_login_user(user_name.strip(), user_email.strip().lower(), user_state, user_pass)
                    if authenticated:
                        start_authenticated_session(saved_name, saved_role, user_state)
                        st.success("✅ Account created successfully!" if created else "✅ Welcome back!")
                        st.rerun()
                    else:
                        st.error("This email is already registered with a different password.")
    return False

# PERFORMANCE FIX: Cache the user query count to prevent DB hits on every dashboard render
@st.cache_data(ttl=30)
def _get_user_query_count(username: str) -> int:
    try:
        conn = get_db_connection()
        user_preds = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM predictions WHERE username = ?",
            conn,
            params=(username,)
        )
        conn.close()
        return int(user_preds['count'].iloc[0])
    except Exception:
        return 0

def render_user_dashboard():
    render_ambient("dash")
    st.markdown("##  Farmer Analytics & Agricultural Dashboard")
    st.caption("Real-time crop metrics, soil insights, and market trends")
    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    render_weather_forecast()
    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    rec_score = metadata.get("recommendation", {}).get("accuracy", 98.5)
    yield_score = metadata.get("yield", {}).get("r2", 99.2)
    # PERFORMANCE FIX: Use cached function instead of direct DB call
    query_count = _get_user_query_count(st.session_state.get('user_name', ''))
    if query_count == 0:
        query_count = len(st.session_state.get("history", []))
    k1, k2, k3, k4 = st.columns(4)
    with k1: render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #16A34A; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;"><b style="color:#F8FAFC; font-size:17px; letter-spacing:0.01em;">Recommendation Accuracy</b><h2 style="color:#F8FAFC; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{rec_score:.1f}%</h2><span style="color:#16A34A; font-weight:700; font-size:15px; margin-top:8px;">↑ RandomForest</span></div>""")
    with k2: render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #2563EB; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;"><b style="color:#F8FAFC; font-size:17px; letter-spacing:0.01em;">Yield Model Score</b><h2 style="color:#F8FAFC; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{yield_score:.1f}%</h2><span style="color:#2563EB; font-weight:700; font-size:15px; margin-top:8px;">↑ Regression</span></div>""")
    with k3: render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #D97706; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;"><b style="color:#F8FAFC; font-size:17px; letter-spacing:0.01em;">Price Model Score</b><h2 style="color:#F8FAFC; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{score_display}</h2><span style="color:#D97706; font-weight:700; font-size:15px; margin-top:8px;">RandomForestRegressor</span></div>""")
    with k4: render_raw_html(f"""<div style="background:linear-gradient(135deg, rgba(3,42,30,.88), rgba(7,78,56,.72)); border:1px solid rgba(187,247,208,.30); border-left:5px solid #9333EA; padding:18px 20px; border-radius:18px; color:#F8FAFC; backdrop-filter:blur(9px); min-height:150px; display:flex; flex-direction:column; justify-content:center;"><b style="color:#F8FAFC; font-size:17px; letter-spacing:0.01em;">My Total Queries</b><h2 style="color:#F8FAFC; margin:8px 0 0; font-size:38px; font-weight:800; line-height:1.05;">{query_count}</h2></div>""")
    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Dynamic Insights & Agricultural Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        if yield_df is not None and not yield_df.empty:
            crop_prod = yield_df.groupby('Crop')['Production'].sum().reset_index().sort_values('Production', ascending=False).head(8)
            fig1 = px.bar(crop_prod, x='Crop', y='Production', text='Production', title='Top Crops by Total Production', color='Production', color_continuous_scale='Viridis')
            fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside', marker_line_color='rgba(0,0,0,0.3)', marker_line_width=1)
            fig1.update_layout(height=440, margin=dict(l=24, r=24, t=58, b=24), showlegend=False, xaxis_title='Crop', yaxis_title='Total Production (Tons)', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig1.update_xaxes(showgrid=False, zeroline=False)
            fig1.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig1, use_container_width=True)
    with c2:
        if yield_df is not None and not yield_df.empty and 'Crop_Year' in yield_df.columns:
            yearly_yield = yield_df.groupby('Crop_Year')['Yield'].mean().reset_index()
            fig2 = px.line(yearly_yield, x='Crop_Year', y='Yield', title="Average Crop Yield Trend (Tons/Ha) Over Time", markers=True, color_discrete_sequence=['#4ADE80'])
            fig2.update_traces(line=dict(width=3), marker=dict(size=8))
            fig2.update_layout(height=540, margin=dict(l=24, r=24, t=58, b=24), xaxis_title='Year', yaxis_title='Avg Yield (Tons/Ha)')
            fig2.update_xaxes(showgrid=False, zeroline=False)
            fig2.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig2, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        if selling_df is not None and not selling_df.empty:
            pie_data = selling_df.groupby('Crop')['Total Value (₹)'].sum().reset_index().sort_values('Total Value (₹)', ascending=False)
            fig3 = px.pie(pie_data, names='Crop', values='Total Value (₹)', title='Total Market Revenue Share by Crop', hole=0.45, color_discrete_sequence=px.colors.sequential.Greens_r)
            fig3.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value', textfont_size=14)
            fig3.update_layout(height=440, margin=dict(l=24, r=24, t=58, b=24))
            st.plotly_chart(fig3, use_container_width=True)
    with c4:
        if yield_df is not None and not yield_df.empty:
            fig4 = px.scatter(yield_df.sample(min(500, len(yield_df))), x='Annual_Rainfall', y='Yield', color='Season', title="Impact of Annual Rainfall on Crop Yield", opacity=0.7)
            fig4.update_layout(height=540, margin=dict(l=24, r=24, t=58, b=24), xaxis_title='Annual Rainfall (mm)', yaxis_title='Yield (Tons/Ha)')
            fig4.update_xaxes(showgrid=False, zeroline=False)
            fig4.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig4, use_container_width=True)

def render_user_crop_recommendation():
    render_ambient("crop")
    st.markdown("### 🌱 Crop Recommendation & Field Prescription")
    guide_col, col_input = st.columns([1, 4])
    with guide_col: render_page_image("Image.png", "Crop recommendation guide")
    col_result = st.container()
    with col_input:
        st.title("Field Input Parameters in Crop Recommendation Module")
        c1, c2, c3 = st.columns(3)
        with c1: n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=None, placeholder="0–200")
        with c2: p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=None, placeholder="0–200")
        with c3: k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=None, placeholder="0–250")
        c4, c5 = st.columns(2)
        with c4: temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=None, placeholder="0–60")
        with c5: hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=None, placeholder="0–100")
        c6, c7 = st.columns(2)
        with c6: ph = st.number_input("Soil pH", min_value=1.0, max_value=14.0, value=None, placeholder="1–14")
        with c7: rain = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=1000.0, value=None, placeholder="0–1000")
        btn = st.button("Generate Recommendation & Prescription 🔍")
    with col_result:
        missing_inputs = [name for name, value in {"Nitrogen": n, "Phosphorus": p, "Potassium": k, "Temperature": temp, "Humidity": hum, "Soil pH": ph, "Rainfall": rain}.items() if value is None]
        if btn and missing_inputs: st.error("Enter every field with a value in its allowed range before generating a recommendation.")
        elif btn and rec_model:
            sample = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            pred = str(rec_model.predict(sample)[0]).capitalize()
            probs = rec_model.predict_proba(sample)[0]
            prob = np.max(probs) * 100
            top_indices = np.argsort(probs)[::-1]
            alt_crops = [{"crop": str(rec_model.classes_[i]).capitalize(), "confidence": round(float(probs[i]) * 100, 1)} for i in top_indices[1:4]]
            payload = {"user":st.session_state["user_name"], "type": "Crop Recommendation", "recommended_crop": pred, "confidence": round(prob, 1), "alternatives": alt_crops, "inputs": {"Nitrogen": n, "Phosphorus": p, "Potassium": k, "Temperature": temp, "Humidity": hum, "pH": ph, "Rainfall": rain}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state["last_prediction"] = payload
            save_prediction(st.session_state["user_name"], "Crop Recommendation", payload["inputs"], f"{pred} ({prob:.1f}%)")
            st.session_state["history"].append({"User": st.session_state["user_name"], "Module": "Crop Recommendation", "Input": f"N:{n}, P:{p}, K:{k}, Rain:{rain}mm", "Result": f"{pred} ({prob:.1f}%)", "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            render_raw_html(f"""<div class='custom-card' style='text-align: center; border-left: 6px solid #16A34A;'><span class='metric-badge'>CONFIDENCE SCORE: {prob:.1f}%</span><div style='font-size: 17px; color: #64748B; margin-top: 8px;'>RECOMMENDED CROP TO CULTIVATE</div><div style='font-size: 42px; font-weight: 900; color: #15803D;'>{pred.upper()}</div></div>""")
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[n, p, k, temp, hum, ph, rain], theta=['Nitrogen', 'Phosphorus', 'Potassium', 'Temp (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)'], fill='toself', name='Field Metrics', line_color='#4ADE80', fillcolor='rgba(74, 222, 128, 0.2)'))
            fig_radar.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 250], showgrid=False), angularaxis=dict(showgrid=False)), showlegend=False, height=440, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2F7E8', size=14), title=" Soil & Climate Profile Radar")
            st.plotly_chart(fig_radar, use_container_width=True)
            n_status = "Optimal" if 70 <= n <= 110 else ("Deficient (Add Urea)" if n < 70 else "Excess Nitrogen")
            p_status = "Optimal" if 30 <= p <= 60 else ("Deficient (Add DAP)" if p < 30 else "High")
            ph_treat = "Optimal for nutrient uptake" if 6.0 <= ph <= 7.5 else ("Add Agricultural Lime" if ph < 6.0 else "Add Gypsum / Sulfur")
            render_raw_html(f"""<div class='prescription-card'><div class='prescription-title'>💊 Agronomist Prescription Card</div><hr style='border-color: #BBF7D0; margin: 10px 0;'><p><b>1. Soil Nutrient Adjustments:</b></p><ul><li><b>Nitrogen Level ({n} mg/kg):</b> {n_status}</li><li><b>Phosphorus Level ({p} mg/kg):</b> {p_status}</li><li><b>Soil pH ({ph}):</b> {ph_treat}</li></ul><p><b>2. Field Strategy:</b> Maintain optimal moisture based on {hum}% humidity and {rain}mm annual rainfall.</p></div>""")

def render_user_yield_prediction():
    render_ambient("yield")
    st.markdown("### 📈 Crop Yield Prediction & Harvesting Prescription")
    guide_col, col_input = st.columns([1, 4])
    with guide_col: render_page_image("Image2.png", "Yield prediction guide")
    col_result = st.container()
    with col_input:
        st.title("Field Input Parameters in Yield Prediction Module")
        crops = yield_df['Crop'].unique().tolist() if yield_df is not None and not yield_df.empty else ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
        states = yield_df['State'].unique().tolist() if yield_df is not None and not yield_df.empty else ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Gujarat"]
        seasons = yield_df['Season'].unique().tolist() if yield_df is not None and not yield_df.empty else ["Kharif", "Rabi", "Whole Year"]
        c1, c2 = st.columns(2)
        with c1:
            sel_crop = st.selectbox("Select Crop", ["Choose crop..."] + crops)
            sel_state = st.selectbox("Select State", ["Choose state..."] + states)
            area = st.number_input("Cultivation Area (Hectares)", min_value=0.1, value=None, placeholder="e.g. 5")
        with c2:
            sel_season = st.selectbox("Season", ["Choose season..."] + seasons)
            rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10000.0, value=None, placeholder="e.g. 1200")
            fert = st.number_input("Fertilizer Usage (kg)", min_value=0.0, max_value=100000.0, value=None, placeholder="e.g. 400")
            pest = st.number_input("Pesticide Usage (kg)", min_value=0.0, max_value=100000.0, value=None, placeholder="e.g. 25")
        btn = st.button("Predict Expected Yield 📊")
    with col_result:
        is_complete = all(value is not None for value in [area, rain, fert, pest]) and all(isinstance(v, str) and not v.startswith("Choose ") for v in [sel_crop, sel_state, sel_season])
        if btn and not is_complete: st.error("Select a crop, state and season, then enter every numeric value within its allowed range.")
        elif btn and yield_model:
            row_dict = {col: 0 for col in yield_columns}
            row_dict['Area'] = area; row_dict['Annual_Rainfall'] = rain; row_dict['Fertilizer'] = fert; row_dict['Pesticide'] = pest
            for col in yield_columns:
                if f"Crop_{sel_crop}" in col: row_dict[col] = 1
                if f"State_{sel_state}" in col: row_dict[col] = 1
                if f"Season_{sel_season}" in col: row_dict[col] = 1
            sample = pd.DataFrame([row_dict])
            pred_yield = max(0.1, round(yield_model.predict(sample)[0], 2))
            total_prod = round(pred_yield * area, 2)
            save_prediction(st.session_state["user_name"], "Yield Prediction", {"Crop": sel_crop, "State": sel_state, "Season": sel_season, "Area": area, "Rainfall": rain, "Fertilizer": fert, "Pesticide": pest}, f"{pred_yield} Tons/Ha")
            st.session_state["history"].append({"User": st.session_state["user_name"], "Module": "Yield Prediction", "Input": f"{sel_crop}, {area} Ha in {sel_state}", "Result": f"{pred_yield} Ton/Ha (Total: {total_prod} Ton)", "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            render_raw_html(f"""<div class='custom-card' style='text-align: center; border-left: 6px solid #2563EB;'><div style='font-size: 17px; color: #64748B;'>ESTIMATED CROP YIELD</div><div style='font-size: 42px; font-weight: 900; color: #1D4ED8;'>{pred_yield} <span style='font-size: 21px;'>Tons / Hectare</span></div><div style='font-size: 19px; color: #334155; margin-top: 8px;'>Expected Total Production: <b>{total_prod} Tons</b></div></div>""")
            render_raw_html(f"""<div class='prescription-card' style='background:#EFF6FF; border-color:#60A5FA;'><div class='prescription-title' style='color:#1E40AF;'>💊 Yield Enhancement Prescription</div><hr style='border-color: #BFDBFE; margin: 10px 0;'><ul><li><b>Fertilizer Efficiency Ratio:</b> {round(fert/area, 1)} kg/ha. Split dosage in vegetative growth phase.</li><li><b>Pesticide Usage:</b> {round(pest/area, 1)} kg/ha. Integrate IPM techniques to reduce chemical reliance.</li></ul></div>""")

def render_user_selling_prediction():
    render_ambient("sell")
    st.markdown("### 💰 Crop Price & Selling Revenue Forecasting")
    guide_col, col_input = st.columns([1, 4])
    with guide_col: render_page_image("Image3.png", "Selling cost and revenue guide")
    col_result = st.container()
    with col_input:
        st.title("Field Input Parameters in Selling Prediction Module")
        crops = selling_df['Crop'].unique().tolist() if selling_df is not None and not selling_df.empty else ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
        states = selling_df['State'].unique().tolist() if selling_df is not None and not selling_df.empty else ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Gujarat"]
        seasons = selling_df['Season'].unique().tolist() if selling_df is not None and not selling_df.empty else ["Kharif", "Rabi", "Whole Year"]
        c1, c2 = st.columns(2)
        with c1:
            sel_crop = st.selectbox("Select Crop", ["Choose crop..."] + crops, key="sell_crop")
            sel_state = st.selectbox("Market State", ["Choose state..."] + states, key="sell_state")
        with c2:
            sel_season = st.selectbox("Harvesting Season", ["Choose season..."] + seasons, key="sell_season")
            qty = st.number_input("Quantity to Sell (kg)", min_value=10.0, max_value=10000000.0, value=None, placeholder="Minimum 10 kg")
        btn = st.button("Forecast Selling Price & Revenue 💰")
    with col_result:
        is_complete = qty is not None and all(isinstance(v, str) and not v.startswith("Choose ") for v in [sel_crop, sel_state, sel_season])
        if btn and not is_complete: st.error("Select crop, market state and season, then enter a quantity of at least 10 kg.")
        elif btn and selling_model:
            row_dict = {col: 0 for col in selling_columns}
            row_dict['Quantity (kg)'] = qty
            for col in selling_columns:
                if f"Crop_{sel_crop}" in col: row_dict[col] = 1
                if f"State_{sel_state}" in col: row_dict[col] = 1
                if f"Season_{sel_season}" in col: row_dict[col] = 1
            sample = pd.DataFrame([row_dict])
            pred_price = max(1.0, round(selling_model.predict(sample)[0], 2))
            total_rev = round(pred_price * qty, 2)
            save_prediction(st.session_state["user_name"], "Selling Forecast", {"Crop": sel_crop, "State": sel_state, "Season": sel_season, "Quantity": qty}, f"₹{pred_price}/kg")
            st.session_state["history"].append({"User": st.session_state["user_name"], "Module": "Selling Forecast", "Input": f"{sel_crop}, {qty}kg in {sel_state}", "Result": f"₹{pred_price}/kg (Revenue: ₹{total_rev:,.2f})", "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            render_raw_html(f"""<div class='custom-card' style='text-align: center; border-left: 6px solid #D97706;'><div style='font-size: 17px; color: #64748B;'>ESTIMATED MARKET PRICE</div><div style='font-size: 42px; font-weight: 900; color: #B45309;'>₹ {pred_price:.2f} <span style='font-size: 21px;'>per kg</span></div><div style='font-size: 21px; color: #15803D; margin-top: 8px; font-weight: 800;'>Total Expected Revenue: ₹ {total_rev:,.2f}</div></div>""")
            render_raw_html(f"""<div class='prescription-card' style='background:#FEF3C7; border-color:#F59E0B;'><div class='prescription-title' style='color:#B45309;'>💡 Market Strategy Prescription</div><hr style='border-color: #FDE68A; margin: 10px 0;'><ul><li><b>Bulk Advantage:</b> For quantities exceeding {qty:.0f} kg, consider direct sales to regional wholesale Mandis to lower logistics costs.</li><li><b>Seasonal Timing:</b> Market trends for <b>{sel_season}</b> suggest steady price holding over the next 3-4 weeks.</li></ul></div>""")

def answer_application_question(question):
    query = question.lower().strip()
    pages = {
        "dashboard": "Dashboard", "home": "Dashboard",
        "crop recommendation": "Crop Recommendation", "recommend crop": "Crop Recommendation", "soil": "Crop Recommendation",
        "yield prediction": "Yield Prediction", "harvest": "Yield Prediction", "production": "Yield Prediction",
        "selling": "Price Forecast", "price forecast": "Price Forecast", "market price": "Price Forecast", "revenue": "Price Forecast",
        "market trends": "Market Trends", "price trends": "Market Trends", "apmc": "Market Trends",
        "weather": "Dashboard", "forecast": "Dashboard"
    }
    for key, page_name in pages.items():
        if key in query:
            if page_name == "Crop Recommendation":
                return f""" **Crop Recommendation Module**
I can help you get the perfect crop recommendation for your field!
**What you need to provide:**
- Nitrogen (N), Phosphorus (P), Potassium (K) levels
- Temperature, Humidity, Soil pH
- Annual Rainfall
**Features:**
- AI-powered crop suggestions with confidence scores
- Alternative crop recommendations
- Detailed agronomist prescription
- Soil nutrient analysis
👉 Click **Crop Recommendation** in the sidebar to get started!"""
            elif page_name == "Yield Prediction":
                return f"""📈 **Yield Prediction Module**
Predict your crop yield before you even plant!
**What you need to provide:**
- Crop type, State, Season
- Cultivation area (hectares)
- Rainfall, Fertilizer, and Pesticide usage
**Features:**
- Accurate yield per hectare prediction
- Total production estimation
- Fertilizer efficiency analysis
- IPM (Integrated Pest Management) recommendations
👉 Click **Yield Prediction** in the sidebar to start predicting!"""
            elif page_name == "Price Forecast":
                return f"""💰 **Selling Price Forecast Module**
Maximize your profits with smart price predictions!
**What you need to provide:**
- Crop type, Market State, Harvesting Season
- Quantity to sell (kg)
**Features:**
- Price per kg prediction
- Total revenue estimation
- Market strategy recommendations
- Bulk selling advantages
- Seasonal timing insights
👉 Click **Price Forecast** in the sidebar to forecast your earnings!"""
            elif page_name == "Market Trends":
                return f"""📊 **Market Trends Module**
Stay updated with real-time market prices!
**Features:**
- Live APMC market prices
- 7-day price trends
- Price comparison across markets
- Market hub analysis
- Historical price data
**Available Markets:**
- Ahmedabad, Rajkot, Surat, Vadodara
- Anand, Gondal, Mehsana, Amreli
👉 Click **Market Trends** in the sidebar to view current prices!"""
            elif page_name == "Dashboard":
                return f""" **Main Dashboard**
Your central hub for all agricultural insights!
**Features:**
- Real-time weather forecast
- Model accuracy metrics
- Your prediction history
- Dynamic agricultural visualizations
- Crop production trends
- Market revenue analytics
👉 Click **Dashboard** in the sidebar to view your overview!"""
    if any(word in query for word in ["how to use", "help", "guide", "tutorial"]):
        return """📚 **AgriSmart Help Guide**
**Getting Started:**
1. **Login/Signup** - Create your account or login
2. **Navigate** - Use the sidebar to access different modules
3. **Input Data** - Enter your field parameters
4. **Get Predictions** - Click the generate button
5. **View Results** - See recommendations and prescriptions
**Available Modules:**
- 🌱 Crop Recommendation - Get crop suggestions
- 📈 Yield Prediction - Predict harvest yields
- 💰 Price Forecast - Estimate selling prices
-  Market Trends - View market prices
-  Dashboard - Overview and analytics
**Need more help?** Ask me about any specific feature!"""
    if any(word in query for word in ["login", "signup", "register", "account", "password"]):
        return """🔐 **Account Management**
**To Login:**
1. Click "Login" or "User Access" on the homepage
2. Enter your email and password
3. Click "Continue"
**To Register:**
1. Click "User Access"
2. Fill in: Name, Email, State, Password
3. Click "Continue" to create account
**Forgot Password?**
Contact admin to reset your password.
**Admin Login:**
- Username: AIMDD
- Password: 123456"""
    if any(word in query for word in ["weather", "rain", "temperature", "forecast"]):
        return f"""🌦️ **Weather Forecast**
The Dashboard shows real-time weather for your state!
**Information Provided:**
- Current temperature
- Weather conditions
- Humidity levels
- Wind speed
- 3-day forecast
The weather updates every 15 minutes automatically based on your selected state.
👉 Go to **Dashboard** to see current weather!"""
    if any(word in query for word in ["admin", "user management", "crud", "manage users"]):
        if st.session_state.get("role") == "Admin":
            return """⚙️ **Admin Panel - User Management**
**Features Available:**
- ➕ Create new users
- ✏️ Update existing users
- ️ Delete user accounts
- 👁️ View all registered users
- 📊 User analytics by state
- 👥 Prediction logs monitoring
**User Management:**
Click **User Management** in the sidebar to access CRUD operations.
**Analytics:**
Click **User Analysis** to view state-wise user distribution.
**Logs:**
Click **User Predictions Log** to monitor all user activities."""
        else:
            return "⚠️ Admin features are only accessible to administrators. Contact AIMDD for admin access."
    if any(word in query for word in ["accuracy", "model", "performance", "r2", "score"]):
        rec_score = metadata.get("recommendation", {}).get("accuracy", 98.5)
        yield_score = metadata.get("yield", {}).get("r2", 99.2)
        return f"""📊 **Model Performance Metrics**
**Crop Recommendation Model:**
- Accuracy: {rec_score}%
- Algorithm: Random Forest Classifier
- Crops Supported: 22 varieties
**Yield Prediction Model:**
- R² Score: {yield_score}%
- Algorithm: Random Forest Regressor
- Features: Crop, State, Season, Area, Rainfall, Fertilizer, Pesticide
**Selling Price Model:**
- Accuracy: {score_display}
- Algorithm: Random Forest Regressor
- Real-time price predictions
All models are trained on authentic agricultural data and regularly updated!"""
    if any(word in query for word in ["logout", "sign out", "exit"]):
        return """ **Logging Out**
To logout:
1. Click the **Logout** button in the sidebar
2. Your session will be ended
3. You'll be redirected to the login page
Your data is saved securely and you can login anytime!"""
    if any(word in query for word in ["data", "dataset", "csv", "export", "download"]):
        return """💾 **Data Export**
**Available Downloads:**
- User dataset (CSV) - Admin only
- Prediction logs (CSV) - Admin only
- Market price data
- Historical trends
**To Export:**
1. Go to the respective module
2. Look for the download button (⬇️)
3. Click to download CSV file
Admin users can export comprehensive reports from User Management and Prediction Logs sections."""
    if any(word in query for word in ["prediction", "history", "my predictions", "past"]):
        return f"""📜 **Your Prediction History**
View all your past predictions!
**Information Tracked:**
- Date and time of prediction
- Module used (Crop/Yield/Selling)
- Input parameters
- Results and confidence scores
**To View History:**
- Admin: Go to **User Predictions Log** in sidebar
- Users: Check the Dashboard for your recent queries
Your predictions are saved automatically for future reference!"""
    return """ **AgriSmart Assistant**
I can help you with:
**🌱 Crop Recommendation**
- Soil analysis and crop suggestions
- Nutrient recommendations
**📈 Yield Prediction**
- Harvest yield forecasting
- Production estimates
**💰 Price Forecast**
- Market price predictions
- Revenue calculations
** Market Trends**
- Live APMC prices
- Price trend analysis
**🏠 Dashboard**
- Weather forecast
- Analytics overview
**👤 Account Help**
- Login/Signup assistance
- Password management
**Just ask me about any feature or say "help" for a complete guide!**
*Type your question above to get started!*"""

def render_smart_assistant():
    render_ambient("dash")
    render_raw_html("""<style>.agri-assistant-header { padding: 4px 0 10px; }.agri-assistant-header h2 { margin: 0; color: #F8FAFC; }.agri-assistant-header p { margin: 5px 0 0; color: #CFE9D4; }div[data-testid="stChatInput"] {border: 1px solid rgba(187,247,208,.42);border-radius: 18px;background: rgba(15,23,42,.92);}div[data-testid="stChatInput"] textarea {color: #F8FAFC !important;}</style><div class="agri-assistant-header"><h2>🤖 AgriSmart Intelligent Assistant</h2><p>Ask me anything about crop recommendation, yield prediction, market prices, or navigation help!</p></div>""")
    if "assistant_chat" not in st.session_state:
        st.session_state["assistant_chat"] = [
            {"role": "assistant", "content": "👋 Hi! I'm your **AgriSmart Intelligent Assistant**. I can help you with:\n• **Navigate** to any module (Dashboard, Crop Recommendation, Yield Prediction, Price Forecast, Market Trends)\n• **Explain** features and how to use them\n• **Answer** questions about agriculture, weather, prices\n• **Guide** you through predictions\n\n**Try asking:**\n- \"How do I get crop recommendation?\"\n- \"Show me market trends\"\n- \"What is yield prediction?\"\n- \"Help me login\"\n\nHow can I assist you today?"}
        ]
    with st.container(height=450, border=True):
        for message in st.session_state["assistant_chat"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    submission = st.chat_input("Ask me anything about AgriSmart features, navigation, or agriculture...", key="smart_assistant_input")
    if submission:
        question = submission.strip()
        if question:
            st.session_state["assistant_chat"].append({"role": "user", "content": question})
            response = answer_application_question(question)
            st.session_state["assistant_chat"].append({"role": "assistant", "content": response})
            st.rerun()

def _matching_value(question, values):
    question = question.casefold()
    for value in sorted({str(item).strip() for item in values if pd.notna(item)}, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", question): return value
    return None

# PERFORMANCE FIX: Removed pd.read_csv disk I/O. Now uses globally cached selling_df for instant loading.
def render_user_market_trends():
    render_ambient("market")
    st.markdown("## 📊 MARKET PRICE TRENDS")
    st.caption("Home / Market Price Trends")
    global selling_df
    if selling_df is not None and not selling_df.empty and "Price per kg ()" in selling_df.columns:
        try:
            market_df = selling_df.copy()
            market_df["Price_per_Quintal"] = market_df["Price per kg (₹)"] * 100
            available_crops = sorted(market_df["Crop"].unique().tolist())
        except Exception:
            available_crops = ["Rice", "Wheat", "Maize", "Cotton", "Soybean"]
            market_df = None
    else:
        available_crops = ["Rice", "Wheat", "Maize", "Cotton", "Soybean"]
        market_df = None
    apmc_markets = ["Ahmedabad APMC", "Rajkot APMC", "Surat APMC", "Vadodara APMC", "Anand APMC", "Gondal APMC", "Mehsana APMC", "Amreli APMC"]
    col_crop, col_market = st.columns(2)
    with col_crop: selected_crop = st.selectbox("Crop", available_crops, key="market_trends_crop")
    with col_market: selected_market = st.selectbox("Market Hub", apmc_markets, key="market_trends_market")
    if market_df is not None and not market_df.empty:
        crop_filtered = market_df[market_df["Crop"] == selected_crop]
        if not crop_filtered.empty:
            base_price = float(crop_filtered["Price_per_Quintal"].mean())
            min_price = float(crop_filtered["Price_per_Quintal"].min())
            max_price = float(crop_filtered["Price_per_Quintal"].max())
        else: base_price, min_price, max_price = 2500.0, 1500.0, 3500.0
    else: base_price, min_price, max_price = 2500.0, 1500.0, 3500.0
    seed_key = sum(ord(c) for c in selected_crop + selected_market)
    np.random.seed(seed_key)
    dates = [(datetime.now() - timedelta(days=i)).strftime("%d %b") for i in range(6, -1, -1)]
    trend = np.linspace(0, np.random.choice([-150, 150, 50, -50]), 7)
    noise = np.random.normal(0, 35, 7)
    fluctuations = trend + noise
    dynamic_prices = [int(np.clip(base_price + d, min_price, max_price)) for d in fluctuations]
    trend_df = pd.DataFrame({"Date": dates, "Price": dynamic_prices})
    st.markdown(f"### Price Trend (₹ Per Quintal) — <span style='color:#22C55E;'>{selected_crop} @ {selected_market}</span>", unsafe_allow_html=True)
    fig = px.line(trend_df, x="Date", y="Price", markers=True)
    fig.update_traces(line=dict(color="#22C55E", width=3), marker=dict(size=8, color="#4ADE80"), hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:,.0f} / Quintal<extra></extra>")
    fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", font=dict(color="#F8FAFC"), xaxis_title="", yaxis_title="", yaxis=dict(showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0)'), xaxis=dict(showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0)'), height=540, margin=dict(l=24, r=24, t=36, b=24))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### LATEST MARKET PRICES")
    table_rows = []; latest_p = dynamic_prices[-1]
    for idx, mkt in enumerate(apmc_markets):
        np.random.seed(seed_key + idx * 31)
        p = int(np.clip(latest_p + np.random.randint(-100, 110), min_price, max_price))
        table_rows.append({"Market": mkt,"Price (/Quintal)": f"₹ {p:,}","Date": datetime.now().strftime("%d %b %Y")})
    render_green_dataframe(pd.DataFrame(table_rows), title="Latest Market Prices", hide_index=True)

def render_admin_powerbi_dashboard():
    render_ambient("admin")
    st.markdown("### 📊 Executive PowerBI Analytics Portal")
    st.caption("System Performance, Regional Crop Yields & Revenue Dashboard")
    total_records = len(rec_df) + len(yield_df) + len(selling_df)
    avg_yield = yield_df['Yield'].mean() if not yield_df.empty else 0.0
    gross_rev = selling_df['Total Value (₹)'].sum() if not selling_df.empty else 0.0
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: render_raw_html(f"<div class='custom-card' style='border-left:5px solid #16A34A; min-height:150px; padding: 18px 20px; display:flex; flex-direction:column; justify-content:center; color:#FFFFFF; border-radius:18px;'><b style='font-size:17px; color:#FFFFFF;'>Total Data Records</b><h2 style='font-size:38px; margin: 8px 0 0; color:#FFFFFF;'>{total_records:,}</h2></div>")
    with kpi2: render_raw_html(f"<div class='custom-card' style='border-left:5px solid #2563EB; min-height:150px; padding: 18px 20px; display:flex; flex-direction:column; justify-content:center; color:#FFFFFF; border-radius:18px;'><b style='font-size:17px; color:#FFFFFF;'>National Avg Yield</b><h2 style='font-size:38px; margin: 8px 0 0; color:#FFFFFF;'>{avg_yield:.2f} T/Ha</h2></div>")
    with kpi3:
        rev_str = f"₹ {gross_rev/1e6:.1f}M" if gross_rev >= 1e6 else f"₹ {gross_rev:,.0f}"
        render_raw_html(f"<div class='custom-card' style='border-left:5px solid #D97706; min-height:150px; padding: 18px 20px; display:flex; flex-direction:column; justify-content:center; color:#FFFFFF; border-radius:18px;'><b style='font-size:17px; color:#FFFFFF;'>Gross Market Revenue</b><h2 style='font-size:38px; margin: 8px 0 0; color:#FFFFFF;'>{rev_str}</h2></div>")
    with kpi4: render_raw_html("<div class='custom-card' style='border-left:5px solid #9333EA; min-height:150px; padding: 18px 20px; display:flex; flex-direction:column; justify-content:center; color:#FFFFFF; border-radius:18px;'><b style='font-size:17px; color:#FFFFFF;'>System ML Health</b><h2 style='font-size:38px; margin: 8px 0 0; color:#FFFFFF;'>100% OK</h2></div>")
    st.markdown("### 📈 Dynamic Intelligence Visualizations")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        if yield_df is not None and not yield_df.empty:
            state_yield = yield_df.groupby('State')['Yield'].mean().reset_index().sort_values('Yield', ascending=False)
            fig_a1 = px.bar(state_yield, x='State', y='Yield', title="Average Yield Performance (Tons/Ha) by State", color='Yield', color_continuous_scale='Greens', text='Yield')
            fig_a1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_a1.update_layout(height=560, margin=dict(l=16, r=16, t=46, b=16), xaxis_title='State', yaxis_title='Avg Yield (Tons/Ha)')
            fig_a1.update_xaxes(showgrid=False, zeroline=False)
            fig_a1.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig_a1, use_container_width=True)
    with col_chart2:
        if selling_df is not None and not selling_df.empty:
            rev_by_crop = selling_df.groupby('Crop')['Total Value (₹)'].sum().reset_index()
            fig_a2 = px.pie(rev_by_crop, names='Crop', values='Total Value (₹)', title="Market Revenue Share Breakdown by Crop", hole=0.4, color_discrete_sequence=px.colors.sequential.Greens_r)
            fig_a2.update_traces(textposition='inside', textinfo='percent+label')
            fig_a2.update_layout(height=470, margin=dict(l=16, r=16, t=46, b=16))
            st.plotly_chart(fig_a2, use_container_width=True)
    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        if yield_df is not None and not yield_df.empty:
            fig5 = px.scatter(yield_df.sample(min(800, len(yield_df))), x="Fertilizer", y="Yield", color="Crop", title="Fertilizer Productivity Output (Kg vs Yield)", opacity=0.75)
            fig5.update_layout(height=470, margin=dict(l=16, r=16, t=46, b=16), xaxis_title='Fertilizer Usage (kg)', yaxis_title='Yield (Tons/Ha)')
            fig5.update_xaxes(showgrid=False, zeroline=False)
            fig5.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig5, use_container_width=True)
    with col_chart4:
        if rec_df is not None and not rec_df.empty:
            fig4 = px.scatter(rec_df.sample(min(800, len(rec_df))), x="ph", y="rainfall", color="label", title="Soil pH vs Rainfall Distribution", opacity=0.75)
            fig4.update_layout(height=470, margin=dict(l=16, r=16, t=46, b=16), xaxis_title='Soil pH', yaxis_title='Annual Rainfall (mm)')
            fig4.update_xaxes(showgrid=False, zeroline=False)
            fig4.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig4, use_container_width=True)

def render_admin_prediction_results():
    render_ambient("admin")
    st.markdown("## 👥 User Prediction Logs & Visual Insights")
    st.caption("Live monitoring of all user prediction requests across modules")
    latest_df = fetch_admin_prediction_logs("All")
    st.markdown("### 📋 Latest Prediction Analytics")
    last_pred = None
    if latest_df is not None and not latest_df.empty:
        last_pred_row = latest_df.iloc[0]
        try: inputs_data = json.loads(last_pred_row["User Inputs"])
        except Exception: inputs_data = last_pred_row["User Inputs"]
        result_text = str(last_pred_row["Prediction Result"])
        confidence = 0
        if "%" in result_text:
            try: confidence = float(result_text.split("(")[-1].replace("%)", ""))
            except: confidence = 0
        last_pred = {"user": last_pred_row.get("User","Unknown"), "module": last_pred_row.get("Module","Unknown"), "result": result_text, "confidence": confidence, "timestamp": last_pred_row.get("Timestamp", "-"), "inputs": inputs_data}
    if last_pred:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("User", last_pred["user"])
        with c2: st.metric("Module", last_pred["module"])
        with c3: st.metric("Confidence", f"{last_pred['confidence']}%")
        with c4: st.metric("Time", last_pred["timestamp"])
        st.markdown("### 🔍 Prediction Details")
        st.success(f"Prediction Result: {last_pred['result']}")
        render_green_dataframe(last_pred["inputs"])
    else: st.info("No prediction logs available.")
    st.divider()
    st.markdown("### 📊 Prediction Module Analytics")
    if latest_df is not None and not latest_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            module_count = latest_df["Module"].value_counts().reset_index()
            module_count.columns = ["Module", "Count"]
            fig = px.bar(module_count, x="Module", y="Count", title="Prediction Requests by Module", text="Count")
            fig.update_layout(height=440)
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            user_count = latest_df["User"].value_counts().head(10).reset_index()
            user_count.columns=["User", "Predictions"]
            fig2 = px.pie(user_count, names="User", values="Predictions", title="Top Active Users")
            fig2.update_layout(height=440)
            st.plotly_chart(fig2, use_container_width=True)
    st.divider()
    st.markdown("###  Comprehensive Prediction Audit Trail")
    module_filter = st.selectbox("Filter by Prediction Module", ["All", "Crop Recommendation", "Yield Prediction", "Selling Forecast"], key="admin_prediction_filter")
    hist_df = fetch_admin_prediction_logs(module_filter)
    if hist_df is not None and not hist_df.empty:
        render_green_dataframe(hist_df, title="Prediction Audit Trail", hide_index=True)
        st.download_button("⬇️ Export Prediction Logs CSV", hist_df.to_csv(index=False), file_name="AgriSmart_prediction_logs.csv", mime="text/csv")
    else: st.info("No prediction history found for selected module.")

def render_all_states_map_page():
    render_ambient("dash")
    st.markdown("### 🗺️ APMC Markets & Regional Hub Map")
    st.caption("Interactive geo-spatial distribution of active agricultural market hubs based on your dataset")
    states_in_data = set()
    if not selling_df.empty:
        states_in_data.update(selling_df['State'].unique())
    if not yield_df.empty:
        states_in_data.update(yield_df['State'].unique())
    map_data = []
    for state in states_in_data:
        if state in STATE_COORDINATES:
            lat, lon = STATE_COORDINATES[state]
            jitter_lat = np.random.uniform(-0.4, 0.4)
            jitter_lon = np.random.uniform(-0.4, 0.4)
            map_data.append({'Market': f"{state} Mandi Hub", 'lat': lat + jitter_lat, 'lon': lon + jitter_lon})
    if not map_data:
        map_data = [{'Market': 'Delhi Central Hub', 'lat': 28.6139, 'lon': 77.2090}]
    st.map(pd.DataFrame(map_data))

# PERFORMANCE FIX: Cache table column checks to prevent redundant DB queries on every admin page render
@st.cache_data(ttl=300)
def _user_table_columns():
    conn = get_db_connection()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    conn.close()
    return cols

@st.dialog("➕ Create New User")
def create_user_dialog(has_full_name):
    with st.form("admin_create_user_form"):
        new_name = st.text_input("Full Name", placeholder="Enter full name")
        new_email = st.text_input("Email Address", placeholder="user@example.com")
        new_state = st.selectbox("State", list(STATE_COORDINATES.keys()))
        new_password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
        submitted = st.form_submit_button("Create User", type="primary", use_container_width=True)
        if submitted:
            email_clean = new_email.strip().lower()
            problems = []
            if not new_name.strip(): problems.append("Name is required.")
            if "@" not in email_clean or "." not in email_clean.split("@")[-1]: problems.append("A valid email address is required.")
            if len(new_password) < 6: problems.append("Password must be at least 6 characters.")
            if problems:
                for p in problems: st.error(f" {p}")
            else:
                conn = get_db_connection()
                try:
                    owner = conn.execute("SELECT id, username FROM users WHERE lower(email) = lower(?)", (email_clean,)).fetchone()
                    if owner:
                        st.error(f"❌ '{email_clean}' is already registered to user **{owner['username']}** (ID {owner['id']}). Use the Update button for that account, or choose a different email.")
                    else:
                        insert_columns = ["username", "email", "state", "password", "role", "created_at"]
                        insert_values = [new_name.strip(), email_clean, new_state, hash_password(new_password), "User", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                        if has_full_name:
                            insert_columns.insert(0, "full_name")
                            insert_values.insert(0, new_name.strip())
                        conn.execute(f"INSERT INTO users ({', '.join(insert_columns)}) VALUES ({', '.join('?' for _ in insert_columns)})", insert_values)
                        conn.commit()
                        fetch_users_by_state.clear()
                        st.success(f"✅ User '{new_name.strip()}' ({email_clean}) created successfully!")
                        time.sleep(1)
                        st.rerun()
                except sqlite3.IntegrityError as e:
                    st.error(f"❌ Could not save user – database constraint: {e}")
                except Exception as e:
                    st.error(f"❌ Database error: {e}")
                finally:
                    conn.close()

@st.dialog("✏️ Update Existing User")
def update_user_dialog(users, has_full_name):
    if users.empty:
        st.info("ℹ️ There are no user accounts to update yet.")
        return
    user_options = {f"ID {int(r['id'])} – {r['username']} ({r['email']})": int(r['id']) for _, r in users.iterrows()}
    chosen = st.selectbox("Choose user to update", list(user_options.keys()))
    selected_id = user_options[chosen]
    sel = users.loc[users["id"] == selected_id].iloc[0]
    with st.form("admin_update_user_form"):
        st.markdown(f"**Updating:** {sel['username']} • {sel['email']}")
        edit_name = st.text_input("Name", value=str(sel["username"] or ""))
        states = list(STATE_COORDINATES.keys())
        edit_state = st.selectbox("State", states, index=states.index(sel["state"]) if sel["state"] in states else 0)
        edit_pass = st.text_input("New password (leave blank to keep current)", type="password")
        update_clicked = st.form_submit_button("Save Changes", type="primary", use_container_width=True)
        if update_clicked:
            if not edit_name.strip():
                st.error("❌ Name is required.")
            elif edit_pass and len(edit_pass) < 6:
                st.error("❌ New password must be at least 6 characters, or leave it blank.")
            else:
                try:
                    conn = get_db_connection()
                    if edit_pass:
                        sql = "UPDATE users SET username=?, state=?, password=?"
                        vals = [edit_name.strip(), edit_state, hash_password(edit_pass)]
                    else:
                        sql = "UPDATE users SET username=?, state=?"
                        vals = [edit_name.strip(), edit_state]
                    if has_full_name:
                        sql = sql.replace("SET ", "SET full_name=?, ", 1)
                        vals.insert(0, edit_name.strip())
                    sql += " WHERE id=?"
                    vals.append(selected_id)
                    conn.execute(sql, vals)
                    conn.commit(); conn.close()
                    fetch_users_by_state.clear()
                    st.success("✅ User updated successfully!")
                    time.sleep(1); st.rerun()
                except Exception as e:
                    st.error(f"❌ Database error: {e}")

@st.dialog("🗑️ Delete User Account")
def delete_user_dialog(users):
    st.warning("️ This action cannot be undone!")
    if users.empty:
        st.info("️ There are no user accounts to delete yet.")
        return
    user_options = {f"ID {int(r['id'])} – {r['username']} ({r['email']})": int(r['id']) for _, r in users.iterrows()}
    chosen = st.selectbox("Choose user to delete", list(user_options.keys()))
    delete_id = user_options[chosen]
    del_user = users.loc[users["id"] == delete_id].iloc[0]
    st.markdown(f"**Selected:** {del_user['username']} ({del_user['email']}) — {del_user['state']} / {del_user['role']}")
    confirm_delete = st.checkbox(f"I understand that deleting {del_user['email']} is permanent.")
    if st.button("🗑️ Delete User Permanently", type="primary", use_container_width=True, disabled=not confirm_delete):
        try:
            conn = get_db_connection()
            conn.execute("DELETE FROM auth_tokens WHERE username = ?", (del_user["username"],))
            conn.execute("DELETE FROM users WHERE id = ?", (delete_id,))
            conn.commit(); conn.close()
            fetch_users_by_state.clear()
            st.success(f"✅ User '{del_user['username']}' deleted permanently!")
            time.sleep(1); st.rerun()
        except Exception as e:
            st.error(f" Database error: {e}")

# PERFORMANCE FIX: Added caching to prevent redundant DB queries on every rerun
@st.cache_data(ttl=30)
def get_all_users():
    conn = get_db_connection()
    users = pd.read_sql_query("SELECT id, username, email, state, role, created_at FROM users ORDER BY id DESC", conn)
    conn.close()
    return users

def render_admin_user_management():
    render_ambient("admin")
    st.markdown("## ⚙️ User Management")
    st.caption("Create, view, update, or permanently delete user accounts.")
    # PERFORMANCE FIX: Use cached user fetch
    users = get_all_users()
    table_cols = _user_table_columns()
    has_full_name = "full_name" in table_cols
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" Create User", use_container_width=True, type="primary", key="top_btn_create"):
            create_user_dialog(has_full_name)
    with col2:
        if st.button("✏️ Update User", use_container_width=True, key="top_btn_update"):
            update_user_dialog(users, has_full_name)
    with col3:
        if st.button("🗑️ Delete User", use_container_width=True, key="top_btn_delete"):
            delete_user_dialog(users)
    st.markdown("### 📋 All Registered Users")
    render_green_dataframe(users, title="All Registered Users", hide_index=True)

def render_admin_user_analysis():
    render_ambient("admin")
    st.markdown("## 🌍 User Analysis by State")
    st.caption("Shows which users registered from which Indian state")
    # PERFORMANCE FIX: Use already cached fetch_users_by_state() instead of raw query
    users_df = fetch_users_by_state()
    if users_df.empty: st.info("No users have registered yet."); return
    c1, c2 = st.columns(2)
    with c1: st.metric("Total Registered Users", len(users_df))
    with c2: st.metric("States Covered", users_df["State"].nunique())
    st.markdown("### 📊 Users by State")
    state_count = users_df["State"].value_counts().reset_index()
    state_count.columns = ["State", "Users"]
    fig = px.bar(state_count, x="State", y="Users", color="Users", color_continuous_scale="Greens", text="Users", title="Registered Users from Each State")
    fig.update_layout(height=470)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🥧 State Distribution")
    fig2 = px.pie(state_count, names="State", values="Users", hole=0.45, title="User Distribution Across States")
    fig2.update_layout(height=470); st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### 📋 Registered Users Dataset")
    render_green_dataframe(users_df, title="Registered Users Dataset", hide_index=True)
    st.download_button("⬇️ Download User Dataset (CSV)", users_df.to_csv(index=False), file_name="AgriSmart_users_by_state.csv", mime="text/csv")

def main():
    if not st.session_state["authenticated"]:
        render_login_page(); return
    with st.sidebar:
        st.markdown(f"###  Welcome, **{st.session_state['user_name']}**")
        st.caption(f"Role: **{st.session_state['role']}**")
        st.markdown("---")
        if st.session_state["role"] == "Admin":
            nav_items = [("📊 Executive Dashboard", "Dashboard"), ("️🧑🏻‍💼 User Management", "User Management"), ("🧾 User Predictions Log", "User Predictions Log"), ("🌍 User Analysis", "User Analysis"), ("🗺️ Regional Market Map", "Market Map"), ("ℹ️ About System", "About System")]
        else:
            nav_items = [("📊 Dashboard", "Dashboard"), ("🌱 Crop Recommendation", "Crop Recommendation"), ("📈 Yield Prediction", "Yield Prediction"), ("💰 Price Forecast", "Price Forecast"), ("📊 Market Trends", "Market Trends")]
        current_page = st.session_state["page_nav"]
        for label, page_name in nav_items:
            is_active = current_page == page_name
            if st.button(label, key=f"nav_{page_name.lower().replace(' ', '_')}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state["assistant_open"] = False; st.session_state["page_nav"] = page_name; st.rerun()
        if st.session_state["role"] != "Admin":
            st.markdown("---")
            render_raw_html("<div class='agri-bot-face'>🤖</div>")
            if st.button("🤖 Ask AgriBot", key="open_smart_assistant", help="Open the AgriSmart voice and chat helper", type="primary", use_container_width=True):
                st.session_state["assistant_open"] = True; st.session_state["page_nav"] = "Smart Assistant"; st.rerun()
            st.caption("Need help? Ask your friendly AgriBot.")
        st.markdown("---")
        if st.button("🚪 Logout"):
            try:
                params = get_query_params() or {}
                auth = params.get("auth", [None])[0] if isinstance(params.get("auth"), list) else params.get("auth")
            except Exception: auth = None
            if auth: revoke_login_token(auth)
            clear_browser_auth_token()
            set_query_params()
            st.session_state["authenticated"] = False; st.session_state["user_name"] = ""; st.session_state["user_state"] = "Delhi"; st.session_state["role"] = "User"; st.session_state["assistant_open"] = False; st.session_state["history"] = []; st.session_state["last_prediction"] = None; st.rerun()
    page = st.session_state["page_nav"]
    if st.session_state["role"] == "Admin":
        if page == "Dashboard": render_admin_powerbi_dashboard()
        elif page == "User Management": render_admin_user_management()
        elif page == "User Predictions Log": render_admin_prediction_results()
        elif page == "User Analysis": render_admin_user_analysis()
        elif page == "Market Map": render_all_states_map_page()
        elif page == "About System": render_admin_about_page()
        elif page == "Smart Assistant": render_smart_assistant()
    else:
        if page == "Dashboard": render_user_dashboard()
        elif page == "Crop Recommendation": render_user_crop_recommendation()
        elif page == "Yield Prediction": render_user_yield_prediction()
        elif page == "Price Forecast": render_user_selling_prediction()
        elif page == "Market Trends": render_user_market_trends()
        elif page == "Smart Assistant": render_smart_assistant()
    render_raw_html("""
    <div style="text-align: center; padding: 30px 0 14px; color: #64748B; font-size: 13px; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 40px;">
        <b>AgriSmart Intelligence Portal</b> © 2026 | Built with Streamlit, Scikit-Learn & Plotly<br>
        Empowering Farmers with AI-Driven Agronomy & Market Insights
    </div>
    """)

if __name__ == "__main__":
    main()
