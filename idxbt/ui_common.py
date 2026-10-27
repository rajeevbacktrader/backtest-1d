"""Helpers shared by the portal tabs."""
import streamlit as st

from . import data

PRESETS = {
    "SMA20 cross + 4-candle hold": "entry: close crossed above sma20 go long\nexit: holding period = 4 candles or tp = 4% or sl = 2% or close crossed below sma 8",
    "Oversold bounce above 200 SMA": "entry: rsi14 < 30 and close > sma 200 go long\nexit: rsi14 > 55 or sl = 3% or holding period = 10 candles",
    "20-day breakout (trend)": "entry: close crossed above 20 day high go long\nexit: close crossed below 10 day low or trailing sl 4%",
    "Short below SMA50": "entry: close crossed below sma50 go short\nexit: tp = 3% or sl = 2% or holding period = 6 candles",
    "MACD cross": "entry: macd crossed above macd signal go long\nexit: macd crossed below macd signal or sl = 3%",
}


@st.cache_data(show_spinner=False)
def load_df(sym, start, end, _stamp):
    return data.load(sym, start or None, end or None)


def pct(x):
    return f"{x:.1%}"


def plotly_chart(fig, **kw):
    """st.plotly_chart without Streamlit's own colour theme, so the chosen graph palette is shown exactly."""
    kw.setdefault("theme", None)
    kw.setdefault("width", "stretch")
    st.plotly_chart(fig, **kw)
