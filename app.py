"""Web portal:  streamlit run app.py   (or ./run.sh / run.bat)"""
import streamlit as st

from idxbt import charts, data, report_view, ui_portfolio, ui_single

st.set_page_config(page_title="Index Backtester", page_icon="📈", layout="wide")
st.markdown('''<style>
.stApp{background:linear-gradient(180deg,#12132a 0%,#1a1633 55%,#102226 100%)}
[data-testid="stSidebar"]{background:#181a33}
[data-testid="stMetric"]{background:#1f2140;border-radius:18px;padding:14px 16px;box-shadow:0 4px 18px rgba(0,0,0,.35)}
[data-testid="stMetricValue"]{color:#aab4ff;font-size:1.55rem}[data-testid="stMetricValue"] div{overflow:visible;text-overflow:clip}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#8e9dfa,#c4a8f0);border:0;border-radius:14px;color:#14152b;font-weight:700}
.stButton>button{border-radius:14px}
h1{background:linear-gradient(90deg,#9aa8ff,#c4a8f0,#f2a9c6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0;padding:8px 16px}
[class*="st-key-rv-"]{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:12px 16px 6px;margin-bottom:12px}
'''+report_view.CSS+'''</style>''', unsafe_allow_html=True)

# --------------------------------------------------------- auto update (once per browser session)
if "auto_done" not in st.session_state:
    st.session_state["auto_done"] = True
    if st.session_state.get("auto_on", True):
        with st.spinner("Checking for new market data…"):
            upd = data.auto_update()
        if upd:
            st.session_state["stamp"] = st.session_state.get("stamp", 0) + 1
            ok = {k: v for k, v in upd.items() if not isinstance(v, str)}
            bad = {k: v for k, v in upd.items() if isinstance(v, str)}
            if ok:
                st.toast("Data updated: " + ", ".join(ok), icon="✅")
            if bad:
                st.toast("Couldn't update " + ", ".join(bad) + " (working offline with stored data)", icon="⚠️")

@st.fragment(run_every="30m")
def keep_fresh():
    """While the portal stays open, quietly top up data every 30 min (throttled, offline-safe)."""
    if st.session_state.get("auto_on", True) and any(not isinstance(v, str) for v in data.auto_update(min_hours=0.4).values()):
        st.session_state["stamp"] = st.session_state.get("stamp", 0) + 1
        st.rerun()


keep_fresh()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    stored = data.summary()
    syms = list(stored["symbol"]) if len(stored) else []
    sym = st.selectbox("Index (single-strategy tab)", syms + (["ALL (compare)"] if len(syms) > 1 else []) if syms else list(data.SYMBOLS))
    c1, c2 = st.columns(2)
    start = c1.text_input("From", placeholder="2015-01-01"); end = c2.text_input("To", placeholder="today")
    fill = st.radio("Trade fills", ["next_open", "close"], horizontal=True,
                    help="next_open: signal at close → trade at next day's open (realistic). close: trade at the signal candle's close.")
    fee = st.number_input("Fee % per side", 0.0, 2.0, 0.03, 0.01)
    slip = st.number_input("Slippage % per side", 0.0, 2.0, 0.0, 0.01)
    cap = st.number_input("Starting capital ₹", 1000.0, 1e10, 100000.0, 10000.0)
    rf = st.number_input("Risk-free rate % (for Sharpe)", 0.0, 15.0, 0.0, 0.5) / 100
    pal = st.selectbox("🎨 Graph colours", list(charts.PALETTES), key="palette",
                       help="Changes only the graphs, not the rest of the portal.")
    st.divider()
    st.subheader("Data")
    st.toggle("Auto-update on startup", value=True, key="auto_on",
              help="When the portal opens (and every 30 min while open), download any missing days. Skips quietly if offline.")
    ls = data.last_success()
    st.caption(f"Last data update: {ls.replace('T', ' ')}" if ls else "Last data update: never")
    if len(stored):
        st.dataframe(stored.rename(columns={"first": "from", "last": "to"}), hide_index=True, width="stretch")
    else:
        st.warning("No data stored yet. Click Update data.")
    if st.button("⬇ Update data (all 4 indices)", width="stretch"):
        with st.spinner("Downloading from Yahoo Finance…"):
            for s in data.SYMBOLS:
                try:
                    st.success(f"{s}: +{data.fetch(s)} rows")
                except Exception as e:
                    st.error(f"{s}: {str(e)[:160]}")
        st.session_state["stamp"] = st.session_state.get("stamp", 0) + 1
        st.rerun()
    up = st.file_uploader("…or import CSV (Date,Open,High,Low,Close)", type="csv")
    if up and st.button("Import CSV"):
        import tempfile
        tgt = st.selectbox("as", list(data.SYMBOLS), key="tgt")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(up.getvalue())
        st.success(f"{data.import_csv(tgt, f.name)} rows imported")

charts.set_palette(pal)

# --------------------------------------------------------------------- main
st.title("📈 Index Backtester")
st.caption("NIFTY · BANKNIFTY · FINNIFTY · SENSEX — daily candles. Describe your strategy in plain English.")
cfg = dict(sym=sym, start=start, end=end, fill=fill, fee=fee, slip=slip, cap=cap, rf=rf)
tab_single, tab_pf = st.tabs(["🎯 Single strategy", "🧩 Portfolio · combine strategies"])
with tab_single:
    ui_single.render(cfg)
with tab_pf:
    ui_portfolio.render(cfg)
st.caption("Backtests use index levels (proxy for futures/ETF). Past performance doesn't predict the future. Not investment advice.")
