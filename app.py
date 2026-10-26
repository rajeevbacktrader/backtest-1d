"""Web portal:  streamlit run app.py   (or ./run.sh / run.bat)"""
import pandas as pd
import streamlit as st

from idxbt import charts, data, engine, metrics, report
from idxbt.dsl import QueryError, parse_query

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
</style>''', unsafe_allow_html=True)

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
    sym = st.selectbox("Index", syms + (["ALL (compare)"] if len(syms) > 1 else []) if syms else list(data.SYMBOLS))
    c1, c2 = st.columns(2)
    start = c1.text_input("From", placeholder="2015-01-01"); end = c2.text_input("To", placeholder="today")
    fill = st.radio("Trade fills", ["next_open", "close"], horizontal=True,
                    help="next_open: signal at close → trade at next day's open (realistic). close: trade at the signal candle's close.")
    fee = st.number_input("Fee % per side", 0.0, 2.0, 0.03, 0.01)
    slip = st.number_input("Slippage % per side", 0.0, 2.0, 0.0, 0.01)
    cap = st.number_input("Starting capital ₹", 1000.0, 1e10, 100000.0, 10000.0)
    rf = st.number_input("Risk-free rate % (for Sharpe)", 0.0, 15.0, 0.0, 0.5) / 100
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

# --------------------------------------------------------------------- main
st.title("📈 Index Backtester")
st.caption("NIFTY · BANKNIFTY · FINNIFTY · SENSEX — daily candles. Describe your strategy in plain English.")

pick = st.pills("Try an example", list(PRESETS), default=None)
if pick:
    st.session_state["q"] = PRESETS[pick]
q = st.text_area("Strategy", st.session_state.get("q", PRESETS["SMA20 cross + 4-candle hold"]), height=120, key="q_box",
                 help="Write 'entry: …' and 'exit: …'. Exit fires on the first rule hit.")
st.session_state["q"] = q

# live parse preview so mistakes show up before running
try:
    preview = parse_query(q)
    st.success("Understood:\n\n```\n" + preview.describe() + "\n```" + ("\n" + " ".join(preview.notes) if preview.notes else ""))
except QueryError as e:
    preview = None
    st.error(f"Can't parse yet: {e}")

with st.expander("Query cheat-sheet"):
    st.markdown("""
- **Prices**: `open high low close`, `prev close`, `close[2]`  · **Indicators**: `sma20`, `ema 50`, `rsi14`, `atr14`, `roc 10`, `macd`, `macd signal`, `bb upper`
- **Breakouts**: `20 day high`, `10 day low` (previous N candles) · **Math**: `sma20 * 1.02`, `sma50 + 2%`
- **Conditions**: `crossed above / below`, `>  <  >=  <=`, join with `and` / `or`; direction `go long` / `go short`
- **Exits**: `holding period = 4 candles`, `tp = 4%`, `sl = 2%`, `trailing sl 3%`, or any condition. `0` disables a rule.
""")

run = st.button("▶ Run backtest", type="primary", disabled=preview is None)
if run:
    targets = list(data.SYMBOLS) if sym.startswith("ALL") else [sym]
    results = {}
    for s in targets:
        try:
            df = load_df(s, start, end, st.session_state.get("stamp", 0))
        except LookupError:
            st.warning(f"No stored data for {s}."); continue
        results[s] = engine.run(preview, df, fill=fill, fee_pct=fee, slippage_pct=slip, capital=cap)
    st.session_state["results"] = results
    st.session_state["rf"] = rf

results = st.session_state.get("results")
if results:
    if len(results) > 1:
        rows = []
        for s, r in results.items():
            m = metrics.compute(r, st.session_state["rf"])
            rows.append({"Index": s, "Trades": m["trades"].get("n", 0), "CAGR": pct(m["strategy"]["cagr"]),
                         "Buy&hold CAGR": pct(m["benchmark"]["cagr"]), "Max DD": pct(m["strategy"]["max_dd"]),
                         "Sharpe": round(m["strategy"]["sharpe"], 2), "Win rate": pct(m["trades"].get("win_rate", 0))})
        st.subheader("Comparison"); st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        name = st.selectbox("Open details for", list(results))
    else:
        name = next(iter(results))
    res = results[name]
    m = metrics.compute(res, st.session_state["rf"]); ins = metrics.insights(res, m)
    s, b, t = m["strategy"], m["benchmark"], m["trades"]
    st.caption(f"{name}: {m['start']:%d %b %Y} → {m['end']:%d %b %Y} · {m['candles']:,} candles")
    k = st.columns(6)
    k[0].metric("Total return", pct(s["total_return"]), f"B&H {pct(b['total_return'])}", delta_color="off")
    k[1].metric("CAGR", pct(s["cagr"]), f"B&H {pct(b['cagr'])}", delta_color="off")
    k[2].metric("Max drawdown", pct(s["max_dd"]), f"B&H {pct(b['max_dd'])}", delta_color="off")
    k[3].metric("Sharpe", f"{s['sharpe']:.2f}", f"B&H {b['sharpe']:.2f}", delta_color="off")
    k[4].metric("Trades", t.get("n", 0))
    k[5].metric("Win rate", pct(t["win_rate"]) if t.get("n") else "—", f"PF {t['profit_factor']:.2f}" if t.get("n") else None, delta_color="off")

    unit = st.radio("Trade graphs in", ["%", "pts"], horizontal=True, key="unit",
                    format_func=lambda u: "Percent (%)" if u == "%" else "Raw index points")
    figs = charts.build_all(res, m, name, unit)
    groups = list(dict.fromkeys(g for g, _ in figs.values()))
    if t.get("n"):
        k2 = st.columns(4)
        k2[0].metric("Points captured", f"{t['total_pts']:+,.0f}")
        k2[1].metric("Avg points / trade", f"{t['avg_pts']:+,.1f}", f"win {t['avg_win_pts']:+,.0f} / loss {t['avg_loss_pts']:+,.0f}", delta_color="off")
        k2[2].metric("Best trade", f"{t['best_pts']:+,.0f} pts", f"{t['best']:+.2f}%", delta_color="off")
        k2[3].metric("Worst trade", f"{t['worst_pts']:+,.0f} pts", f"{t['worst']:+.2f}%", delta_color="off")
    tabs = st.tabs(["💡 Insights"] + [{"Performance": "📈 Performance", "Trades": "🎯 Trades", "Breakdown": "🍩 Breakdown",
                                       "Calendar": "🗓 Calendar"}[g] for g in groups] + ["📋 Trades table"])
    with tabs[0]:
        for i in ins:
            st.markdown(f"- {i}")
        if len(m["by_reason"]):
            st.markdown("**By exit reason**"); st.dataframe(m["by_reason"].round(3), width="stretch")
    for tab, g in zip(tabs[1:-1], groups):
        with tab:
            for key, (gg, fig) in figs.items():
                if gg == g:
                    st.plotly_chart(fig, width="stretch", key=f"{name}_{key}_{unit}")
    with tabs[-1]:
        st.dataframe(res.trades.iloc[::-1], width="stretch", hide_index=True)
        st.download_button("Download trades CSV", res.trades.to_csv(index=False), f"{name}_trades.csv")
        import tempfile, pathlib
        d = pathlib.Path(tempfile.mkdtemp())
        p = report.write_report(res, m, ins, name, d)
        st.download_button("Download full HTML report (interactive)", p.read_text(encoding="utf-8"), f"{name}_report.html", "text/html")
st.caption("Backtests use index levels (proxy for futures/ETF). Past performance doesn't predict the future. Not investment advice.")
