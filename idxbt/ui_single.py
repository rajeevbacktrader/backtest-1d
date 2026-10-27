"""Single-strategy tab."""
import pandas as pd
import streamlit as st

from . import charts, data, engine, metrics, report, report_view
from .dsl import QueryError, parse_query
from .ui_common import PRESETS, load_df, pct


def render(cfg):
    sym, start, end, fill, fee, slip, cap, rf = (cfg[k] for k in ("sym", "start", "end", "fill", "fee", "slip", "cap", "rf"))
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
        tabs = st.tabs(["📑 Report", "💡 Insights"] + [{"Performance": "📈 Performance", "Trades": "🎯 Trades", "Breakdown": "🍩 Breakdown",
                                           "Calendar": "🗓 Calendar"}[g] for g in groups] + ["📋 Trades table"])
        with tabs[0]:
            report_view.render_st(res.equity, res.trades, f"single_{name}", show_points=True, title=f"BACKTEST REPORT · {name}")
        with tabs[1]:
            for i in ins:
                st.markdown(f"- {i}")
            if len(m["by_reason"]):
                st.markdown("**By exit reason**"); st.dataframe(m["by_reason"].round(3), width="stretch")
        for tab, g in zip(tabs[2:-1], groups):
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
