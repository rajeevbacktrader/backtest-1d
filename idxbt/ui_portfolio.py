"""Portfolio tab: several strategy rows, tick the ones to analyse together."""
import numpy as np
import pandas as pd
import streamlit as st

from . import charts, data, engine, metrics
from . import portfolio as pf
from .dsl import QueryError, parse_query
from .ui_common import PRESETS, load_df, pct

DEFAULTS = list(PRESETS.values())


def _add():
    ss = st.session_state
    ss["pf_rows"].append(ss["pf_next"]); ss["pf_next"] += 1


def _remove(rid):
    st.session_state["pf_rows"].remove(rid)


def _tick_all(val):
    for rid in st.session_state["pf_rows"]:
        st.session_state[f"pf_inc_{rid}"] = val


def _fmt_table(t: pd.DataFrame) -> pd.DataFrame:
    d = t.copy()
    for c in ("Total return", "CAGR", "Max DD", "Win rate"):
        d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}")
    d["Avg trade %"] = d["Avg trade %"].map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}%")
    d["Profit factor"] = d["Profit factor"].map(lambda v: "—" if pd.isna(v) else "∞" if np.isinf(v) else f"{v:.2f}")
    for c in ("Sharpe", "Calmar"):
        d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    d["Trades"] = d["Trades"].astype(int)
    return d


def render(cfg):
    ss = st.session_state
    ss.setdefault("pf_rows", [1, 2]); ss.setdefault("pf_next", 3); ss.setdefault("pf_res", {})
    stored = list(data.summary()["symbol"]) if len(data.summary()) else list(data.SYMBOLS)
    run_key = (cfg["start"], cfg["end"], cfg["fill"], cfg["fee"], cfg["slip"], cfg["cap"])

    st.markdown("Add as many strategies as you like (**S1, S2, S3…**). Tick the ones to analyse **together** — "
                "the combined graphs update instantly when you tick or untick.")
    top = st.columns([1.1, 1.1, 1, 3])
    top[0].button("☑ Tick all", on_click=_tick_all, args=(True,), width="stretch")
    top[1].button("☐ Untick all", on_click=_tick_all, args=(False,), width="stretch")
    top[2].button("➕ Add strategy", on_click=_add, type="secondary", width="stretch")
    split = top[3].radio("Capital split", ["Equal", "Custom %"], horizontal=True, key="pf_split",
                         help="Equal: every ticked strategy gets the same share of capital. Custom: set a weight per strategy.")

    rows = {}
    for rid in list(ss["pf_rows"]):
        with st.container(border=True):
            c = st.columns([.55, 1.3, 1.6, 1.1, .5])
            inc = c[0].checkbox("Use", value=True, key=f"pf_inc_{rid}")
            name = c[1].text_input("Name", value=f"S{rid}", key=f"pf_name_{rid}")
            sym = c[2].selectbox("Index", stored, index=stored.index(cfg["sym"]) if cfg["sym"] in stored else 0, key=f"pf_sym_{rid}")
            w = c[3].number_input("Weight %", 0.0, 100.0, 100.0 / max(len(ss["pf_rows"]), 1), 5.0, key=f"pf_w_{rid}",
                                  disabled=split == "Equal")
            c[4].button("🗑", key=f"pf_rm_{rid}", on_click=_remove, args=(rid,), help="Remove this strategy",
                        disabled=len(ss["pf_rows"]) <= 1)
            q = st.text_area("Strategy text", value=DEFAULTS[(rid - 1) % len(DEFAULTS)], key=f"pf_q_{rid}", height=84,
                             label_visibility="collapsed")
            try:
                parsed = parse_query(q)
                st.caption("✅ " + parsed.describe().replace("\n", "  ·  "))
            except QueryError as e:
                parsed = None
                st.error(f"Can't parse: {e}")
            rows[rid] = dict(inc=inc, name=name.strip() or f"S{rid}", sym=sym, w=w, q=q, parsed=parsed)

    names = [r["name"] for r in rows.values()]
    if len(set(names)) != len(names):
        st.error("Give every strategy a different name.")
        return

    # ------------------------------------------------------------------ run
    go = st.button("▶ Run all strategies", type="primary", disabled=not any(r["parsed"] for r in rows.values()), key="pf_run")
    if go:
        prog = st.progress(0.0, text="Backtesting…")
        for i, (rid, r) in enumerate(rows.items()):
            prog.progress((i + 1) / len(rows), text=f"Backtesting {r['name']}…")
            if r["parsed"] is None:
                continue
            key = (r["q"], r["sym"]) + run_key
            if ss["pf_res"].get(rid, {}).get("key") == key:
                continue
            try:
                df = load_df(r["sym"], cfg["start"], cfg["end"], ss.get("stamp", 0))
            except LookupError:
                st.warning(f"{r['name']}: no stored data for {r['sym']}."); continue
            ss["pf_res"][rid] = dict(key=key, res=engine.run(r["parsed"], df, fill=cfg["fill"], fee_pct=cfg["fee"],
                                                            slippage_pct=cfg["slip"], capital=cfg["cap"]))
        prog.empty()

    # ------------------------------------------------------------ analysis
    chosen, stale, missing = [], [], []
    for rid, r in rows.items():
        if not r["inc"]:
            continue
        got = ss["pf_res"].get(rid)
        if got is None:
            missing.append(r["name"])
        elif got["key"] != (r["q"], r["sym"]) + run_key:
            stale.append(r["name"])
        else:
            chosen.append(pf.Sleeve(r["name"], r["sym"], got["res"], r["w"] if split != "Equal" else 1.0))
    if missing or stale:
        st.info("Click **▶ Run all strategies** to " + ("run " + ", ".join(missing) if missing else "") +
                (" and " if missing and stale else "") + ("refresh the edited " + ", ".join(stale) if stale else "") + ".")
    if not chosen:
        if not (missing or stale):
            st.info("Tick at least one strategy to see the combined analysis.")
        return

    try:
        c = pf.combine(chosen, cfg["cap"], cfg["rf"])
    except ValueError as e:
        st.error(str(e)); return
    st.divider()
    st.subheader(f"Combined: {', '.join(s.name for s in chosen)}")
    st.caption(f"{c.start:%d %b %Y} → {c.end:%d %b %Y} · each strategy trades its own share of ₹{cfg['cap']:,.0f} and compounds independently")
    cb, tb, ts = c.stats, c.table.drop(index="COMBINED"), c.stats["trades"]
    k = st.columns(6)
    k[0].metric("CAGR", pct(cb["cagr"]), f"avg {pct(tb.CAGR.mean())}", delta_color="off")
    k[1].metric("Max drawdown", pct(cb["max_dd"]), f"worst {pct(tb['Max DD'].min())}", delta_color="off")
    k[2].metric("Sharpe", f"{cb['sharpe']:.2f}", f"best {tb.Sharpe.max():.2f}", delta_color="off")
    pfv = ts.get("profit_factor", np.nan)
    k[3].metric("Profit factor", "∞" if np.isinf(pfv) else f"{pfv:.2f}" if not pd.isna(pfv) else "—")
    k[4].metric("Trades", ts.get("n", 0), f"win {pct(ts['win_rate'])}" if ts.get("n") else None, delta_color="off")
    k[5].metric("Net P&L", f"₹{cb['net_pnl']:+,.0f}", f"{cb['total_return']:+.1%}", delta_color="off")
    if len(chosen) > 1:
        cd = c.corr_daily.where(~np.eye(len(c.corr_daily), dtype=bool)).stack()
        st.caption(f"Cards compare the combined result with the individual strategies (avg / worst / best). Average pairwise correlation of daily returns: **{cd.mean():.2f}** · Calmar (CAGR ÷ max drawdown): **{cb['calmar']:.2f}**")
    st.dataframe(_fmt_table(c.table), width="stretch")

    tabs = st.tabs(["💡 Insights", "📈 Combined equity & DD", "🔗 Correlation", "📊 Contribution & years", "🗓 Monthly",
                    "🧪 What-if", "📋 Trades"])
    with tabs[0]:
        for i in c.insights:
            st.markdown(f"- {i}")
    with tabs[1]:
        st.plotly_chart(charts.portfolio_equity(c), width="stretch", key="pf_eq")
    with tabs[2]:
        if len(chosen) < 2:
            st.info("Tick two or more strategies to see correlations.")
        else:
            st.plotly_chart(charts.corr_heatmap(c.corr_daily, "Daily-return correlation (−1 = opposite, 0 = independent, +1 = identical)"),
                            width="stretch", key="pf_c1")
            st.plotly_chart(charts.corr_heatmap(c.corr_monthly, "Monthly-return correlation"), width="stretch", key="pf_c2")
            st.plotly_chart(charts.corr_heatmap(c.overlap, "Time overlap: share of active days when both are in a trade", 0, 1, ".0%"),
                            width="stretch", key="pf_c3")
    with tabs[3]:
        st.plotly_chart(charts.contribution(c), width="stretch", key="pf_contrib")
        st.plotly_chart(charts.portfolio_yearly(c), width="stretch", key="pf_year")
    with tabs[4]:
        st.plotly_chart(charts.monthly(metrics.periodic_returns(c.equity, "ME")), width="stretch", key="pf_month")
    with tabs[5]:
        if len(chosen) < 2:
            st.info("Tick two or more strategies for what-if analysis.")
        else:
            st.markdown("**Leave one out** — the combined result if you removed each strategy:")
            lo = c.leave_one_out.copy()
            for col in ("CAGR", "MaxDD"):
                lo[col] = lo[col].map(lambda v: f"{v:.1%}")
            lo["Sharpe"], lo["Calmar"] = lo["Sharpe"].map("{:.2f}".format), lo["Calmar"].map("{:.2f}".format)
            lo["ProfitFactor"] = lo["ProfitFactor"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
            st.dataframe(lo, width="stretch")
            if len(chosen) <= 10:
                if st.button("🔍 Test every possible mix (equal weights)", key="pf_combos"):
                    ss["pf_combo_df"] = (pf.combos(chosen, cfg["cap"], cfg["rf"]), tuple(s.name for s in chosen))
                got = ss.get("pf_combo_df")
                if got and got[1] == tuple(s.name for s in chosen):
                    df = got[0]
                    by = st.selectbox("Rank by", ["Calmar", "Sharpe", "CAGR", "Profit factor"], key="pf_rank")
                    top = df.sort_values(by, ascending=False).head(15).copy()
                    for col in ("CAGR", "Max DD"):
                        top[col] = top[col].map(lambda v: f"{v:.1%}")
                    for col in ("Sharpe", "Calmar", "Profit factor"):
                        top[col] = top[col].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
                    st.dataframe(top, hide_index=True, width="stretch")
                    st.plotly_chart(charts.combos_scatter(df), width="stretch", key="pf_scatter")
                    st.caption("⚠️ Picking the best of many mixes on the same history flatters the result (selection bias). "
                               "Prefer mixes that are good on several measures and still look reasonable on a different date range.")
    with tabs[6]:
        t = c.trades.copy()
        t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.strftime("%Y-%m-%d"); t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.strftime("%Y-%m-%d")
        show = t[["strategy", "symbol", "direction", "entry_date", "exit_date", "entry_price", "exit_price", "bars", "ret_pct", "pts",
                  "pnl", "exit_reason"]].round({"entry_price": 1, "exit_price": 1, "ret_pct": 2, "pts": 1, "pnl": 0})
        st.dataframe(show.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download combined trades CSV", show.to_csv(index=False), "combined_trades.csv")
