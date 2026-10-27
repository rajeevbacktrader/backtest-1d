"""'Backtest report' view: stat chips + stacked EQUITY / DRAWDOWN / P&L cards (cumulative P&L in rupees from zero)."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GREEN, RED, BLUE, INK, MUTED, GRID = "#22c55e", "#ef4444", "#2563eb", "#111827", "#6b7280", "#eef0f3"
G_FILL, R_FILL = "rgba(34,197,94,.13)", "rgba(239,68,68,.13)"


def set_colors(pos: str, neg: str, accent: str):
    """Recolour the report cards (called by charts.set_palette)."""
    global GREEN, RED, BLUE, G_FILL, R_FILL
    rgba = lambda h, a: "rgba(%d,%d,%d,%s)" % (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), a)
    GREEN, RED, BLUE, G_FILL, R_FILL = pos, neg, accent, rgba(pos, .13), rgba(neg, .13)
RANGES = [("6M", 183), ("1Y", 365), ("3Y", 3 * 365), ("5Y", 5 * 365), ("All", None)]
GROUPS = ["Month", "Trade", "Year"]
BASES = {"Closed trades": "trades", "Daily (MTM)": "daily"}
FONT = "Inter, system-ui, sans-serif"


def available_ranges(equity: pd.Series) -> list[str]:
    span = (equity.index[-1] - equity.index[0]).days
    return [n for n, d in RANGES if d is None or d < span]


def _zero_split(x, y):
    """Insert points where the line crosses zero so the green/red parts meet exactly at 0."""
    xs = pd.DatetimeIndex(x).as_unit("ns").asi8.astype(float)   # explicit ns: pandas 3 may hold other units
    ox, oy = [xs[0]], [y[0]]
    for i in range(1, len(y)):
        if (y[i - 1] < 0 < y[i]) or (y[i - 1] > 0 > y[i]):
            t = y[i - 1] / (y[i - 1] - y[i])
            ox.append(xs[i - 1] + t * (xs[i] - xs[i - 1])); oy.append(0.0)
        ox.append(xs[i]); oy.append(y[i])
    return pd.to_datetime(np.array(ox, dtype="int64"), unit="ns"), np.array(oy, float)


def _curve(equity: pd.Series, trades: pd.DataFrame, start: pd.Timestamp, basis: str) -> pd.Series:
    """Cumulative P&L (rupees) starting at 0 on `start`."""
    if basis == "daily":
        e = equity.loc[start:]
        return e - e.iloc[0]
    t = trades[pd.to_datetime(trades.exit_date) >= start].sort_values("exit_date", kind="stable")
    x = pd.DatetimeIndex([start] + list(pd.to_datetime(t.exit_date)))
    y = np.r_[0.0, t.pnl.cumsum().to_numpy()]
    s = pd.Series(y, index=x)
    return s.groupby(level=0).last()        # several exits on one day -> keep the running total


def build(equity: pd.Series, trades: pd.DataFrame, range_name: str = "All", basis: str = "trades", group: str = "Month",
          show_points: bool = True) -> dict:
    days = dict(RANGES).get(range_name)
    end = equity.index[-1]
    start = equity.index[0] if days is None else max(equity.index[0], end - pd.Timedelta(days=days))
    cum = _curve(equity, trades, start, basis)
    tw = trades[pd.to_datetime(trades.exit_date) >= start]

    run_max = np.maximum.accumulate(np.r_[0.0, cum.to_numpy()])[1:]
    dd = cum - run_max
    wins = (tw.ret_pct > 0).mean() if len(tw) else np.nan
    gp, gl = tw.pnl[tw.pnl > 0].sum(), -tw.pnl[tw.pnl <= 0].sum()
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else np.nan)
    f = lambda v: f"{v:+,.0f}"
    chips = [("FINAL", f(cum.iloc[-1]), GREEN if cum.iloc[-1] >= 0 else RED),
             ("PEAK", f"{max(cum.max(), 0):+,.0f}", INK),
             ("MAX DD", f"{dd.min():,.0f}".replace("-", "-"), RED),
             ("WIN RATE", "—" if pd.isna(wins) else f"{wins:.1%}", BLUE),
             ("PROFIT FACTOR", "—" if pd.isna(pf) else "∞" if np.isinf(pf) else f"{pf:.2f}", INK),
             ("TRADES", str(len(tw)), INK)]
    if show_points and len(tw):
        chips.append(("POINTS", f(tw.pts.sum()), GREEN if tw.pts.sum() >= 0 else RED))

    pad = (end - start) * 0.02 + pd.Timedelta(days=2)
    xr = [start - pad, end + pad]
    return dict(chips=chips, start=start, end=end, figs=dict(
        equity=_equity_fig(cum, xr), drawdown=_dd_fig(cum.index, dd, xr), pnl=_pnl_fig(tw, group, xr)))


def _base(fig, xr, height, bottom_axis=False):
    fig.update_layout(height=height, paper_bgcolor="white", plot_bgcolor="white", font=dict(family=FONT, color=MUTED, size=11),
                      margin=dict(l=62, r=22, t=8, b=34 if bottom_axis else 8), showlegend=False, hovermode="x",
                      hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK), bordercolor=GRID))
    fig.update_xaxes(range=xr, showgrid=False, zeroline=False, showline=False, showticklabels=bottom_axis, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=1, showline=False, tickformat="~s", ticks="",
                     fixedrange=True)
    return fig


def _equity_fig(cum: pd.Series, xr):
    x, y = _zero_split(cum.index, cum.to_numpy())
    pos, neg = np.where(y > 0, y, 0.0), np.where(y < 0, y, 0.0)
    f = go.Figure()
    f.add_trace(go.Scatter(x=x, y=pos, fill="tozeroy", fillcolor=G_FILL, line=dict(width=0), hoverinfo="skip"))
    f.add_trace(go.Scatter(x=x, y=neg, fill="tozeroy", fillcolor=R_FILL, line=dict(width=0), hoverinfo="skip"))
    f.add_trace(go.Scatter(x=x, y=np.where(y >= 0, y, np.nan), line=dict(color=GREEN, width=2), hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>"))
    f.add_trace(go.Scatter(x=x, y=np.where(y <= 0, y, np.nan), line=dict(color=RED, width=2), hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>"))
    lim = max(abs(y).max(), 1) * 1.12
    f = _base(f, xr, 300)
    f.update_yaxes(range=[min(y.min() * 1.12, -lim * 0.15), max(y.max() * 1.12, lim * 0.15)])
    return f


def _dd_fig(idx, dd: pd.Series, xr):
    f = go.Figure(go.Scatter(x=idx, y=dd, fill="tozeroy", fillcolor=R_FILL, line=dict(color=RED, width=1.6),
                             hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>"))
    f = _base(f, xr, 190)
    f.update_yaxes(range=[min(dd.min() * 1.15, -1), 0.0 + abs(dd.min()) * 0.03])
    return f


def _pnl_fig(tw: pd.DataFrame, group: str, xr):
    if len(tw) == 0:
        return _base(go.Figure(), xr, 190, True)
    t = tw.copy(); t["d"] = pd.to_datetime(t.exit_date)
    if group == "Trade":
        s = t.set_index("d").pnl
        s = s.groupby(level=0).sum() if False else s
        x, y = s.index, s.to_numpy()
    else:
        per = "M" if group == "Month" else "Y"
        g = t.groupby(t.d.dt.to_period(per)).pnl.sum()
        x, y = g.index.to_timestamp(), g.to_numpy()
    if group == "Month":
        x = x + pd.Timedelta(days=15)
    elif group == "Year":
        x = x + pd.Timedelta(days=182)
    f = go.Figure(go.Bar(x=x, y=y, marker_color=np.where(y >= 0, GREEN, RED), hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>"))
    f = _base(f, xr, 230, bottom_axis=True)
    f.update_xaxes(tickfont=dict(size=10), ticks="outside", tickcolor=GRID)
    f.update_layout(bargap=0.35 if group != "Trade" else 0.2)
    return f


def chips_html(chips) -> str:
    return '<div class="rv-chips">' + "".join(
        f'<div class="rv-chip"><div class="rv-l">{html.escape(l)}</div><div class="rv-v" style="color:{c}">{html.escape(v)}</div></div>'
        for l, v, c in chips) + "</div>"


CSS = """
.rv-chips{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 4px}
.rv-chip{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:9px 13px;min-width:82px}
.rv-l{font:600 10px/1 Inter,system-ui,sans-serif;letter-spacing:.14em;color:#6b7280;margin-bottom:7px}
.rv-v{font:700 18px/1 Inter,system-ui,sans-serif}
.rv-label{font:600 11px/1 Inter,system-ui,sans-serif;letter-spacing:.16em;color:#6b7280;margin:2px 0 6px}
[class*="st-key-rv-chips"]{padding-bottom:16px !important}
.rv-head{font:600 11px/1 Inter,system-ui,sans-serif;letter-spacing:.16em;color:#6b7280;margin-bottom:6px}
"""


def render_st(equity: pd.Series, trades: pd.DataFrame, key: str, show_points: bool = True, title: str = "BACKTEST REPORT"):
    """Streamlit rendering: controls, chip card, then EQUITY / DRAWDOWN / P&L cards."""
    import streamlit as st
    if len(trades) == 0:
        st.info("No trades in this period, nothing to chart."); return
    opts = available_ranges(equity) + ["All"] if available_ranges(equity) else ["All"]
    opts = list(dict.fromkeys(opts))
    c = st.columns([2.0, 2.1, 1.9])
    rng = c[0].segmented_control("Period", opts, default="All", key=f"rv_rng_{key}") or "All"
    basis = c[1].segmented_control("Curve basis", list(BASES), default="Closed trades", help="Closed trades: P&L steps at each exit (clean, like a trade log). Daily (MTM): marks open trades to market every day, so drawdowns are truer.", key=f"rv_basis_{key}") or "Closed trades"
    group = c[2].segmented_control("P&L bars", GROUPS, default="Month", key=f"rv_grp_{key}") or "Month"
    r = build(equity, trades, rng, BASES[basis], group, show_points)
    cfg = {"displayModeBar": False}
    with st.container(key=f"rv-chips-{key}"):
        st.markdown(f'<div class="rv-head">◎ {html.escape(title)} &nbsp;·&nbsp; {r["start"]:%d %b %Y} → {r["end"]:%d %b %Y}</div>' + chips_html(r["chips"]),
                    unsafe_allow_html=True)
    for name, label in (("equity", "EQUITY · cumulative P&L ₹"), ("drawdown", "DRAWDOWN ₹"), ("pnl", f"P&L ₹ · per {group.lower()}")):
        with st.container(key=f"rv-{name}-{key}"):
            st.markdown(f'<div class="rv-label">{label}</div>', unsafe_allow_html=True)
            st.plotly_chart(r["figs"][name], theme=None, config=cfg, key=f"rvfig_{name}_{key}", width="stretch")
