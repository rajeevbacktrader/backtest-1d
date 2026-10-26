"""Pastel, interactive Plotly charts shared by the web portal and the HTML report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .engine import Result

# pastel palette (line colours are a touch deeper so thin lines stay readable)
STRAT, STRAT_FILL = "#9aa8ff", "rgba(154,168,255,.20)"
BENCH, BENCH_FILL = "#7d84a6", "rgba(125,132,166,.22)"
WIN, LOSS = "#7fd4b0", "#f4a58f"
LILAC, SUN, SKY, ROSE = "#c4a8f0", "#f7d58b", "#9bd3f2", "#f2a9c6"
INK, MUTED, GRID, PAPER, PLOT = "#e6e8fa", "#8d92b5", "#2a2c4c", "#161830", "#1c1e3a"
FONT = "Poppins, Inter, system-ui, sans-serif"


def _u(t: pd.DataFrame, unit: str):
    """(values, axis label, suffix, decimals) for per-trade charts. unit: '%' or 'pts'."""
    return (t.pts, "Points", " pts", 0) if unit == "pts" else (t.ret_pct, "Return %", "%", 2)


def _style(fig, title, h=420, legend=True):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=.01, font=dict(size=16, color=INK)),
        height=h, template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        font=dict(family=FONT, color=INK, size=12), margin=dict(l=14, r=14, t=56, b=14) if not fig.layout.margin.t else fig.layout.margin,
        hoverlabel=dict(bgcolor=PLOT, font=dict(family=FONT, color=INK), bordercolor=GRID),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)") if not fig.layout.legend.x else fig.layout.legend, showlegend=legend)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def equity(res: Result, title: str):
    """Equity (top) + underwater (bottom) on one shared, zoomable time axis."""
    eq, bm, t = res.equity, res.benchmark, res.trades
    cap = float(eq.iloc[0])
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.74, .26], vertical_spacing=.04)

    # shaded "in a trade" bands (skipped when there are too many to stay snappy)
    if 0 < len(t) <= 400:
        f.update_layout(shapes=[dict(type="rect", xref="x", yref="y domain", x0=a, x1=b, y0=0, y1=1,
                                     fillcolor="rgba(154,168,255,.06)", line_width=0, layer="below")
                                for a, b in zip(t.entry_date, t.exit_date)])
    f.add_trace(go.Scatter(x=bm.index, y=bm, name="Buy & hold", line=dict(color=BENCH, width=2),
                           hovertemplate="₹%{y:,.0f}"), 1, 1)
    f.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy", line=dict(color=STRAT, width=3.2),
                           fill="tozeroy", fillcolor="rgba(154,168,255,.13)", hovertemplate="₹%{y:,.0f}"), 1, 1)
    # trade results as dots on the curve
    if len(t):
        ey = eq.reindex(pd.DatetimeIndex(t.exit_date)).to_numpy()
        for sel, nm, col in ((t.ret_pct > 0, "Winning trade", WIN), (t.ret_pct <= 0, "Losing trade", LOSS)):
            x = t[sel]
            f.add_trace(go.Scatter(x=x.exit_date, y=ey[sel.to_numpy()], mode="markers", name=nm,
                                   marker=dict(color=col, size=6, line=dict(color=PAPER, width=1)),
                                   text=x.exit_reason + " · " + x.ret_pct.round(2).astype(str) + "% · " + x.pts.round(0).astype(int).astype(str) + " pts",
                                   hovertemplate="%{text}<extra></extra>"), 1, 1)
    log = float(max(eq.max(), bm.max()) / min(eq.min(), bm.min())) > 4
    # end-of-line labels (annotations on a log axis take log10 coordinates)
    for ser, col in ((eq, STRAT), (bm, MUTED)):
        f.add_annotation(x=ser.index[-1], y=float(np.log10(ser.iloc[-1])) if log else float(ser.iloc[-1]), xref="x", yref="y",
                         text=f"<b>₹{ser.iloc[-1]:,.0f}</b><br>{ser.iloc[-1] / cap - 1:+.0%}", showarrow=False,
                         xanchor="left", xshift=8, font=dict(color=col, size=12))

    # underwater
    for ser, nm, col, fc in ((bm, "Buy & hold", BENCH, BENCH_FILL), (eq, "Strategy", LOSS, "rgba(244,165,143,.38)")):
        dd = (ser / ser.cummax() - 1) * 100
        f.add_trace(go.Scatter(x=dd.index, y=dd, name=nm + " drawdown", showlegend=False, line=dict(color=col, width=1.8),
                               fill="tozeroy", fillcolor=fc, hovertemplate="%{y:.1f}%"), 2, 1)
    ddm = (eq / eq.cummax() - 1) * 100
    tr = ddm.idxmin()
    f.add_trace(go.Scatter(x=[tr], y=[ddm.min()], mode="markers+text", showlegend=False,
                           marker=dict(color=LOSS, size=11, symbol="diamond", line=dict(color=PAPER, width=2)),
                           text=[f"max DD {ddm.min():.1f}%"], textposition="middle left", textfont=dict(color=INK, size=11),
                           hovertemplate=f"{tr:%d %b %Y}: %{{y:.1f}}%<extra></extra>"), 2, 1)

    f.update_yaxes(type="log" if log else "linear", tickprefix="₹", title="Equity", row=1, col=1)
    f.update_yaxes(title="Drawdown %", ticksuffix="%", row=2, col=1, rangemode="tozero" if False else "normal")
    f.update_xaxes(rangeselector=dict(
        buttons=[dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"),
                 dict(count=3, label="3Y", step="year", stepmode="backward"), dict(count=5, label="5Y", step="year", stepmode="backward"),
                 dict(step="all", label="All")],
        bgcolor=PLOT, activecolor=STRAT, font=dict(color=INK, size=11), x=0, y=1.0, yanchor="bottom"), row=1, col=1)
    pad = (eq.index[-1] - eq.index[0]) * 0.035
    f.update_xaxes(range=[eq.index[0], eq.index[-1] + pad])
    f.update_layout(
        hovermode="x unified", legend=dict(orientation="h", x=.01, y=.985, bgcolor="rgba(0,0,0,0)"),
        margin=dict(r=90, t=86), updatemenus=[dict(type="buttons", direction="right", x=1, xanchor="right", y=1.0, yanchor="bottom",
            showactive=False, bgcolor=PLOT, bordercolor=GRID, font=dict(color=INK, size=11),
            buttons=[dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                     dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}])])])
    return _style(f, f"{title} · equity curve", 660)


def price(res: Result, title: str):
    d, t = res.df, res.trades
    f = go.Figure(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name=title,
                                 increasing=dict(line=dict(color=WIN), fillcolor=WIN),
                                 decreasing=dict(line=dict(color=LOSS), fillcolor=LOSS)))
    if len(t):
        f.add_trace(go.Scatter(x=t.entry_date, y=t.entry_price, mode="markers", name="Entry",
                               marker=dict(symbol="triangle-up", size=11, color=STRAT, line=dict(color=PAPER, width=1.5)),
                               hovertemplate="Entry %{y:,.1f}"))
        for sel, nm, col in ((t.ret_pct > 0, "Exit · win", "#3fb98a"), (t.ret_pct <= 0, "Exit · loss", "#e8765b")):
            x = t[sel]
            f.add_trace(go.Scatter(x=x.exit_date, y=x.exit_price, mode="markers", name=nm,
                                   marker=dict(symbol="diamond", size=9, color=col, line=dict(color=PAPER, width=1.5)),
                                   text=x.exit_reason + " · " + x.ret_pct.round(2).astype(str) + "% · " + x.pts.round(0).astype(int).astype(str) + " pts",
                                   hovertemplate="%{text}<br>%{y:,.1f}"))
    f.update_layout(xaxis_rangeslider_visible=False); f.update_yaxes(type="log")
    return _style(f, f"{title} · price with your trades", 500)


def waterfall(res: Result, unit="%"):
    t = res.trades; v, lab, suf, dec = _u(t, unit)
    cum = v.cumsum() if unit == "pts" else (1 + t.ret_pct / 100).cumprod().sub(1).mul(100)
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=np.arange(1, len(t) + 1), y=v, name=f"Per trade ({lab.lower()})", marker_color=np.where(v > 0, WIN, LOSS),
                       customdata=np.c_[t.entry_date.astype(str).str[:10], t.exit_reason],
                       hovertemplate="#%{x} · %{customdata[0]} · %{customdata[1]}<br>%{y:,." + str(dec) + "f}" + suf), secondary_y=False)
    f.add_trace(go.Scatter(x=np.arange(1, len(t) + 1), y=cum, name="Running total", line=dict(color=STRAT, width=3),
                           hovertemplate="%{y:,.1f}" + suf), secondary_y=True)
    f.update_xaxes(title="Trade #"); f.update_yaxes(title=f"Per trade ({lab.lower()})", secondary_y=False)
    f.update_yaxes(title="Cumulative" + (" points" if unit == "pts" else " % (compounded)"), secondary_y=True, showgrid=False)
    return _style(f, "Every trade, one bar each · and the running total" + (" (points)" if unit == "pts" else ""), 380)


def distribution(res: Result, unit="%"):
    r, lab, suf, dec = _u(res.trades, unit)
    lim = max(abs(r.quantile(.01)), abs(r.quantile(.99)), 1)
    b = dict(start=-lim, end=lim, size=2 * lim / 30)
    f = go.Figure()
    f.add_trace(go.Histogram(x=r[r > 0].clip(-lim, lim), xbins=b, name="Wins", marker_color=WIN, opacity=.9))
    f.add_trace(go.Histogram(x=r[r <= 0].clip(-lim, lim), xbins=b, name="Losses", marker_color=LOSS, opacity=.9))
    f.add_vline(x=r.mean(), line=dict(color=INK, dash="dot"), annotation_text=f"avg {r.mean():,.{dec}f}{suf}", annotation_font_color=INK)
    f.update_layout(barmode="overlay", bargap=.06); f.update_xaxes(title=lab + " per trade"); f.update_yaxes(title="Trades")
    return _style(f, "Trade result distribution" + (" (points)" if unit == "pts" else ""), 360)


def donuts(res: Result):
    t = res.trades
    f = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                      subplot_titles=("Win vs loss", "How trades ended"))
    w = int((t.ret_pct > 0).sum())
    f.add_trace(go.Pie(labels=["Wins", "Losses"], values=[w, len(t) - w], hole=.62, marker=dict(colors=[WIN, LOSS], line=dict(color=PAPER, width=3)),
                       textinfo="percent", sort=False), 1, 1)
    g = t.exit_reason.value_counts()
    pal = [STRAT, SUN, LILAC, SKY, ROSE, WIN]
    f.add_trace(go.Pie(labels=list(g.index), values=list(g.values), hole=.62, marker=dict(colors=pal[:len(g)], line=dict(color=PAPER, width=3)),
                       textinfo="label+percent"), 1, 2)
    f.update_annotations(font=dict(size=13, color=INK))
    return _style(f, "Outcome mix", 340, legend=False)


def exit_bars(res: Result, unit="%"):
    t = res.trades; v, lab, suf, dec = _u(t, unit)
    g = pd.DataFrame({"avg": v, "r": t.ret_pct, "k": t.exit_reason}).groupby("k").agg(avg=("avg", "mean"), n=("r", "size"), win=("r", lambda x: (x > 0).mean() * 100))
    g = g.sort_values("avg")
    f = go.Figure(go.Bar(y=g.index, x=g.avg, orientation="h", marker_color=np.where(g.avg > 0, WIN, LOSS),
                         text=[f"{a:+,.{dec}f}{suf} · {int(n)} trades · {w:.0f}% win" for a, n, w in zip(g.avg, g.n, g.win)],
                         textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:,.2f}" + suf))
    f.update_xaxes(title=f"Average {lab.lower()} per trade")
    f.update_layout(margin=dict(r=170))
    return _style(f, "Which exits make money?" + (" (points)" if unit == "pts" else ""), 320, legend=False)


def scatter(res: Result, unit="%"):
    t = res.trades; v, lab, suf, dec = _u(t, unit); w = v > 0
    sc = float(v.abs().quantile(.95)) or 1.0
    f = go.Figure()
    for sel, nm, col in ((w, "Win", WIN), (~w, "Loss", LOSS)):
        x, y = t[sel], v[sel]
        f.add_trace(go.Scatter(x=x.bars, y=y, mode="markers", name=nm,
                               marker=dict(color=col, size=np.clip(abs(y) / sc * 14 + 8, 8, 30), opacity=.8, line=dict(color=PAPER, width=1)),
                               text=x.entry_date.astype(str).str[:10] + " · " + x.exit_reason + " · " + x.pts.round(0).astype(int).astype(str) + " pts",
                               hovertemplate="%{text}<br>" + ("%{y:.2f}%" if unit == "%" else "%{y:,.0f} pts")))
    f.add_hline(y=0, line=dict(color=MUTED, width=1)); f.update_xaxes(title="Candles held"); f.update_yaxes(title=lab)
    return _style(f, "Does holding longer help?" + (" (points)" if unit == "pts" else ""), 360)


def yearly(yearly_df: pd.DataFrame):
    f = go.Figure()
    f.add_trace(go.Bar(x=yearly_df.index.astype(str), y=yearly_df["Strategy"] * 100, name="Strategy", marker_color=STRAT,
                       hovertemplate="%{x}: %{y:.1f}%"))
    f.add_trace(go.Bar(x=yearly_df.index.astype(str), y=yearly_df["Buy & hold"] * 100, name="Buy & hold", marker_color=BENCH,
                       hovertemplate="%{x}: %{y:.1f}%"))
    f.update_layout(barmode="group", bargap=.25); f.update_yaxes(title="Return %")
    return _style(f, "Year by year", 360)


def monthly(m: pd.Series):
    p = (m * 100).to_frame("r"); p["y"], p["m"] = p.index.year, p.index.month
    p = p.pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13))
    f = go.Figure(go.Heatmap(z=p.values, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                             y=p.index.astype(str), zmid=0, xgap=3, ygap=3, texttemplate="%{z:.1f}", textfont=dict(size=10, color=INK),
                             colorscale=[[0, "#f4a58f"], [.5, "#1c1e3a"], [1, "#7fd4b0"]], colorbar=dict(title="%", thickness=10),
                             hovertemplate="%{y} %{x}: %{z:.2f}%"))
    f.update_yaxes(autorange="reversed", type="category"); f.update_xaxes(side="top")
    return _style(f, "Monthly returns calendar (%)", max(300, 26 * len(p) + 120), legend=False)


def rolling_year(res: Result):
    f = go.Figure()
    for ser, nm, col in ((res.benchmark, "Buy & hold", BENCH), (res.equity, "Strategy", STRAT)):
        r = ser.pct_change(252).dropna() * 100
        f.add_trace(go.Scatter(x=r.index, y=r, name=nm, line=dict(color=col, width=2.5 if nm == "Strategy" else 2), hovertemplate="%{y:.1f}%"))
    f.add_hline(y=0, line=dict(color=MUTED, width=1)); f.update_layout(hovermode="x unified"); f.update_yaxes(title="1-year return %")
    return _style(f, "Rolling 1-year return · is the edge steady?", 340)


def weekday(res: Result, unit="%"):
    t = res.trades.copy(); v, lab, suf, dec = _u(t, unit); t["v"] = v; t["wd"] = pd.to_datetime(t.entry_date).dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    g = t.groupby("wd").v.agg(["mean", "size"]).reindex(order).fillna(0)
    f = go.Figure(go.Bar(x=g.index.str[:3], y=g["mean"], marker_color=np.where(g["mean"] > 0, WIN, LOSS),
                         text=[f"{int(n)} trades" for n in g["size"]], textposition="outside",
                         hovertemplate="%{x}: %{y:,.2f}" + suf + " avg"))
    f.update_yaxes(title=f"Avg {lab.lower()}")
    return _style(f, "Best day to enter (by entry weekday)" + (" (points)" if unit == "pts" else ""), 320, legend=False)


def rolling_winrate(res: Result, window: int = 20):
    t = res.trades
    if len(t) < window + 5:
        return None
    wr = (t.ret_pct > 0).rolling(window).mean().mul(100)
    f = go.Figure(go.Scatter(x=t.exit_date, y=wr, line=dict(color=LILAC, width=3), fill="tozeroy",
                             fillcolor="rgba(196,168,240,.22)", hovertemplate="%{y:.0f}%"))
    f.add_hline(y=50, line=dict(color=MUTED, dash="dot")); f.update_yaxes(title="Win rate %", range=[0, 100])
    return _style(f, f"Rolling win rate (last {window} trades)", 320, legend=False)


def build_all(res: Result, m: dict, title: str, unit: str = "%") -> dict:
    """name -> (group, figure). Groups drive the portal tabs / report sections."""
    out = {"equity": ("Performance", equity(res, title)),
           "rolling_year": ("Performance", rolling_year(res)), "price": ("Trades", price(res, title))}
    if len(res.trades):
        out["waterfall"] = ("Trades", waterfall(res, unit))
        out["scatter"] = ("Trades", scatter(res, unit))
        rw = rolling_winrate(res)
        if rw is not None:
            out["rolling_win"] = ("Trades", rw)
        out["donuts"] = ("Breakdown", donuts(res)); out["distribution"] = ("Breakdown", distribution(res, unit))
        out["exit_bars"] = ("Breakdown", exit_bars(res, unit)); out["weekday"] = ("Breakdown", weekday(res, unit))
    out["yearly"] = ("Calendar", yearly(m["yearly"])); out["monthly"] = ("Calendar", monthly(m["monthly"]))
    return out


# ------------------------------------------------------------------ portfolio charts
SLEEVE_COLORS = [SKY, SUN, ROSE, LILAC, WIN, "#f0b27a", "#85c1e9", "#d7bde2", "#a3e4d7", "#f5b7b1"]


def portfolio_equity(c):
    """Combined equity + each strategy sleeve (top) and combined drawdown (bottom) on one shared time axis."""
    eq, cap = c.equity, c.capital
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.7, .3], vertical_spacing=.04)
    f.add_trace(go.Scatter(x=c.bench.index, y=c.bench, name="Buy & hold (same weights)", line=dict(color=BENCH, width=2),
                           hovertemplate="₹%{y:,.0f}"), 1, 1)
    for i, col in enumerate(c.sleeves_eq.columns):
        f.add_trace(go.Scatter(x=c.sleeves_eq.index, y=c.sleeves_eq[col], name=f"{col} (standalone)", visible="legendonly" if len(c.sleeves_eq.columns) > 6 else True,
                               line=dict(color=SLEEVE_COLORS[i % len(SLEEVE_COLORS)], width=1.4, dash="dot"), opacity=.9,
                               hovertemplate="₹%{y:,.0f}"), 1, 1)
    f.add_trace(go.Scatter(x=eq.index, y=eq, name="COMBINED", line=dict(color=STRAT, width=3.4), fill="tozeroy",
                           fillcolor="rgba(154,168,255,.12)", hovertemplate="₹%{y:,.0f}"), 1, 1)
    low = min(eq.min(), c.bench.min(), c.sleeves_eq.min().min()); high = max(eq.max(), c.bench.max(), c.sleeves_eq.max().max())
    log = high / low > 4
    for ser, col in ((eq, STRAT), (c.bench, MUTED)):
        f.add_annotation(x=ser.index[-1], y=float(np.log10(ser.iloc[-1])) if log else float(ser.iloc[-1]), xref="x", yref="y",
                         text=f"<b>₹{ser.iloc[-1]:,.0f}</b><br>{ser.iloc[-1] / cap - 1:+.0%}", showarrow=False, xanchor="left", xshift=8,
                         font=dict(color=col, size=12))
    for col in c.sleeves_eq.columns:
        pass
    # drawdowns: each sleeve (on its own capital) thin, combined bold
    for i, col in enumerate(c.norm.columns):
        d = (c.norm[col] / c.norm[col].cummax() - 1) * 100
        f.add_trace(go.Scatter(x=d.index, y=d, name=f"{col} DD", showlegend=False, line=dict(color=SLEEVE_COLORS[i % len(SLEEVE_COLORS)], width=1, dash="dot"),
                               hovertemplate=f"{col}: %{{y:.1f}}%"), 2, 1)
    dd = (eq / eq.cummax() - 1) * 100
    f.add_trace(go.Scatter(x=dd.index, y=dd, name="COMBINED DD", showlegend=False, line=dict(color=LOSS, width=2.4), fill="tozeroy",
                           fillcolor="rgba(244,165,143,.35)", hovertemplate="Combined: %{y:.1f}%"), 2, 1)
    tr = dd.idxmin()
    f.add_trace(go.Scatter(x=[tr], y=[dd.min()], mode="markers+text", showlegend=False,
                           marker=dict(color=LOSS, size=11, symbol="diamond", line=dict(color=PAPER, width=2)),
                           text=[f"max DD {dd.min():.1f}%"], textposition="middle left", textfont=dict(color=INK, size=11),
                           hovertemplate=f"{tr:%d %b %Y}: %{{y:.1f}}%<extra></extra>"), 2, 1)
    f.update_yaxes(type="log" if log else "linear", tickprefix="₹", title="Equity", row=1, col=1)
    f.update_yaxes(title="Drawdown %", ticksuffix="%", row=2, col=1)
    f.update_xaxes(rangeselector=dict(
        buttons=[dict(count=1, label="1Y", step="year", stepmode="backward"), dict(count=3, label="3Y", step="year", stepmode="backward"),
                 dict(count=5, label="5Y", step="year", stepmode="backward"), dict(step="all", label="All")],
        bgcolor=PLOT, activecolor=STRAT, font=dict(color=INK, size=11), x=0, y=1.0, yanchor="bottom"), row=1, col=1)
    pad = (eq.index[-1] - eq.index[0]) * 0.035
    f.update_xaxes(range=[eq.index[0], eq.index[-1] + pad])
    f.update_layout(hovermode="x unified", margin=dict(r=90, t=86), legend=dict(orientation="h", x=.01, y=.985, bgcolor="rgba(0,0,0,0)"),
                    updatemenus=[dict(type="buttons", direction="right", x=1, xanchor="right", y=1.0, yanchor="bottom", showactive=False,
                                      bgcolor=PLOT, bordercolor=GRID, font=dict(color=INK, size=11),
                                      buttons=[dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                                               dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}])])])
    return _style(f, "Combined equity & drawdown · dotted lines = each strategy on its own", 680)


def corr_heatmap(m: pd.DataFrame, title: str, lo=-1.0, hi=1.0, fmt=".2f"):
    f = go.Figure(go.Heatmap(z=m.values, x=list(m.columns), y=list(m.index), zmin=lo, zmax=hi, xgap=4, ygap=4,
                             colorscale=[[0, LOSS], [.5, PLOT], [1, STRAT]] if lo < 0 else [[0, PLOT], [1, STRAT]],
                             texttemplate="%{z:" + fmt + "}", textfont=dict(size=13, color=INK),
                             colorbar=dict(thickness=10), hovertemplate="%{y} vs %{x}: %{z:" + fmt + "}<extra></extra>"))
    f.update_yaxes(autorange="reversed"); f.update_xaxes(side="top")
    f.update_layout(margin=dict(l=14, r=14, t=100, b=14))
    return _style(f, title, max(320, 70 * len(m) + 150), legend=False)


def contribution(c):
    s = c.contribution.sort_values()
    tot = s.sum()
    f = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker_color=np.where(s.values > 0, WIN, LOSS),
                         text=[f"₹{v:+,.0f}" + (f" · {v / tot:.0%} of net" if tot > 0 and v > 0 else "") for v in s.values],
                         textposition="outside", cliponaxis=False, hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
    f.update_layout(margin=dict(r=140)); f.update_xaxes(title="Net P&L ₹ (portfolio)")
    return _style(f, "Who made (or lost) the money?", max(260, 60 * len(s) + 120), legend=False)


def portfolio_yearly(c):
    def yr(s):
        y = s.resample("YE").last().pct_change()
        y.iloc[0] = s.resample("YE").last().iloc[0] / s.iloc[0] - 1
        return y * 100
    f = go.Figure()
    for i, col in enumerate(c.norm.columns):
        y = yr(c.norm[col])
        f.add_trace(go.Bar(x=y.index.year.astype(str), y=y, name=col, marker_color=SLEEVE_COLORS[i % len(SLEEVE_COLORS)], opacity=.85,
                           hovertemplate="%{x}: %{y:.1f}%"))
    y = yr(c.equity)
    f.add_trace(go.Bar(x=y.index.year.astype(str), y=y, name="COMBINED", marker_color=STRAT, hovertemplate="%{x}: %{y:.1f}%"))
    f.update_layout(barmode="group", bargap=.2); f.update_yaxes(title="Return %")
    return _style(f, "Year by year · each strategy vs the combined portfolio", 380)


def combos_scatter(df: pd.DataFrame):
    f = go.Figure(go.Scatter(x=-df["Max DD"] * 100, y=df["CAGR"] * 100, mode="markers", text=df["Mix"],
                             marker=dict(size=np.clip(df["Strategies"] * 5 + 6, 8, 26), color=df["Sharpe"], colorscale=[[0, LOSS], [.5, SUN], [1, WIN]],
                                         colorbar=dict(title="Sharpe", thickness=10), line=dict(color=PAPER, width=1)),
                             hovertemplate="%{text}<br>CAGR %{y:.1f}% · max DD %{x:.1f}%<extra></extra>"))
    f.update_xaxes(title="Max drawdown % (smaller is better →)", autorange="reversed"); f.update_yaxes(title="CAGR %")
    return _style(f, "Every possible mix · return vs risk", 420, legend=False)
