"""Self-contained interactive HTML report (pastel Plotly charts)."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts
from .engine import Result


def _fmt(k, v):
    if isinstance(v, (float, np.floating)):
        return "∞" if np.isinf(v) else "—" if np.isnan(v) else f"{v:,.2f}"
    return str(v)


def write_report(res: Result, m: dict, ins: list[str], title: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    figs = charts.build_all(res, m, title)
    if len(res.trades):                       # raw-points versions of the per-trade charts
        pf = charts.build_all(res, m, title, unit="pts")
        for k in ("waterfall", "distribution", "exit_bars", "scatter", "weekday"):
            figs[k + "_pts"] = ("Points", pf[k][1])
    parts, first = [], True
    for grp in dict.fromkeys(g for g, _ in figs.values()):
        parts.append(f"<h3>{grp}</h3>")
        for k, (g, f) in figs.items():
            if g == grp:
                parts.append(f'<div class="card">{f.to_html(full_html=False, include_plotlyjs=first, config={"displaylogo": False})}</div>')
                first = False
                try:  # PNG copies only if kaleido is installed
                    f.write_image(str(out_dir / f"{k}.png"), scale=2)
                except Exception:
                    pass
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    res.equity.to_frame("strategy").join(res.benchmark.rename("buy_hold")).to_csv(out_dir / "equity.csv")

    s, b, ts = m["strategy"], m["benchmark"], m["trades"]
    P = lambda x: f"{x * 100:.1f}%"
    rows = [("Total return", P(s["total_return"]), P(b["total_return"])), ("CAGR", P(s["cagr"]), P(b["cagr"])),
            ("Volatility (ann.)", P(s["vol"]), P(b["vol"])), ("Sharpe", f"{s['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
            ("Sortino", f"{s['sortino']:.2f}", f"{b['sortino']:.2f}"), ("Max drawdown", P(s["max_dd"]), P(b["max_dd"])),
            ("Calmar", f"{s['calmar']:.2f}", f"{b['calmar']:.2f}")]
    trow = [] if not ts["n"] else [("Trades", ts["n"]), ("Win rate", P(ts["win_rate"])), ("Avg trade", f"{ts['avg_ret']:+.2f}%"), ("Total points captured", f"{ts['total_pts']:+,.0f}"), ("Avg points / trade", f"{ts['avg_pts']:+,.1f}"),
                                    ("Avg win / avg loss", f"{ts['avg_win']:.2f}% / {ts['avg_loss']:.2f}%"),
                                    ("Profit factor", _fmt(0, ts["profit_factor"])), ("Avg candles held", f"{ts['avg_bars']:.1f}"),
                                    ("Time in market", P(m["exposure"])), ("Final equity", f"₹{res.equity.iloc[-1]:,.0f}")]
    t1 = "".join(f"<tr><td>{a}</td><td>{x}</td><td>{y}</td></tr>" for a, x, y in rows)
    t2 = "".join(f"<tr><td>{a}</td><td>{x}</td></tr>" for a, x in trow)
    li = "".join(f"<li>{html.escape(i)}</li>" for i in ins)
    tr = res.trades.copy()
    for c_ in ("entry_date", "exit_date"):
        tr[c_] = pd.to_datetime(tr[c_]).dt.strftime("%Y-%m-%d")
    tr = tr.round({"entry_price": 1, "exit_price": 1, "ret_pct": 2, "pts": 1, "pnl": 0}).drop(columns=["open_at_end"])
    trades_html = tr.iloc[::-1].to_html(index=False, border=0, classes="t")
    imgtags = "".join(parts)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest {html.escape(title)}</title><style>
body{{font:15px/1.55 Poppins,Inter,system-ui,sans-serif;color:#e6e8fa;background:linear-gradient(180deg,#12132a,#1a1633 60%,#102226);max-width:1100px;margin:0 auto;padding:20px}}.card{{background:#161830;border-radius:18px;padding:6px;margin:14px 0;box-shadow:0 4px 18px rgba(0,0,0,.35)}}h3{{margin:1.6em 0 0;color:#9aa8ff}}
h1{{font-size:22px;margin:.2em 0}}h2{{font-size:17px;margin:1.6em 0 .4em}}pre{{background:#1c1e3a;padding:12px;border-radius:14px;white-space:pre-wrap}}
table{{border-collapse:collapse}}td,th{{padding:4px 12px;border-bottom:1px solid #2a2c4c;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}}img{{max-width:100%;margin:8px 0}}li{{margin:.4em 0}}
.sc{{max-height:420px;overflow:auto}}.meta{{color:#8d92b5}}</style></head><body>
<h1>Backtest — {html.escape(title)}</h1>
<div class="meta">{m['start']:%d %b %Y} → {m['end']:%d %b %Y} · {m['candles']} daily candles · fill: {res.params['fill']} · fee {res.params['fee_pct']}%/side · slippage {res.params['slippage_pct']}%</div>
<h2>Strategy</h2><pre>{html.escape(res.strategy.source)}</pre><pre>{html.escape(res.strategy.describe())}</pre>
<div class="g"><div><h2>Returns &amp; risk</h2><table><tr><th></th><th>Strategy</th><th>Buy &amp; hold</th></tr>{t1}</table></div>
<div><h2>Trades</h2><table>{t2}</table></div></div>
<h2>Insights</h2><ul>{li}</ul><h2>Charts</h2>{imgtags}
<h2>All trades (latest first)</h2><div class="sc">{trades_html}</div>
<p class="meta">Backtest on index levels (not directly tradable; a proxy for futures/ETF). Past performance does not predict future results. Not investment advice.</p>
</body></html>"""
    p = out_dir / "report.html"
    p.write_text(page, encoding="utf-8")
    return p
