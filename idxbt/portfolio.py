"""Combine several independently-backtested strategies into one portfolio.

Model: each strategy ("sleeve") runs on its own slice of capital (weight x capital) and compounds
independently - no rebalancing between sleeves. Combined equity = sum of the sleeve equity curves.
All sleeves are cut to their common date window and rebased, so strategies on different indices
(e.g. FINNIFTY's shorter history) are compared on the same days.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from . import metrics
from .engine import Result


@dataclass
class Sleeve:
    name: str
    symbol: str
    res: Result
    weight: float = 1.0


@dataclass
class Combined:
    capital: float
    start: pd.Timestamp
    end: pd.Timestamp
    weights: dict                       # name -> normalised weight (sums to 1)
    norm: pd.DataFrame                  # per-sleeve equity rebased to 1.0 at start
    equity: pd.Series                   # combined equity in rupees
    sleeves_eq: pd.DataFrame            # per-strategy equity in rupees, each on the full capital (standalone view)
    bench: pd.Series                    # weighted buy & hold of the indices used
    trades: pd.DataFrame                # all trades, pnl in rupees on the portfolio, + strategy/symbol/pnl_unit
    stats: dict                         # combined curve stats + trade stats
    table: pd.DataFrame                 # one row per sleeve + Combined
    corr_daily: pd.DataFrame
    corr_monthly: pd.DataFrame
    overlap: pd.DataFrame               # share of in-market days shared by a pair (Jaccard)
    contribution: pd.Series             # net rupee pnl per sleeve
    leave_one_out: pd.DataFrame
    insights: list = field(default_factory=list)


def _norm_weights(sleeves: list[Sleeve]) -> dict:
    w = np.array([max(s.weight, 0.0) for s in sleeves], float)
    if w.sum() <= 0:
        w = np.ones(len(sleeves))
    return dict(zip([s.name for s in sleeves], w / w.sum()))


def _trade_stats_with_pf(tr: pd.DataFrame) -> dict:
    ts = metrics.trade_stats(tr)
    return ts


def _summarise(norm: pd.DataFrame, w: dict, trades_unit: pd.DataFrame, cap: float, rf: float) -> tuple[pd.Series, dict]:
    """Combined equity + stats for an arbitrary weight dict (names missing from w are excluded)."""
    cols = [c for c in norm.columns if w.get(c, 0) > 0]
    wt = pd.Series({c: w[c] for c in cols})
    wt = wt / wt.sum()
    eq = cap * (norm[cols] * wt).sum(axis=1)
    st = metrics.curve_stats(eq, rf)
    tr = trades_unit[trades_unit.strategy.isin(cols)].copy()
    if len(tr):
        tr["pnl"] = tr["pnl_unit"] * tr["strategy"].map(wt) * cap
    st["trades"] = metrics.trade_stats(tr) if len(tr) else dict(n=0)
    st["net_pnl"] = float(eq.iloc[-1] - cap)
    return eq, st


def combine(sleeves: list[Sleeve], capital: float = 100_000.0, rf: float = 0.0) -> Combined:
    if not sleeves:
        raise ValueError("Select at least one strategy")
    names = [s.name for s in sleeves]
    if len(set(names)) != len(names):
        raise ValueError("Strategy names must be unique")
    w = _norm_weights(sleeves)

    start = max(s.res.equity.index[0] for s in sleeves)
    end = min(s.res.equity.index[-1] for s in sleeves)
    if start >= end:
        raise ValueError("The selected strategies have no overlapping dates")

    raw = pd.concat({s.name: s.res.equity for s in sleeves}, axis=1).sort_index().ffill()
    raw = raw.loc[start:end].dropna()
    e0 = raw.iloc[0]
    norm = raw / e0

    bh = pd.concat({s.name: s.res.benchmark for s in sleeves}, axis=1).sort_index().ffill().loc[start:end].dropna()
    bh = bh / bh.iloc[0]
    bench = capital * (bh * pd.Series(w)).sum(axis=1)

    # trades: pnl per rupee of sleeve capital, so any re-weighting stays exact
    parts = []
    for s in sleeves:
        t = s.res.trades
        t = t[pd.to_datetime(t.exit_date) >= start].copy()
        base = float(s.res.equity.loc[:start].iloc[-1])
        t["pnl_unit"] = t["pnl"] / base
        t["strategy"], t["symbol"] = s.name, s.symbol
        parts.append(t)
    trades_unit = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(trades_unit) == 0:
        trades_unit = pd.DataFrame(columns=["entry_date", "exit_date", "direction", "entry_price", "exit_price", "bars", "ret_pct",
                                             "pts", "pnl", "exit_reason", "open_at_end", "pnl_unit", "strategy", "symbol"])

    equity, stats = _summarise(norm, w, trades_unit, capital, rf)
    trades = trades_unit.copy()
    trades["pnl"] = trades["pnl_unit"] * trades["strategy"].map(w) * capital
    trades = trades.sort_values("exit_date", kind="stable").reset_index(drop=True)
    sleeves_eq = norm * capital          # each strategy standalone on the full capital (like-for-like with the combined line)

    # per-strategy table
    rows = []
    for s in sleeves:
        e1 = norm[s.name] * capital                      # each strategy on the full capital, for a like-for-like view
        cs = metrics.curve_stats(e1, rf)
        t1 = trades_unit[trades_unit.strategy == s.name].copy()
        t1["pnl"] = t1["pnl_unit"] * capital
        ts = metrics.trade_stats(t1) if len(t1) else dict(n=0)
        rows.append(_row(s.name, f"{s.symbol} · {w[s.name]:.0%}", cs, ts))
    rows.append(_row("COMBINED", "portfolio", stats, stats["trades"]))
    table = pd.DataFrame(rows).set_index("Strategy")

    rets = norm.pct_change().dropna()
    monthly = norm.resample("ME").last().pct_change().dropna(how="all")
    active = (rets != 0)
    ov = pd.DataFrame(index=norm.columns, columns=norm.columns, dtype=float)
    for a in norm.columns:
        for b in norm.columns:
            u = (active[a] | active[b]).sum()
            ov.loc[a, b] = (active[a] & active[b]).sum() / u if u else np.nan
    contribution = pd.Series({n: float(trades.loc[trades.strategy == n, "pnl"].sum()) for n in names})

    loo = []
    if len(sleeves) > 1:
        for n in names:
            w2 = {k: v for k, v in w.items() if k != n}
            _, st2 = _summarise(norm, w2, trades_unit, capital, rf)
            loo.append(dict(Removed=n, CAGR=st2["cagr"], MaxDD=st2["max_dd"], Sharpe=st2["sharpe"], Calmar=st2["calmar"],
                            ProfitFactor=st2["trades"].get("profit_factor", np.nan), Trades=st2["trades"].get("n", 0)))
    out = Combined(capital, start, end, w, norm, equity, sleeves_eq, bench, trades, stats, table,
                   rets.corr(), monthly.corr() if len(monthly) > 2 else rets.corr() * np.nan, ov, contribution,
                   pd.DataFrame(loo).set_index("Removed") if loo else pd.DataFrame())
    out.insights = insights(out)
    return out


def _row(name, note, cs, ts):
    return {"Strategy": name, "Setup": note, "Total return": cs["total_return"], "CAGR": cs["cagr"], "Max DD": cs["max_dd"],
            "Sharpe": cs["sharpe"], "Calmar": cs["calmar"], "Trades": ts.get("n", 0),
            "Win rate": ts.get("win_rate", np.nan), "Profit factor": ts.get("profit_factor", np.nan),
            "Avg trade %": ts.get("avg_ret", np.nan)}


def combos(sleeves: list[Sleeve], capital: float = 100_000.0, rf: float = 0.0, max_n: int = 10) -> pd.DataFrame:
    """Every non-empty subset of the strategies, equal-weighted - to see which mixes diversify best.
    (A search over many mixes will always find a flattering one in-sample: treat it as a guide, not a result.)"""
    if len(sleeves) > max_n:
        raise ValueError(f"Combination search is limited to {max_n} strategies")
    base = combine(sleeves, capital, rf)
    names = [s.name for s in sleeves]
    rows = []
    for k in range(1, len(names) + 1):
        for sub in combinations(names, k):
            w = {n: 1 / k for n in sub}
            _, st = _summarise(base.norm, w, base.trades.assign(pnl_unit=base.trades.pnl_unit), capital, rf)
            rows.append({"Mix": " + ".join(sub), "Strategies": k, "CAGR": st["cagr"], "Max DD": st["max_dd"], "Sharpe": st["sharpe"],
                         "Calmar": st["calmar"], "Profit factor": st["trades"].get("profit_factor", np.nan), "Trades": st["trades"].get("n", 0)})
    return pd.DataFrame(rows)


def insights(c: Combined) -> list[str]:
    out, p = [], lambda x: f"{x * 100:.1f}%"
    tb = c.table.drop(index="COMBINED")
    cb = c.table.loc["COMBINED"]
    n = len(tb)
    out.append(f"Portfolio of {n} strateg{'y' if n == 1 else 'ies'} over {c.start:%d %b %Y} → {c.end:%d %b %Y}: CAGR {p(cb['CAGR'])}, "
               f"max drawdown {p(cb['Max DD'])}, Sharpe {cb['Sharpe']:.2f}, profit factor "
               f"{'∞' if np.isinf(cb['Profit factor']) else f'{cb['Profit factor']:.2f}'} over {int(cb['Trades'])} trades.")
    if n == 1:
        return out
    wavg = sum(c.weights[k] * tb.loc[k, "Max DD"] for k in tb.index)
    worst = tb["Max DD"].min()
    out.append(f"Drawdown: combined {p(cb['Max DD'])} vs weighted average of the individual drawdowns {p(wavg)} and worst single {p(worst)} "
               + ("— diversification is helping." if cb["Max DD"] > wavg + 0.005 else "— little diversification benefit (they draw down together)."))
    best_sh = tb["Sharpe"].idxmax()
    out.append(f"Sharpe: combined {cb['Sharpe']:.2f} vs best single {tb.loc[best_sh, 'Sharpe']:.2f} ({best_sh}). "
               + ("Combining improved risk-adjusted return." if cb["Sharpe"] > tb.loc[best_sh, "Sharpe"] else
                  f"{best_sh} alone is better risk-adjusted; the others dilute it."))
    cd = c.corr_daily.where(~np.eye(len(c.corr_daily), dtype=bool))
    pairs = cd.stack()
    pairs = pairs[[a < b for a, b in pairs.index]]
    if len(pairs):
        hi, lo = pairs.idxmax(), pairs.idxmin()
        out.append(f"Correlation of daily returns: average {pairs.mean():.2f}; most similar {hi[0]}–{hi[1]} ({pairs.max():.2f}), "
                   f"least similar {lo[0]}–{lo[1]} ({pairs.min():.2f}). "
                   + ("High overlap — these strategies mostly make the same bets." if pairs.mean() > 0.5 else
                      "Low correlation — good diversifiers." if pairs.mean() < 0.2 else "Moderate correlation."))
        ovp = c.overlap.where(~np.eye(len(c.overlap), dtype=bool)).stack()
        ovp = ovp[[a < b for a, b in ovp.index]]
        if len(ovp):
            o = ovp.idxmax()
            out.append(f"Time overlap: {o[0]} and {o[1]} are in a trade on the same day {ovp.max():.0%} of the days either is active.")
    tot = c.contribution.sum()
    top = c.contribution.idxmax()
    out.append(f"P&L contribution: " + ", ".join(f"{k} {v:+,.0f}" for k, v in c.contribution.sort_values(ascending=False).items())
               + (f". {top} provides {c.contribution[top] / tot:.0%} of net profit." if tot > 0 and c.contribution[top] <= tot * 1.0001
                  else f". Largest contributor: {top}."))
    if len(c.leave_one_out):
        gain = c.leave_one_out["Calmar"] - cb["Calmar"]
        if gain.max() > 0.05:
            k = gain.idxmax()
            out.append(f"Dropping {k} would improve the combined Calmar from {cb['Calmar']:.2f} to {c.leave_one_out.loc[k, 'Calmar']:.2f} "
                       f"(CAGR {p(c.leave_one_out.loc[k, 'CAGR'])}, max DD {p(c.leave_one_out.loc[k, 'MaxDD'])}) — it is dragging the portfolio.")
        else:
            out.append("Every strategy pulls its weight: removing any one of them lowers the combined Calmar ratio.")
    return out
