"""Performance statistics and rule-based, plain-English insights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Result

TD = 252


def _dd_stats(eq: pd.Series):
    dd = eq / eq.cummax() - 1
    longest, cur_start, peak_date = pd.Timedelta(0), None, eq.index[0]
    run_peak = eq.iloc[0]
    for d, v in eq.items():
        if v >= run_peak:
            if cur_start is not None:
                longest = max(longest, d - cur_start)
                cur_start = None
            run_peak = v
        elif cur_start is None:
            cur_start = d
    if cur_start is not None:
        longest = max(longest, eq.index[-1] - cur_start)
    return dd, float(dd.min()), longest.days


def curve_stats(eq: pd.Series, rf: float = 0.0) -> dict:
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(TD)
    ex = r - rf / TD
    sharpe = ex.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else np.nan
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TD)
    sortino = ex.mean() * TD / dn if dn > 0 else np.nan
    _, mdd, mdd_days = _dd_stats(eq)
    return dict(total_return=total, cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino, max_dd=mdd,
                max_dd_days=mdd_days, calmar=cagr / abs(mdd) if mdd < 0 else np.nan)


def _streak(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(n=0)
    w, lo = t[t.ret_pct > 0], t[t.ret_pct <= 0]
    gp, gl = w.pnl.sum(), -lo.pnl.sum()
    return dict(
        n=len(t), win_rate=len(w) / len(t), avg_ret=t.ret_pct.mean(), median_ret=t.ret_pct.median(),
        avg_win=w.ret_pct.mean() if len(w) else 0.0, avg_loss=lo.ret_pct.mean() if len(lo) else 0.0,
        payoff=(w.ret_pct.mean() / -lo.ret_pct.mean()) if len(w) and len(lo) and lo.ret_pct.mean() < 0 else np.nan,
        profit_factor=gp / gl if gl > 0 else np.inf if gp > 0 else np.nan,
        best=t.ret_pct.max(), worst=t.ret_pct.min(), avg_bars=t.bars.mean(),
        total_pts=t.pts.sum(), avg_pts=t.pts.mean(), best_pts=t.pts.max(), worst_pts=t.pts.min(),
        avg_win_pts=w.pts.mean() if len(w) else 0.0, avg_loss_pts=lo.pts.mean() if len(lo) else 0.0,
        max_win_streak=_streak(t.ret_pct > 0), max_loss_streak=_streak(t.ret_pct <= 0))


def periodic_returns(eq: pd.Series, freq: str) -> pd.Series:
    return eq.resample(freq).last().pct_change().fillna(eq.resample(freq).last().iloc[0] / eq.iloc[0] - 1)


def compute(res: Result, rf: float = 0.0) -> dict:
    s = curve_stats(res.equity, rf)
    b = curve_stats(res.benchmark, rf)
    ts = trade_stats(res.trades)
    t = res.trades
    pos_days = ((res.equity.diff().fillna(0) != 0) | (res.equity.shift(-1).diff(-1).fillna(0) != 0)).mean()
    exposure = float(t.bars.sum() / len(res.df)) if len(t) else 0.0
    by_reason = (t.groupby("exit_reason").agg(trades=("ret_pct", "size"), win_rate=("ret_pct", lambda x: (x > 0).mean()),
                                               avg_ret=("ret_pct", "mean"), total_pnl=("pnl", "sum"))
                 .sort_values("total_pnl", ascending=False) if len(t) else pd.DataFrame())
    by_dir = (t.groupby("direction").agg(trades=("ret_pct", "size"), win_rate=("ret_pct", lambda x: (x > 0).mean()),
                                          avg_ret=("ret_pct", "mean"), total_pnl=("pnl", "sum")) if len(t) else pd.DataFrame())
    yr_s = periodic_returns(res.equity, "YE"); yr_b = periodic_returns(res.benchmark, "YE")
    yearly = pd.DataFrame({"Strategy": yr_s, "Buy & hold": yr_b}); yearly.index = yearly.index.year
    monthly = periodic_returns(res.equity, "ME")
    return dict(strategy=s, benchmark=b, trades=ts, exposure=exposure, by_reason=by_reason, by_direction=by_dir,
                yearly=yearly, monthly=monthly, start=res.df.index[0], end=res.df.index[-1], candles=len(res.df))


def insights(res: Result, m: dict) -> list[str]:
    out: list[str] = []
    s, b, ts, t = m["strategy"], m["benchmark"], m["trades"], res.trades
    pct = lambda x: f"{x * 100:.1f}%"
    if ts["n"] == 0:
        return ["The entry rule never triggered in this period. Loosen the entry condition or widen the date range."]

    # headline
    verdict = "beat" if s["cagr"] > b["cagr"] else "lagged"
    out.append(f"CAGR {pct(s['cagr'])} vs buy & hold {pct(b['cagr'])} — the strategy {verdict} the index, "
               f"with max drawdown {pct(s['max_dd'])} vs {pct(b['max_dd'])} and only {pct(m['exposure'])} of time invested.")
    if s["cagr"] < b["cagr"] and abs(s["max_dd"]) < abs(b["max_dd"]):
        out.append(f"Lower return but a shallower drawdown: return per unit of drawdown (Calmar) is "
                   f"{s['calmar']:.2f} vs {b['calmar']:.2f} for buy & hold.")
    elif s["cagr"] < b["cagr"]:
        out.append("It underperformed passive holding on both return and risk — hard to justify as-is.")

    # sample size
    if ts["n"] < 30:
        out.append(f"Only {ts['n']} trades — too few to trust. Expect large luck components; use a longer history or a more frequent rule.")
    elif ts["n"] < 100:
        out.append(f"{ts['n']} trades is a modest sample; treat win rate / profit factor as indicative, not precise.")

    # edge
    pf = ts["profit_factor"]
    out.append(f"Win rate {pct(ts['win_rate'])}, average win {ts['avg_win']:.2f}% vs average loss {ts['avg_loss']:.2f}% "
               f"(payoff {ts['payoff']:.2f}); expectancy {ts['avg_ret']:+.2f}% per trade, profit factor "
               f"{'∞' if np.isinf(pf) else f'{pf:.2f}'}.")
    if ts["win_rate"] < 0.4 and ts["payoff"] > 1.5:
        out.append("Classic trend-following profile: many small losses, a few large wins. Needs patience through losing streaks.")
    elif ts["win_rate"] > 0.6 and ts["payoff"] < 1:
        out.append("High win rate but losses are bigger than wins — one regime change can erase many small gains. Check tail risk (worst trade below).")
    out.append(f"Raw points captured: {ts['total_pts']:+,.0f} index points in total (before fees), {ts['avg_pts']:+,.1f} per trade — "
               f"average win {ts['avg_win_pts']:+,.1f} pts vs average loss {ts['avg_loss_pts']:+,.1f} pts; "
               f"best trade {ts['best_pts']:+,.0f} pts, worst {ts['worst_pts']:+,.0f} pts.")
    out.append(f"Worst trade {ts['worst']:.2f}%, best {ts['best']:.2f}%. Longest losing streak: {ts['max_loss_streak']} trades "
               f"(winning streak {ts['max_win_streak']}).")

    # exits
    br = m["by_reason"]
    if len(br):
        parts = [f"{k}: {int(r.trades)} trades, avg {r.avg_ret:+.2f}%, win {pct(r.win_rate)}" for k, r in br.iterrows()]
        out.append("Exit breakdown — " + "; ".join(parts) + ".")
        worst = br.sort_values("total_pnl").iloc[0]
        if worst.total_pnl < 0 and br.index[0] != br.sort_values("total_pnl").index[0]:
            out.append(f"The '{br.sort_values('total_pnl').index[0]}' exit loses money overall — consider loosening or removing it.")

    # long / short
    bd = m["by_direction"]
    if len(bd) > 1:
        out.append("By side — " + "; ".join(f"{k}: {int(r.trades)} trades, avg {r.avg_ret:+.2f}%" for k, r in bd.iterrows()) + ".")

    # years
    y = m["yearly"]["Strategy"]
    pos = int((y > 0).sum())
    out.append(f"Profitable in {pos} of {len(y)} calendar years; best {y.idxmax()} ({pct(y.max())}), worst {y.idxmin()} ({pct(y.min())}).")

    # concentration
    if len(t) >= 10:
        top = t.nlargest(5, "pnl")
        rest = res.trades.drop(top.index)
        cap = res.params["capital"]
        without = (cap + rest.pnl.sum()) / cap - 1
        total = res.equity.iloc[-1] / res.equity.iloc[0] - 1
        if total > 0:
            share = top.pnl.sum() / res.trades.pnl.sum() if res.trades.pnl.sum() > 0 else np.nan
            if share > 0.6:
                out.append(f"Concentrated: the 5 best trades produce {pct(share)} of net profit. Without them the (non-compounded) "
                           f"result is {pct(without)}. The edge may rest on a few big moves.")

    # in-sample vs out-of-sample
    if len(t) >= 20:
        cut = res.df.index[int(len(res.df) * 0.6)]
        a, o2 = t[t.exit_date <= cut], t[t.exit_date > cut]
        if len(a) >= 8 and len(o2) >= 8:
            sa, so = trade_stats(a), trade_stats(o2)
            out.append(f"Stability check — first 60% of history: {sa['n']} trades, avg {sa['avg_ret']:+.2f}%, win {pct(sa['win_rate'])}; "
                       f"last 40%: {so['n']} trades, avg {so['avg_ret']:+.2f}%, win {pct(so['win_rate'])}."
                       + (" The edge faded recently." if sa["avg_ret"] > 0 >= so["avg_ret"] else ""))

    # hold time
    w, lo = t[t.ret_pct > 0], t[t.ret_pct <= 0]
    if len(w) and len(lo):
        out.append(f"Winners are held {w.bars.mean():.1f} candles on average, losers {lo.bars.mean():.1f}.")

    # drawdown
    out.append(f"Longest time under water: {s['max_dd_days']} calendar days (~{s['max_dd_days'] / 365.25:.1f} years).")

    # costs
    fee_drag = 2 * res.params["fee_pct"] * ts["n"] / 100
    if fee_drag > 0.1:
        out.append(f"Trading costs add up to ≈{pct(fee_drag)} of capital over the test. Edge per trade ({ts['avg_ret']:+.2f}%) "
                   f"vs round-trip cost {2 * res.params['fee_pct']:.2f}% — verify it survives real slippage.")
    if t.open_at_end.any():
        out.append("The last trade was still open at the end of data and was closed at the final close.")
    for n in res.strategy.notes:
        out.append("Note: " + n)
    return out
