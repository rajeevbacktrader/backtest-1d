"""Bar-by-bar daily backtester (one position at a time, 100% of equity per trade).

Fill model
  fill="next_open" (default, no look-ahead): a signal seen at the close of day t is filled at the open of t+1.
  fill="close": signal and fill both at the close of day t (what you literally get when you
                'enter when close crossed above...' and trade at the close).
  TP / SL / trailing stops are resting orders: filled intrabar at their level (at the open if it gaps through).
  If TP and SL are both inside one bar's range, SL is assumed to hit first (conservative).
  Holding period N = exit at the close of the Nth candle after the signal candle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dsl import Strategy


@dataclass
class Result:
    strategy: Strategy
    df: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.Series
    benchmark: pd.Series
    params: dict = field(default_factory=dict)


def run(strategy: Strategy, df: pd.DataFrame, fill: str = "next_open", fee_pct: float = 0.03,
        slippage_pct: float = 0.0, capital: float = 100_000.0, allow_overlap_signal: bool = False) -> Result:
    if fill not in ("next_open", "close"):
        raise ValueError("fill must be 'next_open' or 'close'")
    n = len(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    dates = df.index
    ent = strategy.entry_signal(df)
    ext = strategy.exit_signal(df)
    d = strategy.direction
    fee, slip = fee_pct / 100, slippage_pct / 100
    tp, sl, tr = (x / 100 if x else None for x in (strategy.tp_pct, strategy.sl_pct, strategy.trail_pct))
    hold = strategy.hold_bars

    equity = np.full(n, capital, float)
    eq = capital                      # realised equity
    trades = []
    in_pos = pend_entry = pend_exit = False
    pe = peak = 0.0
    ie = first = 0
    eq_before = capital

    def close_trade(i, px, reason, opened=False):
        nonlocal eq, in_pos, pend_exit
        px_net = px * (1 - d * slip)
        ret = d * (px_net / pe - 1) - 2 * fee
        eq = eq_before * (1 + ret)
        trades.append(dict(entry_date=dates[ie], exit_date=dates[i], direction="LONG" if d > 0 else "SHORT",
                           entry_price=pe, exit_price=px_net, bars=i - ie + (0 if fill == "close" else 1),
                           ret_pct=ret * 100, pts=d * (px_net - pe), pnl=eq - eq_before, exit_reason=reason, open_at_end=opened))
        in_pos = pend_exit = False
        return eq

    for i in range(n):
        exited_now = False
        if pend_entry and not in_pos:                     # fill signal from previous close at today's open
            pend_entry = False
            pe, ie, first, eq_before = o[i] * (1 + d * slip), i, i, eq
            peak, in_pos = pe, True
            equity_entry_bar = True
        else:
            equity_entry_bar = False

        if in_pos:
            if pend_exit:                                  # condition exit signalled yesterday -> today's open
                close_trade(i, o[i], "SIGNAL"); exited_now = True
            else:
                px = reason = None
                sl_p = (pe * (1 - d * sl)) if sl else None
                tp_p = (pe * (1 + d * tp)) if tp else None
                tr_p = (peak * (1 - tr) if d > 0 else peak * (1 + tr)) if tr else None
                stops = [p for p in (sl_p, tr_p) if p is not None]
                stop_p = (max(stops) if d > 0 else min(stops)) if stops else None
                if not equity_entry_bar:                   # gap handling at the open
                    if stop_p is not None and (o[i] <= stop_p if d > 0 else o[i] >= stop_p):
                        px, reason = o[i], "TRAIL" if (tr_p is not None and stop_p == tr_p and sl_p != stop_p) else "SL"
                    elif tp_p is not None and (o[i] >= tp_p if d > 0 else o[i] <= tp_p):
                        px, reason = o[i], "TP"
                if px is None:                             # intrabar
                    if stop_p is not None and (l[i] <= stop_p if d > 0 else h[i] >= stop_p):
                        px, reason = stop_p, "TRAIL" if (tr_p is not None and stop_p == tr_p and sl_p != stop_p) else "SL"
                    elif tp_p is not None and (h[i] >= tp_p if d > 0 else l[i] <= tp_p):
                        px, reason = tp_p, "TP"
                if px is not None:
                    close_trade(i, px, reason); exited_now = True
                else:
                    peak = max(peak, h[i]) if d > 0 else min(peak, l[i])
                    candles = i - first + 1
                    if hold and candles >= hold:
                        close_trade(i, c[i], "TIME"); exited_now = True
                    elif ext[i]:
                        if fill == "close":
                            close_trade(i, c[i], "SIGNAL"); exited_now = True
                        elif i + 1 < n:
                            pend_exit = True
                        else:
                            close_trade(i, c[i], "SIGNAL"); exited_now = True

        # mark-to-market equity for this bar
        if in_pos:
            equity[i] = eq_before * (1 + d * (c[i] / pe - 1) - fee)
        else:
            equity[i] = eq

        # new signal (flat, and did not just exit)
        if not in_pos and not exited_now and ent[i]:
            if fill == "close" and i < n - 1:
                pe, ie, eq_before = c[i] * (1 + d * slip), i, eq
                first, peak, in_pos = i + 1, pe, True
                equity[i] = eq_before * (1 + d * (c[i] / pe - 1) - fee)
            elif fill == "next_open" and i < n - 1:
                pend_entry = True

    if in_pos:                                             # force-close at last bar
        close_trade(n - 1, c[-1], "END", opened=True)
        equity[-1] = eq

    tdf = pd.DataFrame(trades, columns=["entry_date", "exit_date", "direction", "entry_price", "exit_price",
                                        "bars", "ret_pct", "pts", "pnl", "exit_reason", "open_at_end"])
    bench = pd.Series(capital * c / c[0], index=dates, name="Buy & hold")
    return Result(strategy, df, tdf, pd.Series(equity, index=dates, name="Strategy"), bench,
                  dict(fill=fill, fee_pct=fee_pct, slippage_pct=slippage_pct, capital=capital))
