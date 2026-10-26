"""Command line:  python -m idxbt <command>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data, engine, metrics, report
from .dsl import QueryError, parse_query

ALL = list(data.SYMBOLS)


def run_query(query, symbol, start=None, end=None, fill="next_open", fee=0.03, slip=0.0, capital=100_000.0,
              out=None, rf=0.0):
    st = parse_query(query)
    df = data.load(symbol, start, end)
    res = engine.run(st, df, fill=fill, fee_pct=fee, slippage_pct=slip, capital=capital)
    m = metrics.compute(res, rf)
    ins = metrics.insights(res, m)
    path = None
    if out:
        path = report.write_report(res, m, ins, data.norm_symbol(symbol), Path(out))
    return res, m, ins, path


def _print(res, m, ins, path, sym):
    s, b, t = m["strategy"], m["benchmark"], m["trades"]
    print(f"\n=== {sym}  {m['start']:%Y-%m-%d} → {m['end']:%Y-%m-%d}  ({m['candles']} candles) ===")
    print(res.strategy.describe())
    print(f"\n{'':22}{'Strategy':>12}{'Buy&Hold':>12}")
    for k, lab in (("total_return", "Total return"), ("cagr", "CAGR"), ("max_dd", "Max drawdown")):
        print(f"{lab:22}{s[k] * 100:>11.1f}%{b[k] * 100:>11.1f}%")
    print(f"{'Sharpe':22}{s['sharpe']:>12.2f}{b['sharpe']:>12.2f}")
    if t["n"]:
        print(f"\nTrades {t['n']} | win {t['win_rate'] * 100:.1f}% | avg {t['avg_ret']:+.2f}% | PF {t['profit_factor']:.2f} "
              f"| avg bars {t['avg_bars']:.1f} | exposure {m['exposure'] * 100:.0f}%")
    print("\nInsights:")
    for i in ins:
        print(" •", i)
    if path:
        print(f"\nReport: {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="idxbt", description="Backtester for NIFTY / BANKNIFTY / FINNIFTY / SENSEX")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download daily OHLC from Yahoo Finance and store locally")
    f.add_argument("--symbols", nargs="*", default=ALL); f.add_argument("--start", default="1990-01-01")
    f.add_argument("--full", action="store_true", help="re-download everything instead of incremental update")

    ic = sub.add_parser("import-csv", help="import Date,Open,High,Low,Close CSV (e.g. from NSE)")
    ic.add_argument("symbol"); ic.add_argument("path")
    sub.add_parser("status", help="show what data is stored")
    sub.add_parser("update", help="fetch only what is missing for all indices (used by the scheduler; logs to data/update.log)")
    sc = sub.add_parser("schedule", help="register automatic background updates with the OS scheduler")
    sc.add_argument("--time", default="17:00", help="weekday update time, local 24h (default 17:00, after NSE close)")
    sc.add_argument("--remove", action="store_true")
    sy = sub.add_parser("synthetic", help="generate fake data under symbol SYNTH (offline demo only)")
    sy.add_argument("--years", type=int, default=15)

    for name in ("run", "shell"):
        r = sub.add_parser(name, help="run a text query" if name == "run" else "interactive query prompt")
        if name == "run":
            r.add_argument("query")
        r.add_argument("--symbol", default="NIFTY", help="NIFTY | BANKNIFTY | FINNIFTY | SENSEX | ALL | SYNTH")
        r.add_argument("--start"); r.add_argument("--end")
        r.add_argument("--fill", choices=["next_open", "close"], default="next_open")
        r.add_argument("--fee", type=float, default=0.03, help="per-side cost in %% (default 0.03)")
        r.add_argument("--slippage", type=float, default=0.0, help="per-side slippage in %%")
        r.add_argument("--capital", type=float, default=100_000)
        r.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate for Sharpe, e.g. 0.065")
        r.add_argument("--out", default="results", help="output folder (report.html, charts, trades.csv)")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "fetch":
            for s in a.symbols:
                try:
                    print(f"{data.norm_symbol(s):10} +{data.fetch(s, a.start, a.full)} rows")
                except Exception as e:
                    print(f"{s:10} FAILED: {e}", file=sys.stderr)
        elif a.cmd == "import-csv":
            print(f"imported {data.import_csv(a.symbol, a.path)} rows")
        elif a.cmd == "update":
            res = data.auto_update(min_hours=0)
            line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M} " + (", ".join(f"{k}={v}" for k, v in res.items()) or "already up to date")
            print(line)
            with open(data.DB_PATH.parent / "update.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return 1 if any(isinstance(v, str) for v in res.values()) else 0
        elif a.cmd == "schedule":
            from . import scheduler
            print(scheduler.remove() if a.remove else scheduler.install(a.time))
        elif a.cmd == "status":
            print(data.summary().to_string(index=False) or "no data")
        elif a.cmd == "synthetic":
            print(f"stored {data.synthetic(a.years)} synthetic rows as SYNTH")
        else:
            syms = ALL if a.symbol.upper() == "ALL" else [a.symbol]
            kw = dict(start=a.start, end=a.end, fill=a.fill, fee=a.fee, slip=a.slippage, capital=a.capital, rf=a.rf)

            def go(q):
                rows = []
                for s in syms:
                    out = Path(a.out) / (data.norm_symbol(s) if len(syms) > 1 else "")
                    try:
                        res, m, ins, p = run_query(q, s, out=out, **kw)
                    except LookupError as e:
                        print(e); continue
                    _print(res, m, ins, p, data.norm_symbol(s))
                    rows.append(dict(symbol=data.norm_symbol(s), trades=m["trades"].get("n", 0),
                                     cagr=m["strategy"]["cagr"], bh_cagr=m["benchmark"]["cagr"], max_dd=m["strategy"]["max_dd"]))
                if len(rows) > 1:
                    print("\nComparison:\n" + pd.DataFrame(rows).round(3).to_string(index=False))

            if a.cmd == "run":
                go(a.query)
            else:
                print("Type a query (entry: ... exit: ...), blank line to quit.")
                while True:
                    try:
                        q = input("> ").strip()
                    except EOFError:
                        break
                    if not q:
                        break
                    try:
                        go(q)
                    except QueryError as e:
                        print("Query error:", e)
    except QueryError as e:
        print("Query error:", e, file=sys.stderr); return 2
    except LookupError as e:
        print(e, file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
