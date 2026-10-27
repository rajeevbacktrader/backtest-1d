import numpy as np
import pandas as pd
import pytest

from idxbt import report_view as rv


def mk():
    idx = pd.bdate_range("2024-01-01", periods=300)
    eq = pd.Series(100_000.0, index=idx)
    pnl = [5000, -2000, -4000, 8000, 3000, -1000]
    ex = [idx[20], idx[60], idx[100], idx[160], idx[220], idx[290]]
    t = pd.DataFrame({"entry_date": [d - pd.Timedelta(days=3) for d in ex], "exit_date": ex, "pnl": pnl,
                      "ret_pct": [p / 1000 for p in pnl], "pts": [p / 10 for p in pnl]})
    cum = pd.Series(np.cumsum(pnl), index=ex).reindex(idx).ffill().fillna(0)
    return (100_000 + cum), t


def chip(r, name):
    return next(v for l, v, c in r["chips"] if l == name)


def test_chips_and_curves():
    eq, t = mk()
    r = rv.build(eq, t, "All", "trades", "Month")
    assert chip(r, "FINAL") == "+9,000"            # 5000-2000-4000+8000+3000-1000
    assert chip(r, "PEAK") == "+10,000"            # after the 5th trade
    assert chip(r, "MAX DD") == "-6,000"           # 5000 -> -1000
    assert chip(r, "WIN RATE") == "50.0%" and chip(r, "TRADES") == "6"
    assert chip(r, "PROFIT FACTOR") == "2.29"      # 16000 / 7000
    assert chip(r, "POINTS") == "+900"


def test_profit_factor_value():
    eq, t = mk()
    r = rv.build(eq, t, "All", "trades", "Month")
    assert float(chip(r, "PROFIT FACTOR")) == pytest.approx(16000 / 7000, abs=0.01)


def test_range_window_filters_trades_and_rebases_to_zero():
    eq, t = mk()
    r = rv.build(eq, t, "6M", "trades", "Month")           # last 183 days: only the trades at idx 220 and 290
    assert int(chip(r, "TRADES")) == 2 and chip(r, "FINAL") == "+2,000"    # 3000-1000
    assert r["figs"]["equity"].data[2].y[0] == 0            # curve starts at zero


def test_daily_basis_matches_closed_when_flat_between_trades():
    eq, t = mk()
    a = rv.build(eq, t, "All", "trades", "Month"); b = rv.build(eq, t, "All", "daily", "Month")
    assert chip(a, "FINAL") == chip(b, "FINAL")


def test_pnl_bars_group_sums():
    eq, t = mk()
    for g in ("Month", "Trade", "Year"):
        f = rv.build(eq, t, "All", "trades", g)["figs"]["pnl"]
        assert sum(f.data[0].y) == 9000
    assert "6M" in rv.available_ranges(eq)


def test_palette_switch_recolours_charts_and_report_cards():
    from idxbt import charts
    try:
        charts.set_palette("Classic · light")
        assert charts.STRAT == "#2563eb" and charts.TEMPLATE == "plotly_white" and rv.BLUE == "#2563eb"
        charts.set_palette("no such palette")                 # unknown name falls back to the default
        assert charts.PALETTE == charts.DEFAULT_PALETTE and charts.TEMPLATE == "plotly_dark"
        eq, t = mk()
        charts.set_palette("Vivid · dark")
        assert rv.build(eq, t)["chips"][0][2] == rv.GREEN == "#16c784"
    finally:
        charts.set_palette(charts.DEFAULT_PALETTE)
