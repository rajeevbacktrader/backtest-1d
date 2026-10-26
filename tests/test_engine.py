import numpy as np
import pandas as pd
import pytest

from idxbt.dsl import QueryError, parse_query
from idxbt.engine import run


def mk(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def flat(n, p=100):
    return [(p, p, p, p)] * n


def test_parse_user_example():
    st = parse_query("entry -close crossed above sma20 go long , exit : holding period =4 candles or tp = 4% , "
                     "sl = 0% or close crossed below sma 8")
    assert st.direction == 1 and st.hold_bars == 4 and st.tp_pct == 4 and st.sl_pct is None
    assert len(st.exit_conds) == 1 and "SMA(8)" in st.exit_conds[0][0].describe()


def test_parse_short_and_rules():
    st = parse_query("entry: close < 20 day low and rsi14 < 40 sell; exit: sl 2% or trailing sl 3% or after 5 days")
    assert st.direction == -1 and st.sl_pct == 2 and st.trail_pct == 3 and st.hold_bars == 5
    assert len(st.entry) == 1 and len(st.entry[0]) == 2


@pytest.mark.parametrize("q", ["close above sma20", "entry: foo bar; exit: tp 2%", "entry: close > sma20; exit:",
                               "entry: volume > 100; exit: tp 1%", "entry: close crossed sma20; exit: tp 1%"])
def test_bad_queries(q):
    with pytest.raises(QueryError):
        parse_query(q)


def test_cross_and_time_exit_close_fill():
    # close 100 -> 99 -> 101 (cross above 100 at idx2) -> 102, 103, 104, 105
    df = mk([(100, 100, 100, 100), (99, 99, 99, 99), (101, 101, 101, 101)] + [(c, c, c, c) for c in (102, 103, 104, 105, 106)])
    st = parse_query("entry: close crossed above 100; exit: hold 3 candles")
    r = run(st, df, fill="close", fee_pct=0)
    t = r.trades.iloc[0]
    assert t.entry_price == 101 and t.exit_price == 104 and t.exit_reason == "TIME" and t.bars == 3
    assert t.ret_pct == pytest.approx((104 / 101 - 1) * 100)


def test_next_open_fill_and_hold():
    df = mk([(100, 100, 100, 100), (99, 99, 99, 99), (101, 101, 101, 101), (102, 103, 102, 103),
             (104, 104, 104, 104), (105, 105, 105, 105), (106, 106, 106, 106), (107, 107, 107, 107)])
    r = run(parse_query("entry: close crossed above 100; exit: hold 3 candles"), df, fill="next_open", fee_pct=0)
    t = r.trades.iloc[0]
    assert t.entry_price == 102                       # open of day after signal
    assert t.exit_date == df.index[5] and t.exit_price == 105   # close of signal-day + 3


def test_tp_sl_intrabar_and_gap():
    base = flat(2) + [(101, 101, 101, 101)]           # cross above 100 at idx2 (prev 100 <=100)
    df = mk(base + [(101, 106, 100, 105), (105, 105, 105, 105)])
    r = run(parse_query("entry: close crossed above 100; exit: tp 4% or sl 3%"), df, fill="close", fee_pct=0)
    assert r.trades.iloc[0].exit_reason == "TP" and r.trades.iloc[0].exit_price == pytest.approx(101 * 1.04)
    # gap down through stop -> filled at open, not at stop level
    df = mk(base + [(95, 96, 94, 95), (95, 95, 95, 95)])
    r = run(parse_query("entry: close crossed above 100; exit: sl 3% or tp 50%"), df, fill="close", fee_pct=0)
    assert r.trades.iloc[0].exit_price == 95 and r.trades.iloc[0].exit_reason == "SL"
    # both inside the bar -> SL first
    df = mk(base + [(101, 110, 90, 100), (100, 100, 100, 100)])
    r = run(parse_query("entry: close crossed above 100; exit: sl 3% or tp 4%"), df, fill="close", fee_pct=0)
    assert r.trades.iloc[0].exit_reason == "SL"


def test_short_and_fees_and_equity_consistency():
    df = mk(flat(2) + [(99, 99, 99, 99)] + [(98, 98, 98, 98), (97, 97, 97, 97), (96, 96, 96, 96), (95, 95, 95, 95)])
    r = run(parse_query("entry: close crossed below 100 short; exit: hold 2 candles"), df, fill="close", fee_pct=0.1)
    t = r.trades.iloc[0]
    assert t.direction == "SHORT" and t.exit_price == 97
    assert t.ret_pct == pytest.approx(-(97 / 99 - 1) * 100 - 0.2)
    assert r.equity.iloc[-1] == pytest.approx(100_000 * (1 + t.ret_pct / 100))


def test_no_lookahead_signal_on_last_bar_ignored():
    df = mk(flat(2) + [(99, 99, 99, 99), (101, 101, 101, 101)])
    r = run(parse_query("entry: close crossed above 100; exit: hold 2 candles"), df, fill="next_open")
    assert r.trades.empty


def test_open_trade_closed_at_end():
    df = mk(flat(2) + [(99, 99, 99, 99), (101, 101, 101, 101), (102, 102, 102, 102), (103, 103, 103, 103)])
    r = run(parse_query("entry: close crossed above 100; exit: hold 50 candles"), df, fill="next_open", fee_pct=0)
    assert r.trades.iloc[0].exit_reason == "END" and bool(r.trades.iloc[0].open_at_end)


def test_stale_and_auto_update_offline(tmp_path, monkeypatch):
    from idxbt import data
    monkeypatch.setattr(data, "DB_PATH", tmp_path / "m.db")
    assert set(data.stale_symbols()) == set(data.SYMBOLS)               # nothing stored -> all stale
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=30)
    data.save("NIFTY", pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}, index=idx))
    assert "NIFTY" not in data.stale_symbols()                          # fresh -> not stale
    # a Monday check with last candle on the previous Friday is NOT stale; older is
    mon = pd.Timestamp("2026-03-02")
    assert data._prev_weekday(mon) == pd.Timestamp("2026-02-27")

    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(data, "fetch", boom)
    out = data.auto_update(force=True)                                  # must not raise
    assert all(isinstance(v, str) and v.startswith("failed") for v in out.values())
    assert data.auto_update() == {}                                     # throttled right after an attempt


def test_expected_last_and_scheduler_commands():
    from idxbt import data, scheduler
    fri_eve, mon_morn = pd.Timestamp("2026-02-27 17:30"), pd.Timestamp("2026-03-02 10:00")
    assert data.expected_last(fri_eve) == pd.Timestamp("2026-02-27")      # after close -> today's candle
    assert data.expected_last(pd.Timestamp("2026-02-27 10:00")) == pd.Timestamp("2026-02-26")
    assert data.expected_last(mon_morn) == pd.Timestamp("2026-02-27")     # Monday morning -> Friday
    assert data.expected_last(pd.Timestamp("2026-02-28 18:00")) == pd.Timestamp("2026-02-27")  # Saturday
    w = scheduler.windows_commands("17:00")
    assert w[0][:4] == ["schtasks", "/Create", "/F", "/TN"] and "MON,TUE,WED,THU,FRI" in w[0] and "ONLOGON" in w[1]
    c = scheduler.cron_lines("17:00")
    assert c[0].startswith("0 17 * * 1-5") and c[1].startswith("@reboot")


def test_points_column():
    df = mk(flat(2) + [(99, 99, 99, 99)] + [(98, 98, 98, 98), (97, 97, 97, 97), (96, 96, 96, 96), (95, 95, 95, 95)])
    r = run(parse_query("entry: close crossed below 100 short; exit: hold 2 candles"), df, fill="close", fee_pct=0.1)
    assert r.trades.iloc[0].pts == pytest.approx(99 - 97)          # short: entry 99 -> exit 97 = +2 pts
    df2 = mk(flat(2) + [(99, 99, 99, 99), (101, 101, 101, 101), (103, 103, 103, 103), (105, 105, 105, 105), (106, 106, 106, 106)])
    r2 = run(parse_query("entry: close crossed above 100; exit: hold 2 candles"), df2, fill="close", fee_pct=0)
    assert r2.trades.iloc[0].pts == pytest.approx(105 - 101)
