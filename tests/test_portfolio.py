import numpy as np
import pandas as pd
import pytest

from idxbt import portfolio as pf
from idxbt.dsl import parse_query
from idxbt.engine import run


def make_df(n=600, seed=1, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * .996, "close": c}, index=idx)


Q1 = "entry: close crossed above sma20 go long; exit: holding period = 4 candles or tp = 4% or close crossed below sma 8"
Q2 = "entry: close crossed below sma20 go short; exit: tp = 3% or sl = 2% or holding period = 6 candles"


@pytest.fixture
def sleeves():
    df = make_df()
    return [pf.Sleeve("S1", "X", run(parse_query(Q1), df)), pf.Sleeve("S2", "X", run(parse_query(Q2), df))]


def test_combined_is_weighted_sum_and_pnl_identity(sleeves):
    c = pf.combine(sleeves, 100_000)
    expect = 100_000 * (0.5 * sleeves[0].res.equity / sleeves[0].res.equity.iloc[0] + 0.5 * sleeves[1].res.equity / sleeves[1].res.equity.iloc[0])
    assert np.allclose(c.equity.values, expect.values)
    # every rupee of equity change is explained by a trade
    assert c.trades.pnl.sum() == pytest.approx(c.equity.iloc[-1] - 100_000, rel=1e-9)
    assert c.contribution.sum() == pytest.approx(c.trades.pnl.sum())


def test_profit_factor_matches_trades(sleeves):
    c = pf.combine(sleeves, 100_000)
    gp, gl = c.trades.pnl[c.trades.pnl > 0].sum(), -c.trades.pnl[c.trades.pnl <= 0].sum()
    assert c.stats["trades"]["profit_factor"] == pytest.approx(gp / gl)
    assert c.stats["trades"]["n"] == len(sleeves[0].res.trades) + len(sleeves[1].res.trades)


def test_single_sleeve_equals_standalone(sleeves):
    c = pf.combine(sleeves[:1], 100_000)
    assert c.equity.iloc[-1] == pytest.approx(sleeves[0].res.equity.iloc[-1])
    assert c.stats["max_dd"] == pytest.approx(pf.metrics.curve_stats(sleeves[0].res.equity)["max_dd"])


def test_custom_weights_and_drawdown_bounds(sleeves):
    sleeves[0].weight, sleeves[1].weight = 3, 1
    c = pf.combine(sleeves, 100_000)
    assert c.weights == pytest.approx({"S1": .75, "S2": .25})
    singles = [pf.metrics.curve_stats(s.res.equity)["max_dd"] for s in sleeves]
    assert c.stats["max_dd"] >= min(singles) - 1e-12        # a mix can't draw down deeper than its worst sleeve


def test_correlation_and_leave_one_out(sleeves):
    c = pf.combine(sleeves, 100_000)
    assert c.corr_daily.loc["S1", "S1"] == pytest.approx(1.0) and -1 <= c.corr_daily.loc["S1", "S2"] <= 1
    assert set(c.leave_one_out.index) == {"S1", "S2"}
    # removing S2 leaves exactly S1 on its own
    assert c.leave_one_out.loc["S2", "CAGR"] == pytest.approx(pf.combine(sleeves[:1], 100_000).stats["cagr"])
    assert len(c.insights) >= 5


def test_different_history_lengths_use_common_window():
    long_df, short_df = make_df(800, 1, "2019-01-01"), make_df(300, 2, "2021-06-01")
    a = pf.Sleeve("A", "X", run(parse_query(Q1), long_df)); b = pf.Sleeve("B", "Y", run(parse_query(Q2), short_df))
    c = pf.combine([a, b], 50_000)
    assert c.start == short_df.index[0] and c.end == min(long_df.index[-1], short_df.index[-1])
    assert c.equity.iloc[0] == pytest.approx(50_000)
    assert (pd.to_datetime(c.trades.exit_date) >= c.start).all()


def test_combos_and_errors(sleeves):
    d = pf.combos(sleeves, 100_000)
    assert len(d) == 3 and set(d.Mix) == {"S1", "S2", "S1 + S2"}
    with pytest.raises(ValueError):
        pf.combine([], 1000)
    with pytest.raises(ValueError):
        pf.combine([sleeves[0], pf.Sleeve("S1", "X", sleeves[1].res)], 1000)
