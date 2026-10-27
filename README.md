# Index backtester (NIFTY · BANKNIFTY · FINNIFTY · SENSEX)

Daily-candle backtests driven by a plain-English rule. **No LLM / API key** — a small deterministic parser turns your text into rules.

## Setup
```bash
pip install -r requirements.txt
python -m idxbt fetch                 # downloads max history for all 4 indices (Yahoo Finance) into data/market.db
python -m idxbt status                # what is stored
python -m idxbt fetch                 # run again any time: incremental update
```
If Yahoo lacks history for an index (FINNIFTY is short), download a CSV from niftyindices.com / bseindia.com and
`python -m idxbt import-csv FINNIFTY file.csv` (columns Date,Open,High,Low,Close).

## Run a query
```bash
python -m idxbt run "entry: close crossed above sma20 go long; exit: holding period = 4 candles or tp = 4% or sl = 2% or close crossed below sma 8" --symbol NIFTY
python -m idxbt run "..." --symbol ALL --start 2015-01-01     # all four indices + comparison
python -m idxbt shell --symbol BANKNIFTY                       # type queries interactively
streamlit run app.py                                           # browser UI
```
Output: console summary + insights, and in `results/`: `report.html` (open in a browser; charts, stats, insights, all trades),
PNG charts, `trades.csv`, `equity.csv`. Options: `--fill next_open|close`, `--fee 0.03` (% per side), `--slippage`, `--capital`, `--rf 0.065`, `--start/--end`.

## Query language
`entry: <conditions> [go long|go short]` and `exit: <rule> or <rule> ...` (exit fires on the first one hit).

| Piece | Examples |
|---|---|
| Prices | `open high low close hl2 hlc3 ohlc4`, `prev close`, `close[2]`, `high 3 candles ago` |
| Indicators | `sma20`, `sma 20`, `20 ema`, `ema(50)`, `wma 10`, `rsi14`, `atr 14`, `roc 10`, `stdev 20`, `macd`, `macd signal`, `macd hist`, `bb upper`, `bb lower 20,2` |
| Breakouts | `20 day high`, `10 day low`, `highest close 55` (= extreme of the *previous* N candles) |
| Arithmetic | `sma20 * 1.02`, `close - 50`, `sma50 + 2%` |
| Conditions | `crossed above/below`, `>`, `<`, `>=`, `<=`, `above`, `below`, `at least` … joined by `and` / `or` (and binds tighter) |
| Exit rules | `holding period = 4 candles` / `after 5 days`, `tp = 4%`, `sl = 2%`, `trailing sl 3%`, or any condition |

A value of `0` (e.g. `sl = 0%`) means that rule is disabled.
Examples: `entry: rsi14 < 30 and close > sma 200 buy; exit: rsi14 > 60 or sl 3%` ·
`entry: close crossed above 20 day high; exit: close crossed below 10 day low or trailing sl 4%` ·
`entry: close crossed below sma50 go short; exit: tp 3% or sl 2% or after 6 candles`

## Fill model (important)
- `next_open` (default): signal on day t's close → filled at day t+1's open. No look-ahead.
- `close`: signal and fill both at day t's close (matches "enter when close crossed…" literally; slightly optimistic).
- TP/SL/trailing are intrabar stop/limit orders (fill at the level, or at the open on a gap). If TP and SL are both inside one candle, SL is assumed first.
- `holding period = N` exits at the close of the Nth candle after the signal candle.
- One position at a time, 100% of equity per trade, compounded, no re-entry on the candle you exit.
- Indices aren't tradable themselves: results approximate index futures/ETFs. Fees default to 0.03%/side.

## Portfolio tab (combine several strategies)
Open the **🧩 Portfolio** tab: add rows S1, S2, S3… (➕), pick an index and write the strategy text in each row, press **Run all strategies**,
then **tick the ones to analyse together** — the combined results update instantly as you tick/untick.
You get: combined equity + drawdown (individual strategies dotted), combined CAGR / max DD / Sharpe / profit factor / win rate,
daily & monthly return correlation, time-overlap, P&L contribution, year-by-year, a monthly calendar, leave-one-out and an
"every possible mix" search. Capital is split equally (or by custom %); each strategy compounds independently on its share.
Strategies on indices with different history lengths are compared on their common date window.

## Graph colours
Sidebar → **🎨 Graph colours**: Pastel · dark (default), Vivid · dark, Classic green/red · dark, Pastel · light, Classic · light.
It recolours only the graphs (and the white report cards), not the rest of the portal. Drawdown is its own graph next to the equity curve.

## Tests
`python -m pytest -q tests`

`python -m idxbt synthetic` creates random-walk data under symbol `SYNTH` for offline trials — never use it for conclusions.

## Getting updates automatically (git)
```
git clone https://github.com/rajeevbacktrader/backtest-1d.git
cd backtest-1d
run.bat            (Windows)   or   ./run.sh
```
`run.bat` / `run.sh` run `git pull` first, so every launch uses the latest code. Your downloaded data (`data/market.db`) is git-ignored and is never overwritten.
Run `setup_auto_update.bat` once to refresh market data in the background every weekday.
