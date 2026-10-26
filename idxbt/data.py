"""Market-data fetch + local SQLite store (daily OHLC)."""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(os.environ.get("IDXBT_DB", Path(__file__).resolve().parent.parent / "data" / "market.db"))

# Yahoo Finance tickers, tried in order until one returns data.
SYMBOLS = {
    "NIFTY": ["^NSEI"],
    "BANKNIFTY": ["^NSEBANK"],
    "FINNIFTY": ["NIFTY_FIN_SERVICE.NS", "^CNXFIN"],
    "SENSEX": ["^BSESN"],
}
ALIASES = {"NIFTY50": "NIFTY", "NIFTY 50": "NIFTY", "NSEI": "NIFTY", "BANKNIFTY": "BANKNIFTY",
           "NIFTYBANK": "BANKNIFTY", "NIFTY BANK": "BANKNIFTY", "BANK NIFTY": "BANKNIFTY", "NSEBANK": "BANKNIFTY",
           "FIN NIFTY": "FINNIFTY", "NIFTY FIN SERVICE": "FINNIFTY", "BSESN": "SENSEX", "BSE SENSEX": "SENSEX"}


def norm_symbol(s: str) -> str:
    s = s.strip().upper()
    s = ALIASES.get(s, s)
    return s


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS ohlc(
        symbol TEXT NOT NULL, date TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        PRIMARY KEY(symbol, date))""")
    c.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    return c


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace(0, np.nan).dropna()
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9) | (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9)
    # repair slightly inconsistent bars instead of dropping them
    df.loc[bad, "high"] = df.loc[bad, ["open", "high", "low", "close"]].max(axis=1)
    df.loc[bad, "low"] = df.loc[bad, ["open", "high", "low", "close"]].min(axis=1)
    return df


def save(symbol: str, df: pd.DataFrame) -> int:
    df = clean(df)
    rows = [(symbol, d.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close) for d, r in df.iterrows()]
    with _conn() as c:
        c.executemany("INSERT OR REPLACE INTO ohlc VALUES (?,?,?,?,?,?)", rows)
    return len(rows)


def load(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    symbol = norm_symbol(symbol)
    q, args = "SELECT date,open,high,low,close FROM ohlc WHERE symbol=?", [symbol]
    if start:
        q += " AND date>=?"; args.append(start)
    if end:
        q += " AND date<=?"; args.append(end)
    with _conn() as c:
        df = pd.read_sql_query(q + " ORDER BY date", c, params=args)
    if df.empty:
        raise LookupError(f"No stored data for {symbol}. Run:  python -m idxbt fetch")
    df.index = pd.to_datetime(df.pop("date"))
    return df


def summary() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT symbol, MIN(date) first, MAX(date) last, COUNT(*) candles FROM ohlc GROUP BY symbol", c)


def fetch(symbol: str, start: str = "1990-01-01", full: bool = False, retries: int = 3) -> int:
    """Download daily OHLC from Yahoo Finance and upsert into the store.
    Incremental by default (re-fetches the last 10 days of stored data)."""
    import yfinance as yf

    symbol = norm_symbol(symbol)
    if symbol not in SYMBOLS:
        raise ValueError(f"Unknown symbol {symbol}; choose from {list(SYMBOLS)}")
    if not full:
        try:
            last = load(symbol).index[-1]
            start = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        except LookupError:
            pass
    err = None
    for tk in SYMBOLS[symbol]:
        for attempt in range(retries):
            try:
                h = yf.Ticker(tk).history(start=start, interval="1d", auto_adjust=False, actions=False)
                if len(h):
                    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                    h = h.rename(columns=str.lower)
                    return save(symbol, h)
                break
            except Exception as e:  # network / rate limit
                err = e
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch {symbol} ({SYMBOLS[symbol]}): {err or 'no data returned'}")


def import_csv(symbol: str, path: str) -> int:
    """Import OHLC from a CSV (e.g. downloaded from NSE/BSE). Columns: Date,Open,High,Low,Close (any case)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    return save(norm_symbol(symbol), df.set_index("date"))


def synthetic(n_years: int = 15, seed: int = 7, symbol: str = "SYNTH") -> int:
    """Random-walk OHLC for offline demos/tests. Stored under its own symbol, never under a real index name."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * n_years)
    r = rng.normal(0.0004, 0.011, len(idx))
    c = 10000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, len(idx)))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, len(idx))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, len(idx))))
    return save(symbol, pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx))


# ----------------------------------------------------------------- auto update
def _meta_get(k):
    with _conn() as c:
        r = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else None


def _meta_set(k, v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))


def _prev_weekday(today: pd.Timestamp) -> pd.Timestamp:
    d = today - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def expected_last(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Date of the newest daily candle that should exist: today after 16:00 on a weekday, else the previous weekday."""
    now = now or pd.Timestamp.now()
    if now.weekday() < 5 and now.hour >= 16:
        return now.normalize()
    return _prev_weekday(now.normalize())


def stale_symbols(now: pd.Timestamp | None = None) -> list[str]:
    """Symbols that are missing or whose newest candle is older than expected_last()."""
    target, have = expected_last(now), summary().set_index("symbol")["last"] if len(summary()) else {}
    return [s for s in SYMBOLS if s not in have or pd.Timestamp(have[s]) < target]


def auto_update(min_hours: float = 4.0, force: bool = False) -> dict:
    """Update stale symbols. Never raises (safe offline). Attempts are throttled to once per `min_hours`.
    Returns {symbol: rows_added | 'error text'}; {} if nothing to do / throttled."""
    last = _meta_get("last_auto_attempt")
    if not force and last and (pd.Timestamp.now() - pd.Timestamp(last)) < pd.Timedelta(hours=min_hours):
        return {}
    todo = list(SYMBOLS) if force else stale_symbols()
    if not todo:
        return {}
    _meta_set("last_auto_attempt", pd.Timestamp.now().isoformat())
    out = {}
    for s in todo:
        try:
            out[s] = fetch(s, retries=1)
        except Exception as e:
            out[s] = f"failed: {str(e)[:120]}"
    if any(not isinstance(v, str) for v in out.values()):
        _meta_set("last_success", pd.Timestamp.now().isoformat(timespec="minutes"))
    return out


def last_success() -> str | None:
    return _meta_get("last_success")
