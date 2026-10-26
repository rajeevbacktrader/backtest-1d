"""Rule-based parser: plain-English strategy text -> Strategy object.

No LLM involved. The grammar is small and deterministic; anything it cannot
understand raises QueryError with a hint pointing at the offending clause.

Example:
    entry: close crossed above sma20 go long
    exit: holding period = 4 candles or tp = 4% or sl = 2% or close crossed below sma 8
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


class QueryError(ValueError):
    pass


# ----------------------------------------------------------------- indicators
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(dn != 0, 100.0)


def atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ------------------------------------------------------------------- operands
@dataclass
class Operand:
    label: str
    fn: Callable[[pd.DataFrame], pd.Series]

    def series(self, df: pd.DataFrame) -> pd.Series:
        return self.fn(df)


PRICE_FIELDS = {"open", "high", "low", "close", "hl2", "hlc3", "ohlc4"}
_NUM = r"\d+(?:\.\d+)?"


def _price_series(df, name):
    if name in ("open", "high", "low", "close"):
        return df[name]
    if name == "hl2":
        return (df["high"] + df["low"]) / 2
    if name == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4


def _check_n(n, text):
    if n < 1 or n > 1000:
        raise QueryError(f"Indicator period out of range in '{text}'")
    return n


def _base_operand(t: str) -> Optional[Operand]:
    if re.fullmatch(r"-?" + _NUM, t):
        v = float(t)
        return Operand(f"{v:g}", lambda df, v=v: pd.Series(v, index=df.index))
    if t in PRICE_FIELDS:
        return Operand(t, lambda df, t=t: _price_series(df, t))
    if t in ("volume", "vol", "oi", "open interest"):
        raise QueryError("Volume / open interest is not available (index OHLC only).")

    # sma(close,20) style
    m = re.fullmatch(r"(sma|ema|wma)\s*\(\s*(open|high|low|close)\s*,\s*(\d+)\s*\)", t)
    if m:
        name, src, n = m.group(1), m.group(2), _check_n(int(m.group(3)), t)
        f = {"sma": sma, "ema": ema, "wma": wma}[name]
        return Operand(f"{name.upper()}({n})" if src == "close" else f"{name.upper()}({src},{n})",
                       lambda df: f(df[src], n))

    # macd [line|signal|hist]
    m = re.fullmatch(r"macd(?:\s*(line|signal|hist|histogram))?\s*(?:\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\))?", t)
    if m:
        part = m.group(1) or "line"
        fa, sl_, sg = (int(m.group(2) or 12), int(m.group(3) or 26), int(m.group(4) or 9))

        def macd_fn(df, part=part, fa=fa, sl_=sl_, sg=sg):
            line = ema(df["close"], fa) - ema(df["close"], sl_)
            sig = line.ewm(span=sg, adjust=False, min_periods=sg).mean()
            return {"line": line, "signal": sig}.get(part, line - sig)

        return Operand(f"MACD {part}", macd_fn)

    # bollinger bands
    m = re.fullmatch(
        rf"(?:bollinger(?:\s+bands?)?|bb)\s*(upper|lower|mid|middle)?(?:\s*bands?)?\s*"
        rf"(?:\(?\s*(\d+)\s*(?:,\s*({_NUM}))?\s*\)?)?\s*(upper|lower|mid|middle)?", t)
    if m and (m.group(1) or m.group(4)):
        part = m.group(1) or m.group(4)
        n = _check_n(int(m.group(2) or 20), t)
        k = float(m.group(3) or 2)

        def bb_fn(df, part=part, n=n, k=k):
            mid, sd = sma(df["close"], n), df["close"].rolling(n).std(ddof=0)
            return mid + k * sd if part == "upper" else mid - k * sd if part == "lower" else mid

        return Operand(f"BB {part}({n},{k:g})", bb_fn)

    # sma20, sma 20, sma(20), rsi14, atr 14 ...
    m = re.fullmatch(r"(sma|ema|wma|rsi|atr|roc|stdev)\s*[\(\[]?\s*(?:of\s*)?(\d+)?\s*[\)\]]?", t)
    if m and (m.group(2) or m.group(1) in ("rsi", "atr")):
        return _indicator(m.group(1), int(m.group(2) or 14), t)
    # 20 sma, 20 day sma, 20-day ema
    m = re.fullmatch(r"(\d+)\s*[- ]?\s*(?:day|candle|bar|period|d)?\s*[- ]?(sma|ema|wma|rsi|atr|roc)", t)
    if m:
        return _indicator(m.group(2), int(m.group(1)), t)

    # rolling extremes of the PREVIOUS n candles (so a breakout can be detected)
    m = re.fullmatch(r"(highest|lowest)\s*(high|low|close)?\s*[\(\[]?\s*(?:of\s*)?(\d+)\s*[\)\]]?", t)
    if m:
        hi = m.group(1) == "highest"
        src = m.group(2) or ("high" if hi else "low")
        return _extreme(hi, src, int(m.group(3)), t)
    m = re.fullmatch(r"(\d+)\s*[- ]?\s*(?:day|candle|bar|period)?\s*[- ]?(?:(closing)\s+)?(high|low)", t)
    if m:
        src = "close" if m.group(2) else m.group(3)
        return _extreme(m.group(3) == "high", src, int(m.group(1)), t)
    return None


def _indicator(name, n, t) -> Operand:
    _check_n(n, t)
    if name == "atr":
        return Operand(f"ATR({n})", lambda df: atr(df, n))
    f = {"sma": sma, "ema": ema, "wma": wma, "rsi": rsi,
         "roc": lambda s, n: s.pct_change(n) * 100, "stdev": lambda s, n: s.rolling(n).std(ddof=0)}[name]
    return Operand(f"{name.upper()}({n})", lambda df: f(df["close"], n))


def _extreme(hi, src, n, t) -> Operand:
    _check_n(n, t)
    word = "High" if hi else "Low"
    return Operand(f"{n}-candle {word}({src})",
                   lambda df: (df[src].rolling(n).max() if hi else df[src].rolling(n).min()).shift(1))


def parse_operand(text: str) -> Operand:
    t = re.sub(r"\s+", " ", text.strip().strip(".").lower())
    t = re.sub(r"^(?:the|its|today'?s|current)\s+", "", t)
    if not t:
        raise QueryError("Missing value on one side of a condition")

    # arithmetic offset:  sma20 * 1.02,  close - 50,  sma20 + 2%
    m = re.fullmatch(rf"(.+?)\s*([*/+\-])\s*({_NUM})\s*(%?)", t)
    if m and not re.fullmatch(r"-?" + _NUM, t):
        try:
            base = parse_operand(m.group(1))
        except QueryError:
            base = None
        if base is not None:
            op, val, pct = m.group(2), float(m.group(3)), bool(m.group(4))
            if pct and op in "*/":
                raise QueryError(f"'{t}': use + or - with %, e.g. 'sma20 + 2%'")
            def arith(df, base=base, op=op, val=val, pct=pct):
                s = base.series(df)
                if pct:
                    return s * (1 + val / 100) if op == "+" else s * (1 - val / 100)
                return {"*": s * val, "/": s / val, "+": s + val, "-": s - val}[op]
            return Operand(f"{base.label} {op} {val:g}{'%' if pct else ''}", arith)

    # shift: prev close / yesterday high / close[1] / close 2 candles ago
    shift = 0
    m = re.fullmatch(r"(?:prev(?:ious)?|yesterday'?s?)\s+(?:(?:day|candle|bar)'?s?\s+)?(.+)", t)
    if m:
        shift, t = 1, m.group(1)
    else:
        m = re.fullmatch(r"(.+?)\s*\[\s*(\d+)\s*\]", t) or \
            re.fullmatch(r"(.+?)\s+(\d+)\s*(?:candles?|bars?|days?)\s+ago", t)
        if m:
            t, shift = m.group(1), int(m.group(2))

    op = _base_operand(t)
    if op is None:
        raise QueryError(
            f"Don't understand '{text.strip()}'. Supported: open/high/low/close, numbers, sma/ema/wma/rsi/atr/roc N, "
            f"macd [signal|hist], bb upper/lower/mid, 'N day high/low', 'highest high N', 'prev close', 'close[2]'.")
    if shift:
        base = op
        return Operand(f"{base.label}[-{shift}]", lambda df: base.series(df).shift(shift))
    return op


# ----------------------------------------------------------------- conditions
_CROSS = re.compile(r"\bcross(?:ed|es|ing)?(?:\s+the)?\s*(above|over|up|upwards|below|under|down|downwards)?\b")
_COMP = re.compile(
    r"(>=|<=|==|!=|>|<|=|greater than or equal to|less than or equal to|greater than|less than|higher than|"
    r"lower than|more than|at least|at most|above|below|over|under|equal to|equals)")
_COMP_MAP = {">=": "ge", "<=": "le", "==": "eq", "=": "eq", "!=": "ne", ">": "gt", "<": "lt",
             "greater than or equal to": "ge", "at least": "ge", "less than or equal to": "le", "at most": "le",
             "greater than": "gt", "higher than": "gt", "more than": "gt", "above": "gt", "over": "gt",
             "less than": "lt", "lower than": "lt", "below": "lt", "under": "lt", "equal to": "eq", "equals": "eq"}
_SYM = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!=",
        "cross_above": "crossed above", "cross_below": "crossed below"}


@dataclass
class Cond:
    kind: str
    left: Operand
    right: Operand
    text: str

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        a, b = self.left.series(df), self.right.series(df)
        k = self.kind
        if k == "cross_above":
            r = (a > b) & (a.shift(1) <= b.shift(1))
        elif k == "cross_below":
            r = (a < b) & (a.shift(1) >= b.shift(1))
        else:
            r = {"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b, "eq": a == b, "ne": a != b}[k]
        return r.fillna(False).astype(bool)

    def describe(self) -> str:
        return f"{self.left.label} {_SYM[self.kind]} {self.right.label}"


def parse_condition(text: str) -> Cond:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"\b(?:is|are|has|have|then)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    m = _CROSS.search(t)
    if m:
        d = m.group(1)
        if not d:
            raise QueryError(f"'{text.strip()}': say 'crossed above' or 'crossed below'")
        kind = "cross_above" if d in ("above", "over", "up", "upwards") else "cross_below"
        l, r = t[:m.start()], t[m.end():]
    else:
        m = _COMP.search(t)
        if not m:
            raise QueryError(f"No comparison found in '{text.strip()}'. Use e.g. 'close crossed above sma20', "
                             f"'rsi14 < 30', 'close > sma 50'.")
        kind = _COMP_MAP.get(m.group(1)) or {"==": "eq"}[m.group(1)]
        l, r = t[:m.start()], t[m.end():]
        r = re.sub(r"^than\s+", "", r.strip())
    return Cond(kind, parse_operand(l), parse_operand(r), text.strip())


# ------------------------------------------------------------------- strategy
@dataclass
class Strategy:
    direction: int                                   # +1 long, -1 short
    entry: List[List[Cond]]                          # OR of AND-groups
    exit_conds: List[List[Cond]] = field(default_factory=list)
    hold_bars: Optional[int] = None
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    trail_pct: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    source: str = ""

    @staticmethod
    def _dnf(groups, df):
        out = np.zeros(len(df), dtype=bool)
        for g in groups:
            m = np.ones(len(df), dtype=bool)
            for c in g:
                m &= c.evaluate(df).to_numpy()
            out |= m
        return out

    def entry_signal(self, df):
        return self._dnf(self.entry, df)

    def exit_signal(self, df):
        return self._dnf(self.exit_conds, df) if self.exit_conds else np.zeros(len(df), dtype=bool)

    def describe(self) -> str:
        j = lambda gs: " OR ".join(" AND ".join(c.describe() for c in g) for g in gs)
        ex = []
        if self.hold_bars:
            ex.append(f"held {self.hold_bars} candles")
        if self.tp_pct:
            ex.append(f"take-profit {self.tp_pct:g}%")
        if self.sl_pct:
            ex.append(f"stop-loss {self.sl_pct:g}%")
        if self.trail_pct:
            ex.append(f"trailing stop {self.trail_pct:g}%")
        if self.exit_conds:
            ex.append(j(self.exit_conds))
        return (f"{'LONG' if self.direction > 0 else 'SHORT'} when {j(self.entry)}\n"
                f"EXIT on first of: " + "  |  ".join(ex))


_N = r"(\d+(?:\.\d+)?)"
_UNIT = r"\s*(%|percent|pct|points?|pts)?"
_RULES = [
    ("hold", re.compile(r"(?:holding\s*period|hold(?:ing)?(?:\s*period)?|max(?:imum)?\s*hold(?:ing)?(?:\s*period)?|"
                        r"time\s*(?:exit|stop)|exit\s*after|after)\s*(?:of|=|:|for|is)?\s*(\d+)\s*"
                        r"(?:candles?|bars?|days?|sessions?|periods?)?")),
    ("hold", re.compile(r"(\d+)\s*(?:candles?|bars?|days?|sessions?)(?:\s*(?:holding|hold|time\s*exit))?")),
    ("trail", re.compile(r"(?:trailing\s*(?:sl|stop(?:\s*loss)?)|tsl|trail(?:ing)?)\s*(?:of|=|:|at|is)?\s*" + _N + _UNIT)),
    ("tp", re.compile(r"(?:tp|take\s*profit|target|profit\s*target|t/p)\s*(?:of|=|:|at|is)?\s*" + _N + _UNIT)),
    ("sl", re.compile(r"(?:sl|stop\s*loss|stoploss|stop|s/l)\s*(?:of|=|:|at|is)?\s*" + _N + _UNIT)),
]


def _match_rule(part: str):
    p = re.sub(r"\s+", " ", part.strip().lower()).strip(" .")
    for kind, rx in _RULES:
        m = rx.fullmatch(p)
        if m:
            if kind != "hold" and m.group(2) and m.group(2).startswith(("p", "P")) and m.group(2) not in ("percent", "pct"):
                raise QueryError(f"'{part.strip()}': only percentage tp/sl are supported (e.g. 'tp = 4%').")
            return kind, float(m.group(1))
    return None


def _split_groups(text: str, and_seps: str = r"\band\b|&|,|;") -> List[List[str]]:
    groups = []
    for g in re.split(r"\bor\b|\|", text):
        parts = [p.strip() for p in re.split(and_seps, g) if p.strip()]
        if parts:
            groups.append(parts)
    return groups


def parse_query(query: str) -> Strategy:
    q = re.sub(r"\s+", " ", query.strip().lower().replace("\n", " "))
    if not q:
        raise QueryError("Empty query")
    q = q.replace("×", "*")
    kw = list(re.finditer(r"\b(entry|enter|entries|exit|exits)\b", q))
    if not kw or not any(k.group(1).startswith("ent") for k in kw):
        raise QueryError("Query needs an 'entry: ...' part and an 'exit: ...' part.")
    segs = {"entry": [], "exit": []}
    for i, k in enumerate(kw):
        end = kw[i + 1].start() if i + 1 < len(kw) else len(q)
        body = re.sub(r"^[\s:\-=]*(?:when|if|on)?\s*", "", q[k.end():end])
        segs["entry" if k.group(1).startswith("ent") else "exit"].append(body.strip(" ,;."))
    entry_txt, exit_txt = " ".join(segs["entry"]).strip(), " ".join(segs["exit"]).strip()
    if not entry_txt:
        raise QueryError("Entry condition is empty.")
    if not exit_txt:
        raise QueryError("Exit condition is empty (use e.g. 'exit: holding period = 5 candles').")

    # direction
    direction = 1
    dm = re.search(r"\b(?:(?:and\s+|then\s+)?go\s+|then\s+|and\s+)?(long|buy|short|sell)\b", entry_txt)
    if dm:
        direction = 1 if dm.group(1) in ("long", "buy") else -1
        entry_txt = (entry_txt[:dm.start()] + " " + entry_txt[dm.end():]).strip(" ,;.")
        entry_txt = re.sub(r"\b(and|or|then)\s*$", "", entry_txt).strip(" ,;.")
    if not entry_txt:
        raise QueryError("Entry condition is empty.")

    st = Strategy(direction=direction, entry=[], source=query.strip())
    for g in _split_groups(entry_txt):
        st.entry.append([parse_condition(p) for p in g])

    # exit: split on or , ; first, then on 'and' inside each clause
    for clause in re.split(r"\bor\b|\||[,;]", exit_txt):
        if not clause.strip():
            continue
        conds = []
        for part in re.split(r"\band\b|&", clause):
            if not part.strip():
                continue
            r = _match_rule(part)
            if r is None:
                conds.append(parse_condition(part))
                continue
            kind, v = r
            if v == 0:
                st.notes.append(f"'{part.strip()}' = 0 is treated as disabled.")
                continue
            if kind == "hold":
                if v != int(v) or v < 1:
                    raise QueryError("Holding period must be a whole number of candles >= 1")
                st.hold_bars = int(v) if st.hold_bars is None else min(st.hold_bars, int(v))
            else:
                attr = {"tp": "tp_pct", "sl": "sl_pct", "trail": "trail_pct"}[kind]
                setattr(st, attr, v)
        if conds:
            st.exit_conds.append(conds)

    if not (st.exit_conds or st.hold_bars or st.tp_pct or st.sl_pct or st.trail_pct):
        raise QueryError("No usable exit rule found.")
    return st
