"""
Position-level return maths.

Shared by portfolio analysis (open positions) and the sales ledger (closed
ones) so the two can never annualize differently — a realized win and an
unrealized one must be judged against the T-bill by the same formula.
"""

from datetime import date, datetime
from typing import Optional

# Below this many days held, annualizing a position's return produces
# nonsense (a +5% week annualizes to five figures). The signal layer and the
# UI both suppress the number instead of showing it.
MIN_DAYS_FOR_ANNUALIZATION = 30


def days_between(start_date_str: str, end: date) -> int:
    """Calendar days from an ISO date string to `end`. 0 if unparseable."""
    try:
        d = datetime.strptime(start_date_str[:10], "%Y-%m-%d").date()
        return (end - d).days
    except Exception:
        return 0


def _as_date(value) -> Optional[date]:
    """
    Normalize to a plain `date`.

    datetime is checked FIRST because it subclasses date, and a pandas Timestamp
    subclasses datetime — the backtest walks a DatetimeIndex, so scoring dates
    arrive as Timestamps while the rate steps parse to plain dates. Comparing
    the two raises TypeError in pandas, and inside `score_symbol` that lands in
    a bare `except: continue`, so every symbol would score zero rows and the run
    would report an empty panel with nothing logged.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def annualized_cash_rate_pct(rate_steps, start_date, end_date) -> Optional[float]:
    """
    What risk-free cash actually returned over ONE window, as an annual rate.

    `rate_steps` is [(effective_date, annual_rate_pct), ...] — a step function,
    which is what a policy rate is: it holds until the next MPC decision.

    WHY THIS IS NOT JUST "THE RATE TODAY". The CBE has been anywhere from 8.25%
    to 27.25% over the history this app covers, so one scalar is wrong for every
    era but the current one — and it errs in BOTH directions, which is the part
    worth knowing. Measured against the real EGINTR series, holding a +25% gain
    for calendar 2024 faced a true hurdle of 26.07% and LOST to cash, where the
    flat 19% called it a win; a +18% gain over 2019 faced 14.69% and BEAT cash,
    where the flat 19% called it a loss. `beat_t_bill_count` on the Winnings
    card was therefore not simply flattering — it was era-dependent noise.

    COMPOUND, DO NOT AVERAGE THE LEVELS. Cash earns each rate for as long as it
    was in force, so the segments multiply. A window that spent eleven months at
    27.25% and one at 19% is near 27.25%, not near their midpoint. The result is
    annualized back out so it is directly comparable to a position's own
    `annualized_return`, and a flat rate therefore returns itself exactly — the
    exponents cancel — which is the property everything else is checked against.

    The earliest known rate is carried BACKWARDS to cover a window that starts
    before the history does. The alternative is falling back to today's rate,
    which is the very bug this function exists to fix; the nearest historical
    rate is always the better estimate.

    Returns None when there is no history or the window is empty, so the caller
    can fall back deliberately rather than have a rate invented for it.
    """
    start, end = _as_date(start_date), _as_date(end_date)
    if start is None or end is None:
        return None
    total_days = (end - start).days
    if total_days <= 0:
        return None

    steps = sorted(
        (d, float(r))
        for d, r in ((_as_date(s), r) for s, r in (rate_steps or []))
        if d is not None
    )
    if not steps:
        return None

    # Carry the earliest rate back over any uncovered prefix.
    steps[0] = (min(steps[0][0], start), steps[0][1])

    growth = 1.0
    for i, (effective, rate) in enumerate(steps):
        seg_start = max(effective, start)
        seg_end = min(steps[i + 1][0], end) if i + 1 < len(steps) else end
        days = (seg_end - seg_start).days
        if days > 0:
            growth *= (1 + rate / 100) ** (days / 365)

    return (growth ** (365 / total_days) - 1) * 100


def annualized_return(total_return_pct: float, days_held: int) -> Optional[float]:
    """
    Annualize a POSITION's return over calendar days held.

    Note this is a different quantity from indicators.annualized_return(),
    which annualizes the STOCK's market return over trading bars. The two
    answer different questions ("how did my purchase do" vs "how did the
    stock do") and must not be compared to each other.

    Returns None when the position was held too briefly to annualize
    meaningfully.
    """
    if days_held < MIN_DAYS_FOR_ANNUALIZATION:
        return None
    base = 1 + total_return_pct / 100
    if base <= 0:
        return -100.0
    return (base ** (365 / days_held) - 1) * 100
