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
