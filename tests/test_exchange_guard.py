"""
No symbol-level fetch may assume a market's primary exchange: ~70% of the
S&P 500 trades on NYSE, and asking NASDAQ for KO waits ~5.5s for nothing.
Only the market registry and the resolver may name `primary_exchange`, and
`exchanges[0]` (the same thing spelled differently) only at the reviewed
sites in ALLOWED_FIRST_EXCHANGE.
"""
import pathlib
import re


ALLOWED = {"app/core/markets.py", "app/core/universe.py"}

# (file, exact stripped line) -> why it is safe. Anything else naming
# `exchanges[0]` fails the build; add a site here only after checking it.
ALLOWED_FIRST_EXCHANGE = {
    # EG-gated multi-symbol branches: EG has exactly one exchange (EGX).
    ("app/routers/compare.py",
     "exchange = market.exchanges[0] if market.code == EG.code else ex_map"): "EG only",
    ("app/routers/historical.py",
     "exchange = market.exchanges[0] if market.code == EG.code else ex_map"): "EG only",
    ("app/routers/intraday.py",
     "exchange = market.exchanges[0] if market.code == EG.code else ex_map"): "EG only",
    # The scanner's exchange FILTER for a one-exchange market, not a symbol fetch.
    ("app/core/tradingview.py",
     'return {"left": "exchange", "operation": "equal", "right": market.exchanges[0]}'):
        "single-exchange filter",
    # Defaults spelled off the EG registry entry; US callers pass the resolved exchange.
    ("app/core/news_fetch.py",
     "def news_url(symbol: str, exchange: str = EG.exchanges[0]) -> str:"): "EG default",
    ("app/core/news_fetch.py",
     "exchange: str = EG.exchanges[0],"): "EG default",
}


def _app_lines():
    root = pathlib.Path(__file__).resolve().parents[1]
    for p in (root / "app").rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if "/vendor/" in rel:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            yield rel, i, line.strip()


def test_primary_exchange_only_in_registry_and_resolver():
    offenders = [f"{rel}:{i}: {line}" for rel, i, line in _app_lines()
                 if rel not in ALLOWED and re.search(r"\bprimary_exchange\b", line)]
    assert offenders == [], "symbol fetches must use universe.resolve_exchange:\n" + "\n".join(offenders)


def test_first_exchange_only_at_reviewed_sites():
    offenders = [f"{rel}:{i}: {line}" for rel, i, line in _app_lines()
                 if rel not in ALLOWED and re.search(r"\bexchanges\s*\[\s*0\s*\]", line)
                 and (rel, line) not in ALLOWED_FIRST_EXCHANGE]
    assert offenders == [], (
        "`exchanges[0]` is `primary_exchange` in disguise; resolve the symbol's exchange "
        "or add a reviewed entry to ALLOWED_FIRST_EXCHANGE:\n" + "\n".join(offenders))


def test_allow_list_has_no_dead_entries():
    present = {(rel, line) for rel, _, line in _app_lines()}
    assert set(ALLOWED_FIRST_EXCHANGE) <= present
