"""
No symbol-level fetch may assume a market's primary exchange: ~70% of the
S&P 500 trades on NYSE, and asking NASDAQ for KO waits ~5.5s for nothing.
Only the market registry and the resolver may name `primary_exchange`.
"""
import pathlib
import re


ALLOWED = {"app/core/markets.py", "app/core/universe.py"}


def test_primary_exchange_only_in_registry_and_resolver():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for p in (root / "app").rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if rel in ALLOWED or "/vendor/" in rel:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\bprimary_exchange\b", line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert offenders == [], "symbol fetches must use universe.resolve_exchange:\n" + "\n".join(offenders)
