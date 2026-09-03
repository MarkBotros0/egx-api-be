"""
The News tab: one HTTP GET per symbol, nothing persisted.

No snapshot table here, deliberately. pe_data and risk_snapshot are cached
because their upstream refuses half the universe at ~6s each; this source
returns 24 symbols in 1.30s and does not refuse. See
docs/superpowers/specs/2026-09-03-news-tab-design.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.index_membership import symbols_in_index


def test_symbols_in_index_returns_members_of_that_tier_only():
    members = symbols_in_index("EGX30")
    assert members, "EGX30 must have members in data/egx_tickers.json"
    assert members == sorted(members), "sorted, so a cache key built from it is stable"
    assert all(s == s.upper() for s in members)
    # The tier is real and disjoint from the others.
    assert set(members).isdisjoint(set(symbols_in_index("NILEX")))


def test_symbols_in_index_is_empty_for_an_unknown_tier():
    """
    Unknown must be empty, never everything. A caller that fans out over the
    result would otherwise hit the whole universe on a typo.
    """
    assert symbols_in_index("EGX42") == []
    assert symbols_in_index("") == []
