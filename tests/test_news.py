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
from app.core.news_fetch import (
    NEWS_ITEM_FIELDS,
    dedupe_stories,
    is_recent,
    normalize_item,
    select_news_symbols,
)


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


_RAW = {
    "id": "tag:reuters.com,2026:newsml_FWN44P224:0",
    "title": "Sodic Signs Medium-Term Facility With CIB",
    "provider": "reuters",
    "published": 1756598400,
    "storyPath": "/news/reuters.com,2026:newsml_FWN44P224:0-sodic-signs/",
    "relatedSymbols": [{"symbol": "EGX:OCDI"}, {"symbol": "EGX:COMI"}],
    "urgency": 2,
    "source": "Reuters",
    "sourceLogoId": "reuters",
    "permission": "provider",
}


def test_normalize_keeps_six_fields_and_absolute_url():
    item = normalize_item(_RAW, "COMI")
    assert set(item) == set(NEWS_ITEM_FIELDS)
    assert item["title"] == "Sodic Signs Medium-Term Facility With CIB"
    assert item["provider"] == "reuters"
    assert item["url"].startswith("https://www.tradingview.com/news/")
    assert item["published_at"].endswith("Z")
    assert item["symbols"] == ["COMI", "OCDI"]


def test_normalize_never_carries_article_body():
    """
    Stories are Reuters/Zawya/LSE copy. The app links out; it must not store or
    render body text. This is a copyright constraint, not a scope preference.
    """
    raw = dict(_RAW, body="Full article text...", shortDescription="A summary")
    item = normalize_item(raw, "COMI")
    for banned in ("body", "shortDescription", "content", "summary", "text"):
        assert banned not in item
    assert "body" not in NEWS_ITEM_FIELDS


def test_normalize_rejects_unusable_rows_rather_than_returning_partials():
    """
    A row with no timestamp cannot be placed in the recency window, and one
    with no storyPath has nowhere to link. Half a story is worse than none.
    """
    assert normalize_item(dict(_RAW, published=None), "COMI") is None
    assert normalize_item(dict(_RAW, published="not-a-number"), "COMI") is None
    assert normalize_item(dict(_RAW, storyPath=None), "COMI") is None
    assert normalize_item(dict(_RAW, title="   "), "COMI") is None
    assert normalize_item({}, "COMI") is None
    assert normalize_item("not a dict", "COMI") is None


def test_normalize_ignores_non_egx_related_symbols():
    raw = dict(_RAW, relatedSymbols=[{"symbol": "NASDAQ:TSLA"}, {"symbol": "EGX:COMI"}])
    assert normalize_item(raw, "COMI")["symbols"] == ["COMI"]


def test_dedupe_merges_tags_instead_of_dropping_the_second_copy():
    """
    One story is returned by BOTH EGX:OCDI and EGX:COMI. Keeping the first and
    discarding the second would lose that it concerns two of the user's
    holdings. Same distinction dedupe_symbol_signals draws in
    portfolio_analysis: dedupe only where the duplicate is truly redundant.

    Each half below carries a single, NON-overlapping symbol, so the merged
    union {"COMI", "OCDI"} is reachable only by dedupe_stories actually
    merging — neither `a` nor `b` alone can produce it. (A prior version of
    this test built both from `_RAW`, whose relatedSymbols already lists both
    tickers; normalize_item on either one alone already yielded the union, so
    a keep-first-drop-second regression would have passed unnoticed.)
    """
    a = normalize_item(dict(_RAW, relatedSymbols=[{"symbol": "EGX:COMI"}]), "COMI")
    b = normalize_item(dict(_RAW, relatedSymbols=[{"symbol": "EGX:OCDI"}]), "OCDI")
    out = dedupe_stories([a, b])
    assert len(out) == 1
    assert out[0]["symbols"] == ["COMI", "OCDI"]


def test_dedupe_returns_newest_first():
    old = normalize_item(dict(_RAW, id="old", published=1700000000), "COMI")
    new = normalize_item(dict(_RAW, id="new", published=1756598400), "COMI")
    assert [i["id"] for i in dedupe_stories([old, new])] == ["new", "old"]


def test_recency_window_boundary():
    now = datetime(2026, 9, 3, tzinfo=timezone.utc)

    def at(days_ago):
        ts = int((now - timedelta(days=days_ago)).timestamp())
        return normalize_item(dict(_RAW, published=ts), "COMI")

    assert is_recent(at(0), now, 30) is True
    assert is_recent(at(29), now, 30) is True
    assert is_recent(at(30), now, 30) is True
    assert is_recent(at(31), now, 30) is False
    # ACGC's real newest story was 275 days old. It is not news.
    assert is_recent(at(275), now, 30) is False


def test_holdings_and_watchlist_merge_into_yours_without_duplicates():
    yours, market, dropped = select_news_symbols(["COMI", "SWDY"], ["SWDY", "ETEL"], ["ABUK"])
    assert yours == ["COMI", "SWDY", "ETEL"], "holdings order first, watchlist appended"
    assert market == ["ABUK"]
    assert dropped == []


def test_a_symbol_you_own_is_never_also_a_market_symbol():
    """Otherwise it is fetched twice and renders in both sections."""
    yours, market, dropped = select_news_symbols(["COMI"], [], ["COMI", "ABUK"])
    assert yours == ["COMI"]
    assert market == ["ABUK"]
    assert dropped == []


def test_holdings_win_the_budget_when_the_cap_binds():
    """
    The cap exists to bound the fan-out. When it binds it must cost the user
    the LEAST relevant symbols — EGX30 names they never asked about — not
    their own holdings.
    """
    yours, market, dropped = select_news_symbols(["A", "B", "C"], ["D"], ["E", "F"], cap=5)
    assert yours == ["A", "B", "C", "D"]
    assert market == ["E"]
    assert dropped == []


def test_the_cap_can_exhaust_the_market_half_entirely():
    yours, market, dropped = select_news_symbols(["A", "B"], [], ["C", "D"], cap=2)
    assert yours == ["A", "B"]
    assert market == []
    assert dropped == []


def test_symbols_are_uppercased_and_blanks_dropped():
    yours, market, dropped = select_news_symbols(["comi", " ", ""], ["etel"], ["abuk"])
    assert yours == ["COMI", "ETEL"]
    assert market == ["ABUK"]
    assert dropped == []


def test_combined_overflows_but_holdings_fit():
    """
    When combined holdings+watchlist overflows the cap but all holdings fit,
    watchlist symbols are dropped and reported.
    """
    yours, market, dropped = select_news_symbols(["A", "B", "C"], ["D", "E"], ["F"], cap=4)
    assert yours == ["A", "B", "C", "D"], "all holdings survive, watchlist D takes the last slot"
    assert market == [], "no budget left for market symbols"
    assert dropped == ["E"], "watchlist E is dropped and reported"


def test_holdings_alone_overflow_cap():
    """
    When holdings alone exceed the cap, something must be dropped and reported.
    This is unavoidable: you cannot fetch 45 symbols under a cap of 40. The
    only requirement is that the dropped holding is REPORTED, not hidden.
    """
    yours, market, dropped = select_news_symbols(["A", "B", "C"], [], ["F"], cap=2)
    assert yours == ["A", "B"], "cap limits yours, so C does not fit"
    assert market == [], "no budget for market"
    assert dropped == ["C"], "C is reported as dropped, not silently lost"
