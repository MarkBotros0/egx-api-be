"""
Arab Finance as a second news source: parse its RSS, tag headlines with the
symbols they name, and merge into /api/news without displacing the wires.

No network: the feed is fixture XML shaped exactly like the live one
(measured 2026-10-11 — guid, link, title, description, a10:updated).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.arabfinance import (
    CATEGORIES,
    PROVIDER,
    fetch_stories,
    merge_duplicates,
    parse_feed,
)
from app.core.news_fetch import NEWS_ITEM_FIELDS, build_feed, feed_status
from app.core.constants import NEWS_MAX_UNTAGGED_ITEMS
from app.core.news_tagging import derive_aliases, tag_symbols


def _rss(*items: tuple[str, str, str]) -> bytes:
    """items: (guid, title, a10:updated)."""
    body = "".join(
        f'<item><guid isPermaLink="false">{g}</guid>'
        f"<link>http://arabfinance.com/en/news/newdetails/{g}</link>"
        f"<title>{t}</title><description>{t} — the article body</description>"
        f"<a10:updated>{u}</a10:updated></item>"
        for g, t, u in items
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<rss xmlns:a10="http://www.w3.org/2005/Atom" version="2.0"><channel>'
        f"<title>Arab Finance RSS Title</title>{body}</channel></rss>"
    ).encode("utf-8")


# --- parsing ---------------------------------------------------------------

def test_parse_feed_shapes_a_news_item_with_a_utc_timestamp_and_https_link():
    xml = _rss(("48489", "Telecom Egypt&#8217;s profits leap 46.7% YoY", "2026-10-10T12:16:00+03:00"))
    [story] = parse_feed(xml, category=6)

    assert story["id"] == "af:48489"
    assert story["title"] == "Telecom Egypt’s profits leap 46.7% YoY"
    assert story["provider"] == PROVIDER == "arab-finance"
    assert story["published_at"] == "2026-10-10T09:16:00Z"
    assert story["url"] == "https://www.arabfinance.com/en/news/newdetails/48489"
    assert story["symbols"] == ["ETEL"]
    assert story["market_news"] is True


def test_parse_feed_never_carries_the_description():
    """Headlines and links only — publishers' copy is never reproduced."""
    [story] = parse_feed(_rss(("1", "Fawry posts profit", "2026-10-10T12:00:00+03:00")), category=2)
    assert "description" not in story
    assert all("article body" not in str(v) for v in story.values())


def test_rows_without_an_id_title_or_offset_date_are_dropped_not_half_kept():
    xml = _rss(
        ("", "No id", "2026-10-10T12:00:00+03:00"),
        ("2", "", "2026-10-10T12:00:00+03:00"),
        ("3", "Bad date", "yesterday"),
        ("4", "Naive date is a guess about Cairo vs UTC", "2026-10-10T12:00:00"),
        ("5", "Kept", "2026-10-10T12:00:00+03:00"),
    )
    assert [s["id"] for s in parse_feed(xml, category=1)] == ["af:5"]


def test_an_untagged_story_is_market_news_only_from_the_macroeconomy_feed():
    xml = _rss(("7", "Egypt’s annual inflation up to 12.8%", "2026-10-10T12:00:00+03:00"))
    assert parse_feed(xml, category=1)[0]["market_news"] is True
    assert parse_feed(xml, category=10)[0]["market_news"] is False  # Education


def test_xml_entity_tricks_are_refused():
    """A third party's XML: entity expansion must raise, never expand."""
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaaaaaaaa">'
        b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><rss><channel>'
        b"<item><guid>1</guid><title>&b;</title></item></channel></rss>"
    )
    with pytest.raises(Exception):
        parse_feed(bomb, category=1)


def test_a_story_in_two_feeds_is_one_row_and_market_news_if_either_copy_says_so():
    xml = _rss(("9", "S&amp;P keeps Egypt’s ratings", "2026-10-10T12:00:00+03:00"))
    rows = parse_feed(xml, category=2) + parse_feed(xml, category=1)
    [merged] = merge_duplicates(rows)
    assert merged["market_news"] is True


def test_fetch_stories_survives_a_failing_feed_and_counts_the_good_ones():
    good = _rss(("1", "Fawry posts profit", "2026-10-10T12:00:00+03:00"))

    def fetcher(category):
        if category == 3:
            raise OSError("boom")
        if category == 4:
            return b"<html>not a feed</html"
        return good

    stories, ok = fetch_stories(fetcher=fetcher)
    assert ok == len(CATEGORIES) - 2
    assert [s["id"] for s in stories] == ["af:1"]  # same story in eight feeds -> one


# --- tagging ---------------------------------------------------------------

@pytest.mark.parametrize(
    "title, expected",
    [
        ("CIB commits up to EGP 2B to MNT-Halan's international offering", ["COMI"]),
        ("Commercial International Bank to Invest EGP 2 Billion", ["COMI"]),
        ("Telecom Egypt’s consolidated profits leap 46.7% YoY", ["ETEL"]),
        ("Fawry's revenue up 30%", ["FWRY"]),
        ("Abu Qir Fertilizers and Chemical Industries posts profit", ["ABUK"]),
        ("Talaat Moustafa Group Holding sells units", ["TMGH"]),
        # The longest match wins: CIB Holdings is CICH, not also COMI.
        ("CIB Holdings announces a board change", ["CICH"]),
        # Whole words only.
        ("CIBC reports quarterly results", []),
        # An alias two ticker rows share names neither.
        ("Global Telecom Holding files results", []),
        # A trimmed name never shrinks to one ordinary word.
        ("Eastern Europe markets fall", []),
        ("S&P keeps Egypt’s ‘B/B’ ratings with stable outlook", []),
        ("Fawry and CIB sign a payments deal", ["COMI", "FWRY"]),
    ],
)
def test_tag_symbols(title, expected):
    assert tag_symbols(title) == expected


def test_derive_aliases_keeps_capital_abbreviations_and_stops_trimming_at_two_words():
    phrases, abbreviations = derive_aliases("Commercial International Bank (CIB)")
    assert phrases == {"Commercial International Bank"} and abbreviations == {"CIB"}

    phrases, _ = derive_aliases("EFG Hermes Holding")
    assert phrases == {"EFG Hermes Holding", "EFG Hermes"}

    phrases, _ = derive_aliases("Eastern Company")
    assert phrases == {"Eastern Company"}

    # Lower-case parentheticals are descriptions, not names.
    _, abbreviations = derive_aliases("Mantrac Group (CAT dealer)")
    assert abbreviations == set()


# --- merging into the feed -------------------------------------------------

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


def _story(story_id, symbols, days_ago, market_news=None):
    when = (NOW - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")
    return {
        "id": story_id,
        "title": f"headline {story_id}",
        "provider": PROVIDER,
        "published_at": when,
        "url": f"https://www.arabfinance.com/en/news/newdetails/{story_id}",
        "symbols": symbols,
        "market_news": bool(symbols) if market_news is None else market_news,
    }


def test_a_tagged_story_lands_on_your_stocks_and_carries_only_the_six_fields():
    feed = build_feed({}, yours=["COMI"], now=NOW, extra=[_story("af:1", ["COMI"], 1)])
    [item] = feed["your_stocks"]
    assert set(item) == NEWS_ITEM_FIELDS
    assert feed["coverage"]["symbols_with_news"] == 1


def test_a_story_tagged_with_someone_elses_stock_is_market_news():
    feed = build_feed({}, yours=["COMI"], now=NOW, extra=[_story("af:2", ["FWRY"], 1)])
    assert feed["your_stocks"] == [] and [i["id"] for i in feed["market"]] == ["af:2"]


def test_non_market_and_stale_stories_are_dropped():
    extra = [
        _story("af:edu", [], 1, market_news=False),
        _story("af:old", ["COMI"], 45),
    ]
    feed = build_feed({}, yours=["COMI"], now=NOW, extra=extra)
    assert feed["your_stocks"] == [] and feed["market"] == []


def test_untagged_macro_stories_are_capped_newest_first():
    extra = [_story(f"af:m{i}", [], i * 0.1, market_news=True) for i in range(40)]
    feed = build_feed({}, yours=[], now=NOW, extra=extra)
    ids = [i["id"] for i in feed["market"]]
    assert len(ids) == NEWS_MAX_UNTAGGED_ITEMS
    assert ids[0] == "af:m0"


def test_the_feed_is_not_unavailable_while_the_second_source_answers():
    assert feed_status({"COMI": []}, ["COMI"], extra_count=3) == "ok"
    assert feed_status({"COMI": []}, ["COMI"], extra_count=0) == "unavailable"
